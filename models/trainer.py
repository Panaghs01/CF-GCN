import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import rasterio
import utils
from models.addGCNnetworks import *

import torch
import torch.optim as optim

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses

from misc.logger_tool import Logger, Timer

from utils import de_norm

def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces all BatchNorm layers in a module with GroupNorm layers.
    
    Args:
        module (nn.Module): The module to traverse and modify.
        num_groups (int): The number of groups for GroupNorm.
                          A common choice is 32.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            # Batch norm 1D is usually applied to sequences/features
            # This is tricky because BatchNorm1d only takes 'num_features'
            # GroupNorm is generally for NCHW data. For sequence data (NCL),
            # this replacement might be conceptually problematic but mathematically possible.
            # However, for robustness, it's often better to focus on 2D/3D.
            # For 1D, if it's acting on the C dimension, we can replace it.
            # Assuming C is the only dimension after batch:
            num_channels = child.num_features
            new_module = nn.GroupNorm(num_groups=min(num_groups, num_channels), 
                                     num_channels=num_channels, 
                                     eps=child.eps, 
                                     affine=child.affine)
            setattr(module, name, new_module)

        elif isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm3d):
            # These are the most common uses of BatchNorm.
            num_channels = child.num_features
            new_module = nn.GroupNorm(num_groups=min(num_groups, num_channels), # Ensure num_groups <= num_channels
                                     num_channels=num_channels, 
                                     eps=child.eps, 
                                     affine=child.affine)
            # Copy momentum/track_running_stats properties for completeness if needed, 
            # though GroupNorm doesn't use them directly.
            
            # Transfer learned parameters (gamma and beta)
            if child.affine:
                 # GroupNorm params are weight/bias, BatchNorm's are weight/bias
                if child.weight is not None:
                    new_module.weight.data = child.weight.data
                if child.bias is not None:
                    new_module.bias.data = child.bias.data
            
            setattr(module, name, new_module)
        
        else:
            # Recursively apply the replacement to submodules
            replace_bn_with_gn(child, num_groups)

class CDTrainer():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders
        self.dataset = args.dataset
        self.data_name = args.data_name
        self.n_class = args.n_class
        self.accumlation_steps = args.accumulation_steps
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)
        replace_bn_with_gn(self.net_G)
        self.net_G.to(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == 'sgd':
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                        momentum=0.9,
                                        weight_decay=5e-4)

        elif args.optimizer == 'adam':
            self.optimizer_G = optim.AdamW(
                self.net_G.parameters(),
                lr=self.lr,           # e.g. 1e-3
                betas=(0.9, 0.999)    
            )

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define the loss functions
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'dice':
            self._pxl_loss = losses.dice_loss
        elif args.loss == 'ce_dice':
            self._pxl_loss = losses.CE_with_Dice
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):

        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255

        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)


        if np.mod(self.batch_id, 4000) == 1:
            #vis_input = utils.make_numpy_grid(self.batch['A']) #de_norm
            #vis_input2 = utils.make_numpy_grid(self.batch['B'])

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])

            # Convert all to 8-channel format
            target_shape = vis_pred.shape[:2]

            """ if self.data_name == 'SenForFlood':
                # Ensure vis_gt has 8 channels using np.stack([vis_gt]*8, axis=2)
                if vis_gt.shape[2] < 8:
                    vis_gt = np.stack([vis_gt[..., 0]] * 8, axis=2)
                # Ensure vis_pred has 8 channels using np.stack([vis_pred]*8, axis=2)
                if vis_pred.shape[2] < 8:
                    vis_pred = np.stack([vis_pred[..., 0]] * 8, axis=2) """

            vis_gt = resize(vis_gt, target_shape)
            #vis_pred = resize(vis_pred, target_shape)
            #print(f"visgt: {vis_gt.shape}, vispred: {vis_pred.shape}, target: {target_shape}")
            
            #print(f"\n\n\nattempting to conacat {vis_pred.shape, vis_gt.shape}")
            vis = np.concatenate([vis_pred[:,:,0], vis_gt[:,:,0]], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=255.0)

            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            #print(vis.shape,type(vis))


            """ if self.data_name == 'SenForFlood':
                with rasterio.open(file_name, 'w', driver='GTiff', height=vis.shape[0],\
                                    width=vis.shape[1], count=vis.shape[2], dtype=vis.dtype) as dst:
                    for i in range(8):
                        dst.write((vis[:,:,i]).astype(np.uint8), i+1)
            else: """
            plt.imsave(file_name, vis)
                    

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        if self.dataset == 'CDDataset_fusion':
            img_in1 = batch['A'].to(self.device)
            img_in2 = batch['B'].to(self.device)
            img_in3 = batch['C'].to(self.device)
            img_in4 = batch['D'].to(self.device)
            self.G_pred = self.net_G(img_in1, img_in2, img_in3, img_in4)
        else:
            img_in1 = batch['A'].to(self.device)
            img_in2 = batch['B'].to(self.device)
            self.G_pred = self.net_G(img_in1, img_in2)
        #print(self.G_pred.min(), self.G_pred.max())


    def _backward_G(self):
        gt = self.batch['L'].to(self.device).long()
        self.G_loss = self._pxl_loss(self.G_pred, gt, ignore_index=255)
        self.G_loss.backward()


    def train_models(self):
        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            #print(self.dataloaders['train'].__len__())
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                
                self._forward_pass(batch)
                #print(f"Prediction min/max: {self.G_pred.min()}, {self.G_pred.max()}\n iomg : {batch['A']}")
                # update G
                #print("mean-std:",batch['A'].mean(), batch['A'].std())
                self._backward_G()
                if self.accumlation_steps > 0:
                    if (self.batch_id + 1) % self.accumlation_steps == 0:
                        self.optimizer_G.step()
                        self.optimizer_G.zero_grad()
                else:
                    self.optimizer_G.step()
                    self.optimizer_G.zero_grad()


                self._collect_running_batch_states()
                self._timer_update()

                del batch
            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()
            

            torch.cuda.empty_cache()

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()

def to_rgb(arr):
    # If arr has 8 channels, select first 3 for visualization
    if arr.shape[2] == 8:
        return arr[..., :3]
    # If arr has 1 channel, repeat to make 3 channels
    elif arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    # If arr has 2 channels, pad to 3 channels
    elif arr.shape[2] == 2:
        return np.concatenate([arr, np.zeros_like(arr[..., :1])], axis=2)
    # If arr has 3 channels, do nothing
    return arr

def resize(arr, shape):
    arr = arr.astype(np.uint8)
    arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    #print(f"array size {arr[:,:,0,:].shape}, target shape {shape}")
    return cv2.resize(arr, (shape[1], shape[0]))

