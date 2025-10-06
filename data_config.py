
class DataConfig:
    data_name = ""
    root_dir = ""
    mean = None
    std = None
    label_transform = None
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            # self.root_dir = '/home/wxs/lc/LEVIR_normal'
            self.root_dir = '/Users/liucong/Desktop/研究生/论文/变化检测/LEVIR_normal'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        elif data_name == 'WHU':
            self.root_dir = '/home/wxs/lc/WHU-CD-256'
        elif data_name == 'DSIFN':
            self.root_dir = '/home/wxs/lc/DSIFN-CD-256'
        elif data_name == 'OMBRIAS2':
            self.root_dir = 'raw_data/OMBRIA/OmbriaS2'
        elif data_name == 'OMBRIAS1':
            self.root_dir = 'raw_data/OMBRIA/OmbriaS1'
        elif data_name == 'SenForFlood':
            self.root_dir = 'raw_data/SenForFlood'
        else:
            raise TypeError('%s has not defined' % data_name)
        
        if data_name == 'SenForFlood':  # 6 channels
            self.mean = [3288.250685293216, 3142.6076760452397,\
                        3167.0665465272696, 3730.833737398937,\
                        2297.8053741934746, 1645.5760596861048]
            self.std = [2730.3334944919493, 2607.3674210507534,\
                        2781.12544738576, 2461.4640886942675,\
                        1373.1495343166732, 1068.134491107072]
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

