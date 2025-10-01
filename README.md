# CF-GCN fork for multispectral image processing
Remodeled the original module to be able to handle 8-channel data and trained on 2 datasets. The purpose of this work is to effectively identify floods and create meaningful flood masks.
- OMBRIA (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9723593)
- Sen4Floods (https://isprs-archives.copernicus.org/articles/XLVIII-M-7-2025/97/2025/isprs-archives-XLVIII-M-7-2025-97-2025.pdf)

### Step 1 download all the data needed for the flood detection
Sen4Floods dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZQCODX
OMBRIA dataset: https://github.com/geodrak/OMBRIA

### Step 2 move the datasets inside the raw_data directory
After moving them inside this directory, run the following scripts
`format2.sh` to format the SenForFlood dataset, `format_masks.sh` to format the masks in the folders and `split_dat.sh` to create a split in .txt form to insert inside the list directory.
Folders A and B should now exist with before being A and after being B

### Step 3 install the requirements
To run the code first install the requirements via
```
pip install requirements.txt
```
### Step 4 Run the model
To run the model check the `main_cd.py` file for the available model parameters and run setup accordingly
A basic way to train the model is to run the following command:
```
python main_cd.py --gpu_ids 0 --data_name SenForFlood --batch_size 16 --split_val test
 --num_workers 8 --checkpoint_root checkpoints --split train --optimizer adam
```

