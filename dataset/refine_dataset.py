import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


train_list = []
valid_list = ["115","117","119","121","122","135","137","139","141","143","145","147"]
test_list = ["001","002","003","004","005","006","007","008","009","010","011","012","013","124","126","128","130","132"]

for i in range(1,212):
    one = str(i).zfill(3)
    if one not in test_list and one not in valid_list:
        train_list.append(one)

class RefineDataset(Dataset):
    def __init__(self, seq_len, stride, mode=0):

        self.motion_dir = "./data/motion"

        self.motions = []
        self.motion_idx = []
        self.filenames = []
        self.seq_len = seq_len

        if mode == 0 :
            target_list = train_list
        elif mode == 1 :
            target_list = valid_list
        else :  
            target_list = test_list

        idx = 0
        for i, filename in enumerate(sorted(glob.glob(os.path.join(self.motion_dir,"*.npy")))):
            basename = os.path.basename(filename).split(".")[0]
            if basename in target_list :
                motion = np.load(filename)
                self.filenames.append(basename)
                self.motions.append(motion)
                for j in range(0, motion.shape[0]-seq_len+1, stride):
                    self.motion_idx.append([idx, j])
                idx = idx+1

    def __len__(self):
        return len(self.motion_idx)
    
    def __getitem__(self, idx) :
        motion_idx, frame_idx = self.motion_idx[idx]

        motion = self.motions[motion_idx][frame_idx:frame_idx+self.seq_len]

        input = torch.from_numpy(motion[:,3:]) # Relative Angel Axis 
        output = torch.from_numpy(motion[:,:3]) # Root Position
        output = output-output[:1] # Normalize root position based on root position of first frame

        filename = f"{self.filenames[motion_idx]}_{frame_idx:04d}"

        return {
            "input":input,
            "output":output,
            "filename":filename,
        }
        