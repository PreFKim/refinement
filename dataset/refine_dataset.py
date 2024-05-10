import torch
from torch.utils.data import Dataset

import numpy as np
import glob
import os

all_list = []
train_list = []
for i in range(1,212):
    all_list.append(str(i).zfill(3))
test_list = ["001","002","003","004","005","006","007","008","009","010","011","012","013","124","126","128","130","132"]
val_list = ["115","117","119","121","122","135","137","139","141","143","145","147"]
for one in all_list:
    if one not in test_list:
        if one not in val_list:
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
            target_list = val_list
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
        motion_idx, frameIdx = self.motion_idx[idx]

        motion = self.motions[motion_idx][frameIdx:frameIdx+self.seq_len]

        input = torch.from_numpy(motion[:,3:])
        output = torch.from_numpy(motion[:,:3])
        output = output-output[:1]
        filename = self.filenames[motion_idx]

        return {
            "input":input,
            "output":output,
            "filename":filename,
        }
        