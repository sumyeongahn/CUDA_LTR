# original code: https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/data_loader/imagenet_lt_data_loaders.py
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random

# from imbalance_data.cv_aug import *
# from imbalance_data.cutout import *
from aug.cuda import *
from aug.transforms import *
from aug.randaug import *

import torchvision.transforms as transforms


class LT_Dataset(Dataset):
    def __init__(self, root, txt, args, dataset, loss_type, use_randaug=False,
                 split='train', aug_prob = 0.5, upgrade=1, downgrade=1):
        self.img_path = []
        self.labels = []
        
        if split=='train':
            transform = get_transform(dataset, loss_type, split='train')
            self.transform = transform
            self.split = 'train'
        else:
            transform = get_transform(dataset, loss_type, split='valid')
            self.transform = transform
            self.split = 'valid'
            
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

        if use_randaug:
            self.aug_transform = transforms.Compose([RandAugment(2, 10)])
        else:
            self.aug_transform = transforms.Compose([])
        
        self.args = args
        self.aug_prob = aug_prob
        self.upgrade = upgrade
        self.downgrade = downgrade
        
        max_mag = 10
        max_ops = 10
        self.max_state = max(max_mag, max_ops) + 1
        self.min_state = 0
        
        states = torch.arange(self.max_state)
        if self.max_state == 1:
            self.ops = torch.tensor([0])
            self.mag = torch.tensor([0])
            
        elif max_mag > max_ops:
            self.ops = (states * max_ops / max_mag).ceil().int()
            self.mag = states.int()
        else:
            self.mag = (states * max_mag / max_ops).ceil().int()
            self.ops = states.int()
        
        print(f"Magnitude set = {self.mag}")
        print(f"Operation set = {self.ops}")
        
        self.curr_state = torch.zeros(len(self.targets))
        self.score_tmp = torch.zeros((len(self.targets), self.max_state))
        self.num_test = torch.zeros((len(self.targets), self.max_state))
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.labels)
    
    def sim_aug(self, img, state, type):
        if type == 'CUDA':
            return CUDA(img, self.mag[state], self.ops[state])
        else:
            return img

    def get_item(self, index, state, train=True):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        if train:
            if len(self.transform) == 1:
                img = self.transform[0][0](img)
                img = self.aug_transform(img)
                img = CUDA(img, self.mag[state], self.ops[state])
                img = self.transform[0][1](img)
                return img, label, index
            
            elif len(self.transform) == 2:
                img1 = self.transform[0][0](img)
                img1 = self.aug_transform(img1)
                img1 = CUDA(img1, self.mag[state], self.ops[state])
                img1 = self.transform[0][1](img1)

                img2 = self.transform[1][0](img)
                img2 = self.sim_aug(img2, state, self.args.sim_type)
                img2 = self.transform[1][1](img2)
                
                return (img1, img2), label, index
            
            elif len(self.transform) == 3:
                img1 = self.transform[0][0](img)
                img1 = self.aug_transform(img1)
                img1 = CUDA(img1, self.mag[state], self.ops[state])
                img1 = self.transform[0][1](img1)

                img2 = self.transform[1][0](img)
                img2 = self.sim_aug(img2, state, self.args.sim_type)
                img2 = self.transform[1][1](img2)
                
                img3 = self.transform[2][0](img)
                img3 = self.sim_aug(img3, state, self.args.sim_type)
                img3 = self.transform[2][1](img3)
                return (img1, img2, img3), label, index

        else:
            if self.split == 'valid':
                img = self.transform(img)
                return img, label, index
            else:
                img = self.transform[0][0](img)
                img = self.aug_transform(img)
                img = CUDA(img, self.mag[state], self.ops[state], rand=False)
                img = self.transform[0][1](img)
                return img, label, index 

    def __getitem__(self, index):
        if self.split == 'train':
            state = self.curr_state[index].int() if torch.rand(1) < self.aug_prob else 0
            img, target, index = self.get_item(index, state, train=True)
        else:
            img, target, index = self.get_item(index, None, train=False)
        return img, target, index

    def update_scores(self, correct, index, state):
        for s in np.unique(state):
            pos = np.where(state == s)
            score_result = np.bincount(index[pos], correct[pos], len(self.score_tmp))
            num_test_result = np.bincount(index[pos], np.ones(len(index))[pos], len(self.score_tmp))
            self.score_tmp[:, s] += score_result
            self.num_test[:, s] += num_test_result
        # score_result = np.bincount(index, correct, len(self.score_tmp))
        # num_test_result = np.bincount(index, np.ones(len(index)), len(self.score_tmp))
        # self.score_tmp += score_result
        # self.num_test += num_test_result

    def update(self):
        # Increase
        pos = torch.where((self.score_tmp == self.num_test) & (self.num_test != 0))
        self.curr_state[pos] += self.upgrade
        
        # Decrease
        pos = torch.where(self.score_tmp != self.num_test)
        self.curr_state[pos] -= self.downgrade
        
        self.curr_state = torch.clamp(self.curr_state, 0, self.max_state-1)
        self.score_tmp *= 0
        self.num_test *= 0

class test_loader(Dataset):
    def __init__(self, indices, state, cifar_dataset):
        self.indices = indices
        self.state = state
        self.dataset = cifar_dataset

    def __getitem__(self,idx):
        data, label, _ = self.dataset.get_item(self.indices[idx], self.state[idx], train=False)
        return data, label, self.indices[idx], self.state[idx]
    
    def __len__(self):
        return len(self.indices)