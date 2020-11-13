# -*- coding: utf-8 -*-
import os, json
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle

TRAIN_SIZE = 0.8
TRAIN_USED = 0.2
MAX_VAL_SIZE = 0.2

class HashTagDataset(Dataset):

    def __init__(self, data_path, split = 'all_train', labels_file = None, transforms = None):
        super(HashTagDataset, self).__init__()
        self.data_path = data_path # 
        self.transforms = transforms
        self.image_names, self.tag, self.targets = [], [], []
        

        #print(labels_file)
        if labels_file is not None: # config = None only for the sake of dirty hack in train.py
            self._parse_root_(labels_file)
        
        
        num_images = self.image_names.shape[0]    
        
        if split == "train":
            arr = np.arange(int(num_images * TRAIN_SIZE))
            idxs = np.random.permutation(arr)
            idxs = idxs[0:int(TRAIN_USED * TRAIN_SIZE * num_images)]
        elif split == "val": 
            arr = np.arange(int(num_images * TRAIN_SIZE), num_images)
            idxs = np.random.permutation(arr)
            idxs = idxs[0:int(MAX_VAL_SIZE * num_images)]
        elif split == "all_train":
            arr = np.arange(int(num_images * TRAIN_SIZE), num_images)
            idxs = np.random.permutation(arr)
    
        self.image_names = self.image_names[idxs]    
        self.tags = self.tags[idxs] 
        self.targets = self.targets[idxs]    
            
    def _parse_root_(self, labels_file):
        
        df = pd.read_csv(labels_file)
        #print(df.head)
        self.image_names = df.urls
        self.tags = df.base_tag.to_numpy()
        self.image_names = [os.path.join(self.data_path, tag, image_name)  for image_name, tag in zip(self.image_names, df.base_tag)]
        self.image_names = np.array(self.image_names)
        
        with open(os.path.join(self.data_path, "category_id.pkl"), 'rb') as f:
            self.category_id = pickle.load(f)
                
        self.targets = np.array([self.category_id[tag] for tag in df.base_tag])
        
        #print(self.image_names)
        #print(self.tags)
        #print(self.category_id)
        #print(self.targets)
        
        

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        sample = {}
        
        image_name = self.image_names[item]
    

        #image = cv2.imread(image_name)
        image = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.
        
        if self.transforms is not None:
            image = self.transforms(image)     
            
        sample["image"] = image
        
        target = np.zeros(22)
        target[self.targets[item]] = 1
        sample["target"] = target
        
        #print(image.shape)
        return sample
