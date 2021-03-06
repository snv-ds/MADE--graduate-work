# -*- coding: utf-8 -*-
import os, json
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle

TRAIN_SIZE = 0.8
TRAIN_USED = 1.000
MAX_VAL_SIZE = 0.2
MAX_ALL_SIZE = 1.0

NUM_LEV1_CATEGORIES = 16 #22
NUM_LEV2_CATEGORIES = 237 #113

class HashTagImageDataset(Dataset):

    def __init__(self, data_path, split = 'all_train', labels_file = None, transforms = None):
        super(HashTagImageDataset, self).__init__()
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
        elif split == "all":
            idxs = np.arange(int(num_images * MAX_ALL_SIZE))
        else:
            idxs = np.arange(num_images)
            
            
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
        
        with open(os.path.join(self.data_path, "lev2_category_id.pkl"), 'rb') as f:
            self.all_category_id = pickle.load(f)    
        
        self.targets = np.array([self.category_id.get(tag, 0) for tag in df.base_tag])
        
        #self.targets = np.array([self.category_id[tag] for tag in row['tags'].split(' ') if tag in self.category_id])
        
        #print(self.image_names)
        #print(self.tags)
        #print(self.category_id)
        #print(self.targets)
        
        

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        sample = {}
        
        image_name = self.image_names[item]
        sample["image_name"] = image_name

        #image = cv2.imread(image_name)
        #print(image_name)
        image = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.
        #print(image)
        
        if self.transforms is not None:
            image = self.transforms(image)   
            
        #print(image)    
        
        sample["image"] = image
        
        target = np.zeros(NUM_LEV1_CATEGORIES)
        target[self.targets[item]] = 1
        sample["target"] = target
        
        #print(image.shape)
        return sample




# данные на основе предрасчитанных фичей
class HashTagFeatureDataset(Dataset):
    
    def __init__(self, data_path, split = 'all_train', features_file = None, labels_file = None):
        super(HashTagFeatureDataset, self).__init__()
        self.data_path = data_path  
        self.tag, self.targets, self.masks  = [], [], []

        self.df = pd.read_csv(os.path.join(data_path, features_file))
        
        df_label = pd.read_csv(labels_file)
        df_label.drop_duplicates(inplace=True)
        
        df_label['img'] = df_label.apply(lambda x: os.path.join(data_path, x['base_tag'].strip(), x['urls'].strip()), axis = 1)
        df_label.reset_index(inplace=True)

        self.df = self.df.merge(df_label)
        self.df = self.df.sort_values('index').reset_index(drop=True)

        with open(os.path.join(self.data_path, "category_id.pkl"), 'rb') as f:
            self.category_id = pickle.load(f)
            
        
            
        with open(os.path.join(self.data_path, "lev2_category_id.pkl"), 'rb') as f:
            self.all_category_id = pickle.load(f)        
            
        with open(os.path.join(self.data_path, "lev1_lev2_category_id.pkl"), 'rb') as f:
            self.lev1_lev2_category_id = pickle.load(f)  
            
        self.num_lev1_categories = np.max([id for tag, id in self.category_id.items()]) + 1    
        self.num_lev2_categories = np.max([id for id1, d in self.lev1_lev2_category_id.items() for tag, id in d.items()]) + 1 - self.num_lev1_categories    
            

        find_indx = []
        for i, row in self.df.iterrows():    
            lev1_tags = [self.category_id[tag] for tag in row['tags'].split(' ') if tag in self.category_id]
            lev2_tags = []
            mask = []
            
            #print(lev1_tags, row['tags'])
            if len(lev1_tags) == 0 and split != "all": 
                continue

            for lev1_tag in lev1_tags:  
                # all_tags = [item.strip("'',") for item in row['tags'].strip('[]').split(' ')]
                all_tags = [item.strip(" ") for item in row['tags'].split(' ')]
                list_tag = [t_tag for t_tag in all_tags if t_tag in self.lev1_lev2_category_id[lev1_tag]]
                lev2_tags += [self.lev1_lev2_category_id[lev1_tag][tag] for tag in list_tag]
                mask += self.lev1_lev2_category_id[lev1_tag].values()
            
            if len(lev1_tags) > 0:
                self.targets.append(np.array(lev1_tags + lev2_tags))
                self.masks.append(mask)
            else:
                self.targets.append(np.array([0]))
                self.masks.append([0])
            find_indx.append(i)
        
        find_idxs = np.array(find_indx)  
        
        self.df = self.df.iloc[find_idxs, :]   
            
            
        #print(1/0)
        #self.targets = np.array()
        
        num_images = self.df.shape[0]    
        
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
        elif split == "all":
            idxs = np.arange(int(num_images * MAX_ALL_SIZE))
        else:
            idxs = np.arange(num_images)
            
        self.df = self.df.iloc[idxs, :]       
        #self.tags = self.tags[idxs] 
        self.targets = np.array(self.targets)[idxs]    
        self.masks = np.array(self.masks)[idxs]  
            
        #print(self.masks)
        
    def _parse_root_(self, labels_file):
        
        #print(df.head)
        self.image_names = df.urls
        self.tags = df.base_tag.to_numpy()
        self.image_names = [os.path.join(self.data_path, tag, image_name)  for image_name, tag in zip(self.image_names, df.base_tag)]
        self.image_names = np.array(self.image_names)
        
        with open(os.path.join(self.data_path, "category_id.pkl"), 'rb') as f:
            self.category_id = pickle.load(f)
                
        #print(self.category_id)    
        self.targets = np.array([self.category_id[tag] for tag in df.base_tag])
        
        #print(self.image_names)
        #print(self.tags)
        #print(self.category_id)
        #print(self.targets)
        
        

    def __len__(self):
        return self.df.shape[0] 

    def __getitem__(self, item):
        sample = {}
  
        sample["image"] = self.df.iloc[item, 1:1001].to_numpy(np.float32)
    
        
        target = np.zeros(self.num_lev1_categories + self.num_lev2_categories)
        
        #print(item)
        #print(self.targets)
        #print(self.targets[item], target.shape)
        target[self.targets[item]] = 1
        sample["target"] = target
        
        mask = np.zeros(self.num_lev1_categories + self.num_lev2_categories)
        #print('mask', self.masks[item])
        mask[self.masks[item]] = 1
        #print(mask)
        sample["mask"] = mask
        
        sample["tags"] = self.df.tags.iloc[item]
        #print(sample)
        return sample