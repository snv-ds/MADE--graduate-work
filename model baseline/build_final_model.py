# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:49:59 2020

@author: ilya-
"""


import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import tqdm
from dataset import HashTagImageDataset, HashTagFeatureDataset
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from transform import Compose, Resize, Crop, Pad, Flip, ToPILImage
import torchvision.models as models
import cv2
import pickle

NUM_LEV1_CATEGORIES = 22
NUM_LEV2_CATEGORIES = 113

class SimpleLayersModel(nn.Module):
    def __init__(self, pretrained_model):
        super(SimpleLayersModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.lev1 = nn.Linear(1000, NUM_LEV1_CATEGORIES, bias=True)
        self.lev2 = nn.Linear(1000 + NUM_LEV1_CATEGORIES, NUM_LEV2_CATEGORIES, bias=True)
        
    def forward(self, x):
        x = F.relu(self.pretrained_model(x))
        x1 = self.lev1(x)
        x2 = torch.cat((x, x1), 1)
        x2 = self.lev2(x2)
        return torch.cat((x1, x2), 1)


class SimpleTwoLevelSingleModel(nn.Module):
    def __init__(self):
        super(SimpleTwoLevelSingleModel, self).__init__()
        self.lev1 = nn.Linear(1000, NUM_LEV1_CATEGORIES, bias=True)
        self.lev2 = nn.Linear(1000 + NUM_LEV1_CATEGORIES, NUM_LEV2_CATEGORIES, bias=True)
    def forward(self, x):
        x1 = self.lev1(x)
        x2 = torch.cat((x, x1), 1)
        x2 = self.lev2(x2)
        return torch.cat((x1, x2), 1)

class HastTagModel():
    
    def __init__(self):
        model = 'resnet101'
        output_dir = 'models_feat'
        data_path = 'data'
        image_size = 256
        
        pretrained_model = models.resnet18(pretrained=True) 
        self.net = SimpleLayersModel(pretrained_model) # Model!!!
        
        net_last = SimpleTwoLevelSingleModel()
        with open(os.path.join(output_dir, model + '-cp-last.pth'), "rb") as fp:
            net_last.load_state_dict(torch.load(fp))
        
    
        copy_layers = ['lev1.weight', 'lev1.bias', 'lev2.weight', 'lev2.bias']
        for layer in copy_layers:
            self.net.state_dict()[layer].data.copy_(net_last.state_dict()[layer])
            
        # load hash tag names category_id
        with open(os.path.join(data_path, "category_id.pkl"), 'rb') as f:
                lev1_category_id = pickle.load(f)   
        self.lev1_category_names = {it[1]:it[0] for it in lev1_category_id.items()}
            
        with open(os.path.join(data_path, "lev2_category_id.pkl"), 'rb') as f:
                all_category_id = pickle.load(f)   
        self.all_category_names = {it[1]:it[0] for it in all_category_id.items()}
        
        with open(os.path.join(data_path, "lev1_lev2_category_id.pkl"), 'rb') as f:
                self.lev1_lev2_category_id = pickle.load(f)          
        
        
    def predict(self, image):
        image_size = 256
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.
 
        val_transforms = Compose([
                Resize(size=(image_size, image_size), keep_aspect=True),
                ToPILImage()
            ])
        image = val_transforms(image)      
        
        # predict
        self.net.eval()
        with torch.no_grad():
            imgs = image.unsqueeze(0)
            pred = sigmoid(self.net(imgs).squeeze(1))  # (b, 1, h, w) -> (b, h, w)
            probs = (pred > 0.6).float()
            v, pred_1 = (pred[0, :NUM_LEV1_CATEGORIES]).topk(1, 0, True, False)
            id_lev1 = pred_1[0].data.tolist()
            cat1_true = self.lev1_category_names[id_lev1] # Level 1 category
            

            cat2_true = []
            if id_lev1 != []:
                available_cat = np.array(list(self.lev1_lev2_category_id[id_lev1].values()))
                v_2, pred_2 = pred[0, available_cat].topk(3, 0, True, False)
                pred_2 = pred_2[v_2 > 0.5]
                pred_t = available_cat[pred_2].tolist()
                #print(pred_2, pred_t, pred_2.shape)
                if pred_2.shape[0] == 1:
                    pred_t = [pred_t]
                    #print('change ', pred_t)
                    
                                # Level 1 category
                cat2_true = [self.all_category_names[ii] for ii in pred_t ]        
        
        return [cat1_true] + cat2_true


def test(model, data_path):
    labels_file = os.path.join(data_path, 'data_set.csv')
    
    df = pd.read_csv(labels_file)
    image_names = [os.path.join(data_path, tag, image_name)  for image_name, tag in zip(df.urls, df.base_tag)]
    
    n = 500
    good = 0
    for i in np.random.choice(df.shape[0], 500):
        image = cv2.imdecode(np.fromfile(image_names[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        tags = model.predict(image)
        res = df.base_tag[i] in tags
        print(tags, df.base_tag[i], res)
        good += res
        
    print('Точность: {:3.3f}'.format(good/n))


def main():

    
    hast_tag_model = HastTagModel()
    
    with open(os.path.join('models', "model_ver_2.pkl"), 'wb') as f:
        pickle.dump(hast_tag_model, f)
        
    data_path = 'data'    
    test(hast_tag_model, data_path)   
        
    
if __name__ == '__main__':
    main()
