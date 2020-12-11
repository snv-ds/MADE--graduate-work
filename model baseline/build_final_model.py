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
from collections import Counter

#NUM_LEV1_CATEGORIES = 22
#NUM_LEV2_CATEGORIES = 113

class SimpleLayersModel(nn.Module):
    def __init__(self, pretrained_model, num_lev1_categories, num_lev2_categories):
        super(SimpleLayersModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.lev1 = nn.Linear(1000, num_lev1_categories, bias=True)
        self.lev2 = nn.Linear(1000 + num_lev1_categories, num_lev2_categories, bias=True)
        
    def forward(self, x):
        x = F.relu(self.pretrained_model(x))
        x1 = self.lev1(x)
        x2 = torch.cat((x, x1), 1)
        x2 = self.lev2(x2)
        return torch.cat((x1, x2), 1)


class SimpleTwoLevelSingleModel(nn.Module):
    def __init__(self, num_lev1_categories, num_lev2_categories):
        super(SimpleTwoLevelSingleModel, self).__init__()
        self.lev1 = nn.Linear(1000, num_lev1_categories, bias=True)
        self.lev2 = nn.Linear(1000 + num_lev1_categories, num_lev2_categories, bias=True)
    def forward(self, x):
        x1 = self.lev1(x)
        x2 = torch.cat((x, x1), 1)
        x2 = self.lev2(x2)
        return torch.cat((x1, x2), 1)

class HastTagModel():
    
    def __init__(self):
        model = 'resnet101'
        model_ver="ver_nk3"
        output_dir = 'models_feat'
        data_path = 'data'
        image_size = 256
        
        
        # load hash tag names category_id
        with open(os.path.join(data_path, "category_id.pkl"), 'rb') as f:
                self.lev1_category_id = pickle.load(f)   
        self.lev1_category_names = {it[1]:it[0] for it in self.lev1_category_id.items()}
            
        with open(os.path.join(data_path, "lev2_category_id.pkl"), 'rb') as f:
                self.all_category_id = pickle.load(f)   
        self.all_category_names = {it[1]:it[0] for it in self.all_category_id.items()}
        
        with open(os.path.join(data_path, "lev1_lev2_category_id.pkl"), 'rb') as f:
                self.lev1_lev2_category_id = pickle.load(f)           
        
        
        self.num_lev1_categories = np.max([id for tag, id in self.lev1_category_id.items()]) + 1      
        self.num_lev2_categories = np.max([id for id1, d in self.lev1_lev2_category_id.items() for tag, id in d.items()]) + 1 - self.num_lev1_categories    


        pretrained_model = models.resnet101(pretrained=True) 
        self.net = SimpleLayersModel(pretrained_model, self.num_lev1_categories, self.num_lev2_categories) 
        
        net_last = SimpleTwoLevelSingleModel(self.num_lev1_categories, self.num_lev2_categories)
        with open(os.path.join(output_dir, model + '-' + model_ver + '.pth'), "rb") as fp:
            net_last.load_state_dict(torch.load(fp))
        
    
        copy_layers = ['lev1.weight', 'lev1.bias', 'lev2.weight', 'lev2.bias']
        for layer in copy_layers:
            self.net.state_dict()[layer].data.copy_(net_last.state_dict()[layer])
            
       
        
        
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
            pred = sigmoid(self.net(imgs).squeeze(1))[0, :]  # (b, 1, h, w) -> (b, h, w)

            v_1, pred_1 = (pred[ :self.num_lev1_categories]).topk(2, 0, True, False)
            pred_1 = pred_1[v_1 > 0.2]
            
            id_lev1 = pred_1.data.tolist()
            cat1_true = [self.lev1_category_names[id] for id in id_lev1] # Level 1 category
            #print('cat1_true', cat1_true)

            cat2_true = []
            if id_lev1 != []:
                for id1 in id_lev1:
                    available_cat = np.array(list(self.lev1_lev2_category_id[id1].values()))
                    
                    print(available_cat)
                    v_2, pred_2 = pred[available_cat].topk(3, 0, True, False)
                    pred_2 = pred_2[v_2 > 0.3]
                    pred_t = available_cat[pred_2].tolist()

                    if pred_2.shape[0] == 1:
                        pred_t = [pred_t]

                cat2_true = [self.all_category_names[ii] for ii in pred_t ]        
            #print('cat2_true', cat2_true)
        #print('cat_true', cat1_true, cat2_true)    
        return cat1_true + cat2_true


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
    
def test_level(model, data_path, n = 100, test = False):
    labels_file = os.path.join(data_path, 'data_set.csv')
    
    df = pd.read_csv(labels_file)

    image_names = [os.path.join(data_path, tag, image_name)  for image_name, tag in zip(df.urls, df.base_tag)]
    
    stats = {}

    samples = np.random.choice(int(df.shape[0]* 0.8), n)
    if test:
        samples = np.random.choice(int(df.shape[0] * 0.2), n) + int(df.shape[0]* 0.8)
    
    for step, i in enumerate(samples):
        image = cv2.imdecode(np.fromfile(image_names[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        pred_tags = model.predict(image)
        
        lev1_tags = [model.lev1_category_id[tag] for tag in df.tags[i].split(' ') if tag in model.lev1_category_id]
        lev1_tags = [tag for tag in df.tags[i].split(' ') if tag in model.lev1_category_id]
        
        if len(lev1_tags) == 0:
            print('-')
            continue       
        print('df.tags[i]', df.tags[i])
        print('lev1_tags', lev1_tags, 'pred_tags', pred_tags)
        print()

        for lev1_tag in lev1_tags:
            res = lev1_tag in pred_tags
            if stats.get(lev1_tag, -1) == -1:
                
                errors_tag = None if res else [pred_tags[0]]
                stats[lev1_tag] = [1, int(res), errors_tag]
                
            else:
                previos = stats[lev1_tag]
                errors_tags = previos[2]
                if not res:
                    errors_tag = None if res else [pred_tags[0]]
                    errors_tags = errors_tag if previos[2] is None else previos[2] + errors_tag
                    
                stats[lev1_tag] = [previos[0] + 1, previos[1] + res, errors_tags]
                
            if step % 100 == 0:
                print(step)

    return stats

def main():

        
    hast_tag_model = HastTagModel()
    
    with open(os.path.join('models', "model_ver_3.pkl"), 'wb') as f:
        pickle.dump(hast_tag_model, f)
        
    data_path = 'data'    
    #test(hast_tag_model, data_path)   
    
    stats = test_level(hast_tag_model, data_path, 50, True)    
        
    n_all = 0
    n_good = 0
    for key, stat in stats.items(): 
        count_errors = Counter(stat[2]).most_common()
        print('Для {:} примеров: {:}, точность: {:3.3f} ошибки:'.format(key, stat[0], stat[1] / stat[0]),
                  count_errors[:min(3, len(count_errors))])
        n_all += stat[0]
        n_good += stat[1]
        
    print('Точность: {:3.3f}'.format(n_good/n_all))
    stats_true = stats    
       

    stats = test_level(hast_tag_model, data_path, 300, False)    
        
    n_all = 0
    n_good = 0
    for key, stat in stats.items(): 
        count_errors = Counter(stat[2]).most_common()
        print('Для {:} примеров: {:}, точность: {:3.3f} ошибки:'.format(key, stat[0], stat[1] / stat[0]),
                  count_errors[:min(3, len(count_errors))])
        n_all += stat[0]
        n_good += stat[1]
        
    print('Точность: {:3.3f}'.format(n_good/n_all))

if __name__ == '__main__':
    main()
