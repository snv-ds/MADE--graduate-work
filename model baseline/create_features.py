# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:17:25 2020

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
from dataset import HashTagImageDataset
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from transform import Compose, Resize, Crop, Pad, Flip, ToPILImage
import torchvision.models as models

# the proper way to do this is relative import, one more nested package and main.py outside the package
# will sort this out
sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from utils import get_logger


def apply_model(net, train_dataloader, logger, args=None, device=None):
    net.eval()
    num_batches = len(train_dataloader)

    logger.info('Starting apply_model.')
    res_features = []
    tqdm_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, batch in tqdm_iter:
        imgs = batch["image"]
        target = batch["target"]
        #print(batch["image_name"], imgs)
        
        pred = net(imgs.to(device)).squeeze(1)
        #print(pred.detach().numpy())
        res_features.append(pred.detach().numpy())

    logger.info('apply_model finished!')
    return res_features
        

def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=str, default=None ,help='path to the data')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-s', '--image_size', dest='image_size', default=256, type=int, help='input image size')
    parser.add_argument('-m', '--model', dest='model', default='resnet18', choices=('resnet18', 'resnet101', 'wide_resnet50'))
    parser.add_argument('-l', '--load', dest='load', default=False, help='load file model')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='/data/', help='dir to save log and models')
    parser.add_argument('-f', '--features_file',  help="file for used features", default=None, type=str)
    args = parser.parse_args()
    #
    os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, 'train.log'))
    logger.info('Start training with params:')
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    
    if args.model == 'resnet18':
        net = models.resnet18(pretrained=True)
    if args.model == 'resnet101':
        net = models.resnet101(pretrained=True)     
    if args.model == 'wide_resnet50':
        net = models.wide_resnet50_2(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, 1000, bias=True)

    # TODO: img_size=256 is rather mediocre, try to optimize network for at least 512
    logger.info('Model type: {}'.format(net.__class__.__name__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.load:
        if args.continue_calc:
            try:
                with open(os.path.join(args.output_dir, args.model + 'cp-last.pth'), "rb") as fp:
                    net.load_state_dict(torch.load(fp))
            except:
                pass
        else:
            net.load_state_dict(torch.load(args.load))
            
    net.to(device)

    # dataset
    train_transforms = Compose([
        Resize(size=(args.image_size, args.image_size), keep_aspect=True),
        ToPILImage()
    ])

    train_dataset = HashTagImageDataset(args.data_path, labels_file = os.path.join(args.data_path, 'data_set.csv'),
                                 split = 'all', transforms=train_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                  shuffle=False, drop_last=False)

    logger.info('Length of train=%d', len(train_dataset))
    logger.info('Number of batches of train=%d', len(train_dataloader))
    
    try:
        features = apply_model(net, train_dataloader, logger=logger, args=args,
              device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        sys.exit(0)
        
    features = np.concatenate(features)
    df = pd.DataFrame(features)
    df.insert(loc = 0, column="img", value=train_dataset.image_names)
    
    ffile = os.path.join(args.data_path, 'features_'+args.model+'.csv')
    if args.features_file is not None:
        ffile = os.path.join(args.data_path, args.features_file)
    df.to_csv(ffile, index=False)

if __name__ == '__main__':
    main()
