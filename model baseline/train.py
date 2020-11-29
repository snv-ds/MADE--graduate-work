import os
import sys
from argparse import ArgumentParser

import numpy as np
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

# the proper way to do this is relative import, one more nested package and main.py outside the package
# will sort this out
sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from utils import get_logger

NUM_LEV1_CATEGORIES = 22
NUM_LEV2_CATEGORIES = 113

class SimpleLayersModel(nn.Module):
    def __init__(self, pretrained_model):
        super(SimpleLayersModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.last_layer = nn.Linear(1000, NUM_CATEGORIES, bias=True)

    def forward(self, x):
        x = F.relu(self.pretrained_model(x))
        return self.last_layer(x)

class SimpleSingleModel(nn.Module):
    def __init__(self):
        super(SimpleSingleModel, self).__init__()
        self.last_layer = nn.Linear(1000, NUM_CATEGORIES, bias=True)

    def forward(self, x):
        return self.last_layer(x)

class SimpleTwoLevelSingleModel(nn.Module):
    def __init__(self):
        super(SimpleTwoLevelSingleModel, self).__init__()
        self.lev1 = nn.Linear(1000, NUM_LEV1_CATEGORIES, bias=True)
        self.lev2 = nn.Linear(1000 + NUM_LEV1_CATEGORIES, NUM_LEV2_CATEGORIES, bias=True)
    def forward(self, x):
        x1 = self.lev1(x)
        x2 = torch.cat((x, x1), 1)
        #print(x.shape, x1.shape, x2.shape)
        x2 = self.lev2(x2)
        return torch.cat((x1, x2), 1)
        
    
def accuracy_k(output, target, topk=3):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0)

    #_, pred = output.topk(topk, 1, True, True)
    _, pred = output.topk(topk, 1, True, False)
    #print('output ', output[0, :], pred[0, :])
    res = 0
    for j in range(batch_size):
        res += torch.sum(target[j, pred[j]]) / torch.sum(target[j])
    
    return res / batch_size

def eval_net(net, dataset, device):
    net.eval()
    total3_level1 = 0
    total3_level2 = 0
    total3 = 0
    
    total1_level1 = 0
    total1_level2 = 0
    total1 = 0
    
    cat_lev1 = {it[1]:it[0] for it in dataset.dataset.category_id.items()}
    cat_all = {it[1]:it[0] for it in dataset.dataset.all_category_id.items()}
    #print(cat_all)
    
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            imgs = batch["image"]
            target = batch["target"]
            mask = batch["mask"]
            #print(imgs.shape)
            pred = sigmoid(net(imgs.to(device)).squeeze(1))  # (b, 1, h, w) -> (b, h, w)
            probs = (pred > 0.3).float()
            
            #for j in range(pred.shape[0]):
                
                # cat1_true = [cat_lev1[ii] for ii, v in enumerate(target[j, :NUM_LEV1_CATEGORIES].tolist()) if v == 1]
                # print('lev1 ', cat1_true)
                # for l in range(NUM_LEV1_CATEGORIES):
                #     if pred[j, l] > 0.1:
                #         print(cat_lev1[l], pred[j, l].item())
                        
                # cat2_true = [i for i, v in enumerate(target[j, NUM_LEV1_CATEGORIES:].tolist()) if v == 1]       
                # print(cat2_true)
                # cat2_true = [cat_all[ii + NUM_LEV1_CATEGORIES] for ii, v in enumerate(target[j, NUM_LEV1_CATEGORIES:].tolist()) if v == 1] 
                # num_mask = [ii for ii, v in enumerate(mask[j,:].tolist())  if v == 1]
                
                # print('lev2 ', cat2_true)
                # print(num_mask)
                # for l in num_mask:
                #     id = l# + NUM_LEV1_CATEGORIES
                #     if pred[j, id] > 0.1:
                #         print(cat_all[id], pred[j, id].item())

            total3_level1 += accuracy_k(pred[:, :NUM_LEV1_CATEGORIES], target[:, :NUM_LEV1_CATEGORIES], 3)
            non_zero = torch.sum(target[:, NUM_LEV1_CATEGORIES:],  1) > 0
            total3_level2 += accuracy_k(pred[non_zero, NUM_LEV1_CATEGORIES:], target[non_zero, NUM_LEV1_CATEGORIES:], 3)
            total3 += accuracy_k(pred, target, 3)
            
            total1_level1 += accuracy_k(pred[:, :NUM_LEV1_CATEGORIES], target[:, :NUM_LEV1_CATEGORIES], 1)
            non_zero = torch.sum(target[:, NUM_LEV1_CATEGORIES:],  1) > 0
            total1_level2 += accuracy_k(pred[non_zero, NUM_LEV1_CATEGORIES:], target[non_zero, NUM_LEV1_CATEGORIES:], 1)
            total1 += accuracy_k(pred, target, 1)

    return total3 / i, total3_level1 / i, total3_level2 / i, total1 / i, total1_level1 / i, total1_level2 / i


def train(net, optimizer, criterion, scheduler, train_dataloader, val_dataloader, logger, args=None, device=None):
    num_batches = len(train_dataloader)

    best_model_info = {'epoch': -1, 'val': 0., 'train': 0., 'train_loss': 0.}

    for epoch in range(args.epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        net.train()


        epoch_loss = 0.
        tqdm_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        mean_bce, mean_dice = [], []
        for i, batch in tqdm_iter:
            imgs = batch["image"]
            target = batch["target"]
            mask = batch["mask"]
            
            pred = net(imgs.to(device))
            probs = sigmoid(pred)

            #print(probs.shape, target.float().shape)
            loss = criterion(probs[:, :NUM_LEV1_CATEGORIES], target.float()[:, :NUM_LEV1_CATEGORIES])
            
            
            loss2 = criterion(probs * mask, target.float() * mask)
            
            num_mask = [ii for ii, v in enumerate(mask[1,:].tolist())  if v == 1]
            #print(num_mask)
            #print((probs * mask)[1, num_mask].tolist(), 
            #      (target.float() * mask)[1, num_mask].tolist(), 
            #      target.float()[1, :NUM_LEV1_CATEGORIES].tolist())
            #print("loss {:}, {:}".format(loss, loss2))
            loss += loss2
            
            epoch_loss += loss.item()
            tqdm_iter.set_description('mean loss: {:.4f}'.format(epoch_loss / (i + 1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if scheduler is not None:
            scheduler.step(epoch)    

        logger.info('Epoch finished! Loss: {:.5f}'.format(epoch_loss / num_batches))

        val3_score, val3_score1, val3_score2, val1_score, val1_score1, val1_score2 = eval_net(net, val_dataloader, device=device)
        logger.info('Validation acc3: {:.4f} for lev1: {:.4f}, for lev2: {:.4f}  acc1: {:.4f} for lev1: {:.4f}, for lev2: {:.4f} '.format(val3_score, val3_score1, val3_score2, 
                                                                                                                                          val1_score, val1_score1, val1_score2))
        
        # if val_dice > best_model_info['val_dice']:
        #     best_model_info['val_dice'] = val_dice
        #     best_model_info['train_loss'] = epoch_loss / num_batches
        #     best_model_info['epoch'] = epoch
        #     torch.save(net.state_dict(), os.path.join(args.output_dir, 'cp-best.pth'))
        #     logger.info('Validation Dice Coeff: {:.5f} (best)'.format(val_dice))
        # else:
        #     logger.info('Validation Dice Coeff: {:.5f} (best {:.5f})'.format(val_dice, best_model_info['val_dice']))

        torch.save(net.state_dict(), os.path.join(args.output_dir, args.model + '-cp-last.pth'))
        

def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=str, default=None ,help='path to the data')
    parser.add_argument('-e', '--epochs', dest='epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-s', '--image_size', dest='image_size', default=256, type=int, help='input image size')
    parser.add_argument('-lr', '--learning_rate', dest='lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('-lrs', '--learning_rate_step', dest='lr_step', default=10, type=int, help='learning rate step')
    parser.add_argument('-lrg', '--learning_rate_gamma', dest='lr_gamma', default=0.5, type=float,
                        help='learning rate gamma')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('-m', '--model', dest='model', default='resnet18', choices=('resnet18', 'resnet101', 'wide_resnet50'))
    parser.add_argument('-l', '--load', dest='load', default=False, help='load file model')
    parser.add_argument('-v', '--val_split', dest='val_split', default=0.8, help='train/val split')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='/tmp/logs/', help='dir to save log and models')
    parser.add_argument('-c', '--continue_calc',  help="continue calculation", action="store_true")
    parser.add_argument('-f', '--features_file',  help="use features", default=None, type=str)
    args = parser.parse_args()
    #
    os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, 'train.log'))
    logger.info('Start training with params:')
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    
    if args.features_file is None:
        if args.model == 'resnet18':
            pretrained_model = models.resnet18(pretrained=True)
            pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 1000, bias=True)
            net = SimpleLayersModel(pretrained_model)
        if args.model == 'wide_resnet50':
            pretrained_model = models.wide_resnet50_2(pretrained=True)
            pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 1000, bias=True)
            net = SimpleLayersModel(pretrained_model)
    else:
        net = SimpleTwoLevelSingleModel()

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
    # net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # TODO: loss experimentation, fight class imbalance, there're many ways you can tackle this challenge
    #criterion = lambda x, y: (args.weight_bce * nn.BCELoss()(x, y), (1. - args.weight_bce) * dice_loss(x, y))
    criterion = nn.BCELoss()
    # TODO: you can always try on plateau scheduler as a default option
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma) \
        if args.lr_step > 0 else None
                
    if args.features_file is None:
        # dataset
        # TODO: to work on transformations a lot, look at albumentations package for inspiration
        train_transforms = Compose([
            Crop(min_size=1 - 1 / 3., min_ratio=1.0, max_ratio=1.0, p=0.5),
            Flip(p=0.05),
            Pad(max_size=0.6, p=0.25),
            Resize(size=(args.image_size, args.image_size), keep_aspect=True),
            ToPILImage()
        ])
        # TODO: don't forget to work class imbalance and data cleansing
        val_transforms = Compose([
            Resize(size=(args.image_size, args.image_size), keep_aspect=True),
            ToPILImage()
        ])
        
        train_dataset = HashTagDataset(args.data_path, labels_file = os.path.join(args.data_path, 'data_set.csv'),
                                     split = 'train', transforms=train_transforms)
        val_dataset = HashTagDataset(args.data_path, labels_file = os.path.join(args.data_path, 'data_set.csv'), 
                                     split = 'val', transforms=val_transforms)
    else:
        train_dataset = HashTagFeatureDataset(args.data_path, labels_file = os.path.join(args.data_path, 'data_set.csv'),
                                     features_file = args.features_file,
                                     split = 'train')
        val_dataset = HashTagFeatureDataset(args.data_path, labels_file = os.path.join(args.data_path, 'data_set.csv'), 
                                     features_file = args.features_file,       
                                     split = 'val')


    #print(train_size)
    # TODO: always work with the data: cleaning, sampling
    #sampler = RandomSampler(train_dataset, replacement=True, num_samples=1000)
    #val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1200)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                  shuffle=False, drop_last=True)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,
                                shuffle=False, drop_last=False)
    
    logger.info('Length of train/val=%d/%d', len(train_dataset), len(val_dataset))
    logger.info('Number of batches of train/val=%d/%d', len(train_dataloader), len(val_dataloader))
    
    try:
        train(net, optimizer, criterion, scheduler, train_dataloader, val_dataloader, logger=logger, args=args,
              device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        sys.exit(0)


if __name__ == '__main__':
    main()
