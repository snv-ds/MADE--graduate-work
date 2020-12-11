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
import pickle

# the proper way to do this is relative import, one more nested package and main.py outside the package
# will sort this out
sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from utils import get_logger

#NUM_LEV1_CATEGORIES = 16 #22
#NUM_LEV2_CATEGORIES = 237 #113

class SimpleLayersModel(nn.Module):
    def __init__(self, pretrained_model, num_lev1_categories):
        super(SimpleLayersModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.last_layer = nn.Linear(1000, num_lev1_categories, bias=True)

    def forward(self, x):
        x = F.relu(self.pretrained_model(x))
        return self.last_layer(x)

class SimpleSingleModel(nn.Module):
    def __init__(self, num_lev1_categories):
        super(SimpleSingleModel, self).__init__()
        self.last_layer = nn.Linear(1000, num_lev1_categories, bias=True)

    def forward(self, x):
        return self.last_layer(x)

class SimpleTwoLevelSingleModel(nn.Module):
    def __init__(self, num_lev1_categories, num_lev2_categories):
        super(SimpleTwoLevelSingleModel, self).__init__()
        self.lev1 = nn.Linear(1000, num_lev1_categories, bias=True)
        self.lev2 = nn.Linear(1000 + num_lev1_categories, num_lev2_categories, bias=True)
    
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

def eval_net(net, dataset, num_lev1_categories, device):
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

            total3_level1 += accuracy_k(pred[:, :num_lev1_categories], target[:, :num_lev1_categories], 3)
            non_zero = torch.sum(target[:, num_lev1_categories:],  1) > 0
            total3_level2 += accuracy_k(pred[non_zero, num_lev1_categories:], target[non_zero, num_lev1_categories:], 3)
            total3 += accuracy_k(pred, target, 3)
            
            total1_level1 += accuracy_k(pred[:, :num_lev1_categories], target[:, :num_lev1_categories], 1)
            non_zero = torch.sum(target[:, num_lev1_categories:],  1) > 0
            total1_level2 += accuracy_k(pred[non_zero, num_lev1_categories:], target[non_zero, num_lev1_categories:], 1)
            total1 += accuracy_k(pred, target, 1)

    return total3 / i, total3_level1 / i, total3_level2 / i, total1 / i, total1_level1 / i, total1_level2 / i


def train(net, optimizer, criterion, scheduler, train_dataloader, val_dataloader, num_lev1_categories, logger,  args=None, device=None):
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
            loss = criterion(probs[:, :num_lev1_categories], target.float()[:, :num_lev1_categories])
            
            
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

        val3_score, val3_score1, val3_score2, val1_score, val1_score1, val1_score2 = eval_net(net, val_dataloader, num_lev1_categories, device=device)
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

        torch.save(net.state_dict(), os.path.join(args.output_dir, args.model + '-' + args.model_ver + '.pth'))
        

def test_lev1_cat(net, dataset, num_lev1_categories, args):
    net.eval()
    total3_level1 = 0
    total3_level2 = 0
    total3 = 0
    
    total1_level1 = 0
    total1_level2 = 0
    total1 = 0
    
    cat_lev1 = {it[1]:it[0] for it in dataset.dataset.category_id.items()}
    cat_all = {it[1]:it[0] for it in dataset.dataset.all_category_id.items()}
    
    stats = {key:{'tag': value, 'num': 0, 'good': 0, 'pd': 0, 
                  'num2': 0, 'good2': 0, 'pd2': 0, 
                  'num_fact': 0, 'good_fact': 0} for key, value in cat_lev1.items() }
    
    pred_lev1_first = []
    pred_lev1_second = []
    with torch.no_grad():
        for i, batch in enumerate(dataset): #tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            imgs = batch["image"]
            target = batch["target"]
            mask = batch["mask"]
            #print(imgs.shape)
            
            pred = sigmoid(net(imgs).squeeze(1))  # (b, 1, h, w) -> (b, h, w)
            probs = (pred > 0.3).float()
            
            pred_lev1 = pred[:, :num_lev1_categories]
            target_lev1 = target[:, :num_lev1_categories]
            
            
            pred_values, pred_index = pred_lev1.topk(2, 1, True, False)
            
            k_loc = max(torch.sum(pred_lev1 > 0.25).item(), 1)
            
            curr_accc = accuracy_k(pred_lev1, target_lev1, k_loc)
            total1_level1 += curr_accc            
            
            # first predict for cat
            tag_id = pred_index[0, 0].item()
            stats[tag_id]['num'] += 1
            stats[tag_id]['good'] += target_lev1[0, pred_index[0, 0]].item()
            stats[tag_id]['pd'] += pred_values[0, 0].item()
            
            pred_lev1_first.append(
                [stats[tag_id]['tag'], pred_values[0, 0].item(), target_lev1[0, pred_index[0, 0]].item()])
            
             # second predict for cat
            tag_id = pred_index[0, 1].item()
            stats[tag_id]['num2'] += 1
            stats[tag_id]['good2'] += target_lev1[0, pred_index[0, 1]].item()
            stats[tag_id]['pd2'] += pred_values[0, 1].item()   
            
            pred_lev1_second.append(
                [stats[tag_id]['tag'], pred_values[0, 1].item(), target_lev1[0, pred_index[0, 1]].item()])
            
            tag_real_id = torch.where(target_lev1[0,:] > 0)[0]
            for id in tag_real_id.tolist():
                stats[id]['num_fact'] += 1
                stats[id]['good_fact'] += (pred_lev1[0, id].item() > 0.2) 
            
    #print(stats)

    with open(os.path.join(args.output_dir, "stats_1_" + args.model + '-' + args.model_ver + '.pkl'), 'wb') as f:
        pickle.dump(stats, f)
        
    df = pd.DataFrame(pred_lev1_first)
    df.to_csv(os.path.join(args.output_dir, "pred_1_1_" + args.model + '-' + args.model_ver + '.pkl'), index = False)
    
    df = pd.DataFrame(pred_lev1_second)
    df.to_csv(os.path.join(args.output_dir, "pred_1_2_" + args.model + '-' + args.model_ver + '.pkl'), index = False)

    return total1_level1 / i      

def test_lev2_cat(net, dataset, num_lev1_categories, categories, args, cutoff_lev1 = 0.3, cutoff_lev2 = 0.4, ver_name = ''):
    net.eval()
    total3_level1 = 0
    total3_level2 = 0
    total3 = 0
    
    total1_level1 = 0
    total1_level2 = 0
    total1 = 0
    
    cat_lev1 = {it[1]:it[0] for it in dataset.dataset.category_id.items()}
    cat_all = {it[1]:it[0] for it in dataset.dataset.all_category_id.items()}
    
    category_id = categories['category_id'] 
    category_top = categories['category_top'] 
    lev1_lev2_category_id = categories['lev1_lev2_category_id'] 
    category_all_names = categories['category_all_names']
    category_all_id = {id:tag for tag, id in categories['category_all_names'].items()}
    #category_top_names = [tag for tag, id in categories['category_top'.items()]
    #print(category_top[100])

    
    stats = {key:{'tag': value, 'num': 0, 'good': 0, 'pd': 0, 
                  'num2': 0, 'good2': 0, 'pd2': 0, 
                  'num_fact': 0, 'good_fact': 0} for key, value in cat_lev1.items() }
    
    stats = {}
    pred_lev1_second = []
    score_all = 0
    score_100 = 0
    score_300 = 0
    score_500 = 0
    with torch.no_grad():
        for i, batch in enumerate(dataset): #tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            imgs = batch["image"]
            target = batch["target"]
            mask = batch["mask"]
            all_target_tags = batch["tags"][0]
            
            pred = sigmoid(net(imgs).squeeze(1))[0,:] # only for batch_size = 1
            #probs = (pred > 0.3).float()
            
            pred_lev1 = pred[:num_lev1_categories]
            target = target[0,:]
            target_lev1 = target[:num_lev1_categories]
            #print('target', target)
            
            pred_values, pred_index = pred_lev1.topk(2, 0, True, False)
            
            k_loc = min(5, max(torch.sum(pred_lev1 > cutoff_lev1).item(), 1)) # number of pred in lev1
            
            pred_index = pred_index[:k_loc]
            
            #print(pred_values, pred_index)
            id_lev1 = pred_index.tolist()
            cat1_true = [cat_lev1[id] for id in id_lev1]
            #print('id_lev1', id_lev1, cat1_true)
            cat2_true = []
            if id_lev1 != []:
                available_cat = np.array(list(set([item for id1 in id_lev1 for item in lev1_lev2_category_id[id1].values()])))
                #available_cat = np.array(list(lev1_lev2_category_id[id_lev1].values()))
                #print('available_cat', available_cat)
                if k_loc < 0 or k_loc > 5:
                    print('k_loc', k_loc, pred_values[:k_loc])
                v_2, pred_2 = pred[available_cat].topk(5 - k_loc, 0, True, False)
                #print('pred_2', v_2, pred_2)
                pred_2 = pred_2[v_2 > cutoff_lev2]
                pred_t = available_cat[pred_2].tolist()
                #print(pred_2, pred_t, pred_2.shape)
                if pred_2.shape[0] == 1:
                    pred_t = [pred_t]

                cat2_true = [category_all_id[ii] for ii in pred_t ] 
            
            #target_name = [for id in target_lev1.tolist()]
            #print('cat ', cat1_true, cat2_true, )
            
            all_target_tags = all_target_tags.split(" ")
            
            new_score_all = np.sum([1 for tag in (cat1_true + cat2_true) if tag in all_target_tags])
            new_score_100 = np.sum([1 for tag in (cat1_true + cat2_true) if (tag in all_target_tags and tag in category_top[100])])
            new_score_300 = np.sum([1 for tag in (cat1_true + cat2_true) if (tag in all_target_tags and tag in category_top[300])])
            new_score_500 = np.sum([1 for tag in (cat1_true + cat2_true) if (tag in all_target_tags and tag in category_top[500])])
            #print(new_score_all, new_score_100, new_score_500)
            
            score_all += new_score_all
            score_100 += new_score_100 
            score_300 += new_score_300 
            score_500 += new_score_500
            
            for tag in (cat1_true + cat2_true):
                line = stats.get(tag, {"tag": tag, "n":0, "n_plus":0, "n_minus":0})
                if tag in all_target_tags:
                    line["n_plus"] += 1
                else:
                    line["n_minus"] += 1
                stats[tag] = line   
            
            for tag in all_target_tags:
                if tag in category_top[300]:
                    line = stats.get(tag, {"tag": tag, "n":0, "n_plus":0, "n_minus":0})
                    line["n"] += 1                
                    stats[tag] = line       
                    
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.to_csv(os.path.join(args.output_dir, "stats_all_" + args.model + '-' + args.model_ver + ver_name + '.csv'), index = False)

    n = dataset.__len__()
    print(n, score_all, score_100, score_all / n, score_100/ n, score_300/ n, score_500/ n)

    return score_all / n, score_100/ n, score_300/ n, score_500/ n      



def test(net, train_dataloader, val_dataloader, test_dataloader,  num_lev1_categories, categories, 
         logger,  args=None, device=None):
    
    val1_score= test_lev1_cat(net, val_dataloader, num_lev1_categories, args)
    logger.info('Validation acc1 for lev1: {:.4f}'.format(val1_score))
    
    #score_all, score_100, score_300, score_500 = test_lev2_cat(net, val_dataloader, num_lev1_categories, categories, args)
    
    #for cat1 in [0.125, 0.15, 0.175]:
    #    for cat2 in [0.275, 0.3, 0.325]:        
    #        score_all, score_100, score_300, score_500 = test_lev2_cat(net, val_dataloader, num_lev1_categories, categories, args, cat1, cat2)
    #        print(cat1, cat2, score_all)
    
    score_all, score_100, score_300, score_500 = test_lev2_cat(net, val_dataloader, num_lev1_categories, categories, args, 0.15, 0.3, '')
    score_all, score_100, score_300, score_500 = test_lev2_cat(net, test_dataloader, num_lev1_categories, categories, args, 0.15, 0.3, '_all')

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
    parser.add_argument('-mv', '--model_ver', dest='model_ver', default='', help='version to save models')
    parser.add_argument('-c', '--continue_calc',  help="continue calculation", action="store_true")
    parser.add_argument('-f', '--features_file',  help="use features", default=None, type=str)
    parser.add_argument('-t', '--test', dest='test_model', default=False, help='run test for test model')
    args = parser.parse_args()
    #
    os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, 'train.log'))
    logger.info('Start training with params:')
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
        
    #     
    with open(os.path.join(args.data_path, "category_id.pkl"), 'rb') as f:
            category_id = pickle.load(f)

    with open(os.path.join(args.data_path, "lev1_lev2_category_id.pkl"), 'rb') as f:
            lev1_lev2_category_id = pickle.load(f)  
            
    with open(os.path.join(args.data_path, "lev2_category_id.pkl"), 'rb') as f:
        category_all_names = pickle.load(f)    
        
    with open(os.path.join(args.data_path, "category_top.pkl"), 'rb') as f:
        category_top = pickle.load(f)  
        
    categories = {'category_id': category_id, 'lev1_lev2_category_id':lev1_lev2_category_id, 
                  'category_all_names':category_all_names, 'category_top':category_top}  
    
    num_lev1_categories = np.max([id for tag, id in category_id.items()]) + 1      
    num_lev2_categories = np.max([id for id1, d in lev1_lev2_category_id.items() for tag, id in d.items()]) + 1 - num_lev1_categories    

       
    if args.features_file is None:
        if args.model == 'resnet18':
            pretrained_model = models.resnet18(pretrained=True)
            pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 1000, bias=True)
            net = SimpleLayersModel(pretrained_model, num_lev1_categories)
        if args.model == 'wide_resnet50':
            pretrained_model = models.wide_resnet50_2(pretrained=True)
            pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 1000, bias=True)
            net = SimpleLayersModel(pretrained_model, num_lev1_categories)
    else:
        net = SimpleTwoLevelSingleModel(num_lev1_categories, num_lev2_categories)

    logger.info('Model type: {}'.format(net.__class__.__name__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.load or args.continue_calc: # working with existence model
        try:
            with open(os.path.join(args.output_dir, args.model + '-' + args.model_ver + '.pth'), "rb") as fp:
                net.load_state_dict(torch.load(fp))
        except:
            pass
            
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
        if args.continue_calc or not args.load:
            train(net, optimizer, criterion, scheduler, train_dataloader, val_dataloader, num_lev1_categories, logger=logger, args=args, device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        sys.exit(0)
        
    if args.test_model:
        test_dataset = HashTagFeatureDataset(args.data_path, labels_file = os.path.join(args.data_path, 'data_set.csv'), 
                             features_file = args.features_file,       
                             split = 'all')
        print('test',test_dataset.__len__() )
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0,
                            shuffle=False, drop_last=False)
                
        test(net, train_dataloader, val_dataloader, test_dataloader, num_lev1_categories, 
             categories, logger=logger, args=args, device=device)

if __name__ == '__main__':
    main()
