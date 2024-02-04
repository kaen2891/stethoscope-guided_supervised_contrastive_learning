from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import pickle
import argparse
import numpy as np

import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torchvision import transforms

from util.icbhi_dataset_for_tsne import ICBHIDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class, Projector
from method import MetaCL


def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--m_cls', type=int, default=0,
                        help='set k-way classification problem for domain (meta)')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])

    # model
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')

    parser.add_argument('--method', type=str, default='ce')
    parser.add_argument('--domain_adaptation', action='store_true')
    parser.add_argument('--domain_adaptation2', action='store_true')    
    # Meta Domain CL loss
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--target_type', type=str, default='project_flow',
                        help='how to make target representation', choices=['project_flow', 'grad_block1', 'grad_flow1', 'project_block1', 'grad_block2', 'grad_flow2', 'project_block2', 'project_block_all'])
                        
    parser.add_argument('--name', type=str, default='ce')
    # Meta for SCL
    parser.add_argument('--meta_mode', type=str, default='none',
                        help='the meta information for selecting', choices=['none', 'age', 'sex', 'loc', 'dev', 'label'])
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    
    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.method in ['mscl'] or args.domain_adaptation or args.domain_adaptation2:
        if args.meta_mode in ['age', 'sex']:
            args.m_cls = 2
        elif args.meta_mode in ['dev', 'label']:
            args.m_cls = 4
        elif args.meta_mode in ['loc']:
            args.m_cls = 7    
    
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.method in ['mscl'] or args.domain_adaptation or args.domain_adaptation2:
                #single
                if args.meta_mode == 'age':
                    args.meta_cls_list = ['Adult', 'Child']
                elif args.meta_mode == 'sex':
                    args.meta_cls_list = ['Male', 'Female']
                elif args.meta_mode == 'loc':
                    args.meta_cls_list = ['Tc', 'Al', 'Ar', 'Pl', 'Pr', 'Ll', 'Lr']
                elif args.meta_mode == 'dev':
                    args.meta_cls_list = ['Meditron', 'LittC2SE', 'Litt3200', 'AKGC417L']
                elif args.meta_mode == 'label':
                    args.meta_cls_list = ['None', 'Crackle', 'Wheeze', 'Both']
                            
            if not args.method in ['mscl']:
                if args.n_cls == 4:
                    args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
                elif args.n_cls == 2:
                    args.cls_list = ['normal', 'abnormal']
                
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3 and args.method not in ['mscl']:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2 and args.method not in ['mscl']:
                args.cls_list = ['healthy', 'unhealthy']
            else:
                raise NotImplementedError
        
    else:
        raise NotImplementedError
    
    if args.n_cls == 0 and args.m_cls !=0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args


def set_loader(args):
    if args.dataset == 'icbhi':        
        args.h = int(args.desired_length * 100 - 2)
        args.w = 128
        '''
        train_transform = [transforms.ToTensor(),
                            SpecAugment(args),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        '''
        #train_transform = [transforms.ToTensor(), transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        ##
        #train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)

        #train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True) if args.method not in ['mscl'] else None

        args.class_nums = val_dataset.class_nums
        if args.domain_adaptation or args.domain_adaptation2:
            args.domain_nums = val_dataset.domain_nums
    else:
        raise NotImplemented    
    
    sampler = None
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True, sampler=None)
        
    return None, val_loader, args
    


def set_model(args):
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        if args.domain_adaptation:
            kwargs['domain_label_dim'] = args.m_cls

    model = get_backbone_class(args.model)(**kwargs)
    if args.domain_adaptation:
        class_classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
        domain_classifier = nn.Linear(model.final_feat_dim, args.m_cls) if args.model not in ['ast'] else deepcopy(model.domain_mlp_head)
    elif args.domain_adaptation2:
        class_classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
        domain_classifier = Projector(model.final_feat_dim, args.proj_dim)
    else:
        classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.domain_adaptation2 else nn.Identity()
    
    
    criterion = nn.CrossEntropyLoss()
    
    if args.domain_adaptation:
        criterion2 = nn.CrossEntropyLoss()
        
    elif args.domain_adaptation2:
        criterion2 = MetaCL(temperature=args.temperature)
    if args.model not in ['ast'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')

    
    
    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'domain_mlp_head' in k: #del domain_mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        if ckpt.get('classifier', None) is not None:
            class_classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    
    if args.method == 'ce':
        if args.domain_adaptation or args.domain_adaptation2:
            criterion = [criterion.cuda(), criterion2.cuda()]
        else:
            criterion = [criterion.cuda()]
    elif args.method == 'mscl':
        criterion = MetaCL(temperature=args.temperature).cuda()
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    if args.domain_adaptation or args.domain_adaptation2:
        classifier = [class_classifier.cuda(), domain_classifier.cuda()]
    else:
        classifier.cuda()
    projector.cuda()
    
    return model, classifier, projector, criterion, None

def main():
    args = parse_args()

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_model = None
    if args.dataset == 'icbhi' and args.method not in ['mscl']:
        best_acc = [0, 0, 0]  # Specificity, Sensitivity, Score
    
    args.transforms = SpecAugment(args)
    train_loader, val_loader, args = set_loader(args)
    model, classifier, projector, criterion, _ = set_model(args)

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
    
    from copy import deepcopy
        
    icbhi_labels = ['None', 'Crackle', 'Wheeze', 'Both']
    stetho_labels = ['Meditron', 'LittC2SE', 'Litt3200', 'AKGC417L']
    
    from MulticoreTSNE import MulticoreTSNE as TSNE
    
    # pip install seaborn
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    if not os.path.isdir('./t_sne_results'):
        os.makedirs('./t_sne_results')
    
    name = args.name
    
    
    ## eval
    embedding_eval = torch.zeros((2756, 768))
    lab_eval = torch.zeros((2756,))
    meta_eval = torch.zeros((2756,))
    
    for idx, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        if args.domain_adaptation or args.domain_adaptation2:
            class_labels = labels[0].cuda(non_blocking=True)
            meta_labels = labels[1].cuda(non_blocking=True)
    
        lab_eval[idx] = deepcopy(class_labels)
        meta_eval[idx] = deepcopy(meta_labels)
    
        with torch.no_grad():
            feat = model(images)
    
            embedding_eval[idx] = deepcopy(feat.cpu())
    
    emb = TSNE(n_jobs=20, perplexity=50).fit_transform(embedding_eval)
    
       
    df3 = pd.DataFrame()
    
    df3["y"] = lab_eval
    df3["comp-1"] = emb[:,0]
    df3["comp-2"] = emb[:,1]
    for_visual_icbhi_test_label = [icbhi_labels[int(x)] for x in df3.y.tolist()]
    
    #hue=[label_dict[x] for x in df.y.tolist()],
    sns.scatterplot(x="comp-1", y="comp-2", hue=for_visual_icbhi_test_label,
                    palette=sns.color_palette("hls", 4),
                    data=df3).set(title="ICBHI dataset T-SNE figure by labels on test set")
    
    plt.savefig(os.path.join('./t_sne_results/{}_test_label_icbhi.png'.format(name)))
    plt.cla() # Clear the current axes
    plt.clf() # Clear the current figure
    
    df4 = pd.DataFrame()
    df4["y"] = meta_eval
    df4["comp-1"] = emb[:,0]
    df4["comp-2"] = emb[:,1]
    for_visual_icbhi_test_stetho = [stetho_labels[int(x)] for x in df4.y.tolist()]
    
    sns.scatterplot(x="comp-1", y="comp-2", hue=for_visual_icbhi_test_stetho,
                    palette=sns.color_palette("hls", 4),
                    data=df4).set(title="ICBHI dataset T-SNE figure by stethoscope-labels on test set")
    
    plt.savefig(os.path.join('./t_sne_results/{}_test_device_icbhi.png'.format(args.name)))
    plt.cla() # Clear the current axes
    plt.clf() # Clear the current figure
    
    
    

if __name__ == '__main__':
    main()
