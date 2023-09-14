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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class, Projector
from method import MetaCL, PatchMixLoss, PatchMixConLoss


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
    # for SSAST
    parser.add_argument('--ssast_task', type=str, default='ft_avgtok', 
                        help='pretraining or fine-tuning task', choices=['ft_avgtok', 'ft_cls'])
    parser.add_argument('--fshape', type=int, default=16, 
                        help='fshape of SSAST')
    parser.add_argument('--tshape', type=int, default=16, 
                        help='tshape of SSAST')
    parser.add_argument('--ssast_pretrained_type', type=str, default='Patch', 
                        help='pretrained ckpt version of SSAST model')

    parser.add_argument('--method', type=str, default='ce')
    parser.add_argument('--domain_adaptation', action='store_true')
    parser.add_argument('--domain_adaptation2', action='store_true')    
    # Meta Domain CL & Patch-Mix CL loss
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--mix_beta', default=1.0, type=float,
                        help='patch-mix interpolation coefficient')
    parser.add_argument('--time_domain', action='store_true',
                        help='patchmix for the specific time domain')

    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--negative_pair', type=str, default='all',
                        help='the method for selecting negative pair', choices=['all', 'diff_label'])
    parser.add_argument('--target_type', type=str, default='project_flow',
                        help='how to make target representation', choices=['project_flow', 'grad_block1', 'grad_flow1', 'project_block1', 'grad_block2', 'grad_flow2', 'project_block2', 'project_block_all', 'representation_all'])
    
    # Meta for SCL
    parser.add_argument('--meta_mode', type=str, default='none',
                        help='the meta information for selecting', choices=['none', 'age', 'sex', 'loc', 'dev', 'label'])
                        
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method) if args.meta_mode == 'none' else '{}_{}_{}_{}'.format(args.dataset, args.model, args.method, args.meta_mode)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    if args.method in ['patchmix', 'patchmix_cl']:
        assert args.model in ['ast', 'ssast']
    
    if args.domain_adaptation:
        args.save_folder = os.path.join(args.save_dir, 'da', args.model_name)
    elif args.domain_adaptation2:
        args.save_folder =  os.path.join(args.save_dir, 'da2', args.model_name)
    else:
        args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.domain_adaptation or args.domain_adaptation2:
        if args.meta_mode in ['age', 'sex']:
            args.m_cls = 2
        elif args.meta_mode in ['dev', 'label']:
            args.m_cls = 4
        elif args.meta_mode in ['loc']:
            args.m_cls = 7
            
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.domain_adaptation or args.domain_adaptation2:
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
                            
            
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']
                
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
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
        #args.h, args.w = 798, 128
        train_transform = [transforms.ToTensor(),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        ##
        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)

        args.class_nums = train_dataset.class_nums
        if args.domain_adaptation or args.domain_adaptation2:
            args.domain_nums = train_dataset.domain_nums
    else:
        raise NotImplemented    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, val_loader, args
    

def set_model(args):
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        if args.domain_adaptation:
            kwargs['domain_label_dim'] = args.m_cls
    elif args.model == 'ssast':
        kwargs['label_dim'] = args.n_cls
        kwargs['fshape'], kwargs['tshape'] = args.fshape, args.tshape
        kwargs['fstride'], kwargs['tstride'] = 10, 10
        kwargs['input_tdim'] = 798
        kwargs['task'] = args.ssast_task
        kwargs['pretrain_stage'] = not args.audioset_pretrained
        kwargs['load_pretrained_mdl_path'] = args.ssast_pretrained_type
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        if args.domain_adaptation:
            kwargs['domain_label_dim'] = args.m_cls # not debugging yet

    model = get_backbone_class(args.model)(**kwargs)
    if args.domain_adaptation:
        class_classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
        domain_classifier = nn.Linear(model.final_feat_dim, args.m_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.domain_mlp_head)
    elif args.domain_adaptation2:
        class_classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
        domain_classifier = Projector(model.final_feat_dim, args.proj_dim)
    else:
        classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method in ['patchmix_cl'] or args.domain_adaptation2 else nn.Identity()
    
    
    if not args.weighted_loss:
        weights = None
        criterion = nn.CrossEntropyLoss()
    else:
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        
    if args.domain_adaptation:
        criterion2 = nn.CrossEntropyLoss()
        
    elif args.domain_adaptation2:
        criterion2 = MetaCL(temperature=args.temperature)
           
    if args.model not in ['ast', 'ssast'] and args.from_sl_official:
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
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        if ckpt.get('classifier', None) is not None:
            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    if args.method == 'ce':
        if args.domain_adaptation or args.domain_adaptation2:
            criterion = [criterion.cuda(), criterion2.cuda()]
        else:
            criterion = [criterion.cuda()]
    elif args.method == 'patchmix':
        
        if args.domain_adaptation:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda(), PatchMixLoss(criterion=criterion2).cuda()]
        elif args.domain_adaptation2:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]
        else:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda()]
    elif args.method == 'patchmix_cl':
        criterion = [criterion.cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    if args.domain_adaptation or args.domain_adaptation2:
        classifier = [class_classifier.cuda(), domain_classifier.cuda()]
    else:
        classifier.cuda()
    projector.cuda()
    
    
    if args.domain_adaptation or args.domain_adaptation2:
        optim_params = list(model.parameters()) + list(classifier[0].parameters()) + list(classifier[-1].parameters()) + list(projector.parameters())
    else:
        optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, projector, criterion, optimizer

def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    if args.domain_adaptation or args.domain_adaptation2:
        classifier[0].train()
        classifier[1].train()
    else:
        classifier.train()
    projector.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        # data load
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        if args.domain_adaptation or args.domain_adaptation2:
            class_labels = labels[0].cuda(non_blocking=True)
            meta_labels = labels[1].cuda(non_blocking=True)
        else:
            labels = labels.cuda(non_blocking=True)
        bsz = class_labels.shape[0] if args.domain_adaptation or args.domain_adaptation2 else labels.shape[0]
        
        
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                if args.domain_adaptation or args.domain_adaptation2:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict()), deepcopy(projector.state_dict())]
                    p = float(idx + epoch * len(train_loader)) / args.epochs + 1 / len(train_loader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]
                    alpha = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                features = model(args.transforms(images), args=args, alpha=alpha, training=True)
                if args.domain_adaptation:
                    #features = (features, domain_features) # domain_features -> ReverseLayerF                    
                    
                    output = classifier[0](features[0])
                    class_loss = criterion[0](output, class_labels)
                                   
                    meta_output = classifier[1](features[1])
                    meta_loss = criterion[1](meta_output, meta_labels)
                    
                    loss = class_loss + args.alpha * meta_loss
                
                elif args.domain_adaptation2:
                    output = classifier[0](features[0])
                    class_loss = criterion[0](output, class_labels)
                    
                    features2 = model(args.transforms(images), args=args, alpha=alpha, training=True)
                    
                    
                    if args.target_type == 'project_flow_all':
                        proj1, proj2 = classifier[1](features[1]), classifier[1](features2[1]) # classifier[1] = projector #
                    
                    
                    elif args.target_type == 'representation_all':
                        proj1, proj2 = features[1], features2[1]
                    
                    #target type
                    #elif args.target_type == 'grad_block1':
                    elif args.target_type == 'z1block_project':
                        proj1 = deepcopy(features[1].detach())
                        proj2 = classifier[1](features2[1])
                    
                        
                    #elif args.target_type == 'grad_flow1':
                    elif args.target_type == 'z1_project2':
                        proj1 = features[1]
                        proj2 = classifier[1](features2[1])
                    
                    
                    #elif args.target_type == 'project_block1':
                    elif args.target_type == 'project1block_project2':
                        proj1 = deepcopy(classifier[1](features[1]).detach())
                        proj2 = classifier[1](features2[1])
                    
                    
                    #elif args.target_type == 'grad_block2':
                    elif args.target_type == 'project1_r2block':
                        proj1 = classifier[1](features[0])
                        proj2 = deepcopy(features2[1].detach())
                    
                        
                    #elif args.target_type == 'grad_flow2':
                    elif args.target_type == 'project1_r2':
                        proj1 = classifier[1](features[0])
                        proj2 = features2[1]
                    
                    
                    #elif args.target_type == 'project_block2':
                    elif args.target_type == 'project1_project2block':
                        proj1 = classifier[1](features[1])
                        proj2 = deepcopy(classifier[1](features2[1]).detach())
                    
                    elif args.target_type == 'project_block_all':
                        proj1 = deepcopy(classifier[1](features[1]).detach())
                        proj2 = deepcopy(classifier[1](features2[1]).detach())
                    
                    meta_loss = criterion[1](proj1, proj2, meta_labels) #meta cl loss
                    
                    loss = class_loss + args.alpha * meta_loss
                
                else:
                    output = classifier(features)
                    loss = criterion[0](output, labels)
                    

            elif args.method == 'patchmix':
                mix_images, labels_a, labels_b, lam, index = model(args.transforms(images), y=class_labels if args.domain_adaptation or args.domain_adaptation2 else labels, 
                    y2=meta_labels if args.domain_adaptation or args.domain_adaptation2 else None, da_index=None, args=args, alpha=alpha, patch_mix=True, time_domain=args.time_domain, training=True)
                
                if args.domain_adaptation:
                    output = classifier[0](mix_images[0])
                    class_loss = criterion[1](output, labels_a[0], labels_b[0], lam)
                    
                    meta_output = classifier[1](mix_images[1])
                    meta_loss = criterion[2](meta_output, labels_a[1], labels_b[1], lam)
                    
                    loss = class_loss + args.alpha * meta_loss
                    
                elif args.domain_adaptation2:
                    output = classifier[0](mix_images[0])
                    class_loss = criterion[1](output, labels_a[0], labels_b[0], lam)
                    
                    mix_images2, labels_a2, labels_b2, lam2, index2 = model(args.transforms(images), y=class_labels if args.domain_adaptation or args.domain_adaptation2 else labels, 
                        y2=meta_labels if args.domain_adaptation or args.domain_adaptation2 else None, da_index=(lam, index), args=args, alpha=alpha, patch_mix=True, time_domain=args.time_domain, training=True)
                    proj1, proj2 = classifier[1](mix_images[1]), classifier[1](mix_images2[1]) # classifier[1] = projector // #deepcopy for here? for ablation?
                    
                    meta_loss = criterion[2](proj1, proj2, labels_b, labels_b2, lam, index, args) #patchmix cl loss

                    loss = class_loss + args.alpha * meta_loss
                
                else:
                    output = classifier(mix_images)
                    loss = criterion[1](output, labels_a, labels_b, lam)

            elif args.method == 'patchmix_cl':
                features = model(args.transforms(images))
                output = classifier(features)
                loss = criterion[0](output, labels)

                if args.target_type == 'grad_block':
                    proj1 = deepcopy(features.detach())
                elif args.target_type == 'grad_flow':
                    proj1 = features
                elif args.target_type == 'project_block':
                    proj1 = deepcopy(projector(features).detach())
                elif args.target_type == 'project_flow':
                    proj1 = projector(features)

                # use 'patchmix_cl' for augmentation
                mix_images, labels_a, labels_b, lam, index = model(args.transforms(images), y=labels, args=args, patch_mix=True, time_domain=args.time_domain)
                proj2 = projector(mix_images)
                loss += args.alpha * criterion[1](proj1, proj2, labels, labels_b, lam, index, args)

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], class_labels if args.domain_adaptation or args.domain_adaptation2 else labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                if args.domain_adaptation or args.domain_adaptation2:
                    classifier[0] = update_moving_average(args.ma_beta, classifier[0], ma_ckpt[1])
                    classifier[1] = update_moving_average(args.ma_beta, classifier[1], ma_ckpt[2])
                else:
                    classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    
    if args.domain_adaptation or args.domain_adaptation2:
        classifier = classifier[0]
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            

            with torch.cuda.amp.autocast():
                features = model(images, args=args, training=False)
                output = classifier(features)
                loss = criterion[0](output, labels)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels[idx].item()] += 1.0

            sp, se, sc = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_model = None
    if args.dataset == 'icbhi':
        best_acc = [0, 0, 0]  # Specificity, Sensitivity, Score
    
    args.transforms = SpecAugment(args)
    train_loader, val_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
     
    if not args.eval:
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            loss, acc = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
            
            # eval for one epoch
            best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
            
            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)

        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier[0].load_state_dict(best_model[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(val_loader, model, classifier, criterion, args, best_acc)

    if args.domain_adaptation:
        update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results_da.json'))
    elif args.domain_adaptation2:
        update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results_da2.json'))
    else:
        update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    
if __name__ == '__main__':
    main()
