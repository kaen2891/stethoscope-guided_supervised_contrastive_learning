from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio


class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):
        data_folder = os.path.join(args.data_folder, 'icbhi_dataset/audio_test_data')
        folds_file = os.path.join(args.data_folder, 'icbhi_dataset/patient_list_foldwise.txt')
        official_folds_file = os.path.join(args.data_folder, 'icbhi_dataset/official_split.txt')
        test_fold = args.test_fold
        
        self.data_folder = data_folder
        self.train_flag = train_flag
        self.split = 'train' if train_flag else 'test'
        self.transform = transform
        self.args = args
        self.mean_std = mean_std

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.n_mels = args.n_mels

        # ==========================================================================
        """ get ICBHI dataset meta information """
        # store stethoscope device information for each file or patient
        device_id, self.device_to_id = 0, {}
        self.device_id_to_patient, self.file_to_device = {}, {}
        self.device_to_id = {'Meditron': 0, 'LittC2SE': 1, 'Litt3200': 2, 'AKGC417L': 3}
        self.device_id_to_patient = {0: [], 1: [], 2: [], 3: []}

        filenames = sorted(glob(data_folder+'/*')) #--> 1840: 920 wav + 920 txt
        filenames = set(f.strip().split('/')[-1].split('.')[0] for f in filenames if '.wav' in f or '.txt' in f)
        filenames = sorted(list(set(filenames))) #--> 920
        
        for f in filenames:
            f += '.wav'
            # get the total number of devices from original dataset (icbhi dataset has 4 stethoscope devices)
            device = f.strip().split('_')[-1].split('.')[0] #-->Meditron / LittC2SE / Litt3200 / AKGC417L 
            # get the device information for each wav file            
            self.file_to_device[f.strip().split('.')[0]] = self.device_to_id[device]

            pat_id = f.strip().split('_')[0] # 101
            if pat_id not in self.device_id_to_patient[self.device_to_id[device]]:
                self.device_id_to_patient[self.device_to_id[device]].append(pat_id) #0: ['101']
                
        # store all metadata (age, sex, adult_BMI, child_weight, child_height, device_index)
        self.file_to_metadata = {}
        meta_file = pd.read_csv(os.path.join(args.data_folder, 'icbhi_dataset/metadata.txt'), names=['age', 'sex', 'adult_BMI', 'child_weight', 'child_height', 'chest_location'], delimiter= '\t')
        
        meta_file['chest_location'].replace({'Tc':0, 'Al':1, 'Ar':2, 'Pl':3, 'Pr':4, 'Ll':5, 'Lr':6}, inplace=True) # Tc --> 0, Al -> 1, ...
        for f in filenames:
            pat_idx = int(f.strip().split('_')[0])
            info = list(meta_file.loc[pat_idx])
            info[1] = 0 if info[1] == 'M' else 1 # --> Man:0, Woman:1
            
            info = np.array(info)
            for idx in np.argwhere(np.isnan(info)):
                info[idx] = -1
            
            self.file_to_metadata[f] = torch.tensor(np.append(info, self.file_to_device[f.strip()])) #age, sex, adult_BMI, child_weight, child_height, chest_location, device
            
        
        """ train-test split based on train_flag and test_fold """
        if test_fold in ['0', '1', '2', '3', '4']:  # from RespireNet, 80-20% split
            
            patient_dict = {}
            all_patients = open(folds_file).read().splitlines()
            for line in all_patients:
                idx, fold = line.strip().split(' ')
                if train_flag and int(fold) != int(test_fold):
                    patient_dict[idx] = fold
                elif train_flag == False and int(fold) == int(test_fold):
                    patient_dict[idx] = fold
            
            if print_flag:
                print('*' * 20)
                print('Train and test 80-20% split with test_fold {}'.format(test_fold))
                print('Patience number in {} dataset: {}'.format(self.split, len(patient_dict)))
        else:
            patient_dict = {}
            all_fpath = open(official_folds_file).read().splitlines()
            
            for line in all_fpath:
                fpath, fold = line.strip().split('\t') #--> fpath: '101_1b1_Al_sc_Meditron' / fold --> test
                if train_flag and fold == 'train': # using for training set
                    patient_dict[fpath] = fold
                elif not train_flag and fold == 'test': # using for test set
                    patient_dict[fpath] = fold

            if print_flag:
                print('*' * 20)
                print('Train and test 60-40% split with test_fold {}'.format(test_fold))
                print('File number in {} dataset: {}'.format(self.split, len(patient_dict)))
        
        annotation_dict = get_annotations(args, data_folder)

        self.filenames = []
        for f in filenames:
            # for 'official' test_fold, two patient dataset contain both train and test samples
            idx = f.split('_')[0] if test_fold in ['0', '1', '2', '3', '4'] else f
            
            if idx in patient_dict:
                self.filenames.append(f)
    
        self.audio_data = []  # each sample is a tuple with (audio_data, label, metadata)

        if print_flag:
            print('*' * 20)  
            print("Extracting individual breathing cycles..")

        self.cycle_list = []
        for idx, filename in enumerate(self.filenames):
            sample_data = get_individual_cycles_torchaudio(args, annotation_dict[filename], self.file_to_metadata[filename], data_folder, filename, self.sample_rate, args.n_cls)
            
            cycles_with_labels = [(data[0], data[1], data[2]) for data in sample_data]
            self.cycle_list.extend(cycles_with_labels) 
        
        for sample in self.cycle_list:
            self.audio_data.append(sample)
        
        self.class_nums = np.zeros(args.n_cls)
        if args.m_cls:
            self.domain_nums = np.zeros(args.m_cls)
            
        for sample in self.audio_data:
            if args.domain_adaptation or args.domain_adaptation2:
                self.class_nums[sample[1]] += 1
                self.domain_nums[sample[2]] += 1
            else:
                self.class_nums[sample[1]] += 1
            
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        if args.m_cls:
            self.domain_ratio = self.domain_nums / sum(self.domain_nums) * 100
        
        if print_flag:
            print('[Preprocessed {} dataset information]'.format(self.split))
            print('total number of audio data: {}'.format(len(self.audio_data)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
            
            if args.domain_adaptation or args.domain_adaptation2:
                print('*' * 25)
                print('For the Meta Label Distribution')            
                for i, (n, p) in enumerate(zip(self.domain_nums, self.domain_ratio)):
                    print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.meta_cls_list[i]+')', int(n), p))
        
        # ==========================================================================
        """ convert fbank """
        self.audio_images = []
        for index in range(len(self.audio_data)): #for the training set, 4142
            audio, label, meta_label = self.audio_data[index][0], self.audio_data[index][1], self.audio_data[index][2] # wav, label, metadata --> [1, 128000], 0~3, [1, 7]
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if aug_idx > 0:
                    if self.train_flag:
                        audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)                
                    
                    image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                    audio_image.append(image)
                else:
                    image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                    audio_image.append(image)
            self.audio_images.append((audio_image, label, meta_label))
        
        # ==========================================================================

    def __getitem__(self, index):
        audio_images, label, meta_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]

        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        if meta_label is None or not self.train_flag:
            return audio_image, label
        else:
            return audio_image, (label, meta_label)

    def __len__(self):
        return len(self.audio_data)