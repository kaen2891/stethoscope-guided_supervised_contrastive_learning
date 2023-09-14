from collections import namedtuple
import os
import math
import random
from tkinter import W
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torchaudio
from torchaudio import transforms as T

from .augmentation import augment_raw_audio

__all__ = ['get_annotations', 'save_image', 'get_mean_and_std', 'get_individual_cycles_torchaudio', 'generate_mel_spectrogram', 'generate_fbank', 'get_score']


# ==========================================================================
""" ICBHI dataset information """
def _extract_lungsound_annotation(file_name, data_folder):
    tokens = file_name.strip().split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient Number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_folder, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')

    return recording_info, recording_annotations


def get_annotations(args, data_folder):
    if args.class_split == 'lungsound' or args.class_split in ['lungsound_meta', 'meta']: # 4-class
        filenames = sorted(glob(data_folder+'/*')) #--> 1840: 920 wav + 920 txt
        filenames = set(f.strip().split('/')[-1].split('.')[0] for f in filenames if '.txt' in f)
        filenames = sorted(list(set(filenames))) #--> 920

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            annotation_dict[f] = ann

    elif args.class_split == 'diagnosis':
        filenames = sorted(glob(data_folder+'/*')) #--> 1840: 920 wav + 920 txt
        filenames = set(f.strip().split('/')[-1].split('.')[0] for f in filenames if '.txt' in f)
        filenames = sorted(list(set(filenames))) #--> 920
        tmp = pd.read_csv(os.path.join(args.data_folder, 'icbhi_dataset/patient_diagnosis.txt'), names=['Disease'], delimiter='\t')

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            ann.drop(['Crackles', 'Wheezes'], axis=1, inplace=True)

            disease = tmp.loc[int(f.strip().split('_')[0]), 'Disease']
            ann['Disease'] = disease

            annotation_dict[f] = ann
            
    return annotation_dict


def save_image(image, fpath):
    save_dir = os.path.join(fpath, 'image.jpg')
    cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    

def get_mean_and_std(dataset):
    """ Compute the mean and std value of mel-spectrogram """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    cnt = 0
    fst_moment = torch.zeros(1)
    snd_moment = torch.zeros(1)
    for inputs, _, _ in dataloader:
        b, c, h, w = inputs.shape
        nb_pixels = b * h * w

        fst_moment += torch.sum(inputs, dim=[0,2,3])
        snd_moment += torch.sum(inputs**2, dim=[0,2,3])
        cnt += nb_pixels

    mean = fst_moment / cnt
    std = torch.sqrt(snd_moment/cnt - mean**2)

    return mean, std
# ==========================================================================


# ==========================================================================
""" data preprocessing """

def _get_lungsound_label(crackle, wheeze, n_cls, args):
    if n_cls == 4 or args.method in ['mscl']:
        if crackle == 0 and wheeze == 0:
            return 0
        elif crackle == 1 and wheeze == 0:
            return 1
        elif crackle == 0 and wheeze == 1:
            return 2
        elif crackle == 1 and wheeze == 1:
            return 3
    
    elif n_cls == 2:
        if crackle == 0 and wheeze == 0:
            return 0
        else:
            return 1


def _get_diagnosis_label(disease, n_cls):
    if n_cls == 3:
        if disease in ['COPD', 'Bronchiectasis', 'Asthma']:
            return 1
        elif disease in ['URTI', 'LRTI', 'Pneumonia', 'Bronchiolitis']:
            return 2
        else:
            return 0

    elif n_cls == 2:
        if disease == 'Healthy':
            return 0
        else:
            return 1


def _slice_data_torchaudio(start, end, data, sample_rate):
    """
    SCL paper..
    sample_rate denotes how many sample points for one second
    """
    max_ind = data.shape[1]
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[:, start_ind: end_ind]


def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
        if data.dim() == 1:
            data = data.unsqueeze(0)
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data


def get_meta_infor(metadata, label, args):
    age = int(metadata[0])
    sex = int(metadata[1])
    loc = int(metadata[-2])
    dev = int(metadata[-1])
    
    
    if args.meta_mode == 'none':
        return None
    
    if args.meta_mode == 'age': # Age Meta-SCL
        if age >= 19:
            meta_label = 0
        else:
            meta_label = 1
        
    if args.meta_mode == 'sex':
        if sex == 0:
            meta_label = 0
        else:
            meta_label = 1
    
    if args.meta_mode == 'loc':
        meta_label = loc
    
    if args.meta_mode == 'dev':
        meta_label = dev
    
    if args.meta_mode == 'label':
        meta_label = label
                
    return meta_label

def get_individual_cycles_torchaudio(args, recording_annotations, metadata, data_folder, filename, sample_rate, n_cls):
    sample_data = []
    fpath = os.path.join(data_folder, filename+'.wav')
    data, sr = torchaudio.load(fpath)
    
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        data = resample(data)
    
    fade_samples_ratio = 16
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
    
    for idx in recording_annotations.index:
        row = recording_annotations.loc[idx]

        start = row['Start'] # start time (second)
        end = row['End'] # end time (second)
        audio_chunk = _slice_data_torchaudio(start, end, data, sample_rate)

        if args.class_split == 'lungsound':
            crackles = row['Crackles']
            wheezes = row['Wheezes']
            label = _get_lungsound_label(crackles, wheezes, n_cls, args)
            meta_label = get_meta_infor(metadata, label, args)
            
            sample_data.append((audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls, args), meta_label))
        elif args.class_split == 'diagnosis':
            disease = row['Disease']            
            sample_data.append((audio_chunk, _get_diagnosis_label(disease, n_cls), meta_label))
        
    padded_sample_data = []
    for data, label, m_label in sample_data:
        data = cut_pad_sample_torchaudio(data, args) # --> resample to [1, 128000] --> 8 seconds
        padded_sample_data.append((data, label, m_label))

    return padded_sample_data


def generate_fbank(args, audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    if args.model in ['ast']:
        mean, std =  -4.2677393, 4.5689974
    else:
        mean, std = fbank.mean(), fbank.std()
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank 


# ==========================================================================


# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================
