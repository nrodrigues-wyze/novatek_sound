import argparse
import os

import numpy as np
import torch
import pandas as pd
import soundfile as sf
from math import ceil
from openpyxl import load_workbook
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
#from wyze_audio_detection_inference.classify import ClassificationEngine
from classify import ClassificationEngine
from models import *
import h5py
import librosa
import scipy
import torch.utils.data as data
from torch.utils.data import DataLoader
import glob
from utils.torch_utils import pad_split_input, move_data_to_device, audio_sample_pad

test_path = "/data/data/sound_data/test_set_1103_wyze_clip_hfdata.h5"
device='cuda:0'
batch_size = 1024
nw =32
model_paths=['weights/model_novatek_32bit_may_1_input_255_new_clsweights/backup54000.pt','weights/model_novatek_32bit_16k_group_relu_input/backup1500.pt','weights/model_novatek_32bit_may_1_input_255_new_clsweights_2_pre/backup6000.pt']
#model_paths = glob.glob("weights/model_novatek_32bit_may_1_input_255_new_clsweights_2_int16/*")
model_paths.sort()
cfg = "cfg/sounddet_16x_8cls_stride_32bit_maxpool_v7_256.cfg"

labels = [b'sound_other', b'dog_barking', b'cat_meowing', b'baby_crying' , b'people_talking', b'glass_breaking', b'gun_shooting', b'background']

class Hdf5Dataset(data.Dataset):

    def __init__(self, hdf5file, waveform_key='waveform', label_key='targets'):
        self.hdf5file = hdf5file
        self.waveform_key = waveform_key
        self.label_key = label_key
        
        with h5py.File(self.hdf5file, 'r') as db:
            self.len = len(db[self.label_key])
            self.target = db[self.label_key][:].tolist()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        if idx >= self.len:
            return None
        
        with h5py.File(self.hdf5file, 'r') as db:

            waveform = torch.tensor(db[self.waveform_key][idx])
            target = db[self.label_key][idx]
        
        return waveform, target



for model_path in model_paths:
    eval_data = Hdf5Dataset(test_path)


    eval_loader = torch.utils.data.DataLoader(eval_data,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = nw, pin_memory = True)



    clf = ClassificationEngine(model_path, cfg)
    y_true_dummies = pd.get_dummies(eval_data.target, drop_first = False)

    prob_dict = dict()
    results = dict()
    for label in labels:
        prob_dict[label] = []

    #import pdb; pdb.set_trace()
    precision_dict = dict()
    recall_dict = dict()
    threshold_dict = dict()
    data = dict()

    for batch_eval in tqdm(eval_loader):
        batch_input = move_data_to_device(batch_eval[0], device)
        #print(batch_input)
        results = clf.val_audio_ingenic(batch_input)

        for result in results['probs']:
            for n in range(len(labels)):
                prob_dict[labels[n]].append(result[n])        

    #import pdb; pdb.set_trace()
    results = dict()
    for label in labels:

        results[label] = average_precision_score(y_true_dummies[label], prob_dict[label])
    print(model_path)
    print(results)

