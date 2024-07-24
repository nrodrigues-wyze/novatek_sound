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

import librosa
import scipy
import torch.utils.data as data
from torch.utils.data import DataLoader
from utils.torch_utils import pad_split_input, move_data_to_device, audio_sample_pad


class AudioDataset(data.Dataset):
    def __init__(self, eval_data):

        self.target_sample_rate = 16000
        self.audio_clip_length = 2
        self.sample_length = self.audio_clip_length * self.target_sample_rate
        self.window_size = 1024
        self.hop_size = 320
        self.fmin = 20
        self.fmax = 14000
        self.mel_bins = 64
        self.sample_rate = 16000
        self.segment_length = 2
        self.overlap_length = 0



        self.data = pd.read_csv(eval_data)
        #import pdb; pdb.set_trace()
        self.data = self.data[self.data.targets != "not_sure"]
        self.y_true_dummies = pd.get_dummies(self.data.targets, drop_first = False)
        
    def __getitem__(self, index):
        entry = self.data.iloc[index]
        #print(entry['audio_name'])
        #print(entry['audio_name'])
        (waveform, orig_sample_rate) = sf.read(entry['audio_name'])

        start_time = max(0, entry['start_time'])
        end_time = min(len(waveform) / orig_sample_rate, entry['end_time'])
        waveform = waveform[int(start_time * orig_sample_rate): int(end_time * orig_sample_rate)]        
        
        if len(waveform) != waveform.size:  # double channe
            #print("Double Audio file", entry['audio_name'])
            waveform = waveform[:, 0]

        
        if orig_sample_rate != self.target_sample_rate:
            gcd = np.gcd(orig_sample_rate, self.target_sample_rate)
            # there are many other ways to do resampling
            waveform = scipy.signal.resample_poly(
                waveform, self.target_sample_rate // gcd, orig_sample_rate // gcd, axis=-1)

        audio_length = len(waveform)/self.target_sample_rate
        waveform = torch.tensor(waveform[None, :])


        #print(waveform.shape)
        if waveform.shape[1] < self.sample_length:
            n_clips = 1
            n_segs = int(ceil(
                waveform.shape[1] / ((self.segment_length - self.overlap_length) * self.target_sample_rate)))
            audio_segments = pad_split_input(
                waveform, n_clips, n_segs, self.segment_length, self.overlap_length, self.target_sample_rate)
        else:
            n_clips = int(ceil(waveform.shape[1] / self.sample_length))
            audio_clips = audio_sample_pad(
                waveform, n_clips, self.sample_length)
            n_segs = int(ceil(self.sample_length / ((self.segment_length -
                         self.overlap_length) * self.target_sample_rate)))
            audio_segments = pad_split_input(
                audio_clips, n_clips, n_segs, self.segment_length, self.overlap_length, self.target_sample_rate)

        audio_segments = np.array(audio_segments)[0]
        #print(audio_segments.shape)
        librosaspectrogram = librosa.stft(audio_segments, n_fft=self.window_size, hop_length=self.hop_size,
                                              win_length=self.window_size, window='hann', center=True,
                                              pad_mode='reflect') ** 2
        
        melfb = librosa.filters.mel(sr=self.sample_rate, n_fft=self.window_size, n_mels=self.mel_bins, fmin=self.fmin,
                                    fmax=self.fmax).T

        ggg = np.sqrt(librosaspectrogram.real ** 2 + librosaspectrogram.imag ** 2).T

        kkkkk = np.matmul(ggg, melfb)

        #import pdb; pdb.set_trace()

        waveform = (np.log10(np.clip(kkkkk, a_min=1e-10, a_max=np.inf)) * 10) - 10.0 * np.log10(np.maximum(1e-10, 1.0))
    
        label = entry['targets']
        return waveform, label

    def __len__(self):
        return self.data.shape[0]



def val(model_path, cfg, data_path, device, batch_size, training=True, results_file=""):
    labels = ['sound_other', 'dog_barking', 'cat_meowing', 'baby_crying' , 'people_talking', 'glass_breaking', 'gun_shooting', 'background']

    print(model_path)

    #labels = ['sound_other', 'dog_barking', 'cat_meowing', 'baby_crying' , 'people_talking']
    #print(labels)
    prob_dict = dict()
    results = dict()
    for label in labels:
        prob_dict[label] = []

    
    eval_data = AudioDataset(data_path)
    nw = 8#min(os.cpu_count(),batch_size)
    print(nw)
    eval_loader = torch.utils.data.DataLoader(eval_data,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = nw, pin_memory = True)
    if not training:
        clf = ClassificationEngine(model_path, cfg)

    
    #import pdb; pdb.set_trace()
    precision_dict = dict()
    recall_dict = dict()
    threshold_dict = dict()
    data = dict()
    #print(eval_data.y_true_dummies)
    for batch_eval in tqdm(eval_loader):
        batch_input = move_data_to_device(batch_eval[0], device)
        results = clf.val_audio_ingenic(batch_input)
        
        '''
        import pdb; pdb.set_trace()
        print("predictions", "Labels")
        for pred,gt in zip(results['probs'], batch_eval[1]):
            print(labels[pred.argmax(axis=0)],gt)
        import pdb; pdb.set_trace()
        '''
        #print(results)
        for result in results['probs']:
            #print(result)
            for n in range(len(labels)):
                #print(result)
                prob_dict[labels[n]].append(result[n])        
    results = dict()
    for label in labels:
        y_true = eval_data.y_true_dummies.eval(label)
        #data[label] = prob_dict[label]
        results[label] = average_precision_score(y_true, prob_dict[label])
        #y_true = eval_data.y_true_dummies.eval(label)
        #precision_dict[label], recall_dict[label], threshold_dict[label] = precision_recall_curve(y_true,
                                                                                          #prob_dict[label])
    

    if results_file!="":
        with open(results_file, "a") as f:
            avg_map = (results['dog_barking'] + results['cat_meowing'] + results['baby_crying'] + results['people_talking'] + results['glass_breaking'] + results['gun_shooting']) / 6
            f.write("%s %f %f %f %f %f %f %f\n"%(model_path.split('/')[-1], results['dog_barking'], results['cat_meowing'], results['baby_crying'], results['people_talking'], results['glass_breaking'], results['gun_shooting'], avg_map))
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Model AP analysis')
    parser.add_argument('--model_path', type = str, required = True)
    parser.add_argument('--cfg', type = str, required = True)
    parser.add_argument('--data_path', type = str, required = True)
    parser.add_argument('--results-file', type = str, required = True)
    parser.add_argument('--device', type = int, required = True)
    parser.add_argument('--batch_size', type = int, required = True)
    parameters = parser.parse_args()

    parameters.device='cuda:0'
    final_result = val(parameters.model_path,parameters.cfg, parameters.data_path,parameters.device, parameters.batch_size, False, parameters.results_file)
    print(final_result)
