import pandas as pd
import numpy as np
import scipy
import soundfile as sf
import librosa
import torch
import h5py
from math import ceil

def pad_split_input(x, batch_size, n_segs, segment_length, overlap_length, signal_rate):                   
    """                                                                                                    
    Pad and split audio frames into short frames                                                           
    Args:                                                                                                  
      x: (batch_size, audio_length)                                                                        
      n_segs: target number of short segmentations                                                         
      segment_length: target length for segmentation                                                       
      overlap_length: target length for overlapping between two short frames                               
      signal_rate: audio signal rate                                                                       
    Outputs:                                                                                               
     output: (batch_size * n_segs, segment_length * signal_rate)                                           
    """                                                                                                    
    #import pdb; pdb.set_trace()                                                                           
    total_length = int(((n_segs - 1) * (segment_length - overlap_length) + segment_length) * signal_rate)  
    if x.shape[1] < total_length:                                                                          
        x = torch.cat((x.float(), torch.zeros(batch_size, total_length - x.shape[1])), axis = 1)           
    res = torch.zeros((batch_size * n_segs, int(segment_length * signal_rate)))                            
    for i in range(batch_size):                                                                            
        # print(i)                                                                                         
        for j in range(n_segs):                                                                            
            # print(j)                                                                                     
            start_time = (segment_length - overlap_length) * j                                             
            end_time = start_time + segment_length                                                         
            # print(i* batch_size + j)                                                                     
            res[i * n_segs + j, :] = x[i, int(start_time * signal_rate): int(end_time * signal_rate)]      
    return res  

def float32_to_int16(x):
    # assert np.max(np.abs(x)) <= 1.
    if torch.max(torch.abs(x)) > 1.:
        x /= torch.max(torch.abs(x))
    x = x.numpy()
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

target_sample_rate, sample_rate = 16000, 16000
segment_length = 2 
overlap_length = 0 
audio_clip_length = 2
sample_length = audio_clip_length * target_sample_rate   
window_size = 1024                                                 
hop_size = 320                                                     
fmin = 20                                                          
fmax = 14000                                                       
mel_bins = 64                                                      
n_clips = 1 
df = pd.read_csv('audio_annotation_latest.csv')

#full_path,start_time,end_time,targets
#'audio_path', 'target', 'waveform'

labels = ['sound_other', 'dog_barking', 'cat_meowing', 'baby_crying' , 'people_talking', 'glass_breaking', 'gun_shooting', 'background']


mapper = {'siren':'sound_other',
          'gunshot':'gun_shooting'}

audio_paths = []
targets = []
waveforms = []

for index, row in df.iterrows():
    if index<1418865:
        continue
    if index%10000==0:
        print(index)
    file_name = row['full_path'].replace('/mnt/data/', '/data/data/')
    start_time = row['start_time']
    end_time = row['end_time']
    try:
        (waveform, orig_sample_rate) = sf.read(file_name)

        waveform = waveform[int(start_time * orig_sample_rate): int(end_time * orig_sample_rate)]
        if len(waveform) != waveform.size:  # double channe                                 
            print("Double Audio file", entry['audio_name'])                                 
            waveform = waveform[:, 0]                                                       
                                                                                        
        audio_length = len(waveform)/target_sample_rate   
        waveform = torch.tensor(waveform[None, :])     
        
        n_segs = int(ceil(                                                                                
            waveform.shape[1] / ((segment_length - overlap_length) * target_sample_rate))) 
        
        audio_segments = pad_split_input(                                                                 
            waveform, n_clips, n_segs, segment_length, overlap_length, target_sample_rate) 

        for audio_segment in audio_segments:
            # audio_segments = np.array(audio_segments)[0]
            if row['targets'] not in labels:
                row['targets'] = mapper[row['targets']]
            waveform_16 = float32_to_int16(audio_segment)
            audio_paths.append(row['full_path'])
            onehot = np.array([0.0]*len(labels)).astype(np.float32)
            onehot[labels.index(row['targets'])] = 1.0
            targets.append(onehot)
            waveforms.append(waveform_16)

    except:
        print(file_name)

with h5py.File('new_train_data.h5','w') as f:
    f['audio_path'] = audio_paths
    f['target'] = targets
    f['waveform'] = waveforms

# df = pd.DataFrame({'audio_path': audio_paths, 'target': targets, 'waveform':waveforms}) 
# df.to_hdf('new_train_.h5', key='df', mode='w')  

#audio_length = len(waveform)/target_sample_rate   
#waveform = torch.tensor(waveform[None, :])             

import pdb; pdb.set_trace()
'''
if len(waveform) != waveform.size:  # double channe                                 
    print("Double Audio file", entry['audio_name'])                                 
    waveform = waveform[:, 0]                                                       
                                                                                    
audio_length = len(waveform)/target_sample_rate   
waveform = torch.tensor(waveform[None, :])             

n_clips = 1                                                                                       
n_segs = int(ceil(                                                                                
    waveform.shape[1] / ((segment_length - overlap_length) * target_sample_rate))) 
audio_segments = pad_split_input(                                                                 
    waveform, n_clips, n_segs, segment_length, overlap_length, target_sample_rate) 
audio_segments = np.array(audio_segments)[0]


if orig_sample_rate != target_sample_rate:                                     
    gcd = np.gcd(orig_sample_rate, target_sample_rate)                         
    # there are many other ways to do resampling                                    
    waveform = scipy.signal.resample_poly(                                          
        waveform, target_sample_rate // gcd, orig_sample_rate // gcd, axis=-1) 

librosaspectrogram = librosa.stft(audio_segments, n_fft=window_size, hop_length=hop_size,   
                                      win_length=window_size, window='hann', center=True,        
                                      pad_mode='reflect') ** 2                                        
                                                                                                      
melfb = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
                            fmax=fmax).T                                                         
                                                                                                      
ggg = np.sqrt(librosaspectrogram.real ** 2 + librosaspectrogram.imag ** 2).T                          
                                                                                                      
kkkkk = np.matmul(ggg, melfb)                                                                         
                                                                                                      
#import pdb; pdb.set_trace()                                                                          
                                                                                                      
waveform = (np.log10(np.clip(kkkkk, a_min=1e-10, a_max=np.inf)) * 10) - 10.0 * np.log10(np.maximum(1e-10, 1.0))
'''
                                                                                           
                                                                                                   

import pdb; pdb.set_trace()
