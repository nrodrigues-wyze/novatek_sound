import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

file_1 = h5py.File('/data/data/SD_zyu/waveforms/all_train_20240103_real16k.h5','r+')
file_2 = h5py.File('new_train_data.h5','r+')

audio_path_1 = file_1['audio_path'][:]
audio_path_2 = file_2['audio_path'][:]

merged_audio_path = np.concatenate([audio_path_1, audio_path_2])

target_1 = file_1['target'][:]
target_2 = file_2['target'][:]

targets = np.concatenate([target_1, target_2])

# waveform_1 = file_1['waveform'][:]
waveform_2 = file_2['waveform'][:]

size_1 = file_1['waveform'].shape[0]
size_2 = waveform_2.shape[0]
waveform = np.zeros((size_1+size_2, 32000), dtype=np.int16)
waveform[size_1:size_1+size_2] = waveform_2
for i in range(0,size_1,100000):
    end = min(size_1, i+100000)
    waveform[i:end] = file_1['waveform'][i:end]
    # end = min(size_1, (i+1)*100000)
    # waveform_2.append(file_1['waveform'][i*100000:end])
    # import pdb; pdb.set_trace()

# waveforms = np.concatenate([waveform_1, waveform_2])

with h5py.File('all_train_20240503_real16k.h5','w') as f:
    f['audio_path'] = merged_audio_path
    f['target'] = targets
    f['waveform'] = waveform

