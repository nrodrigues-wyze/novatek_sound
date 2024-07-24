from torch.utils import data
import h5py
import numpy as np
import os
import sys


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)



class Hdf5Dataset(data.Dataset):

    def __init__(self, hdf5file, waveform_key='waveform', label_key='target',
                 transform=None):
        self.hdf5file = hdf5file
        self.waveform_key = waveform_key
        self.label_key = label_key
        self.transform = transform

        with h5py.File(self.hdf5file, 'r') as db:
            self.len = len(db[self.label_key])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            return None
        
        with h5py.File(self.hdf5file, 'r') as db:
            waveform = int16_to_float32(db[self.waveform_key][idx])
            target = db[self.label_key][idx]
        sample = {'waveform': waveform, 'target': target}
        if self.transform:
            sample = self.transform(sample)
        return sample


def make_weights_for_balanced_classes(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        targets = hf['target'][:].astype(np.float32)

    (audios_num, classes_num) = targets.shape
    indexes_per_class = []

    for i in range(classes_num):
        indexes_per_class.append(np.sum(targets[:, i] == 1))
    weight_per_class = [0.] * classes_num

    for j in range(classes_num):
        weight_per_class[j] = audios_num / indexes_per_class[j]
    print("OLD", weight_per_class)
    #weight_per_class = [3.0,8.1,20.0,30.0,4.0,10.0,10.0,2.0]
    weight_per_class = [0.5,5.0,20.0,20.0,10.0,5.0,5.0,0.5]

    print("NEW",weight_per_class)
    weight = [0] * audios_num

    for k in range(audios_num):
        if len(np.where(targets[k, :] == 1)[0]) == 0:
            weight[k] = 0
        else:
            weight[k] = weight_per_class[np.where(targets[k, :] == 1)[0][0]]
    return weight

