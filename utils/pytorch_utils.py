import sys
import os
import json
import glob
from easydict import EasyDict as edict
import warnings
import numpy as np
import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
pjoin = os.path.join

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0:: 2].transpose(0, -1) * mixup_lambda[0:: 2] +
           x[1:: 2].transpose(0, -1) * mixup_lambda[1:: 2]).transpose(0, -1)
    return out

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0:: 2].transpose(0, -1) * mixup_lambda[0:: 2] +
           x[1:: 2].transpose(0, -1) * mixup_lambda[1:: 2]).transpose(0, -1)
    return out

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def forward(model, generator, return_input=False,
            return_target=False):
    """Forward data to a model.
    Args:
      model: object
      generator: object
      return_input: bool
      return_target: bool
    Returns:
      clipwise_output: (audios_num, classes_num)
      (ifexist) segmentwise_output: (audios_num, segments_num, classes_num)
      (ifexist) framewise_output: (audios_num, frames_num, classes_num)
      (optional) return_input: (audios_num, segment_samples)
      (optional) return_target: (audios_num, classes_num)
    """
    output_dict = {}
    device = next(model.parameters()).device
    print(device)
    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        # print(n)
        batch_input = batch_data_dict['waveform']
        batch_input = move_data_to_device(batch_input, device)
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            model.eval()
            
            output = model(batch_input)
            batch_output = {'clipwise_output': output[0]}

        append_to_dict(output_dict, 'clipwise_output',
                       batch_output['clipwise_output'].data.cpu().numpy())

        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])

        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis = 0)

    return output_dict


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

class MyCollator(object):
    def __init__(self, augment, signal_rate):
        self.augment = augment
        self.signal_rate = signal_rate
        # self.augment_fn = Compose([
        #     AddGaussianNoise(min_amplitude = 0.001, max_amplitude = 0.015, p = 0.5),
        #     TimeStretch(min_rate = 0.8, max_rate = 1.25, p = 0.5),
        #     PitchShift(min_semitones = -4, max_semitones = 4, p = 0.5),
        #     Shift(min_fraction = -0.5, max_fraction = 0.5, p = 0.5),
        # ])
        self.augment_fn = Compose([
            TimeStretch(min_rate = 0.8, max_rate = 1.2, p = 0.5),
            PitchShift(min_semitones = -2, max_semitones = 2, p = 0.5),
            Shift(min_shift = -0.2, max_shift = 0.2, p = 0.5),
        ])

    def __call__(self, list_data_dict):
        """Collate data.
        Args:
            list_data_dict, e.g., ['waveform': (clip_samples,), 'target':(class_nums)},
                                  {'waveform': (clip_samples,), 'target':(class_nums)},
                                    ...]
        Returns:
            np_data_dict, dict, e.g.,
                {'waveform': (batch_size, clip_samples), 'targets': (batch_size, class_nums)}
        """
        test_data = torch.tensor(np.array([data_dict['waveform'] for data_dict in list_data_dict]))

        #print(test_data, test_data.shape)
        #import pdb; pdb.set_trace()

        np_data_dict = {}
        if self.augment:
            np_data_dict['waveform'] = np.array([data_dict['waveform'] for data_dict in list_data_dict])
            np_data_dict['waveform'] = torch.tensor(self.augment_fn(np_data_dict['waveform'], self.signal_rate))
        else:
            np_data_dict['waveform'] = torch.tensor(np.array([data_dict['waveform'] for data_dict in list_data_dict]))
        np_data_dict['target'] = torch.tensor(np.array([data_dict['target'] for data_dict in list_data_dict]))
        return np_data_dict
