import math
import os
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

def init_seeds(seed=0):
    torch.manual_seed(seed)

    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False


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


def move_data_to_device(x, device):
    """
    Convert data into torch tensor and move it to cpu or gpu device
    Args:
        x: input data
        device: 'cpu' or 'gpu' device
    Returns: x in the device
    """
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def audio_sample_pad(x, n_clips, sample_length):
    res = torch.zeros((n_clips, sample_length))
    pad_size = x.shape[1] - (n_clips - 1) * sample_length
    for i in range(n_clips - 1):
        res[i, :] = x[0, i * sample_length: (i + 1) * sample_length]
    res[(n_clips - 1), 0:pad_size] = x[0, (n_clips - 1) * sample_length:(pad_size + (n_clips - 1) * sample_length)]
    return res

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '# + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))

class ModelEMA:

    def __init__(self, model, decay=0.9999, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.updates = 0  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 1000))  # decay exponential ramp (to help early epochs)
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.ema, k, getattr(model, k))
