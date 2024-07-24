import torch.nn.functional as F
import torch.nn as nn
from utils.parse_config import *
from utils.utils import *
import sys
# from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

def create_modules(module_defs, img_size):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    ##quantize
    target_device = hyperparams['target_device']
    bita = int(hyperparams['bita'])
    bitw = int(hyperparams['bitw'])
    is_quantize = bool(hyperparams['is_quantize'])

    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    window_size, hop_size, window_size, sample_rate, mel_bins, fmin, fmax = 1024, 320, 1024, 16000, 64, 20, 14000
    # Spectrogram extractor
    '''
    module_list.append(Spectrogram(n_fft = window_size, hop_length = hop_size,
                                            win_length = window_size, window = window, center = center,
                                            pad_mode = pad_mode,
                                            freeze_parameters = True))

    # Logmel feature extractor
    module_list.append(LogmelFilterBank(sr = sample_rate, n_fft = window_size,
                                            n_mels = mel_bins, fmin = fmin, fmax = fmax, ref = ref, amin = amin,
                                            top_db = top_db,
                                            freeze_parameters = True))
    '''

    #modue_list.append()
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            size = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            
            ##activation just support relu6
            if mdef['activation'] == 'relu6':
                act_fn = nn.ReLU(inplace=False)
            elif mdef['activation'] == 'None':
                act_fn = None
            else:
                print('convolutional no support activation funtion!!')
                exit()

            modules.add_module('Conv2d',nn.Conv2d(in_channels = output_filters[-1],
                               out_channels = filters,
                               kernel_size = (size, size), stride = stride,
                               padding = (size - 1) // 2 if mdef['pad'] else 0, bias = not bool(bn)))

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            
            else:
                routs.append(i)  # detection output (goes into yolo layer)  ##only conv 1x1 should be detection output

            if act_fn:
                modules.add_module("activation", act_fn)

        elif mdef['type'] == 'conv_dw':
            if bool(mdef['first_layer']) or bool(mdef['last_layer']):
                print('depthwise not support first or last layer!')
                exit()
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            size = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            ##activation just support relu6
            if mdef['activation'] == 'relu6':
                act_fn = nn.ReLU(inplace=False)
            elif mdef['activation'] == 'None':
                act_fn = None
            else:
                print('no support activation funtion!!')
                exit()
            
            #nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
            modules.add_module('Conv2d_dw', nn.Conv2d(in_channels = output_filters[-1],
                               out_channels = output_filters[-1],
                               kernel_size = (size, size), stride = stride, groups = output_filters[-1],
                               padding = (size - 1) // 2 if mdef['pad'] else 0, bias = not bool(bn)))
            if bn:
                modules.add_module("BatchNorm2d_dw", nn.BatchNorm2d(output_filters[-1], momentum=0.03, eps=1E-4))
            if act_fn:
                modules.add_module("activation_dw", act_fn)

            modules.add_module('Conv2d1x1', nn.Conv2d(in_channels=output_filters[-1], 
                                                       out_channels=filters, 
                                                       kernel_size = (1, 1), stride = 1,
                                                       padding = 0, bias = not bool(bn)))
            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            if act_fn:
                modules.add_module("activation", act_fn)


        elif mdef['type'] == 'maxpool':
            size = mdef['size']
            stride = mdef['stride']
            
            modules.add_module('maxpool', nn.MaxPool2d((size,size), stride=stride))
            # ops.Maxpool2D(kernel_h=size, kernel_w=size, stride=stride, target_device=target_device)
           

        # elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
        #     layers = mdef['layers']
        #     filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
        #     routs.extend([i + l if l < 0 else l for l in layers])
            
        elif mdef["type"] == "shortcut":
            # print(mdef["from"])
            filters = output_filters[1:][int(mdef["from"][0])]
            modules.add_module(f"shortcut_{i}", nn.Sequential())


        elif mdef['type'] == 'fcLayer':
            filters = mdef['filters']
            # print("############ Input size ##############", img_size)
            #linear_input = output_filters[-1]*(img_size[0]//16)*(img_size[1]//16
            '''
            linear_input =  (img_size[1]//32 + (img_size[1]//32)%2)*(img_size[0]//32 + (img_size[0]//32)%2)*output_filters[-1]
            '''
            modules.add_module('FlattenInput', nn.Flatten())
            # modules.add_module('FlattenInput', ops.Flatten(
            #     shape_list=(-1,512),
            #     target_device=target_device)
            #         )
            
            #temp = torch.mean(x[0],dim=3)     
            #(x1, _) = torch.max(temp, dim=2)  
            #x2 = torch.mean(temp, dim=2)      
                              
            #x = (x1 + x2, x[1], x[2])
            
            # print("Model weights and activation", bita, bitw)
            modules.add_module('FC_Layer', nn.Linear(512, filters))

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers):
        super(YOLOLayer, self).__init__()
        #import pdb; pdb.set_trace()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        print("############################# number of classes \n",nc)
        self.no = nc + 5  # number of outputs (85)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

    def forward(self, p, img_size, out):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)

        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction
        #print(p.shape)
        if self.training:
            return p

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    def __init__(self, cfg, img_size=(416, 416)):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)

        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info()  # print model description
        #print(self.module_list)
        
    def forward(self, x, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        if verbose:
            str1 = ''
            print('0', x.shape)
        
        # x = ops.Preprocess(0, 255.)(x)
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'unpooling', 'maxpool','conv_dw']:
                x = module(x)
            elif mtype == 'shortcut':  # sum
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str1 = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                x = module([x] + [out[j] for j in self.module_defs[i]["from"]])
            # elif mtype == 'route':  # concat
            #     layers = mdef['layers']
            #     if verbose:
            #         l = [i - 1] + layers  # layers
            #         s = [list(x.shape)] + [list(out[i].shape) for i in layers]  # shapes
            #         str1 = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
            #     if len(layers) == 1:
            #         x = out[layers[0]]
            #     else:
            #         x = ops.Route()([out[i] for i in layers])
                        
            elif mtype == 'yolo':
                yolo_out.append(module(x, img_size, out))
                import pdb; pdb.set_trace()
                #print(yolo_out[0].shape)
            #print(out, i)
            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), mtype), list(x.shape), str)
                str1 = ''
        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


class MobileNetv2(nn.Module):
    def __init__(self, cfg, img_size=(416, 416)):
        super(MobileNetv2, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        #self.b0 = ops.batchNorm2d(64)
        #self.yolo_layers = get_yolo_layers(self)
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.count = 0
        window_size, hop_size, window_size, sample_rate, mel_bins, fmin, fmax = 1024, 320, 1024, 16000, 64, 20, 14000

        self.spec = Spectrogram(n_fft = window_size, hop_length = hop_size,
                                        win_length = window_size, window = window, center = center,
                                        pad_mode = pad_mode,
                                        freeze_parameters = True)

        # Logmel feature extractor
        self.logmel = LogmelFilterBank(sr = sample_rate, n_fft = window_size,
                                        n_mels = mel_bins, fmin = fmin, fmax = fmax, ref = ref, amin = amin,
                                        top_db = top_db,
                                        freeze_parameters = True)

        self.spec_augmenter = SpecAugmentation(time_drop_width=4, time_stripes_num=2,
                                               freq_drop_width=4, freq_stripes_num=1)
        
        #self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        #self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info()  # print model description
        #print(self.module_list)
        



    def forward(self, x, verbose=False):
        img_size = x.shape[-2:]
        sound_out, out = [], []
        if verbose:
            str1 = ''
            print('0', x.shape)
        #x = x.reshape(-1,1,51,64)
        #x = ops.Preprocess(0, 255.)(x)
        #print(x.shape)
        #import pdb; pdb.set_trace()
        

        #x = self.spec(x)
        #x = self.logmel(x)
    
        
        #if self.count==130:
        #    import pdb; pdb.set_trace()
        self.count+=1
        #print(self.training)
        # x = x.reshape(-1,1,51,64)
        if self.training:
            #print("Augmenting")
            x = self.spec(x)
            x = self.logmel(x)
            x = self.spec_augmenter(x)
        else:
            x = x.reshape(-1,1,101,64)
        x = ((torch.clamp(x,-100,28)+100.0)/128.0)*255
        

       
        #import pdb; pdb.set_trace()
        #x = ops.batchNorm2d(x[0])
        #x = self.bn_forwardpass(x)
        #print("here")
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            # print(mtype,x.shape, module)
            if isinstance(x,list):
                import pdb; pdb.set_trace()
            if mtype in ['convolutional', 'maxpool','conv_dw']:
                #print('Before',x[0].shape)
                # if mtype=='maxpool':
                #     print(module)
                x = module(x)
                #print('After',x[0].shape)
                #import pdb; pdb.set_trace()
            
            elif mtype == "shortcut":
                # print(mdef)
                layer_i = int(mdef["from"][0])
                x = out[-1] + out[layer_i]
                # import pdb; pdb.set_trace()

            
            elif mtype == 'fcLayer':
                #print(x[0].shape)
                #import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()
                
                '''
                temp = torch.mean(x[0],dim=3)     
                (x1, _) = torch.max(temp, dim=2)  
                x2 = torch.mean(temp, dim=2)                       
                x = (x1 + x2, x[1], x[2])        
                '''
                #x = (x2, x[1], x[2])

                
                
                fcout = module(x)
                
                
                if self.training:        
                    clipwise_output = torch.log_softmax(fcout, dim=-1)
                else:
                    clipwise_output = torch.log_softmax(fcout, dim=-1)
                
                sound_out.append(clipwise_output)
                # use for export
                #sound_out.append(fcout)
                return sound_out
                

            out.append(x )
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), mtype), list(x.shape), str)
                str1 = ''
        return sound_out
        
    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny





def load_darknet_weights(self, weights, cutoff=-1, pt=False, FPGA=False):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            #import pdb; pdb.set_trace()
            conv_layer = module[0].Conv2d
            if mdef['batch_normalize']:
                if FPGA:
                    # Load BN bias, weights, running mean and running variance
                    num_b = conv_layer.beta.numel()
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.beta)
                    conv_layer.beta.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.gamma)
                    conv_layer.gamma.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.running_mean)
                    conv_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.running_var)
                    conv_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[0].BatchNorm2d.batch_norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                # Load conv. weights
                import pdb;pdb.set_trace()
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
            else:
                # if os.path.basename(file) == 'yolov3.weights' or os.path.basename(file) == 'yolov3-tiny.weights':
                # pt标识使用coco预训练模型，读取参数时yolo层前面的一层输出为255
                if pt and os.path.basename(file).split('.')[-1] == 'weights':
                    num_b = 255
                    ptr += num_b
                    num_w = int(self.module_defs[i - 1]["filters"]) * 255
                    ptr += num_w
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
        elif mdef['type'] == 'depthwise':
            depthwise_layer = module[0]
            if mdef['batch_normalize']:
                if FPGA:
                    # Load BN bias, weights, running mean and running variance
                    num_b = conv_layer.beta.numel()
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.beta)
                    conv_layer.beta.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.gamma)
                    conv_layer.gamma.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.running_mean)
                    conv_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.running_var)
                    conv_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
            # Load conv. weights
            num_w = depthwise_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(depthwise_layer.weight)
            depthwise_layer.weight.data.copy_(conv_w)
            ptr += num_w
        elif mdef['type'] == 'se':
            se_layer = module[0]
            fc = se_layer.fc
            fc1 = fc[0]
            num_fc1 = fc1.weight.numel()
            fc1_w = torch.from_numpy(weights[ptr:ptr + num_fc1]).view_as(fc1.weight)
            fc1.weight.data.copy_(fc1_w)
            ptr += num_fc1
            fc2 = fc[2]
            num_fc2 = fc2.weight.numel()
            fc2_w = torch.from_numpy(weights[ptr:ptr + num_fc2]).view_as(fc2.weight)
            fc2.weight.data.copy_(fc2_w)
            ptr += num_fc2

    # 确保指针到达权重的最后一个位置
    print("loaded Darknet weights", ptr, len(weights))
    assert ptr == len(weights)

