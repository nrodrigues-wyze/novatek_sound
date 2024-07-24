import torch
import torchvision
from models import *
from torch.autograd import Variable
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="weights/tmp.pt", help="path to model")
    parser.add_argument('--cfg', type=str, default='cfg/persondet.cfg', help='*.cfg path')
    parser.add_argument('--width', type=int, default=416, help='Input Image height')
    parser.add_argument('--height', type=int, default=416, help='Input Image width')



    cfg = parser.parse_args()
    
    trained_model = MobileNetv2(cfg.cfg, (cfg.width, cfg.height)).cuda()
    print('train_model: ', trained_model)
    trained_model.load_state_dict(torch.load(cfg.model)['model'], True)

    #dummy_input = Variable(torch.randn(1,3,cfg.height, cfg.width)).cuda()
    dummy_input = Variable(torch.randn(1,1,cfg.height, cfg.width)).cuda()
    
    dst_path = cfg.model.rsplit('.',1)[0] +  ".onnx"
    
    #print(dst_path)
    #import pdb; pdb.set_trace()


    torch.onnx.export(trained_model, dummy_input, dst_path, verbose=True, do_constant_folding=False, opset_version=13)
    
    print("======convert over=======save in: %s\n"%dst_path)


if __name__ == '__main__':
    main()
