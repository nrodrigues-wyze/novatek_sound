import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.torch_utils import intersect_dicts
from utils.pytorch_utils import move_data_to_device, MyCollator, do_mixup
from utils.generator import Hdf5Dataset, make_weights_for_balanced_classes
from utils.evaluate import Evaluator
from utils.utilities import Mixup
from utils.losses import get_loss_func


#if not os.path.exists(wdir):
#    os.makedirs(wdir)


def train():
    wdir = opt.save_dir + os.sep  # weights dir     
    last = wdir + 'last.pt'                                                            
    best = wdir + 'best.pt'                                                            
    results_file = wdir + 'results.txt'                                                
                                                                                   
    #print(wdir)
    if not os.path.exists(wdir):                                                       
        os.makedirs(wdir)                                                              

    
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    start_epoch = 0
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    mixup = opt.mixup
    signal_rate = 16000
    learning_rate = 1e-3
    weight_decay = 0 #0.000484
    class_nums = 8
    loss_func = get_loss_func('clip_nll')

    # Initialize
    init_seeds()
    
    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = int(data_dict['classes'])  # number of classes

    # Remove previous results
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    #model = Darknet(cfg).to(device)
    model = MobileNetv2(cfg,img_size=(101,64)).to(device)
    #import pdb; pdb.set_trace()
    # Optimizer


    train_data = Hdf5Dataset(train_path)
    eval_data = Hdf5Dataset(test_path)
    my_collator = MyCollator(opt.augment, signal_rate)
    nw=32

    if opt.balanced:
        weights = make_weights_for_balanced_classes(train_path)
        weights = torch.DoubleTensor(weights)
        #import pdb; pdb.set_trace()
        print("Sampler weights: ", weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size = batch_size * 2 if mixup else batch_size,
                                                   sampler = sampler, num_workers = nw, pin_memory = True,
                                                   collate_fn = my_collator)

    eval_loader = torch.utils.data.DataLoader(eval_data,
                                              batch_size = batch_size,
                                              shuffle = False,
                                              num_workers = nw, pin_memory = True)
    #import pdb; pdb.set_trace()

    if mixup:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    weights = opt.weights
    if weights.endswith('.pt'):  # pytorch format
        #import pdb; pdb.set_trace()
        print("loaded pretrained weights", weights)
        
        model.load_state_dict(torch.load(weights, map_location='cuda:0')['model'])
        '''
        chkpt = torch.load(weights, map_location='cuda:0')

        # load model
        #print(model)
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(chkpt['model'], strict=False)
        '''
    #print(model)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999),
                           eps = 1e-08, weight_decay = weight_decay, amsgrad = True)   
    
    #import pdb; pdb.set_trace()
    #lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05 
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    iteration=0
    checkpoint = {
        'iteration': iteration,
        'model': model.state_dict()}

    torch.save(checkpoint, wdir + '/init_model.pt' )     

    evaluator = Evaluator(model = model, classes_num = class_nums)
    #import pdb; pdb.set_trace()
    #good_model = torch.load('data/7000_iterations.pth', map_location='cuda:0')

    #if  have_load_model and opt.resume:  # Calculate mAP
    '''
    if weights.endswith('.pt'):
        print('test the preload model is {}'.format(weights))
        statistics = evaluator.evaluate(eval_loader)
        print("Preloaded weights accuracy:",statistics['accuracy'])
    '''
    # Start training
    nb = len(train_loader)  # number of batches

    t0 = time.time()
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    iteration=0
    for epoch in range(epochs):
        print("Training Epoch Number", epoch)
        
        for batch_data_dict in train_loader:
            model.train()
            batch_input = move_data_to_device(batch_data_dict['waveform'], device)
            if mixup:
                batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                    batch_size = len(batch_data_dict['waveform']))
                mixup_lambda = move_data_to_device(batch_data_dict['mixup_lambda'], device)
                """{'clipwise_output': (batch_size, classes_num), ...}"""
                batch_output_dict = model(batch_input, mixup_lambda)
                batch_target_dict = dict(
                    target = move_data_to_device(do_mixup(batch_data_dict['target'], batch_data_dict['mixup_lambda']).float(),
                                                 device))

            else:
                
                #import pdb; pdb.set_trace()
                output = model(batch_input, None)
                #import pdb; pdb.set_trace()
                batch_output_dict = {'clipwise_output': output[0]}
                """{'clipwise_output': (batch_size, classes_num), ...}"""
                batch_target_dict = {'target': move_data_to_device(batch_data_dict['target'], device)}

            # loss
            #import pdb; pdb.set_trace()
            loss = loss_func(batch_output_dict, batch_target_dict)
            if iteration%100==0:
                print(iteration, loss)
            #writer.add_scalar("Loss/train", loss, iteration)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
            
            if iteration > 0 and iteration % opt.save_iter_interval == 0:
                #import pdb; pdb.set_trace()
                checkpoint = {
                    'iteration': iteration,
                    'model': model.state_dict()}
                print("savin model to :", wdir + '/backup%d'%(iteration))

                torch.save(checkpoint, wdir + '/backup%g.pt' % iteration)
                del checkpoint
 
        print(optimizer)
        # Update Scheduler
        #scheduler.step()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--save_iter_interval', type=int, default=200, help='interations after which to save ckpt')
    parser.add_argument('--cfg', type=str, default='cfg/persondet.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco_person.data', help='*.data path')
    parser.add_argument('--save-dir', type=str, default='weights/oct_model_32', help='*.data path')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--balanced', action='store_true', help='sound balancd training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 1 or cpu)')
    parser.add_argument('--mixup', action = 'store_true', help = 'whether or not using mixup augmentation')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--augment', action='store_true', help='Augment input data')
    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    print(opt)
    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)

    tb_writer = None

    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter()
    except:
        pass

    train()  # train normally
