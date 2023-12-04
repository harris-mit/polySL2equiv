from generate_data import pickle_to_dataloader
import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pickle
#from pytorch_lightning.tuner import Tuner
from models.sl2models import SL2Net, MLP_for_Comparison
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.model1 import SimplePolyModel
from models.lightningmodel import PLModel
from models.sl2models import SL2Net, MLP_for_Comparison, ScaleNet
import numpy as np
import time
from utils import normalized_mse_loss, prepare_for_logger, fraction_psd
from torch import nn
from tests.helper import equiv_transform, equiv_transform_for_min_poly, equiv_transform_from_presaved_fxn

def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        accelerator = "gpu"
    else:
        device = torch.device('cpu')
        accelerator = "cpu"
    print('device', device)
    print('normalize data', args.normalize_data)

    if args.mode == 'min_poly' or args.mode == 'minimizer_poly':
        if args.mode == 'min_poly':
            equiv_function = (lambda x,y: equiv_transform_for_min_poly(x, y, thresh=args.thresh, cond_lb=args.cond_lb, rotation=args.only_test_rotations)) 
            #additional_loss_function = loss_fn
        else:
            equiv_function = None # this is possible to add, just need to write versions with appropriate output transformation 
            #additional_loss_function = None
    else:
        equiv_function = (lambda x,y: equiv_transform(x, y, thresh=args.thresh, cond_lb=args.cond_lb, rotation=args.only_test_rotations))

    full_data_dir = os.path.join(args.data_home_dir, args.data_dir)
    dsets, dloaders = pickle_to_dataloader(full_data_dir, args.batch_size, return_datasets=True, transvectant_scaling = False, transform=equiv_function, normalize=args.normalize_data)

    saved_run_info_path = os.path.join(args.model_dir, 'saved_run_info') #os.path.join(os.path.abspath(os.path.join(os.path.dirname(checkpoint_path), os.path.pardir)), 'saved_run_info')
    print('saved_run_info_path', saved_run_info_path)
    out = torch.load(saved_run_info_path)
    kwargs = out['model_kwargs']
    if 'mode' not in kwargs.keys():
        kwargs['mode'] = args.mode
    kwargs['device'] = device
    best_chkpt_path = out['best_chkpt_path']

    print('best_chkpt_path', best_chkpt_path)

    if 'max_irrep_degree' in kwargs.keys():
        base_model = SL2Net(**kwargs).to(device)# SL(2)-equivariant net
    else:
        base_model = MLP_for_Comparison(**kwargs).to(device) # generic model

    if args.unnormalized_loss:
        loss_fn = nn.functional.mse_loss
    else:
        loss_fn = normalized_mse_loss
    
    if args.mode == 'min_poly' or args.mode == 'minimizer_poly':
        if args.mode == 'min_poly':
            equiv_function = (lambda x,y: equiv_transform_for_min_poly(x, y, thresh=args.thresh, cond_lb=args.cond_lb, rotation=args.only_test_rotations)) 
        else:
            equiv_function = None # this is possible to add, just need to write versions with appropriate output transformation 
    else:
        equiv_function = (lambda x,y: equiv_transform(x, y, thresh=args.thresh, cond_lb=args.cond_lb, rotation=args.only_test_rotations, return_A=True))

    mymodel = PLModel(base_model, loss_fn=loss_fn, use_lr_scheduler=False, equiv_function=equiv_function, use_eval_mode=False, lr = 3e-4, additional_loss_function=None, normalize_val=args.normalize_val).to(device)

    mymodel.load_state_dict(torch.load(best_chkpt_path, map_location=device)['state_dict'])

    # forward pass time:
    batch_times = []

    with torch.no_grad():
        val_losses = []
        total_time = 0
        total_instances = 0
        for batch in dloaders['val']:
            start = time.time()
            x, y = batch
            x, y = x.to(device), y.to(device)

            x_transformed, y_transformed, A_things = equiv_function(x, y)

            A, A_induced, A_out_induced = A_things['A'], A_things['A_induced'], A_things['A_out_induced']

            print('A', A, 'cond of A_induced', torch.linalg.cond(A_induced), 'cond of A out induced', torch.linalg.cond(A_out_induced))
            
            #pred_of_transformed = 

            pred = mymodel.net(x, eval_mode=True)
            end = time.time()
            
            lss = loss_fn(pred, y)
            val_losses.append(lss.detach().cpu().data)

            total_instances += x.shape[0]
            total_time += (end - start)

            break
    avg_val_loss = torch.mean(torch.tensor(val_losses))
    avg_time_for_one_instance = total_time / total_instances
    print('batch', args.batch_size, 'last one leftover', x.shape[0], 'time (s) for one instance', avg_time_for_one_instance)
    print('average val loss', avg_val_loss)

    return

    mymodel.to(device)
    mymodel.net.to(device)
    mymodel.net.device = device
    trainer = pl.Trainer(accelerator=accelerator, devices=1, 
                         log_every_n_steps=1, 
                         accumulate_grad_batches=1,
                         gradient_clip_val=0.0) 
    out = trainer.validate(model=mymodel, ckpt_path=best_chkpt_path, dataloaders=dloaders['val'])
    for elt in out:
        print(elt, type(elt))


    # mymodel = PLModel.load_from_checkpoint(checkpoint_path)
    

    # time the forward pass
    # 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate a trained model on validation set')
    #parser.add_argument('--saved_dir', type=str, default='/home/gridsan/hanlaw/SL2equivariance/trained_models_and_logs/',
                        #help='Directory where the checkpoint is')
    parser.add_argument('--model_dir', type=str, default='/home/gridsan/hanlaw/SL2equivariance/trained_models_and_logs/May_9_yes_training_val_norm_no_sl2net_norm/epoch_1000_lay_4_channel_50_mx_14_MLP_100_100_e1000_test_all_sl2_do_norm_val_thresh_5_lb_4/my_model/version_1', help='The rest of the path to checkpoint')

    parser.add_argument('--normalize_data', action='store_true', default=False,
                        help='--Whether or not to normalize the input data. Turn off for min poly!')
    parser.add_argument('--normalize_val', action='store_true', default=False,
                        help='--Whether or not to normalize the data at the automatic validation step. Turn off for min poly!')
                
    parser.add_argument('--data_home_dir', type=str, default='/home/gridsan/hanlaw/TransvectantNets_shared/data/equivariant')

    parser.add_argument('--data_dir', type=str, default='deg_6_train_5000_val_100_test_100')

    parser.add_argument('--mode', type=str, default='max_det',
                        help='--Mode for generating labels y. --max_det for Gram matrix of maximal determinant (equivariant, matrix task), --def for definiteness (invariant, scalar task), --min_poly for minimizing a polynomial, --minimizer_poly for getting the location of the minimizer of a polynomial.')
    parser.add_argument('--thresh', type=float, default=4, 
                        help='--Condition number maximum for generating SL2 matrices at validation time')
    parser.add_argument('--cond_lb', type=float, default=5, 
                        help='--Condition number lower bound for generating SL2 matrices at validation time')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances per batch')
    
    parser.add_argument('--only_test_rotations', default=False, action='store_true',
                        help='--Only test rotations (note: not unitary after being induced, though) with the loss function on the transformed data')

    parser.add_argument('--unnormalized_loss', default=False, action='store_true',
                        help='--If included, use mse_loss instead of normalizing by y (normalizing is the default behavior).')
                    
    args = parser.parse_args()
    main(args)

