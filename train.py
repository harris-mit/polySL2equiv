from generate_data import pickle_to_dataloader
import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.tensorboard import SummaryWriter
import argparse
from models.model1 import SimplePolyModel
from models.lightningmodel import PLModel
from models.sl2models import SL2Net, MLP_for_Comparison, ScaleNet
from models.so2models import SO2Net
import numpy as np
import time
from utils import normalized_mse_loss, prepare_for_logger, fraction_psd
from torch import nn
from tests.helper import equiv_transform, equiv_transform_for_min_poly, equiv_transform_from_presaved_fxn, equiv_transform_for_min_poly_from_presaved_fxn

def main(args: argparse) -> None:

    # Generate random data

    if torch.cuda.is_available():
        device = torch.device('cuda')
        accelerator = "gpu"
    else:
        device = torch.device('cpu')
        accelerator = "cpu"

    print('device', device)

    args.device = device

    if args.unnormalized_loss:
        loss_fn = nn.functional.mse_loss
    else:
        loss_fn = normalized_mse_loss
        print('Using normalized loss')

    if args.mode == 'min_poly' or args.mode == 'minimizer_poly':
        if args.mode == 'min_poly':
            equiv_function = (lambda x,y: equiv_transform_for_min_poly(x, y, thresh=args.thresh, cond_lb=args.cond_lb, rotation=args.only_test_rotations, A_and_induceds=None)) 
            additional_loss_function = loss_fn
        else:
            equiv_function = None # this is possible to add, just need to write versions with appropriate output transformation 
            additional_loss_function = None
    else:
        equiv_function = (lambda x,y: equiv_transform(x, y, thresh=args.thresh, cond_lb=args.cond_lb, rotation=args.only_test_rotations))

    if args.data_aug:
        if args.precomputed_aug_file is not None:
        # transform is dictionary with 'transform_file' key for precomputed file full path and 'transform_from_presaved_fxn'
        # key that takes as input x, y, A, induced_mats
            if args.mode == 'min_poly':
                equiv_presaved_fxn = equiv_transform_for_min_poly_from_presaved_fxn 
            elif args.mode == 'max_det':
                equiv_presaved_fxn = equiv_transform_from_presaved_fxn
            transform = {'transform_file': args.precomputed_aug_file, 'transform_from_presaved_fxn': equiv_presaved_fxn}
        else:
            # transform is just fxn that takes in x, y and spits out augmented x, y
            transform = equiv_function
    else:
        transform = None

    print('normalize_data', args.normalize_data, 'normalize_val', args.normalize_val)
    dsets, dloaders = pickle_to_dataloader(args.data_dir, args.batch_size, return_datasets=True, transvectant_scaling = args.transvectant_normalize, transform=transform, normalize=args.normalize_data)    

    print('have dataloaders')

    # Includes computation for the MLP which is not used for an equivariant model, but doesn't matter (since fast).
    
    for x, y  in dloaders['train']:
        if args.mode == 'min_poly' or args.mode == 'minimizer_poly':
            x_keys = list(x.keys())
            deg = max(x_keys)
            input_degs = x_keys
            if args.mode == 'min_poly':
                output_size = 1
            else:
                output_size = 2
        else: #
            deg = x.shape[-1] - 1
            input_degs = [deg]
            if args.mode == 'def':
                output_size = 1
            elif args.mode == 'max_det':
                output_size = (int(deg / 2) + 1)**2
            additional_loss_function = (lambda pred,y: 1 - fraction_psd(pred, cutoff=-1*1e-5))
        break

    if args.generic_model:
        # how many outputs depends on the application; may or may not need to reshape
        base_model = MLP_for_Comparison(num_hidden = [sum(input_degs) + len(input_degs)] + args.mlp_arch + [output_size], mode=args.mode)
    elif args.so2_model:
        # previously said max(input_degs)???
        if args.mode == 'min_poly':
            inindeg = input_degs 
        else:
            inindeg = max(input_degs)
        base_model = SO2Net(input_deg = inindeg, num_layers = args.num_layers, num_internal_channels = args.num_internal_channels, max_irrep_degree = args.max_irrep, mode= args.mode, device=device, num_hidden_for_invar = args.invar_arch, precomputed_T_dir=os.path.join(args.precompute_dir, 'CGTmats')).to(device)
    else:
        mlp_args = {'on_input': args.use_input_mlp, 'input_arch': args.mlp_arch, 'before_output': args.use_output_mlp, 'output_arch': args.mlp_arch}

        def gated_and_self(deg_i, deg_j):
            if deg_i == 0 or deg_j == 0:
                return True
            if deg_i == deg_j:
                return True
            return False
        
        if args.gated_and_self:
            do_tensor_fxn = gated_and_self
        else:
            do_tensor_fxn = (lambda x, y: True)
        base_model = SL2Net(input_degs = input_degs, num_layers=args.num_layers,
                            num_internal_channels=args.num_internal_channels, max_irrep_degree=args.max_irrep,
                            device=device, num_hidden_for_invar=args.invar_arch, batch_norm=not args.no_batch_norm,
                            do_skip=not args.no_skip, mode=args.mode, transvectant_normalize = False, # turned off from args.transvectant_normalize
                                    scale_set_2=args.scale_set_2, precomputed_T_dir=os.path.join(args.precompute_dir, 'CGTmats'), mlp_args=mlp_args, do_tensor_fxn=do_tensor_fxn).to(device)

    
    tangent_loss_ops={
                    'hermite': args.hermite, 
                    'hermite_scaling': 1, 
                    'pure_equivariance': args.pure_equivariance, 
                    'pure_equivariance_scaling': 1,
                    }

    # Initialize model
    mymodel = PLModel(base_model, loss_fn=loss_fn, use_lr_scheduler=args.use_lr_scheduler, equiv_function=equiv_function, use_eval_mode=args.use_eval_mode, lr = args.learning_rate, additional_loss_function=additional_loss_function, normalize_val=args.normalize_val, tangent_loss_ops=tangent_loss_ops)
    # Compute number of parameters in model, so that comparisons are fair 
    num_param = mymodel.num_param
    print('Number of parameters in model', num_param)

    if args.just_print_param:
        return

    start = time.time()
    # Train 
    logger = TensorBoardLogger(save_dir=args.save_dir, name="my_model")
    #summary_logger = SummaryWriter(save_dir=args.save_dir)
    #logged_hparams = prepare_for_logger(mymodel.net.kwargs).update({"lr": mymodel.lr})
    #logger.log_hyperparams(mymodel.hparams, logged_hparams)

    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, save_top_k=2, monitor="val_loss")

    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator=accelerator, devices=1, 
                         log_every_n_steps=1, logger=logger, 
                         accumulate_grad_batches=1,
                         gradient_clip_val=0.0,
                         callbacks=[checkpoint_callback]) 
                        #limit_train_batches, other args poss.
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(mymodel, mode="power")
    trainer.fit(model=mymodel, train_dataloaders=dloaders['train'], val_dataloaders=dloaders['val']) #, auto_lr_find=True)
    
    end = time.time()
    elapsed_time = (end-start)/60.0
    print('generic_model ', args.generic_model, 'ELAPSED TIME (min): ', elapsed_time, 'save_dir', args.save_dir)

    best_chkpt_path = trainer.checkpoint_callback.best_model_path
    print('best_chkpt_path', best_chkpt_path)

    directory_for_misc_saves = os.path.dirname(best_chkpt_path) # removed one dirname

    saved_run_info_dict = {'best_chkpt_path': best_chkpt_path, 'model_kwargs': base_model.kwargs, 'num_param': num_param,
                           'learning_rate':args.learning_rate, 'elapsed_time': elapsed_time, 'train_losses': torch.tensor(mymodel.train_losses)}

    torch.save(saved_run_info_dict, os.path.join(directory_for_misc_saves, 'saved_run_info'))
          
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training an SL(2)-equivariant model')

    # General training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances per batch')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.expanduser('~'), 'TransvectantNets_shared', 'data/equivariant/deg_8_train_5000_val_500_test_1000/'),
                        help='--')
    parser.add_argument('--save_dir', type=str, default='trained_models_and_logs/',
                        help='--Directory in which to save pytorch_lightning outputs, model checkpoint, and Tensorboard log.')
    parser.add_argument('--precompute_dir', type=str, default=os.path.join(os.path.expanduser('~'), 'TransvectantNets_shared/precomputations'),
                        help='--Directory in which to save general precomputations.')
    parser.add_argument('--just_print_param', default=False, action='store_true',
                        help='--If included, just print the number of parameters in the specified model and return -- do not train.')
    parser.add_argument('--unnormalized_loss', default=False, action='store_true',
                        help='--If included, use mse_loss instead of normalizing by y (normalizing is the default behavior).')
    parser.add_argument('--mode', type=str, default='max_det',    
                        help='--Mode by which the labels y were generated -- must match the mode used to generate data_dir via generate_data.py! --max_det for Gram matrix of maximal determinant (equivariant, matrix task), --def for definiteness (invariant, scalar task), --min_poly for minimizing a polynomial, --minimizer_poly for getting the location of the minimizer of a polynomial.')
    parser.add_argument('--use_eval_mode', default=False, action='store_true',
                        help='--If included, use eval mode for the validation loss. Note that eval_mode is always used for the equivariance test.')
    parser.add_argument('--data_aug', default=False, action='store_true',
                        help='--If included, use data augmentation. Defaults to SL2, but if flag for only_test_rotations, then just rotations.')

    parser.add_argument('--precomputed_aug_file', type=str, default=None,
                        help='--If included, use precomputed from given file path to pkl file (if there\'s also a flag for data augmentation). ')
    parser.add_argument('--normalize_data', action='store_true', default=False,
                        help='--Whether or not to normalize the input data. Turn off for min poly!')
    parser.add_argument('--normalize_val', action='store_true', default=False,
                        help='--Whether or not to normalize the data at the automatic validation step. Turn off for min poly!')
    
    parser.add_argument('--hermite', action='store_true', default=False,
                        help='--use hermite first order information in loss function. only for max det!')
    
    parser.add_argument('--pure_equivariance', action='store_true', default=False,
                        help='--use basic equivariance in loss function. only for max det!')


    # Arguments for an equivariant model
    parser.add_argument('--max_irrep', type=int, default=10,
                        help='--')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='--')
    parser.add_argument('--num_internal_channels', type=int, 
                        default=2, help='--')
    parser.add_argument('--invar_arch', nargs="+", type=int, default=[10, 10], 
                        help='--Architecture for the MLP operating on the invariant irreps')

    parser.add_argument('--no_skip', default=False, action='store_true', 
                        help='--If included, no skip connections')
    parser.add_argument('--no_batch_norm', default=False, action='store_true', 
                        help='--If included, no batch norm')
    parser.add_argument('--gated_and_self', default=False, action='store_true', 
                        help='--If included, only do self-tensor products and tensor products with the invariant')

    # Arguments for a non-equivariant model
    parser.add_argument('--generic_model', default=False, action='store_true',
                        help='--If included, train a generic (non-equivariant) model instead of an equivariant one.')
    parser.add_argument('--so2_model', default=False, action='store_true',
                        help='--If included, train an SO2 equivariant model instead of SL2.')
    parser.add_argument('--mlp_arch', nargs="+", type=int, default=[20, 20], 
                        help='--If included, train a generic (non-equivariant) model instead of an equivariant one.')
   
    # Some more general arguments
    parser.add_argument('--use_lr_scheduler', default=False, action='store_true',
                        help='--If included, use a scheduler for the learning rate.')
    parser.add_argument('--thresh', type=float, default=10, 
                        help='--Condition number maximum for generating SL2 matrices at validation time')
    parser.add_argument('--cond_lb', type=float, default=1, 
                        help='--Condition number lower bound for generating SL2 matrices at validation time')
    parser.add_argument('--learning_rate', type=float, default = 3e-4, help="--Learning rate for optimizer.")
    parser.add_argument('--transvectant_normalize', default=False, action='store_true',
                        help='--Normalize the inputs by psi_d(p,p)')

    parser.add_argument('--scale_set_2', default=False, action='store_true',
                        help='--Set scale factor to 2')
    
    parser.add_argument('--only_test_rotations', default=False, action='store_true',
                        help='--Only test rotations (note: not unitary after being induced, though) with the loss function on the transformed data')

    parser.add_argument('--use_input_mlp', default=False, action='store_true',
                        help='--Use input mlp')

    parser.add_argument('--use_output_mlp', default=False, action='store_true',
                        help='--Use output mlp')

    args = parser.parse_args()
    main(args)


