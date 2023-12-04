from generate_data import pickle_to_dataloader
import argparse
import torch
from models.sl2models import SL2Net, MLP_for_Comparison
from models.lightningmodel import PLModel
import os
from utils import count_parameters
from torch import nn
from tests.bernstein_comparison import ispsd, solve_bernstein

def load_model_from_directory(run_dir, model_type='SL2Net', device=torch.device('cpu')):
    # Must have saved dictionary of input arguments required by the model_type in saved_run_info file, in run_dir. Models trained by train.py will satisfy this requirement 

    run_dict = torch.load(os.path.join(run_dir, 'saved_run_info'))
    best_chkpt_path = run_dict['best_chkpt_path']
    model_kwargs = run_dict['model_kwargs']
    print('model kwargs', model_kwargs)
    if model_type == 'SL2Net': # I think there's a cleaner way to do this for many multiple types, but for two, this is fine
        model_kwargs['device'] = device 
        base_model = SL2Net(**model_kwargs)
    else:
        print('model_kwargs', model_kwargs)
        base_model = MLP_for_Comparison(**model_kwargs)
    plmodel = PLModel(base_model)
    PL_chkpt = torch.load(best_chkpt_path, map_location=device)
    plmodel.load_state_dict(PL_chkpt['state_dict'])
    model = plmodel.net.to(device)
    return model

def main(args: argparse) -> None:

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Get dataloaders from saved data files in args.data_dir
    dsets, dloaders = pickle_to_dataloader(args.data_dir, args.batch_size,
                                           return_datasets = True)

    # Get trained model from args.chkpt_path

    #equiv_model = load_model_from_directory(args.run_dir_equivariant, model_type='SL2Net', device=device)
    #print('Number of parameters in equivariant model', count_parameters(equiv_model))

    run_generic = (args.run_dir_generic is not None)
    if run_generic:
        generic_model = load_model_from_directory(args.run_dir_generic, model_type=
    'MLP_for_Gram', device=device)
        print('Number of parameters in generic model', count_parameters(generic_model))

    # TODO: baselines on dsets['test']
    # NOTE: this will depend on which type of data we have (min_poly, )
    

    # EXAMPLE: iterate over test dataloader and evaluate the model
    model_results = []
    for x, y in dloaders['test']:
        x, y = x.to(device), y.to(device)

        import time
        
        if run_generic:
            start = time.time()
            #equiv_model_pred = equiv_model(x)
            generic_model_pred = generic_model(x)
            end = time.time()
            elapsed = (end - start) 
            print('Forward pass of deg', x.shape[-1]-1, 'batch', x.shape[0], f'time: {elapsed:.9f} sec')
            

        model_results += [ispsd(generic_model_pred)]
        
    bernstein_results = [solve_bernstein(p) for p in dsets['test'].tensors[0]]
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compare an SL(2)-equivariant model to baselines on the Gram matrix (equivariant) task')

    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of instances per batch')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='--')
    parser.add_argument('--run_dir_equivariant', type=str, default='trained_models_and_logs/most_recent/my_model/version_2',
                        help='--Assume it\'s a directory containing torch loadable file saved_run_info, corresponding to an equivariant model')

    parser.add_argument('--run_dir_generic', type=str, default=None,
                        help='--Assume it\'s a directory containing torch loadable file saved_run_info, coresponding to a generic model. If argument not included, will not compare to a generic model as one of the baselines')
                        
    args = parser.parse_args()
    main(args)
