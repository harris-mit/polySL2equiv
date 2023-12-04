from generate_data import pickle_to_dataloader
import numpy as np
from thewalrus import perm
import math
import cvxpy
from utils import sum_antidiagonals, batch_sum_antidiagonals
import torch
import argparse
import pickle
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
from models.lightningmodel import PLModel
import sympy
from scipy.optimize import brute, fmin
from sympy.utilities.lambdify import lambdify
from collections import defaultdict
import torch
from cg import transvectant
from itertools import combinations
# seed random numbers so datasets are predictable!
torch.manual_seed(0)
np.random.seed(0)
import time
from tqdm import tqdm
import pickle
from generate_data import get_many_random_pos_poly, get_max_log_det
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from compute_test_errors import get_model_type, get_model
from utils import normalized_mse_loss


def main(args):
    res = {}
    res_list = []

    device = torch.device('cpu')

    MLPs = {'6': '/home/gridsan/hanlaw/SL2equivariance/trained_models_and_logs/May_14_deg_6_mega_run/seed_2/generic_sched_none_MLP_100_1000_nobnorm_e700_test_all_sl2_do_data_norm_both',
        '8': '/home/gridsan/hanlaw/SL2equivariance/trained_models_and_logs/May_11_high_degs/deg_8/generic_sched_none_MLP_100_1000_nobnorm_e500_test_all_sl2_do_data_norm/my_model/version_0',
    '10': '/home/gridsan/hanlaw/SL2equivariance/trained_models_and_logs/May_11_high_degs/deg_10/generic_sched_none_MLP_100_1000_nobnorm_e500_test_all_sl2_do_data_norm/my_model/version_0',
    '12': '/home/gridsan/hanlaw/SL2equivariance/trained_models_and_logs/May_11_high_degs/deg_12/generic_sched_none_MLP_100_1000_nobnorm_e500_test_all_sl2_do_data_norm/my_model/version_0',
    '14': '/home/gridsan/hanlaw/SL2equivariance/trained_models_and_logs/May_11_high_degs/deg_14/generic_sched_none_MLP_100_1000_nobnorm_e500_test_all_sl2_do_data_norm/my_model/version_0'}

    batch_size = 100

    loss_fn = normalized_mse_loss
    MLP_times = {}
    solver_times = {}
    for deg_key in MLPs.keys():
        deg = int(deg_key)
        full_data_dir = f'/home/gridsan/hanlaw/TransvectantNets_shared/data/equivariant/fresh_deg_{deg}_train_5000_val_100_test_100'
        dsets, dloaders = pickle_to_dataloader(full_data_dir, batch_size, return_datasets=True, transvectant_scaling = False, transform=None, normalize=True)

        split = 'train'

        # load trained MLP
        model_type = get_model_type(MLPs[deg_key])
        base_model, best_chkpt_path = get_model(MLPs[deg_key], model_type, loss_fn, device=device)

        mymodel = PLModel(base_model, loss_fn=loss_fn, use_lr_scheduler=False, equiv_function=None, use_eval_mode=False, lr = 3e-4, additional_loss_function=None, normalize_val=True).to(device)

        mymodel.load_state_dict(torch.load(best_chkpt_path, map_location=device)['state_dict'])

        main_net = mymodel.net

        with torch.no_grad():
            start = time.time()
            for batch in dloaders[split]:
                x, y = batch
                pred = main_net(x, eval_mode=True)
                
            end = time.time() # just for timing
            MLP_time = (end - start)
            

            # now for eval
            losses = []
            for batch in dloaders[split]:
                x, y = batch
                pred = main_net(x, eval_mode=True)
                loss = loss_fn(pred, y)
                losses.append(loss.detach().data)
            print('losses', losses)
            avg_loss = torch.mean(torch.tensor(losses))
            MLP_times[deg_key] = [MLP_time, avg_loss]
            print('deg', deg)
            print('MLP time',MLP_time,'loss',avg_loss)
        
        start = time.time()
        for i in range(len(dsets[split])):
            x = dsets[split][i][0]
            mat = get_max_log_det(x)
        end = time.time()

        solver_time = (end - start)
        solver_times[deg_key] = [solver_time]
        print('solver_time', solver_time)
    with open('timing_test_results_2.pkl','wb') as f:
        pickle.dump({'solver': solver_times, 'MLP': MLP_times}, f)

            

    # 1) Use consistent dataset
    # 2) Include timing test for trained MLPs also
    # 3) Save data to generate a bar plot
    
    """for deg in args.degrees:
        solve_times = []
        x = get_many_random_pos_poly(args.num_samples, int(deg / 2)) + 1
        for i in range(args.num_samples):
            start = time.time()
            mat = get_max_log_det(x[i, :]).float()
            end = time.time()
            solve_times.append(end - start)
        mean_sol_time = np.mean(np.array(solve_times))
        res[deg] = mean_sol_time
        res_list.append(mean_sol_time)
    print(res)
    plt.plot(args.degrees, res_list)
    plt.xlabel('Degree')
    plt.ylabel('Runtime, in seconds')
    plt.savefig('runtimes.jpg')"""




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Timing test for baseline methods on max_det problem')
    parser.add_argument('--degrees', type=int, nargs="+", default=[4, 5, 6, 7, 8],
                        help='Degrees to test')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='How many samples of each degree to draw')
                
    #parser.add_argument('--data_dir', type=str, default=None,
                        #help='--Path to directory where data files are; will use these')
    parser.add_argument('--mode', type=str, default='max_det',
                        help='--Mode for generating labels y. --max_det for Gram matrix of maximal determinant (equivariant, matrix task), --def for definiteness (invariant, scalar task), --min_poly for minimizing a polynomial, --minimizer_poly for getting the location of the minimizer of a polynomial.')
                        
    args = parser.parse_args()
    main(args)

