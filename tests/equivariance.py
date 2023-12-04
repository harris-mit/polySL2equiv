from thewalrus import perm
import os
import torch
import sys
sys.path.append('..')
from cg import transvectant
from models.model1 import SimplePolyModel
from models.sl2models import SL2Net
from generate_data import get_many_random_pos_poly
from helper import get_SL2_matrices, induced
import numpy as np 
import math

# problem arises: deg 4, max irrep 16, layers 5. batch_norm True, don't set weights to one

deg = 8 #8 #4
max_irrep = 16 #16 #16

numsamples = 1
n = int((deg / 2) + 1)
num_layers = 4 #3 #5 #4 #6
batch_norm = True #True
num_internal_channels = 1 #1 #0

do_invariant = False
set_weights_to_one = False

output_gram_matrix = not do_invariant
precomputed_T_dir = os.path.join(os.path.expanduser('~'), 'TransvectantNets_shared/precomputations/CGTmats') # change for your own home dir
print('precomputed_T_dir', precomputed_T_dir)

torch.manual_seed(1)
np.random.seed(1)

example_sl2_model = SL2Net(input_degs = [deg], num_layers=num_layers, num_internal_channels=num_internal_channels, mode='max_det',
                           max_irrep_degree=max_irrep,
                           numerical_norm = False, num_hidden_for_invar=[10,10], batch_norm = batch_norm, do_skip=False, precomputed_T_dir=precomputed_T_dir,
                           numerically_stable = True, force_recompute_T=False, transvectant_normalize=True) #.to(device)
if set_weights_to_one:
    for sl2layer in example_sl2_model.SL2Layers:
        for linlayerky in sl2layer.linear_layers.keys():
            sl2layer.linear_layers[linlayerky].weight.data.fill_(1)

A = get_SL2_matrices(1, return_tensor=False)[0, ...]
#A = np.matmul(A, A)
print('norm of A', np.linalg.norm(A))
A_induced = induced(A, k=deg, scaled=False, return_tensor=True)
print('norm of induced A', torch.norm(A_induced))
print('det of A', torch.linalg.det(A_induced))


poly = get_many_random_pos_poly(numsamples, int(deg / 2))
poly_transformed = torch.matmul(A_induced.permute(1,0), poly.unsqueeze(-1))[:,:,0]

print('norm of poly', torch.norm(poly))
print('norm of poly_transformed', torch.norm(poly_transformed))

#out = example_sl2_model(poly) # out is batch x n x n
#out_transformed = example_sl2_model(poly_transformed)
print('input', torch.cat([poly, poly_transformed]).shape)
results = example_sl2_model(torch.cat([poly, poly_transformed], dim=0), eval_mode=True) # added explicit dim on 1/22

def check_layers(results, print_vals=False):
    for k in results.keys():
        num_channels_this_layer = results[k].shape[1]
        print('num_channels_this_layer', num_channels_this_layer)
        break
    out = {k: results[k][0:numsamples].unsqueeze(-1) for k in results.keys()} 
    out_transformed = {k: results[k][numsamples:] for k in results.keys()}
    Adeg_induced = {k: induced(A, k=k, scaled=False, return_tensor=True) for k in out.keys()}
    out_expected_transformed_dict = {k: torch.matmul(Adeg_induced[k].permute(1,0), out[k].permute(0,1,2,3)).reshape(numsamples, num_channels_this_layer,-1)
                                     for k in out.keys()}
    for k in out.keys():
        #print('k', k, 'Adeg_induced[k]', Adeg_induced[k].shape,'out_k', out[k].shape, 'out_transformed[k]',
        #out_transformed[k].shape, 'out_expected_transformed_dict[k]', out_expected_transformed_dict[k].shape)
        # OK IT IS A SHAPE ISSUE HERE
        print('k', k, 'Adeg_induced[k]', Adeg_induced[k].shape,'out_k', out[k].shape, 'out_transformed[k]', out_transformed[k].shape, 'out_expected_transformed_dict[k]', out_expected_transformed_dict[k].shape)
        rel_error = float(torch.norm(out_expected_transformed_dict[k] - out_transformed[k]) / (torch.norm(out_transformed[k])))
        log_rel_error = math.log10(rel_error + 1e-8)
        print("\nDeg", k, f'norms: {float(torch.norm(out_expected_transformed_dict[k])):.3f} {float(torch.norm(out_transformed[k])):.3f} \
        rel error {rel_error:.3f} = 10^\
        {log_rel_error:.3f}')
        if print_vals:
            print(out_transformed[k].data)
            print(out_expected_transformed_dict[k].data)

print('poly', poly.shape, 'poly_transformed', poly_transformed.shape)
#print('input to layer', torch.cat([poly, poly_transformed], dim = 0).unsqueeze(1).shape)
x2 = {deg:torch.cat([poly, poly_transformed], dim = 0).unsqueeze(1)}
print('SL2Layers',)
for l in range(num_layers):
    print('l', l)
    for ky in x2.keys():
        print('  ky', ky, x2[ky].shape)
    x2 = example_sl2_model.SL2Layers[l](x2, eval_mode=True)
    print(f'\n\n----Layer {l}')
    check_layers(x2, False)

# Check final output  matrices for equivariance
out = results[0:numsamples,...]
out_transformed = results[numsamples:,...]
A_out_induced = induced(A, k=n-1, scaled=False, return_tensor=True)

print('A cond', np.linalg.cond(A), 'A induced cond', torch.linalg.cond(A_induced), 'A_out_induced cond', torch.linalg.cond(A_out_induced))

out_expected_transformed = torch.matmul(torch.matmul(A_out_induced.permute(1, 0), out), A_out_induced)
print('out_transformed', out_transformed.shape, 'out_expected_transformed', out_expected_transformed.shape)
print('Individual norms: ', torch.norm(out_transformed), torch.norm(out_expected_transformed))
print('Normalized equivariance error:', torch.norm(out_transformed - out_expected_transformed) / torch.norm(out_transformed))

