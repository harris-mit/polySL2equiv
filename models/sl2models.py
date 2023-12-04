import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("~/SL2equivariance")
from utils import batch_multiply_poly_tensors
from cg import generate_linear_map, get_irrep_dims, transvectant
import torch
from collections import defaultdict
import copy
import numpy as np
import os
import pickle
from utils import batch_sum_antidiagonals

def get_linear_map(m, n, precomputed_T_dir, force_recompute=False, numerically_stable = True):
    # Like generate_linear_map, but check for precomputed version in precomputed_T_dir according to naming convention below, instead of doing from scratch
    # pass in None to force return to previous function, not editing file system
    
    if precomputed_T_dir is None:
        return generate_linear_map(m, n, numerically_stable)
    filename = os.path.join(precomputed_T_dir, f'T_{m}_{n}.pt')
    if os.path.isfile(filename) and not force_recompute:
        with open(filename, 'rb') as f:
            T = torch.load(f) #pickle.load(f)
    else:
        T = generate_linear_map(m, n, numerically_stable)
        os.makedirs(precomputed_T_dir, exist_ok=True)
        with open(filename, 'wb') as f:
            torch.save(T,f)  #pickle.dump(T, f)
    return T

def all_pairwise_tensors(input_degs, do_tensor_fxn=(lambda x,y: True)):
    # input_degs is a list of degrees that show up in the input
    degree_counts = defaultdict(lambda: 0)
    output_sizes_and_orders = {}
    for ind_i, deg_i in enumerate(input_degs):
        for ind_j, deg_j in enumerate(input_degs):
            if deg_i <= deg_j and do_tensor_fxn(deg_i, deg_j):
                tnsizes, tnorders = get_irrep_dims(deg_i + 1, deg_j + 1)
                output_sizes_and_orders[(deg_i, deg_j)] = (tnsizes, tnorders)
                for k in tnsizes:
                    degree_counts[k - 1] += 1 
    return degree_counts, output_sizes_and_orders


class SL2Net(nn.Module):
    def __init__(self, input_degs, num_layers, num_internal_channels, precomputed_T_dir, max_irrep_degree=10, mode='max_det', device=torch.device('cpu'), numerical_norm=False, num_hidden_for_invar=[10, 10], batch_norm=True, do_skip=True, numerically_stable = True, force_recompute_T = False, transvectant_normalize=False, mlp_args=None, do_tensor_fxn=(lambda x,y: True), scale_set_2=False): # could later add num_input_channels option, when data processed to dictionary outside of forward pass 
        # input_degs should be a list 
        super().__init__()
        self.kwargs = {'input_degs': input_degs, 
                       'num_layers': num_layers, 'num_internal_channels': num_internal_channels, 'max_irrep_degree': max_irrep_degree,'numerical_norm': numerical_norm, 'num_hidden_for_invar': num_hidden_for_invar, 'batch_norm': batch_norm, 'do_skip': do_skip, 'mode':mode, 'precomputed_T_dir':precomputed_T_dir, 'transvectant_normalize':transvectant_normalize, 'numerically_stable':numerically_stable, 'force_recompute_T':force_recompute_T}
        self.scale_set_2 = scale_set_2
        if mode == 'max_det' or mode == 'min_poly':
            num_output_channels = 1
        else:
            num_output_channels = 2
        self.transvectant_normalize = transvectant_normalize
        self.input_degs = input_degs
        self.SL2Layers = nn.ModuleList([])
        next_input_degs = input_degs
        next_input_channels = 1
        next_output_channels = num_internal_channels
        self.device = device
        for layer_ind in range(num_layers):
            if layer_ind <= 1:
                use_do_tensor_fxn = (lambda x,y: True)
            else:
                use_do_tensor_fxn = do_tensor_fxn
            next_layer = SL2Layer(input_degs=next_input_degs, num_input_channels=next_input_channels,
                                  num_output_channels=next_output_channels, max_irrep_degree=max_irrep_degree,
                                  device=device, numerical_norm=numerical_norm, num_hidden_for_invar=num_hidden_for_invar,
                                  batch_norm=batch_norm, do_skip=do_skip, precomputed_T_dir=precomputed_T_dir,
                                  numerically_stable=numerically_stable, force_recompute_T=force_recompute_T,
                                  transvectant_normalize=(transvectant_normalize and layer_ind == 0), do_tensor_fxn=use_do_tensor_fxn, scale_set_2=scale_set_2)

            self.SL2Layers.append(next_layer)

            next_input_degs = next_layer.output_degrees
            next_input_channels = next_output_channels
            if layer_ind == num_layers - 2:
                next_output_channels = num_output_channels
        self.mode = mode
        if mode == 'max_det':
            self.Irreps2SymGramMatrix = Irreps2SymGramMatrix(input_degs[0], device=device, precomputed_T_dir=precomputed_T_dir)
       
        self.mlp_args = mlp_args # for gently breaking equivariance 

        # Option: MLP on raw input, to be added at the end
        # Option: MLP on last representation, to the correct output size
        if self.mlp_args is not None and self.mlp_args['on_input']:
            if self.mode == 'min_poly':
                num_in = sum(input_degs) + len(input_degs)
                num_out = 1
            else:
                deg = input_degs[0]
                num_in = deg + 1
                num_out = (int(deg / 2) + 1)**2
            self.mlp_input = MLP_for_Comparison(num_hidden=[num_in] + self.mlp_args['input_arch'] + [num_out], mode=self.mode)
        if self.mlp_args is not None and self.mlp_args['before_output']:
            penultimate_degs = self.SL2Layers[-1].output_degrees
            num_in = sum(penultimate_degs) + len(penultimate_degs)
            if self.mode == 'min_poly':
                num_out = 1
            else:
                deg = input_degs[0]
                num_out = (int(deg / 2) + 1)**2
            self.mlp_output = MLP_for_Comparison(num_hidden=[num_in] + self.mlp_args['output_arch'] + [num_out], mode=self.mode)

    def forward(self, x, eval_mode=False):
        # Assume x is NOT already a dictionary 
        # x is a tensor, batch x deg+1
        if type(x) == dict or type(x) == defaultdict:
            x_dict = x.copy()
            for ky in x_dict.keys():
                elt = x_dict[ky]
                # reshaping from batch x deg + 1 to batch, 1, deg+1
                x_dict[ky] = elt.reshape(-1, 1, elt.shape[-1])
            if self.mlp_args is not None and self.mlp_args['on_input']:
                mlp_on_input = self.mlp_input(x_dict)
            else:
                mlp_on_input = 0   
        else:
            # try normalizing by sqrt(psi(p,p,d))
            if self.mlp_args is not None and self.mlp_args['on_input']:
                mlp_on_input = self.mlp_input(x)
            else:
                mlp_on_input = 0
            x_dict = {x.shape[-1]-1: x.unsqueeze(1)}
        
        lay_count = 0
        for layer in self.SL2Layers:
            x_dict = layer(x_dict, eval_mode=eval_mode)
            lay_count += 1
        # output degrees for the next layer: self.SL2Layer.degree_counts.keys()
        #return x_dict
        if self.mlp_args is not None and self.mlp_args['before_output']:
            penultimate_result = self.mlp_output(x_dict)
        else:
            if self.mode == 'max_det':
                x_dict[self.input_degs[0]] = x.unsqueeze(1)
                # The following zeroing is because we want a symmetric matrix
                k = self.input_degs[0]
                for i in range((k-2)//4 + 1):
                    x_dict[k - 2 - 4 * i] = torch.zeros(x_dict[k - 2 - 4 * i].shape, device = self.device)

                # make sure original input degree irrep is the input, to enforce the linear constraint (antidiagonal sums)
                penultimate_result = self.Irreps2SymGramMatrix.apply(x_dict)
            else:
                penultimate_result = x_dict[0][:,0,0] 
        return penultimate_result + mlp_on_input

class Irreps2SymGramMatrix():
    def __init__(self, output_degree, device, precomputed_T_dir):
        n = int((output_degree/2) + 1)
        self.n = n
        T = get_linear_map(n, n, precomputed_T_dir)
        self.Tinv = torch.inverse(T).to(device)
        tnsizes, tnorders = get_irrep_dims(n, n)
        self.tnsizes = tnsizes
        self.tnorders = tnorders

    def apply(self, x):
        # x is a dictionary with x[deg] a tensor of size batch x 1 x deg+1 
        vx = torch.cat([x[self.tnsizes[i]-1][:, 0, :] for i in range(len(self.tnsizes))], dim=-1).unsqueeze(-1)
        batch = vx.shape[0]
        asymm_mat = torch.matmul(self.Tinv, vx).reshape(batch, self.n, self.n).double() # batch x vx.shape[1] x 1, before reshape
        
        # symmetrize asymm_mat, which is (possibly?) asymmetric
        return (asymm_mat + asymm_mat.permute(0, 2, 1)).float() / 2  
        
class MLP(nn.Module):
    def __init__(self, num_hidden=[1, 10, 10, 1], input_output_match=True, sigmoid_activation = False):
        super().__init__()
        self.linear_layers = nn.ModuleList([])
        if input_output_match:
            assert num_hidden[0] == num_hidden[-1], "Number of input and output features to the MLP should match (num channels)"
        self.num_hidden = num_hidden
        num_in_next = num_hidden[0]
        self.sigmoid_activation = sigmoid_activation
        for ind in range(1, len(num_hidden)):
            i = num_hidden[ind]
            self.linear_layers.append(nn.Linear(num_in_next, i))
            # initialize weights
            torch.nn.init.kaiming_uniform_(self.linear_layers[ind-1].weight,
                                           a=0, mode='fan_in')#, nonlinearity='leaky_relu')
            #torch.nn.init.zeros_(self.linear_layers[ind-1].bias)
            num_in_next = i
    
    def forward(self, x):
        # x is: batch, degree 

        for ind, lay in enumerate(self.linear_layers):
            x = lay(x)
            if ind <= len(self.linear_layers) - 2:
                if self.sigmoid_activation:
                    x = torch.sigmoid(x)
                else:
                    x = F.leaky_relu(x)
        return x 

class MLP_for_Comparison(nn.Module): # this could be put in another file
    def __init__(self, num_hidden=[1, 10, 10, 36], mode='max_det'):
        # final number of hidden units should be a square

        super().__init__()
        self.kwargs = {'num_hidden': num_hidden, 'mode': mode} # mode added recently
        self.MLP = MLP(num_hidden = num_hidden, input_output_match=False)
        self.num_hidden = num_hidden
        self.mode = mode
    
    def forward(self, x_in, eval_mode=None):
        # ignore eval_mode, not relevant here
        if self.mode == 'max_det':
            dim_of_output = int(np.sqrt(self.num_hidden[-1]))
            if type(x_in) == torch.Tensor:
                raw_result = self.MLP(x_in).reshape(x_in.shape[0], dim_of_output, dim_of_output)
                

                antidiag_sums = batch_sum_antidiagonals(raw_result)
                for i in range(2*dim_of_output-1):
                    if i % 2 == 0:
                        ind = int(i/2)
                        raw_result[:, ind,ind] = raw_result[:, ind,ind] - antidiag_sums[:, i] + x_in[:, i]
                    else:
                        ind = int((i+1)/2)
                        raw_result[:, ind, ind-1] = raw_result[:, ind, ind-1] - antidiag_sums[:, i] + x_in[:, i]

                symmetrized_result = (raw_result + raw_result.permute(0,2,1))/2
                return symmetrized_result
            else: # dict
                x = x_in.copy()
                x_all = []
                for ky in x.keys():
                    x_all.append(x[ky].reshape(x[ky].shape[0], -1))
                x_all = torch.cat(x_all, dim=1)
                return self.MLP(x_all).reshape(x_all.shape[0], dim_of_output, dim_of_output)
        elif self.mode == 'def':
            # the input is a torch tensor, not a dictionary
            
            return self.MLP(x_in)
        elif self.mode == 'min_poly' or self.mode == 'minimizer_poly':
            # need to gather up the tensors of x and concatenate them (not along batch dimension)
            x = x_in.copy()
            x_all = []
            for ky in x.keys():
                x_all.append(x[ky].reshape(x[ky].shape[0], -1))
            x_all = torch.cat(x_all, dim=1)
            if self.mode == 'min_poly':
                out = self.MLP(x_all)
                res = out[:, 0]
                return res
            else:
                return self.MLP(x_all)
       
            

class ScaleNet(nn.Module):
    def __init__(self, net, mode = 'max_det'):
        super().__init__()
        self.net = net
        self.mode = mode
        self.kwargs = {'mode': mode, 'net_type': type(net), 'net_kwargs': net.kwargs}
    
    def forward(self, x_in, eval_mode=False):
        # x is batch x deg+1 (original input tensor)
        if self.mode == 'max_det':
            x = x_in
            norms = torch.norm(x, dim=1).unsqueeze(-1)
            normalized_x = torch.divide(x, norms)
            net_output = self.net(normalized_x, eval_mode=eval_mode)
            return torch.multiply(net_output, norms.unsqueeze(-1))
        elif self.mode == 'min_poly':
            x = {}
            normalizers = torch.norm(torch.cat([x_in[i] for i in x_in.keys()], dim = 0), dim = 1).unsqueeze(-1)
            for ky in x_in.keys():
                x[ky] = torch.divide(x_in[y], normalizers)
            net_output = self.net(x, eval_mode=eval_mode)
            return torch.multiply(net_output, normalizers[:,0])
        else: # minimizer poly
            print("ERROR NOT CONSIDERED")
                


class SL2Layer(nn.Module):
    def __init__(self, input_degs, num_input_channels, num_output_channels, precomputed_T_dir, max_irrep_degree=10, device=torch.device('cpu'), numerical_norm=False, num_hidden_for_invar=[10, 10], batch_norm=True, do_skip=True, numerically_stable=True, force_recompute_T=False, transvectant_normalize=False, do_tensor_fxn=(lambda x,y: True), initial_transvectant_normalize=False, scale_set_2=False):
        # We will maintain the same number of input and output channels for each irrep
        super().__init__() 

        # Precomputed linear (Clebsch-Gordan) mapping from tensor products back to irreps; not learnable
        self.Ts = {}
        for ind_i, deg_i in enumerate(input_degs):
            for ind_j, deg_j in enumerate(input_degs):
                if deg_i <= deg_j and do_tensor_fxn(deg_i, deg_j):
                    self.Ts[(deg_i, deg_j)] = get_linear_map(deg_i+1, deg_j+1, precomputed_T_dir, force_recompute = force_recompute_T,
                                                             numerically_stable = numerically_stable).to(device)

        degree_counts, output_sizes_and_orders = all_pairwise_tensors(input_degs, do_tensor_fxn=do_tensor_fxn)

        self.scale_set_2 = scale_set_2
        self.do_tensor_fxn = do_tensor_fxn
        self.do_skip = do_skip
        if self.do_skip:
            for deg in input_degs:
                degree_counts[deg] += 1

        self.degree_counts = degree_counts
        self.output_sizes_and_orders = output_sizes_and_orders
        self.transvectant_normalize = transvectant_normalize
        self.output_degrees = set(range(max_irrep_degree + 1)).intersection(set(degree_counts.keys()))
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.max_irrep_degree = max_irrep_degree
        self.input_degs = input_degs
        self.linear_layers = nn.ModuleDict({})
        self.numerical_norm = numerical_norm
        self.batch_norm = batch_norm
        self.device = device
        self.running_var_est = None # originally set to None, then updated with each forward pass
        if batch_norm:
            self.batch_norm_params = nn.ParameterDict({})
            for i in input_degs:
                self.batch_norm_params[f'{i}'] = nn.Parameter(torch.ones(num_input_channels), requires_grad=True) # could be torch.ones, but do this ot check equivariance

        if 0 in self.output_degrees:
            num_invar_features_to_mlp = self.num_input_channels * degree_counts[0]
            self.mlp_for_invar = MLP(num_hidden = [num_invar_features_to_mlp] + num_hidden_for_invar + [num_invar_features_to_mlp]) # should be num_input_channels since done before the linear mixing
        for i in self.output_degrees: 
            self.linear_layers[f'lin_deg_{i}'] = nn.Linear(num_input_channels * degree_counts[i], num_output_channels, bias=False) 
            #torch.nn.init.kaiming_uniform_(self.linear_layers[f'lin_deg_{i}'].weight) #using defaults for now...
            #torch.nn.init.normal_(self.linear_layers[f'lin_deg_{i}'].weight)
            #torch.nn.init.normal_(self.linear_layers[f'lin_deg_{i}'].weight) / 100
            self.linear_layers[f'lin_deg_{i}'].weight.data.uniform_(0.0, np.sqrt(3/2)/(num_input_channels * degree_counts[i]))
        

    def forward(self, x_in, eval_mode=False):
        # x is going to be a dictionary with keys = irrep degrees = self.input_degs
        # x[deg] = tensor of size batch channel deg+1, where batch and channel are constant over all keys deg
        assert set(x_in.keys()) == set(self.input_degs), "Input has degrees present which do not match those given to initialize SL2Layer"

        for ky in x_in.keys():
            batch, channel, _ = x_in[ky].shape # any key to get batch and channel size
            break 
        assert channel == self.num_input_channels, "Num input channels does not match"
        x = {} # don't change the inputs in place...
        scales_dict = {}
        if self.batch_norm:
            for ky in x_in.keys():
                thisx = torch.einsum('bcs->bsc',x_in[ky]).view(-1,channel)
                # Only scale by variance...
                # Just compute the variance directly from the batch. #TODO: Learn the scale...
                if eval_mode:
                    if self.running_var_est is None:
                        var_for_normalization = torch.ones(thisx.shape[1:], device=thisx.device)# identity of the correct size...
                    else:
                        var_for_normalization = self.running_var_est
                else:
                    var_for_normalization = torch.var(thisx, dim = 0, unbiased = False)
                    if self.running_var_est is None:
                        self.running_var_est = var_for_normalization
                    else:
                        self.running_var_est = 0.9*self.running_var_est + 0.1*var_for_normalization
                normalizers = torch.sqrt(var_for_normalization + 1e-5)
                #normalizers = torch.maximum(normalizers, 1e-7*torch.ones(normalizers.shape).to(self.device))
                transformed = torch.div(thisx, normalizers)
                transformed = torch.mul(transformed, self.batch_norm_params[f'{ky}'])
                x[ky] = transformed.view(batch, ky+1, channel).permute(0,2,1)
        elif self.transvectant_normalize:
            for ky in x_in.keys():
                if ky != 0:
                    outer_product = torch.bmm(x_in[ky].reshape(batch * channel, ky + 1, 1),
                                              x_in[ky].reshape(batch * channel, 1, ky+1)).reshape(
                                                  batch * channel, (ky + 1) * (ky + 1), 1)
                    irreps_from_tensor = torch.matmul(self.Ts[(ky, ky)].double(), outer_product.double()).float()
                    
                    scales = torch.sqrt(torch.abs(irreps_from_tensor[:, -1, 0]) + 1e-5) # invariant irrep
                    scales_dict[ky] = scales.reshape(batch, channel, -1)
                    x[ky] = torch.div(x_in[ky].reshape(batch * channel, -1), scales.unsqueeze(1)).reshape(batch, channel, -1)
                else:
                    x[ky] = x_in[ky]
        else:
            x = x_in
        intermediate_output = defaultdict(lambda: [])
        
        for ind_i, deg_i in enumerate(self.input_degs):
            for ind_j, deg_j in enumerate(self.input_degs):
                if deg_i <= deg_j and self.do_tensor_fxn(deg_i, deg_j):
                    outer_product = torch.bmm(x[deg_i].reshape(batch * channel, deg_i+1, 1),
                                            x[deg_j].reshape(batch * channel, 1, deg_j+1)).reshape(
                                                batch * channel, (deg_i + 1) * (deg_j + 1) , 1)
                    if self.numerical_norm:
                        fac = torch.norm(outer_product[:, :, 0], dim=1).view(-1, 1, 1) + 1
                    else:
                        fac = torch.ones(batch * channel, 1, 1, device=self.device)
                    irreps_from_tensor = torch.mul(fac, torch.matmul(self.Ts[(deg_i, deg_j)].double(),
                                                                    torch.div(outer_product, fac).double())).float() # extra precision to avoid numerical problems, where large numbers should cancel to 0

                    tnsizes = self.output_sizes_and_orders[(deg_i, deg_j)][0]
                    tnorders = self.output_sizes_and_orders[(deg_i, deg_j)][1]

                    for ind_k, deg_k_plus_one in enumerate(tnsizes):
                        deg_k = deg_k_plus_one - 1
                        #if deg_k_plus_one == 1 and deg_i == deg_j and deg_i != 0:
                        #    intermediate_output[deg_k].append(scales_dict[deg_i]) # These come from the scalings above
                        if deg_k <= self.max_irrep_degree: # only store small-deg irreps, which we care about tracking
                            assert irreps_from_tensor.shape[-1] == 1, "irreps from tensor does not have last dim size = 1"
                            irreps_of_deg_k_from_tensor = irreps_from_tensor[
                                :, sum(tnsizes[0:ind_k]):sum(tnsizes[0:ind_k+1]), 0].view(batch, channel, -1)
                            intermediate_output[deg_k].append(irreps_of_deg_k_from_tensor)

                            # append correct slice of irreps_from_tensor
                            # should have # rows = deg_k + 1, # columns = self.degree_counts[deg_k]
                            # instead of appending, make a list and horizontally concatenate (torch.cat) at the end 
        
        # not sure if skip connection is done correctly with gated nonlinearity at the moment? 
        if self.do_skip:
            for ky in x.keys():
                intermediate_output[ky].append(x[ky]) # use the unnormalized inputs... or maybe normalized...

        final_output = defaultdict(lambda: [])

        for ky in intermediate_output.keys():

            all_irreps_of_deg_ky = torch.einsum('bcs->bsc', torch.cat(intermediate_output[ky], dim=1))
            #stddevs = torch.std(torch.einsum('bsc->bcs',all_irreps_of_deg_ky).view(-1, ky+1), 0)
            #means = torch.mean(torch.einsum('bsc->bcs',all_irreps_of_deg_ky).view(-1, ky+1), 0)
            

            if ky == 0: # new code for mlp before linear recombination
                all_irreps_of_deg_ky = self.mlp_for_invar(all_irreps_of_deg_ky.view(batch, -1)).view(batch, 1, -1)
            
            denom = (self.num_input_channels * self.degree_counts[ky])
            scale_factor = (1 / denom) # clean this up if it works
            # if scale_factor != (1 / all_irreps_of_deg_ky.shape[2]):
            #     print('scale_factor issue')
            #     breakpoint()
            # assert scale_factor == (1 / all_irreps_of_deg_ky.shape[2]), "Scale factor is a different size"
            scale_factor = 1  #TO PUT BACK IN
            if self.scale_set_2:
                scale_factor = 0.5
            if f'lin_deg_{ky}' not in self.linear_layers.keys():
                print('lin deg ky is not in keys, ky is ', ky)
                breakpoint()
            final_output[ky] = scale_factor * torch.einsum('bsc->bcs', self.linear_layers[f'lin_deg_{ky}'](all_irreps_of_deg_ky))

        return final_output
