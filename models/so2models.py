import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("~/SL2equivariance")
from basis_change import get_poly2fourier, get_fourier2poly, get_fourier2poly_full
import torch
from collections import defaultdict
import copy
import numpy as np
import os
import pickle
from utils import batch_sum_antidiagonals
from models.sl2models import MLP, Irreps2SymGramMatrix

class SO2Net(nn.Module):
    def __init__(self, input_deg, num_layers, num_internal_channels, precomputed_T_dir, max_irrep_degree=10, mode='max_det', device=torch.device('cpu'), num_hidden_for_invar=[10, 10], output_polys = False): # could later add num_input_channels option, when data processed to dictionary outside of forward pass 
        # input_degs should be a list 
        self.mode = mode
        super().__init__()
        self.kwargs = {'input_deg': input_deg, 
                       'num_layers': num_layers, 'num_internal_channels': num_internal_channels, 'max_irrep_degree': max_irrep_degree,'num_hidden_for_invar': num_hidden_for_invar, 'output_polys': output_polys, 'precomputed_T_dir': precomputed_T_dir}
        self.input_deg = input_deg
        self.SO2Layers = nn.ModuleList([])
        if self.mode == "max_det":
            next_input_channels = 1
            next_input_deg = input_deg
            self.F = torch.tensor(get_poly2fourier(input_deg)).cfloat().to(device) # Fourier matrix
        else: # "min_poly"
            next_input_channels = len(input_deg)
            next_input_deg = max(input_deg) # assume list in this case
            self.F = {d: torch.tensor(get_poly2fourier(d)).cfloat().to(device) for d in input_deg}
        next_output_channels = num_internal_channels
        self.device = device
        self.output_polys = output_polys
        self.max_irrep_degree = max_irrep_degree
        self.precomputed_T_dir = precomputed_T_dir
        for layer_ind in range(num_layers):
            next_layer = SO2Layer(input_deg=next_input_deg,
                                  num_input_channels=next_input_channels,
                                  num_output_channels=next_output_channels, max_irrep_degree=max_irrep_degree,
                                  device=device, num_hidden_for_invar=num_hidden_for_invar)
            self.SO2Layers.append(next_layer)
            next_input_deg = next_layer.output_degrees
            next_input_channels = next_output_channels
            if layer_ind == num_layers - 2:
                if mode == "min_poly":
                    next_output_channels = 1
                elif mode == "max_det":
                    next_output_channels = input_deg + 1 #int(input_deg/2) + 1
        #self.zt = torch.zeros(x.shape).double().to(self.device)
        if mode == 'max_det' and not output_polys:
            assert input_deg <= 8, "The last layer is not currently implemented beyond degree 8 -- needs more trig identities"
            self.Finvs = get_fourier2poly_full(device=self.device)
            self.Irreps2SymGramMatrix = Irreps2SymGramMatrix(input_deg, device=device, precomputed_T_dir=precomputed_T_dir)
       
    def forward(self, x, eval_mode=False):
        if self.mode == "max_det":
            if type(x) == type({}):
                assert len(list(x)) == 1, "Assumes just one input polynomial"
                deg = list(x)[0]
                assert x[deg].shape[1] == 1, "Assumes just one input channel"
                x = x[deg][:,0,:] #.to(self.device)
            # x is a tensor, batch x deg+1
            zt = torch.zeros(x.shape, device=self.device) #.double()
            xcomplex = torch.complex(x, zt).cfloat() # x.double()
            xfourier = torch.matmul(self.F, xcomplex.T).T
            xfourier = xfourier.unsqueeze(1)
            # xfourier below is batch x channel x 2(+ or - deg) x deg + 1
            # The xfourier[:,:,0,:] is positive frequencies
            # xfourier[:,:,1,:] is negative frequencies
            xfourier = torch.stack([xfourier, torch.conj(xfourier)], dim = 2)
        elif self.mode == "min_poly":
            # x is now a dictionary and x[deg] has shape deg + 1
            # transform to xfourier of shape 
            xfouriers = []
            for i in self.F:
                zt = torch.zeros(x[i].shape, device=self.device) #.double()
                zt2 = torch.zeros((x[i].shape[0], max(self.input_deg) - i), device = self.device).cfloat() #.unsqueeze(0)
                # zero pad the frequencies
                xcomplex = torch.complex(x[i], zt).cfloat() # x.double()
                xfourieri = torch.matmul(self.F[i], xcomplex.T).T
                xfourieri = torch.cat([xfourieri, zt2], dim = 1)
                #xfourieri = xfourieri.unsqueeze(0)
                # xfourier below is batch x 2(+ or - deg) x deg + 1
                # The xfourier[:,0,:] is positive frequencies
                # xfourier[:,1,:] is negative frequencies
                xfourieri = torch.stack([xfourieri, torch.conj(xfourieri)], dim = 1)
                xfouriers += [xfourieri]
            xfourier = torch.stack(xfouriers, dim = 1)
            # After all stacking, shape is batch x channel x 2 x max_deg + 1
        else:
            print("Error -- mode not implemented")
        lay_count = 0
        for layer in self.SO2Layers:
            xfourier = layer(xfourier, eval_mode=eval_mode)
            lay_count += 1
        if self.output_polys: # useful for testing and debugging
            return xfourier
        if self.mode == "min_poly":
            return xfourier[:, 0, 0, 0] # get the first invariant
        # Convert xfourier to a matrix
        polyx = {}
        for (inddegp1, degp1) in enumerate(self.Irreps2SymGramMatrix.tnsizes):
            if degp1 <= self.max_irrep_degree + 1 and degp1-1 in self.Finvs.keys():
                # The even frequencies are the ones that show up in self.Finvs columns
                xhat0 = torch.matmul(self.Finvs[degp1-1][0], #.double().to(self.device),
                                     torch.real(xfourier[:,(degp1-1)//2, 0, range(0,degp1, 2)]).T).T.unsqueeze(1).float()
                xhat1 = torch.matmul(self.Finvs[degp1-1][1], #.double().to(self.device),
                                     torch.imag(xfourier[:, (degp1-1)//2, 0, range(0,degp1, 2)]).T).T.unsqueeze(1).float()
                polyx[degp1 - 1] = xhat0 + xhat1
            else:
                polyx[degp1 - 1] = torch.zeros((x.shape[0], 1, degp1), device=self.device).float()
        #print("Setting p[d] = 0 for debugging. Don't leave like this")
        #polyx[x.shape[1] - 1] = torch.zeros(x.shape[0], 1, x.shape[1])
        polyx[x.shape[1] - 1] = x.unsqueeze(1)
        # polyx should be a dictionary with x[deg] a tensor of size batch x 1 x deg+1 
        return self.Irreps2SymGramMatrix.apply(polyx)

def all_fourier_tensor_degs(input_deg, max_irrep_degree):
    # input_deg is the degree of the input
    degree_counts = defaultdict(lambda: 0)
    for deg_i in range(input_deg + 1):
        for deg_j in range(input_deg + 1):
            degree_counts[-deg_i + deg_j] += 1
            if deg_i <= deg_j and deg_i + deg_j <= max_irrep_degree:
                degree_counts[deg_i + deg_j] += 1
                degree_counts[-deg_i - deg_j] += 1
    return degree_counts

class SO2Layer(nn.Module):
    def __init__(self, input_deg, num_input_channels, num_output_channels, max_irrep_degree=10, device=torch.device('cpu'), num_hidden_for_invar=[10, 10]):
        # We will maintain the same number of input and output channels for each irrep
        super().__init__() 
        degree_counts = all_fourier_tensor_degs(input_deg, max_irrep_degree)
        for deg in range(input_deg+1):
            degree_counts[deg] += 1
            if deg > 0: # don't do this twice
                degree_counts[-deg] += 1
        self.degree_counts = degree_counts
        self.output_degrees = min(max_irrep_degree, max(degree_counts.keys()))
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.max_irrep_degree = max_irrep_degree
        self.input_deg = input_deg
        self.linear_layers = nn.ModuleDict({})
        self.device = device

        num_invar_features_to_mlp = self.num_input_channels * degree_counts[0]
        self.mlp_for_invar = MLP(num_hidden = [num_invar_features_to_mlp] + num_hidden_for_invar + [num_invar_features_to_mlp], sigmoid_activation=True).to(device) # should be num_input_channels since done before the linear mixing
        for i in range(-self.output_degrees, self.output_degrees + 1): 
            self.linear_layers[f'lin_deg_{i}'] = nn.Linear(num_input_channels * degree_counts[i], num_output_channels, bias=False).to(torch.cfloat) # HL got rid of .to(device)
            # comment out below initializations to use defaults
            #self.linear_layers[f'lin_deg_{i}'].weight.data.uniform_(-1/(num_input_channels * degree_counts[i]), 1/(num_input_channels * degree_counts[i]))
            #self.linear_layers[f'lin_deg_{i}'].weight.data.uniform_(0, 10)
        

    def forward(self, x, eval_mode=False):
        # x is a tensor indexed by batch, channel, irrep (each irrep is a number)
        assert x.shape[3] == self.input_deg+1, "Input has degrees present which do not match those given to initialize SO2Layer"
        assert x.shape[2] == 2, "Input should have positive and negative frequencies"
        assert x.shape[1] == self.num_input_channels, "Input has wrong number of channels"
        batch = x.shape[0]
        intermediate_output = defaultdict(lambda: [])
        
        for deg_i in range(self.input_deg+1):
            for deg_j in range(self.input_deg+1):
                intermediate_output[-deg_i + deg_j].append(
                        torch.multiply(x[:,:,1,deg_i],x[:,:,0,deg_j]) / 2)
                if deg_i <= deg_j and deg_i + deg_j <= self.max_irrep_degree:
                    intermediate_output[deg_i + deg_j].append(
                        torch.multiply(x[:,:,0,deg_i],x[:,:,0,deg_j]) / 2)
                    intermediate_output[-deg_i - deg_j].append(
                        torch.multiply(x[:,:,1,deg_i],x[:,:,1,deg_j]) / 2)
        for degi in range(x.shape[3]):
            intermediate_output[degi].append(x[:,:,0,degi])
            if degi > 0: # don't do this to zero twice...
                intermediate_output[-degi].append(x[:,:,1,degi])

        final_output = torch.zeros(batch, self.num_output_channels, 2, self.output_degrees+1, dtype = torch.cfloat, device = self.device)

        for ky in intermediate_output.keys():
            if abs(ky) <= self.max_irrep_degree:
                xky = torch.stack(intermediate_output[ky],dim = 2).view(batch, -1)
                if ky == 0: # mlp for invariants
                    xky = self.mlp_for_invar(torch.real(xky).float()).type(torch.complex64) # HL changed from 128 # The invariant will be real...
                if ky >= 0:
                    signidx = 0
                else:
                    signidx = 1
                final_output[:,:,signidx,abs(ky)] = self.linear_layers[f'lin_deg_{ky}'](xky)

        return final_output
