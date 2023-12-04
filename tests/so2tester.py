%load_ext autoreload
%autoreload 2
import sys
sys.path.append('..')
from models.so2models import SO2Layer, SO2Net
import torch
from basis_change import get_poly2fourier
import numpy as np
from generate_data import induced
if torch.cuda.is_available():
    device = torch.device('cuda')
    accelerator = "gpu"
else:
    device = torch.device('cpu')
    accelerator = "cpu"
d = 8
num_in_channels = 1
num_out_channels = 4
num_batches = 2
max_irrep_degree=10
num_layers = 6; num_internal_channels = 5; 
so2net = SO2Net(d, num_layers, num_internal_channels, max_irrep_degree, device=device, num_hidden_for_invar=[10, 10], output_polys=True)

# Test the input layer /fourier transform for equivariance
p = torch.rand(num_batches, d+1)
F = torch.tensor(get_poly2fourier(d))
th = np.pi/8
g = np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])
#g, r = np.linalg.qr(np.random.randn(2,2))
#g[0,:] *= -1
gd = induced(g, d, return_tensor = True)
gp = torch.matmul(gd.T, p.T).T
def input2fourier(x):
    zt = torch.zeros(x.shape).double().to(device)
    xcomplex = torch.complex(x.double(), zt)
    xfourier = torch.matmul(so2net.F, xcomplex.T).T
    xfourier = xfourier.unsqueeze(1)
    xfourier = torch.stack([xfourier, torch.conj(xfourier)], dim = 2)
    return xfourier
Fp = input2fourier(p)
Fgp = input2fourier(gp)
rotations = np.array([np.exp(-j * 1j * th) for j in range(d+1)])
rotations = np.stack([rotations, np.conj(rotations)])
repeatedrotations = torch.tensor(rotations.reshape(1, 1, 2, -1).repeat(num_batches, axis = 0).repeat(num_out_channels, axis = 1)).to(device)
rotFp = torch.multiply(repeatedrotations, Fp)
torch.linalg.norm(rotFp - Fgp) / torch.linalg.norm(Fgp)

# test single layer (below)
so2layer = SO2Layer(d, num_in_channels, num_out_channels, max_irrep_degree, device = device, num_hidden_for_invar=[10, 10])
Fp = input2fourier(p)
Fgp = input2fourier(gp)
sFgp = so2layer(Fgp.to(device))
sFp = so2layer(Fp.to(device))
rotations = np.array([np.exp(-j * 1j * th) for j in range(so2layer.output_degrees+1)])
rotations = np.stack([rotations, np.conj(rotations)])
repeatedrotations = torch.tensor(rotations.reshape(1, 1, 2, -1).repeat(num_batches, axis = 0).repeat(num_out_channels, axis = 1)).to(device)
rotsFp = torch.multiply(repeatedrotations, sFp)
print(torch.linalg.norm(rotsFp - sFgp))

# test net with composition of layers
num_layers = 4; num_internal_channels = 5; 
so2net = SO2Net(d, num_layers, num_internal_channels, max_irrep_degree, device=device, num_hidden_for_invar=[10, 10], output_polys=True)

sFgp = so2net(gp.to(device))
sFp = so2net(p.to(device))
num_out_channels = sFp.shape[1]
rotations = np.array([np.exp(-j * 1j * th) for j in range(so2layer.output_degrees+1)])
rotations = np.stack([rotations, np.conj(rotations)])
repeatedrotations = torch.tensor(rotations.reshape(1, 1, 2, -1).repeat(num_batches, axis = 0).repeat(num_out_channels, axis = 1)).to(device)
rotsFp = torch.multiply(repeatedrotations, sFp)
print(torch.linalg.norm(rotsFp - sFgp) / torch.linalg.norm(rotsFp))

# Full SO(2)
num_layers = 4
g, r = np.linalg.qr(np.random.randn(2,2))
g[:,0] *= -1
gd = induced(g, d, return_tensor = True)
gp = torch.matmul(gd.T, p.T).T
so2net = SO2Net(d, num_layers, num_internal_channels=10, max_irrep_degree=12, device = device, num_hidden_for_invar=[50, 50]).to(device)
sFgp = so2net(gp.to(device))
sFp = so2net(p.to(device))
gd2 = induced(g, int(d/2), return_tensor = True).float().to(device)
print(torch.linalg.norm(gd2.T @ sFp @ gd2 - sFgp) / torch.linalg.norm(sFgp))
#G(p,so2net)

# mldp = get_max_log_det(p[0], solver = "MOSEK").float()
# mldgp = get_max_log_det(gp[0], solver = "MOSEK").float()
# torch.linalg.norm(gd2.T @ mldp @ gd2 - mldgp)/torch.linalg.norm(mldp)