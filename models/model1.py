import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("~/SL2equivariance")
from utils import batch_multiply_poly_tensors

class SimplePolyModel(nn.Module):
    def __init__(self, input_degree):
        super().__init__()
        self.lin1 = nn.Linear(3*(input_degree+1) - 2, 1)
        
    def forward(self, x):
        squared = batch_multiply_poly_tensors(x, x)
        cubed = batch_multiply_poly_tensors(squared, x)
        result = self.lin1(cubed)
        result = F.relu(result)
        return result



