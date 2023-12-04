import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from utils import count_parameters, normalized_mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sys
sys.path.append('/home/gridsan/hanlaw/SL2equivariance') 
from generate_data import normalize_pair
from cg import compute_induced_from_lie
from torch.autograd.functional import jvp

class PLModel(pl.LightningModule):
    def __init__(self, net, loss_fn=nn.functional.mse_loss, lr=3e-4,
                    use_lr_scheduler = False, equiv_function=None, additional_loss_function=None, use_eval_mode=False, normalize_val=False, 
                    tangent_loss_ops={
                        'hermite': False, 
                        'hermite_scaling': 1, 
                        'pure_equivariance': False, 
                        'pure_equivariance_scaling': 1,
                        },
                    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.num_param = count_parameters(net)
        self.loss_fn = loss_fn
        self.tangent_loss_ops = tangent_loss_ops
        self.lr = lr
        self.use_lr_scheduler = use_lr_scheduler
        # self.equiv_function(x, pred) is call format
        if equiv_function is None:
            self.do_equiv_loss = False
        else:
            self.do_equiv_loss = True
            self.equiv_function = equiv_function
        self.use_eval_mode = use_eval_mode
        self.additional_loss_function = additional_loss_function
        self.normalize_val = normalize_val # only used for transform validation
        self.train_losses = []
        
    def forward(self, batch):
        return self.net(batch)
    
    # induced matrix of exponential of t*X_1 <-- callable function using torch autodiff operations


    # 1 def rotation_lie_deriv(model,imgs):
    # 12 """ Lie deriv. of model w.r.t. rotation, can be scalar/image"""
        # 13 def rotated_model(theta):
            # 14 z = model(rotate(imgs,theta))
            # 15 img_like = (len(z.shape) == 4) # more complex for ViT/Mixer
            # 16 return rotate(z,-theta) if img_like else z
    # 17 return jvp(rotated_model, torch.zeros(1,requires_grad=True))[-1]
    # 18
    # 19 def e_l

    # ------------ EQUATION ------------
    #
    # d/dt (e^{tX}^[d]^T * f(p) * e^{tX}^[d])_{t = 0} = d/dt [ net( e^{tX}^[d] p ) ]_{t=0} 
    # 
    # If the network is already equivariant, this error term becomes:
    #   d/dt (g f(p))_{t=0} - d/dt [ net( g p ) ]_{t=0} 
    # = d/dt (g f(p))_{t=0} - d/dt [ g net( p ) ]_{t=0} 
    # = d/dt (g(t) [f(p) - net(p)]_{t=0} 
    # = d/dt (g(t))_{t=0} [f(p) - net(p)]
    # = constant * [f(p) - net(p)], so then this first order information doesn't add anything (unsurprising)

    def compute_grad_net_at_transformed(self, p): 
        # "RHS"
        # Compute d/dt [ net( e^{tX}^[d] p ) ]_{t=0} 
        # Here, p is treated as an unchanging polynomial. dimensions: batch x (deg + 1) 
        # X is a basis element of the algebra

        d = p.shape[-1] - 1

        def net_at_transformed_t_1(t):
            rep_matrix = compute_induced_from_lie(t, d=d, basis=0).to(p.device) # should be d+1 x d+1
            return self.net(torch.matmul(rep_matrix.permute(1, 0).unsqueeze(0), p.unsqueeze(-1)).squeeze(-1)) # 1 x d+1 x d+1 times batch x d+1 x 1
        
        t1_out, t1_grad = jvp(net_at_transformed_t_1, torch.zeros(1, requires_grad=True), create_graph=True)

        def net_at_transformed_t_2(t):
            rep_matrix = compute_induced_from_lie(t, d=d, basis=1).to(p.device) # should be d+1 x d+1
            return self.net(torch.matmul(rep_matrix.permute(1, 0).unsqueeze(0), p.unsqueeze(-1)).squeeze(-1)) # 1 x d+1 x d+1 times batch x d+1 x 1
        
        t2_out, t2_grad = jvp(net_at_transformed_t_2, torch.zeros(1, requires_grad=True), create_graph=True)

        def net_at_transformed_t_3(t):
            rep_matrix = compute_induced_from_lie(t, d=d, basis=2).to(p.device) # should be d+1 x d+1
            return self.net(torch.matmul(rep_matrix.permute(1, 0).unsqueeze(0), p.unsqueeze(-1)).squeeze(-1)) # 1 x d+1 x d+1 times batch x d+1 x 1
        
        t3_out, t3_grad = jvp(net_at_transformed_t_3, torch.zeros(1, requires_grad=True), create_graph=True)

        # same for net_at_transformed_t2, net_at_transformed_t3

        return t1_grad, t2_grad, t3_grad # t2_grad, t3_grad
    
    def compute_hermite_loss(self, p, f_at_p, mode='max_det'): 
        # Compute d/dt (e^{tX}^[d]^T * f(p) * e^{tX}^[d])_{t = 0}
        # f_at_p is batch x (deg/2 + 1) x (deg/2 + 1)

        d = p.shape[-1] - 1

        if mode is 'max_det':
            def transform_label_1(t):
                rep_matrix = compute_induced_from_lie(t, d=(d/2), basis=0).to(p.device) # should be d+1 x d+1
                return torch.matmul(rep_matrix.permute(1, 0).unsqueeze(0), torch.matmul(f_at_p, rep_matrix.unsqueeze(0)))
            
            lhs_t1_out, lhs_t1_grad = jvp(transform_label_1, torch.zeros(1, requires_grad=True), create_graph=True)

            def transform_label_2(t):
                rep_matrix = compute_induced_from_lie(t, d=(d/2), basis=1).to(p.device) # should be d+1 x d+1
                return torch.matmul(rep_matrix.permute(1, 0).unsqueeze(0), torch.matmul(f_at_p, rep_matrix.unsqueeze(0)))
            
            lhs_t2_out, lhs_t2_grad = jvp(transform_label_2, torch.zeros(1, requires_grad=True), create_graph=True)

            def transform_label_3(t):
                rep_matrix = compute_induced_from_lie(t, d=(d/2), basis=2).to(p.device) # should be d+1 x d+1
                return torch.matmul(rep_matrix.permute(1, 0).unsqueeze(0), torch.matmul(f_at_p, rep_matrix.unsqueeze(0)))
            
            lhs_t3_out, lhs_t3_grad = jvp(transform_label_3, torch.zeros(1, requires_grad=True), create_graph=True)


            # etc for lhs_t2_grad, lhs_t3_grad
        else: # min_poly
            assert False, 'unimplemented but should return 0'
        
        rhs_t1_grad, rhs_t2_grad, rhs_t3_grad = self.compute_grad_net_at_transformed(p)

        # print('rhs_t1_grad', rhs_t1_grad[0:2,0:4])
        # print('lhs_t1_grad', lhs_t1_grad[0:2,0:4])
        # breakpoint()
        return (torch.norm(rhs_t1_grad - lhs_t1_grad)**2 + torch.norm(rhs_t2_grad - lhs_t2_grad)**2 + torch.norm(rhs_t3_grad - lhs_t3_grad)**2) / (torch.norm(lhs_t1_grad)**2 + torch.norm(lhs_t2_grad)**2 + torch.norm(lhs_t3_grad)**2)  # + torch.norm(rhs_t2_grad - lhs_t2_grad) + torch.norm(rhs_t3_grad - lhs_t3_grad) 


    def compute_pure_equivariance_loss(self, p, mode='max_det'):
        return self.compute_hermite_loss(p, self.net(p), mode=mode)

    def compute_overall_loss(self, pred, x, y):
        value_loss = self.loss_fn(pred, y)
        overall_loss = value_loss
        
        with torch.no_grad():
            normalized_loss = normalized_mse_loss(pred, y)

        # Gradient-based loss options
        if self.tangent_loss_ops['hermite']:
            hermite_grad_loss = self.compute_hermite_loss(x, y, mode='max_det')
            hermite_grad_loss *= self.tangent_loss_ops['hermite_scaling']
            overall_loss += hermite_grad_loss
        else:
            hermite_grad_loss = -1
        
        if self.tangent_loss_ops['pure_equivariance']:
            pure_equivariance_grad_loss = self.compute_pure_equivariance_loss(x, mode='max_det')
            pure_equivariance_grad_loss *= self.tangent_loss_ops['pure_equivariance_scaling']
            overall_loss += pure_equivariance_grad_loss 
        else:
            pure_equivariance_grad_loss = -1
        
        return value_loss, normalized_loss, overall_loss, hermite_grad_loss, pure_equivariance_grad_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x)
        # loss = self.loss_fn(pred, y) # replace with compute_overall_loss
        
        # with torch.no_grad():
        #     normalized_loss = normalized_mse_loss(pred, y)

        value_loss, normalized_loss, loss, hermite_grad_loss, pure_equivariance_grad_loss = self.compute_overall_loss(pred, x, y)

        self.log("train_hermite_loss", hermite_grad_loss, on_epoch=True) 
        self.log("train_pure_equiv_loss", pure_equivariance_grad_loss, on_epoch=True) 
        self.log("train_orig_normalized_loss", normalized_loss, on_epoch=True) 
        self.log("train_loss", loss, on_epoch=True) 
        self.log("value_loss", value_loss, on_epoch=True) 
        self.train_losses.append(loss.detach().data)
        
        self.log("train_loss_normalized", normalized_loss, on_epoch=True) 
        if self.additional_loss_function is not None:
            train_additional_loss = self.additional_loss_function(pred, y)
            self.log("train_additional_loss", train_additional_loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # PL automatically turns off gradient computation, dropout, batchnorm :)
        x, y = batch
        if self.use_eval_mode:
            eval_mode = True
        else:
            eval_mode = False
        pred = self.net(x, eval_mode=eval_mode)
        #loss = self.loss_fn(pred, y)
        
        value_loss, normalized_loss, loss, hermite_grad_loss, pure_equivariance_grad_loss = self.compute_overall_loss(pred, x, y)

        self.log("val_hermite_loss", hermite_grad_loss, on_epoch=True) 
        self.log("val_pure_equiv_loss", pure_equivariance_grad_loss, on_epoch=True) 
        self.log("val_orig_normalized_loss", normalized_loss, on_epoch=True) 

        if self.do_equiv_loss:
            with torch.no_grad():
                if pred.dtype != torch.complex64 and pred.dtype != torch.complex:
                    mse_loss = nn.functional.mse_loss(pred, y)
                x_transformed, expected_pred_transformed = self.equiv_function(x, pred) #, thresh=np.sqrt(3)) #thresh included via wrapper in equiv_function now
                
                # normalize x_transformed, expected_transformed
                if self.normalize_val:
                    x_transformed, expected_pred_transformed = normalize_pair(x_transformed, expected_pred_transformed)

                pred_transformed = self.net(x_transformed, eval_mode=True) # eval_mode here should always be true
                equiv_loss = self.loss_fn(pred_transformed, expected_pred_transformed)
                if self.additional_loss_function is not None:
                    val_additional_loss_on_transformed = self.additional_loss_function(pred_transformed, expected_pred_transformed)
                    self.log("val_additional_loss_on_transformed", val_additional_loss_on_transformed, on_epoch=True)

                # overwrite previous x_transformed and pred_transformed!
                x_transformed, y_transformed = self.equiv_function(x, y)

                pred_transformed_before_norm = self.net(x_transformed, eval_mode=True)
                self.log("val_loss_on_transformed_unnormalized", self.loss_fn(pred_transformed_before_norm, y_transformed), on_epoch=True)
                self.log("val_numerator_loss_unnormalized", torch.norm(pred_transformed_before_norm - y_transformed), on_epoch=True)
                # normalize x_transformed, y_transformed
            
                if self.normalize_val:
                    x_transformed, y_transformed = normalize_pair(x_transformed, y_transformed)

                pred_transformed = self.net(x_transformed, eval_mode=True)
                val_loss_on_transformed = self.loss_fn(pred_transformed, y_transformed)

                if pred.dtype != torch.complex64 and pred.dtype != torch.complex:
                    val_unnorm_loss_on_transformed = nn.functional.mse_loss(pred_transformed, y_transformed)
                    self.log("val_unnorm_loss_on_transformed", val_unnorm_loss_on_transformed, on_epoch=True)

                self.log("val_norm_of_pred", torch.norm(pred), on_epoch=True)
                self.log("val_norm_of_y", torch.norm(y), on_epoch=True)
                self.log("val_norm_of_pred_minus_y", torch.norm(pred - y), on_epoch=True)
                if type(x) == torch.tensor:
                    self.log("val_norm_of_x_should_const", torch.norm(x), on_epoch=True)
                self.log("val_norm_of_pred_transformed", torch.norm(pred_transformed), on_epoch=True)
                self.log("val_norm_of_y_transformed", torch.norm(y_transformed), on_epoch=True)
                self.log("val_norm_of_pred_minus_y_transformed", torch.norm(pred_transformed-y_transformed), on_epoch=True)

                self.log("val_loss_on_transformed", val_loss_on_transformed, on_epoch=True)
                
            self.log("log10_val_equivariance_loss", torch.log10(equiv_loss), on_epoch=True) 
            if pred.dtype != torch.complex64 and pred.dtype != torch.complex:
                self.log("mse_loss", mse_loss, on_epoch=True) 



        if self.additional_loss_function is not None:
            val_additional_loss = self.additional_loss_function(pred, y)
            self.log("val_additional_loss", val_additional_loss, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True) 
        self.log("num_param", self.num_param, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = optim.RMSprop(self.parameters(), lr = 1e-3)#, lr= self.lr, momentum = .9)
        out_dict = {"optimizer":optimizer}
        if self.use_lr_scheduler:
                out_dict["lr_scheduler"] = {
                    "scheduler": ReduceLROnPlateau(optimizer),
                    "monitor": "train_loss",
                    "frequency": 1
                        # If "monitor" references validation metrics, then "frequency" should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                }

        return out_dict
