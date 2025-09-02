import torch
from torch import nn
import numpy as np
# from layers import sft_module
import mxnet as mx
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, laplacian_kernel, sigmoid_kernel
from layers import WeightNormLayer, BN_self, LN_self, IN_self, GN_self, BGN_self, IBN_self
## FINER SIREN WIRE GUASS PEMLP

import tqdm
import pdb
import math
from torch.functional import align_tensors
from math import sqrt
import torchvision.models as models

class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def scale_transformation(coords, scale_input_method='gaussian', input_scale_factor=1.0, input_shift_factor=0.0, size=None):
    """
    Scales the input coordinates based on the specified scaling method.

    Args:
        coords (torch.Tensor): Input coordinates with shape (N, D), where N is the number of coordinates and D is the dimension.
        scale_input_method (str): Method for scaling the input coordinates. Options are 'gaussian', 'exponential', 'laplacian'.
        input_scale_factor (float): Scaling factor to apply after distance transformation.
        input_shift_factor (float): Shifting factor to apply after distance transformation.

    Returns:
        torch.Tensor: Scaled coordinates with the same shape as input `coords`.
    """
    input = coords[0, ...]
    dimention = input.shape[1]
    centres = torch.zeros(dimention, dimention).cuda()
    sigmas = torch.ones_like(centres)[0].cuda()
    
    # centres = torch.zeros(2, 2).cuda()
    # sigmas = torch.tensor([1.0, 1.0]).cuda()
    if size is not None:
        size = size
    else:
        size = (input.size(0), 2, 2)
    
    x = input.unsqueeze(1).expand(size)
    c = centres.unsqueeze(0).expand(size)
    
    if scale_input_method == 'gaussian':
        distances = (x - c).pow(2).sum(-1) / (2 * sigmas.unsqueeze(0).pow(2))
    elif scale_input_method == 'exponential':
        distances = torch.abs(x - c).sum(-1) / (2 * sigmas.unsqueeze(0).pow(2))
    elif scale_input_method == 'laplacian':
        distances = torch.abs(x - c).sum(-1) / sigmas.unsqueeze(0)
    else:
        raise ValueError(f"Unknown scale_input_method: {scale_input_method}")

    scaled_coords = input_scale_factor * torch.exp(-distances).unsqueeze(0) + input_shift_factor
    
    return scaled_coords

## SIREN
class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model. If it is, no
            activation is applied and 0.5 is added to the output. Since we
            assume all training data lies in [0, 1], this allows for centering
            the output of the model.
        use_bias (bool): Whether to learn bias in linear layer.
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if not self.is_last:
            out = self.activation(out)
        return out


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)    
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)


    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out
    
class SineLayerConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        # self.conv_kernel_hidden = nn.Conv1d(256, 256, kernel_size=5, padding=2, groups=256, bias=False)
        self.conv_kernel_hidden = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False)
        custom_weights_hidden = torch.ones_like(self.conv_kernel_hidden.weight)
        with torch.no_grad():
            self.conv_kernel_hidden.weight.copy_(custom_weights_hidden) 
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)    
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)


    def forward(self, input):
        linear_out = self.linear(input)
        #2d conv
        linear_out = linear_out.permute(0, 2, 1).reshape(1,256,256,256)
        linear_out = self.conv_kernel_hidden(linear_out)
        linear_out = linear_out.reshape(1,256,-1)
        linear_out = linear_out.permute(0, 2, 1)
        # ##1d conv
        # linear_out = linear_out.permute(0, 2, 1)
        # linear_out = self.conv_kernel_hidden(linear_out)
        # linear_out = linear_out.permute(0, 2, 1)
        out = torch.sin(self.omega_0 * linear_out)
        return out

class SineLayerNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, norm_type=None):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm_type = norm_type
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)    
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)


    def forward(self, input):
        input = self.linear(input)
        if self.norm_type == 'BN':
            norm = BN_self(input.shape[2]).cuda()
        elif self.norm_type == 'LN':
            norm = LN_self(input.shape[2]).cuda()
        elif self.norm_type == 'GN':
            norm = GN_self(input.shape[2], groups=32).cuda() ## default groups=32
        elif self.norm_type == 'BGN':
            norm = BGN_self(input.shape[2], groups=32).cuda()  ## default groups=64
        elif self.norm_type == 'IN':
            norm = IN_self(input.shape[2]).cuda()
        else:
            raise ValueError(f"Invalid norm type: {self.norm_type}. Please choose 'BN', 'LN', 'GN', or 'IN'.") 
        input_norm = norm(input.squeeze(0)[:,:,None,None])
        input_norm = input_norm.squeeze().unsqueeze(0)
        out = torch.sin(self.omega_0 * input_norm)
        # out = torch.sin(self.omega_0 * self.linear(input))
        return out
    
class SineLayerNormImage(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, norm_type=None, resolution=None):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm_type = norm_type
        self.resolution = resolution
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)    
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)


    def forward(self, input):
        input = self.linear(input)
        if self.norm_type == 'BN':
            norm = BN_self(input.shape[2]).cuda()
        elif self.norm_type == 'IBN':
            norm = IBN_self(input.shape[2], groups=2).cuda()
        elif self.norm_type == 'LN':
            norm = LN_self(input.shape[2]).cuda()
        elif self.norm_type == 'GN':
            norm = GN_self(input.shape[2], groups=32).cuda()
        elif self.norm_type == 'IN':
            norm = IN_self(input.shape[2]).cuda()
        else:
            raise ValueError(f"Invalid norm type: {self.norm_type}. Please choose 'BN', 'LN', 'GN', or 'IN'.")     
                
        height, width = self.resolution
        input_norm = norm(input.permute(0,2,1).reshape(input.shape[0],input.shape[2], height, width))
        input_norm = input_norm.reshape(input.shape[0],input.shape[2],height*width).permute(0,2,1)
        out = torch.sin(self.omega_0 * input_norm)
        # out = torch.sin(self.omega_0 * self.linear(input))
        return out


class SineLayerNormaudio(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, norm_type=None):
        super(SineLayerNormaudio, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm_type = norm_type
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)    
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)
                

        # import pdb;   pdb.set_trace()

    def forward(self, input):
        input = self.linear(input)
        
        # Apply the appropriate normalization based on norm_type
        if self.norm_type == 'BN':
            norm = nn.BatchNorm1d(input.shape[1]).cuda()
        elif self.norm_type == 'LN':
            norm = nn.LayerNorm(input.shape[1:]).cuda()
        elif self.norm_type == 'GN':
            norm = nn.GroupNorm(32, input.shape[1]).cuda()
        elif self.norm_type == 'IN':
            norm = nn.InstanceNorm1d(input.shape[1]).cuda()
        else:
            raise ValueError(f"Invalid norm type: {self.norm_type}. Please choose 'BN', 'LN', 'GN', or 'IN'.") 
        
        
        # Apply the normalization and activation
        input_norm = norm(input.transpose(1, 2))
        input_norm = input_norm.transpose(1, 2)
        out = torch.sin(self.omega_0 * input_norm)
        
        return out

    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True,  scale_input_method=None, input_scale_factor=1, input_shift_factor=0, scale_output=0, scale_output_method=None, bias_prior_add=None, output_scale_factor=1, output_shift_factor=0, norm_type=None, norm_layer_num=-1, resolution=None):
        super().__init__()

        
        self.net = []
        if norm_type is not None and norm_layer_num < 0: 
            self.net.append(SineLayerNorm(in_features, hidden_features, is_first=True, omega_0=first_omega_0, bias=bias, norm_type=norm_type))
            # self.net.append(SineLayerNormImage(in_features, hidden_features, is_first=True, omega_0=first_omega_0, bias=bias, norm_type=norm_type, resolution=resolution))
        else:
            # self.net.append(SineLayerConv(in_features, hidden_features, is_first=True, omega_0=first_omega_0, bias=bias))
            self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, bias=bias))
        
        for i in range(hidden_layers):
            if i==-1:
                self.net.append(SineLayerConv(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
                continue
            if i == norm_layer_num:
                self.net.append(SineLayerNorm(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, norm_type=norm_type)) 
                continue
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
            
                

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)
        self.bias_prior_add = bias_prior_add
        self.scale_output = scale_output
        self.scale_input_method = scale_input_method
        self.scale_output_method = scale_output_method
        self.input_scale_factor = input_scale_factor
        self.input_shift_factor = input_shift_factor
        self.output_scale_factor = output_scale_factor
        self.output_shift_factor = output_shift_factor
        
        # self.conv_kernel_input = nn.Conv1d(in_features, in_features, kernel_size=5, padding=2, groups=in_features, bias=False)
        # self.conv_kernel_output = nn.Conv1d(out_features, out_features, kernel_size=9, padding=4, groups=out_features, bias=False)
        # self.conv_kernel_hidden = nn.Conv1d(hidden_features, hidden_features, kernel_size=5, padding=2, groups=hidden_features, bias=False)
        # custom_weights_input = torch.ones_like(self.conv_kernel_input.weight)
        # custom_weights_output = torch.ones_like(self.conv_kernel_output.weight)
        # custom_weights_hidden = torch.ones_like(self.conv_kernel_hidden.weight)
        # with torch.no_grad():
        #     self.conv_kernel_input.weight.copy_(custom_weights_input)
        #     self.conv_kernel_output.weight.copy_(custom_weights_output)
        #     self.conv_kernel_hidden.weight.copy_(custom_weights_hidden) 

    def forward(self, coords):
       
        ######input scale
        if self.scale_input_method == 'linear':
            #scale kernel
            coords = self.input_scale_factor*coords + self.input_shift_factor
            # coords = (coords * 0.5 + 0.5) * 255            
            #convolution kernel
            # coords_inter = coords.permute(0,2,1)
            # coords_inter = self.conv_kernel_input(coords_inter)
            # coords = coords_inter.permute(0,2,1)
        elif self.scale_input_method == 'polynomial':
            coords = self.input_scale_factor*(coords**3) + self.input_shift_factor
        elif self.scale_input_method == 'cosine':
            coords = self.input_scale_factor * torch.cos(30 * coords) + self.input_shift_factor
        elif self.scale_input_method in ['gaussian', 'exponential', 'laplacian']:
            coords = scale_transformation(coords, scale_input_method=self.scale_input_method, input_scale_factor=self.input_scale_factor, input_shift_factor=self.input_shift_factor)
        else:
            coords = coords
        output = self.net(coords) 
        if self.scale_output:
            if self.bias_prior_add is not None:
                output = self.output_scale_factor * output + self.bias_prior_add
            elif self.scale_output_method in ['polynomial', 'gaussian', 'exponential', 'laplacian']:
                if self.scale_output_method == 'polynomial':
                    coords = self.input_scale_factor*(output**3) + self.input_shift_factor
                else:
                    size = (output.size(1), 3, 3)
                    output = scale_transformation(output, scale_input_method=self.scale_output_method, input_scale_factor=self.output_scale_factor, input_shift_factor=self.output_shift_factor, size=size)     
            else:
                #scale kernel
                output = self.output_scale_factor * output + self.output_shift_factor
                
                #convolution kernel
                # output_inter = output.permute(0,2,1)
                # output_inter = self.conv_kernel_output(output_inter)
                # output = output_inter.permute(0,2,1)
        
        
        ##### Calculate each layer output before sine function
        # output_layer1 = self.net[0].linear(coords)
        # output_layer2 = self.net[1].linear(self.net[0](coords))
        # output_layer3 = self.net[2].linear(self.net[1](self.net[0](coords)))
        # output_layer4 = self.net[3].linear(self.net[2](self.net[1](self.net[0](coords))))  
        # with open('./output_value_scale_weight.txt', 'a') as f:
        #   for layer_num in range(1, 5):
        #     output_layer = eval(f'output_layer{layer_num}')
        #     abs_mean = torch.abs(output_layer).mean().item()
        #     max_value = output_layer.max().item()
        #     min_value = output_layer.min().item()
        #     f.write(f'output_layer{layer_num}_abs_mean: {abs_mean}, max: {max_value}, min: {min_value}\n')
        return output

## FINER sin⁡(𝑜𝑚𝑒𝑔𝑎∗𝑠𝑐𝑎𝑙𝑒∗(𝑊𝑥+𝑏𝑖𝑎𝑠)) 𝑠𝑐𝑎𝑙𝑒=|𝑊𝑥+𝑏𝑖𝑎𝑠|+1
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)
    # import pdb;pdb.set_trace()
    def generate_scale(self, x):
        if self.scale_req_grad: 
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out

class FinerLayerNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False, norm_type=None):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        self.norm_type = norm_type
        if self.first_bias_scale != None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)
    # import pdb;pdb.set_trace()
    def generate_scale(self, x):
        if self.scale_req_grad: 
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        if self.norm_type == 'BN':
            norm = BN_self(x.shape[2]).cuda()
        elif self.norm_type == 'LN':
            norm = LN_self(x.shape[2]).cuda()
        elif self.norm_type == 'BGN':
            norm = BGN_self(x.shape[2], groups=32).cuda()  ## default groups=64
        elif self.norm_type == 'GN':
            norm = GN_self(x.shape[2], groups=32).cuda()
        elif self.norm_type == 'IN':
            norm = IN_self(x.shape[2]).cuda()
        else:
            raise ValueError(f"Invalid norm type: {self.norm_type}. Please choose 'BN', 'LN', 'GN', or 'IN'.")  
        x_norm = norm(x.squeeze(0)[:,:,None,None])
        x = x_norm.squeeze().unsqueeze(0)
        out = torch.sin(self.omega_0 * scale * x)
        return out
    

class Finer(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
                 first_bias_scale=None, scale_req_grad=False, norm_type=None, norm_layer_num=-1):
        super().__init__()
        self.net = []
        if norm_type is not None and norm_layer_num < 0:
            self.net.append(FinerLayerNorm(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad, norm_type=norm_type))
        else:
            self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

        for i in range(hidden_layers):
            if i == norm_layer_num:
                self.net.append(FinerLayerNorm(hidden_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad, norm_type=norm_type))
                continue
            self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output        


#han20241124
class RandomFourierFeatures(nn.Module):
    """Random Fourier Features with default Pytorch initialization.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        use_bias (bool): Whether to learn bias in linear layer.
        w0 (float):
    """
    def __init__(self,
            dim_in,
            dim_out,
            use_bias=True,
            w0=30.0
                 ):
        super().__init__()  
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.use_bias = use_bias
        self.act_fn = Sine(w0=w0)
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

    def forward(self, x):
        return self.act_fn(self.linear(x))

        
class SCONE(nn.Module):
    """Spatially-Collaged Coordinated Network (SCONE) model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden units.
        dim_out (int): Dimension of output.
        use_bias (bool): Whether to learn bias in linear layer.
        w0 (float):
    """
    def __init__(
            self,
            dim_in,
            dim_out,
            scone_configs=None
                 ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = 256#250#scone_configs.dim_hidden
        self.dim_out = dim_out
        self.num_layers = 4#scone_configs.num_layers
        self.omegas = [90,60,30]#scone_configs.omegas
        self.w0 = 30#scone_configs.w0
        self.use_bias = True#scone_configs.use_bias

        layers = []
        for ind in range(self.num_layers - 1):
            is_first = ind == 0
            layer_dim_in = dim_in if is_first else self.dim_hidden
            # import pdb;pdb.set_trace()
            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=self.dim_hidden,
                    w0=self.w0,
                    use_bias=self.use_bias,
                is_first=is_first,
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayer(
            dim_in=self.dim_hidden, dim_out=dim_out, w0=self.w0, use_bias=self.use_bias, is_last=True
        )

        self.rffs = nn.ModuleList(
            [RandomFourierFeatures(dim_in=dim_in, dim_out=self.dim_hidden, w0=self.omegas[i]) 
             for i in range(self.num_layers-1)]) 

    def forward(self, x, step=None, exp_name=None, **kwargs):
        input = x
        for i, module in enumerate(self.net):
            bases = self.rffs[i](input)
            masks = module(x)
            masks = 0.5 * (1 - torch.cos(2 * masks)) #masks*masks # sin^2
            x = bases*masks

        x = self.last_layer(x)
        return x

    
class MLP(torch.nn.Sequential):
    '''
    Args:
        in_channels (int): Number of input channels or features.
        hidden_channels (list of int): List of hidden layer sizes. The last element is the output size.
        mlp_bias (float): Value for initializing bias terms in linear layers.
        activation_layer (torch.nn.Module, optional): Activation function applied between hidden layers. Default is SiLU.
        bias (bool, optional): If True, the linear layers include bias terms. Default is True.
        dropout (float, optional): Dropout probability applied after the last hidden layer. Default is 0.0 (no dropout).
    '''
    def __init__(self, MLP_configs, bias=True, dropout = 0.0):
        super().__init__()

        in_channels=MLP_configs['in_channels'] 
        hidden_channels=MLP_configs['hidden_channels']
        self.mlp_bias=MLP_configs['mlp_bias']
        activation_layer=MLP_configs['activation_layer']

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if MLP_configs['task'] == 'denoising':
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.001)
            torch.nn.init.constant_(m.bias, self.mlp_bias)

    def forward(self, x):
        out = self.layers(x)
        return out