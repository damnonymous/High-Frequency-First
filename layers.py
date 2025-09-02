import torch
import torch.nn as nn
from torch import linalg as LA
import mxnet as mx
from torch.nn.parameter import Parameter
import torch.nn.functional as F

###############  Helper functions ###############
def compute_windowed_variance(x, window_size):
    pad = window_size // 2
    x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
    
    # x_unfold = x_padded.unfold(2, window_size, 1).unfold(3, window_size, 1)
    # mean = x_unfold.mean(dim=[2, 3])
    # mean_square = (x_unfold ** 2).mean(dim=[2, 3])
    
    N, C, H, W = x.shape
    mean = torch.zeros(N, C, H, W).cuda()
    mean_square = torch.zeros(N, C, H, W).cuda()
    for i in range(window_size):
        for j in range(window_size):
            mean += x_padded[:, :, i:i+H, j:j+W] / (window_size * window_size)
            mean_square += (x_padded[:, :, i:i+H, j:j+W] ** 2) / (window_size * window_size)
    
    var = mean_square - mean ** 2
    return mean, var


def instance_std(x, eps=1e-5):
    var = torch.var(x, dim = (2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape).to(x.device)
    return torch.sqrt(var + eps)

def group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

def group_mean(x, groups=32):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    mean = torch.mean(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(mean, (N, C, H, W))

def batch_group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    batch_groups = groups * 1
    x = torch.reshape(x, (batch_groups, N // batch_groups, groups, C // groups, H, W))
    var = torch.var(x, dim = (1, 3, 4, 5), keepdim = True).expand_as(x)
    # var = torch.var(x, dim=(0, 1, 2, 3, 4, 5), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

def batch_group_mean(x, groups=32):
    N, C, H, W = x.size()
    batch_groups = groups * 1
    x = torch.reshape(x, (batch_groups, N // batch_groups, groups, C // groups, H, W))
    mean = torch.mean(x, dim = (1, 3, 4, 5), keepdim = True).expand_as(x)
    # mean = torch.mean(x, dim = (0, 1, 2, 3, 4, 5), keepdim = True).expand_as(x)
    return torch.reshape(mean, (N, C, H, W))


def batch_group_mean_std(x, groups=32, eps = 1e-5):
    N, C, H, W = x.size()
    # import pdb;pdb.set_trace()
    import pdb;pdb.set_trace()
    batch_groups = 32   ####default batch_groups = 32
    groups = 32
    x_origin = x
    x = torch.reshape(x, (batch_groups, N // batch_groups, groups, C // groups, H, W))
    mean = torch.mean(x, dim = (1, 3, 4, 5), keepdim = True).expand_as(x)
    # mean = torch.mean(x, dim = (0, 1, 2, 3, 4, 5), keepdim = True).expand_as(x)
    
    batch_groups_var = 64  ###default batch_gbatch_groups_var = 128
    groups_var = 64
    # import pdb;pdb.set_trace()
    x = torch.reshape(x_origin, (batch_groups_var, N // batch_groups_var, groups_var, C // groups_var, H, W))
    var = torch.var(x, dim = (1, 3, 4, 5), keepdim = True).expand_as(x)
    
    return torch.reshape(mean, (N, C, H, W)), torch.reshape(torch.sqrt(var + eps), (N, C, H, W)) 

# def batch_group_mean_std(x, groups=32, eps=1e-5):
#     N, C, H, W = x.size()
#     total_elements = N * C * H * W  
    
#     while total_elements % (groups * 8 * H * W) != 0:
#         groups -= 1
#         if groups == 1:
#             break 
    
#     batch_groups = N // groups 
#     remaining_dim = total_elements // (batch_groups * groups * 8 * H * W)  
    
#     try:
#         x = torch.reshape(x, (batch_groups, remaining_dim, groups, C // groups, H, W))
#     except RuntimeError as e:
#         raise

#     # 检查 reshape 后的数据形状
#     print(f"Reshaped tensor shape: {x.shape}")

#     mean = torch.mean(x, dim=(1, 3, 4, 5), keepdim=True).expand_as(x)
#     var = torch.var(x, dim=(1, 3, 4, 5), keepdim=True).expand_as(x)
#     return torch.reshape(mean, (N, C, H, W)), torch.reshape(torch.sqrt(var + eps), (N, C, H, W))






def intra_batch_group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    batch_groups = groups * 1
    x = torch.reshape(x, (batch_groups, N // batch_groups, groups, C // groups, H, W))
    var = torch.var(x, dim = (1, 3, 4, 5), keepdim = True).expand_as(x)
    # var = torch.var(x, dim=(0, 1, 2, 3, 4, 5), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))


def intra_batch_mean_var(x, groups=32, eps = 1e-5):
    N, C, H, W = x.size()
    # kernel_size = 11
    # weight = ((torch.ones(256, 1, kernel_size, kernel_size)) / (kernel_size**5)).float().cuda()   ## kernel_size **5
    # mean = F.conv2d(x, weight, padding=kernel_size//2, groups=256) 
    # mean_square = F.conv2d(x**2, weight, padding=kernel_size//2, groups=256) 
    # var = mean_square - mean**2
    x_origin = x
    groups = 1 
    channel_groups = 1
    x = torch.reshape(x, (N, channel_groups, C //channel_groups, groups, H // groups, groups, W //groups))
    mean = torch.mean(x, dim = (0, 2, 4, 6), keepdim = True).expand_as(x) 
    mean = torch.reshape(mean, (N, C, H, W))

    var_groups = 1 
    x_var = torch.reshape(x_origin, (N, channel_groups, C //channel_groups, var_groups, H // var_groups, var_groups, W //var_groups))
    var = torch.var(x_var, dim = (0, 2, 4, 6), keepdim = True).expand_as(x_var) 
    var = torch.reshape(var, (N, C, H, W))
   
    return mean, torch.sqrt(var + eps)

class WCConv2d(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
            bias=True, gamma=1.0, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

    def get_weight(self):
        weight = self.weight - torch.mean(self.weight, dim=[1, 2, 3], keepdim=True)
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


############### Parametric Normalization Layers ############### 
### Scaled Weight Standardization (Brock et al., 2021) ###
# (based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/nfnet.py) 
class ScaledStdConv2d(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
            bias=True, scale_activ=1.0, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gamma = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.scale = scale_activ / (self.weight[0].numel() ** 0.5)  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (self.weight - mean) / (std + self.eps)
        return (2 * pi / (pi - 1)) ** 0.5 * self.gamma * self.scale * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


### Weight Normalization (Salimans and Kingma, 2016) ###
class WN_self(nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gamma = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.eps = eps

    def get_weight(self):
        intermediate_norms = torch.linalg.norm(self.weight, dim=[2, 3], keepdim=True)
        denom = torch.linalg.norm(intermediate_norms, dim=[1], keepdim=True)
        # denom = torch.linalg.norm(self.weight, dim=[1, 2, 3], keepdim=True)
        weight = self.weight / (denom + self.eps)
        return self.gamma * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

# Scaled activation function for WeightNorm (Performs scaled/bias correction; Arpit et al., 2016)
class WN_scaledReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return (2 * pi / (pi - 1))**0.5 * (F.relu(x, inplace=self.inplace) - (1 / (2 * pi))**(0.5))


############### Activations-based normalization layers ###############
### BatchNorm (Ioffe and Szegedy, 2015) ###
# (based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d)
class BN_self(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('moving_mean', torch.ones(shape))
        self.register_buffer('moving_var', torch.ones(shape))
        self.reset_parameters()

    def reset_parameters(self):
        self.moving_var.fill_(1)

    def forward(self, X):
        if self.training:
            var, mean = torch.var_mean(X, dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.moving_mean.mul_(self.momentum)
            self.moving_mean.add_((1 - self.momentum) * mean)
            self.moving_var.mul_(self.momentum)
            self.moving_var.add_((1 - self.momentum) * var)
        else:
            var = self.moving_var
            mean = self.moving_mean
        X = (X - mean) * torch.rsqrt(var+self.eps)
        return X * self.gamma + self.beta
    

# ##with momentum
# class IBN_self(nn.Module):
#     def __init__(self, num_features, momentum=0.9, eps=1e-5, groups=32):
#         super(IBN_self, self).__init__()
#         shape = (1, num_features, 256, 256)
#         self.gamma = nn.Parameter(torch.ones(shape))
#         self.beta = nn.Parameter(torch.zeros(shape))
#         self.momentum = momentum
#         self.eps = eps
#         self.groups = groups

#         self.register_buffer('moving_mean', torch.ones(shape))
#         self.register_buffer('moving_var', torch.ones(shape))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.moving_var.fill_(1)

#     def forward(self, X, eps=1e-5):
#         if self.training:
#             mean, var = intra_batch_mean_var(X, groups=self.groups, eps=eps)
#             self.moving_mean.mul_(self.momentum)
#             self.moving_mean.add_((1 - self.momentum) * mean)
#             self.moving_var.mul_(self.momentum)
#             self.moving_var.add_((1 - self.momentum) * var)
#         else:
#             var = self.moving_var
#             mean = self.moving_mean
#         X = (X - mean) * torch.rsqrt(var+self.eps)
#         return X * self.gamma + self.beta


class IBN_self(nn.Module):
    def __init__(self, num_features, groups=32):
        super(IBN_self, self).__init__()
        self.num_features = num_features
        self.groups = groups
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x, eps=1e-5):
        # me2 = intra_batch_group_mean(x, groups=self.groups)
        # nu2 = intra_batch_group_std(x, groups=self.groups, eps=eps)
        me2, nu2 = intra_batch_mean_var(x, groups=self.groups, eps=eps)
        x = (x-me2) / (nu2)
        return self.gamma * x + self.beta    



### LayerNorm (Ba et al., 2016) ###
class LN_self(nn.Module):
    def __init__(self, num_features, data_type=None):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.data_type = data_type

    def forward(self, X, eps=1e-5):
        if self.data_type == 'complex_type':
            X = X.real
        var, mean = torch.var_mean(X, dim=(1, 2, 3), keepdim=True, unbiased=False)
        X = (X - mean) / torch.sqrt(var + eps) 
        return self.gamma * X + self.beta  


### InstanceNorm (Ulyanov et al., 2017) ###
class IN_self(nn.Module):
    def __init__(self, num_features):
        super(IN_self, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, X, eps=1e-5):
        var, mean = torch.var_mean(X, dim=(2, 3), keepdim=True, unbiased=False)
        X = (X - mean) / torch.sqrt(var + eps) 
        return self.gamma * X + self.beta


### GroupNorm (Wu and He, 2018) ###
class GN_self(nn.Module):
    def __init__(self, num_features, groups=32):
        super(GN_self, self).__init__()
        self.num_features = num_features
        self.groups = groups
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x, eps=1e-5):
        me2 = group_mean(x, groups=self.groups)
        nu2 = group_std(x, groups=self.groups, eps=eps)
        x = (x-me2) / (nu2)
        return self.gamma * x + self.beta

class BGN_self(nn.Module):
    def __init__(self, num_features, groups=32):
        super(BGN_self, self).__init__()
        self.num_features = num_features
        self.groups = groups
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x, eps=1e-5):
        # me2 = batch_group_mean(x, groups=self.groups)
        # nu2 = batch_group_std(x, groups=self.groups, eps=eps)
        me2, nu2  = batch_group_mean_std(x, groups=self.groups, eps=eps)
        x = (x-me2) / (nu2)
        return self.gamma * x + self.beta

### Filter Response Normalization (Singh and Krishnan, 2019) ###
class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)


class FRN_self(nn.Module):
    def __init__(self, num_features, eps=1e-5, is_eps_learnable=True):
        super(FRN_self, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_learnable = is_eps_learnable

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        if self.is_eps_learnable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        return self.gamma * x + self.beta


### Variance Normalization (Daneshmand et al., 2020) ###
# Essentially an ablation of BatchNorm that the authors found to be as successful as BatchNorm
class VN_self(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('moving_var', torch.ones(shape))
        self.reset_parameters()

    def reset_parameters(self):
        self.moving_var.fill_(1)

    def forward(self, X):
        if self.training:
            var = torch.var(X, dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.moving_var.mul_(self.momentum)
            self.moving_var.add_((1 - self.momentum) * var)
        else:
            var = self.moving_var

        X = X * torch.rsqrt(var+self.eps)
        return X * self.gamma + self.beta


############### AutoML designed layers (based on https://github.com/digantamisra98/EvoNorm) ###############
### EvoNormSO (Liu et al., 2020) ###
class EvoNormSO(nn.Module):
    def __init__(self, num_features, eps = 1e-5, groups = 32):
        super(EvoNormSO, self).__init__()
        self.groups = groups
        self.eps = eps
        self.num_features = num_features

        self.gamma = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))
        self.v = nn.Parameter(torch.ones(1,self.num_features, 1, 1))

    def forward(self, x):
        num = x * torch.sigmoid(self.v * x)   
        return num / group_std(x, groups = self.groups, eps = self.eps) * self.gamma + self.beta

### EvoNormBO (Liu et al., 2020) ###
class EvoNormBO(nn.Module):
    def __init__(self, num_features, momentum = 0.9, eps = 1e-5, training = True):
        super(EvoNormBO, self).__init__()
        self.training = training
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features

        self.gamma = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))
        self.v = nn.Parameter(torch.ones(1,self.num_features, 1, 1))
        self.register_buffer('moving_var', torch.ones(1, self.num_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.moving_var.fill_(1)

    def forward(self, x):
        if self.moving_var.device != x.device:
            self.moving_var = self.moving_var.to(x.device)
        if self.training:
            var = torch.var(x, dim = (0, 2, 3), unbiased = False, keepdim = True)
            self.moving_var.mul_(self.momentum)
            self.moving_var.add_((1 - self.momentum) * var)
        else:
            var = self.moving_var

        den = torch.max((var+self.eps).sqrt(), self.v * x + instance_std(x, eps = self.eps))
        return x / den * self.gamma + self.beta






class WeightNormLayer(nn.Module):
    def __init__(self, module):
        super(WeightNormLayer, self).__init__()

        weight = getattr(module, 'weight')
        delattr(module, 'weight')

        module.register_parameter('weight_g', nn.Parameter(self.norm_except_dim_0(weight)))
        module.register_parameter('weight_v', nn.Parameter(weight))
        self.module = module

        self.compute_weight()

    def compute_weight(self):
        g = getattr(self.module, 'weight_g')
        v = getattr(self.module, 'weight_v')
        # w = v / self.norm_except_dim_0(v)
        w = g * v / self.norm_except_dim_0(v)
        setattr(self.module, 'weight', w)

    @staticmethod
    def norm_except_dim_0(weight):
        output_size = (weight.size(0),) + (1,) * (weight.dim() - 1)
        out = LA.norm(weight.view(weight.size(0), -1), ord=2, dim=1).view(*output_size)
        return out

    def forward(self, x):
        self.compute_weight()
        return self.module.forward(x)

class WeightNormLayerReLU(nn.Module):
    def __init__(self, module):
        super(WeightNormLayerReLU, self).__init__()
        weight = getattr(module, 'weight')
        delattr(module, 'weight')

        module.register_parameter('weight_g', nn.Parameter(self.norm_except_dim_0(weight)))
        module.register_parameter('weight_v', nn.Parameter(weight))
        self.module = module

        self.compute_weight()

    def compute_weight(self):
        g = getattr(self.module, 'weight_g')
        v = getattr(self.module, 'weight_v')
        # w = v / self.norm_except_dim_0(v)
        w = g * v / self.norm_except_dim_0(v)
        setattr(self.module, 'weight', w)

    @staticmethod
    def norm_except_dim_0(weight):
        output_size = (weight.size(0),) + (1,) * (weight.dim() - 1)
        out = LA.norm(weight.view(weight.size(0), -1), ord=2, dim=1).view(*output_size)
        return out

    def forward(self, x):
        self.compute_weight()
        return self.module.forward(x)

class MeanOnlyBatchNormLayer(nn.Module):
    def __init__(self, module):
        super(MeanOnlyBatchNormLayer, self).__init__()
        self.module = module

        if isinstance(self.module, WeightNormLayer):
            rootModule = self.module.module
        elif isinstance(self.module, (nn.Conv2d, nn.Linear, NINLayer)):
            rootModule = self.module
        else:
            self.module = None
            raise ValueError('Unsupported module:', module)

        weight = getattr(rootModule, 'weight')
        if getattr(rootModule, 'bias', None) is not None:
            delattr(rootModule, 'bias')
            rootModule.bias = None

        self.register_parameter('bias', nn.Parameter(torch.zeros((weight.size(0),), device=weight.device)))
        self.register_buffer('avg_batch_mean', torch.zeros(size=(weight.size(0),)))

    def forward(self, x):
        activation_prev = self.module.forward(x)
        output_size = (1,) + (activation_prev.size(1),) + (1,) * (activation_prev.dim() - 2)
        if not self.training:
            activation = activation_prev - self.avg_batch_mean.view(*output_size)
        else:
            num_outputs = activation_prev.size(1)
            mu = torch.mean(activation_prev.swapaxes(1, 0).contiguous().view(num_outputs, -1), dim=-1)
            activation = activation_prev - mu.view(*output_size)
            self.avg_batch_mean = 0.9 * self.avg_batch_mean + 0.1 * mu
        if hasattr(self, 'bias'):
            activation += self.bias.view(*output_size)
        return activation


class GaussianNoiseLayer(nn.Module):
    def __init__(self, device, sigma=0.15, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float32).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class NINLayer(nn.Module):
    def __init__(self, input_features, output_features,  activation=None, bias=True):
        super().__init__()
        self.num_units = output_features

        self.register_parameter('weight', nn.Parameter(torch.randn(output_features, input_features)))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(output_features,)))
        else:
            self.register_parameter('bias', None)

        self.apply(self._init_weights)

        self.activation = activation

    def _init_weights(self, module):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, x):
        out_r = torch.tensordot(self.weight, x, dims=([1], [1]))
        remaining_dims = range(2, x.ndim)
        out = out_r.permute(1, 0, *remaining_dims)

        if self.bias is not None:
            remaining_dims_biases = (1,) * (x.ndim - 2)  # broadcast
            b_shuffled = self.bias.view(1, -1, *remaining_dims_biases)
            out = out + b_shuffled
        if self.activation is not None:
            out = self.activation(out)
        return out


def sft_module(data, temperature=1):
    '''
    The implementation of spectral feature transformation module.

    Parameters
    ----------
    data : Symbol
        The input of the module.
    temperature : Symbol
        The temperature of the softmax.

    Returns
    -------
    Symbol
        The result symbol.

    Symbol
        The result similarity matrix.
    '''
    in_sim = mx.symbol.L2Normalization(data=data)
    sim = mx.symbol.dot(in_sim, in_sim, transpose_b=True)

    aff = mx.symbol.softmax(sim, temperature=temperature, axis=1)
    feat = mx.symbol.dot(aff, data)

    return feat, sim



if __name__ == '__main__':
    m = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    w = MeanOnlyBatchNormLayer(WeightNormLayer(m))

