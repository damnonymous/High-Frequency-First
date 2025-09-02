# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils_multy, training_multy, loss_functions_multy, modules
from models import Finer, Siren, Gauss, PEMLP, Wire,Siren_relu,RELU_bias,DinerSiren,DinerMLP,AdaptiveMultiSiren,FR_INR,SCONE
from layers import WeightNormLayer, WeightNormLayerReLU
from torch.utils.data import DataLoader
import configargparse
from functools import partial
from warmup_scheduler import GradualWarmupScheduler
import cv2
import torch
import numpy as np
import math
import torchvision
from PIL import Image
import torch.nn as nn
from utils import compute_high_frequency_ratio
from scipy.ndimage import gaussian_filter



p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--counter_end_times',  type=str, default='mse')
p.add_argument('--lr_warm_up',  type=int, default=0)
p.add_argument('--blur',  type=int, default=0)
p.add_argument('--image_path',  type=str, default=0)
p.add_argument('--first_run',  type=int, default=0)
# p.add_argument('--loss_function',  type=str, default='mse')
p.add_argument("--loss_function", choices=['mse','mse_2stage_mix_image', 'mse_rect','mse_random_select','mse_mix_rect','mse_mix_value_rect','mse_mix_random_select','mse_mix_image_rect','mse_mix_image_random_select','mse_mix_noise_random_select','mse_mix_noise_gradient_select','mse_mix_value_random_select','mse_mix_pattern_random_select',
                                           'mse_mix_patterndivide16_random_select','mse_mix_patterndivide64_random_select','mse_mix_patterndivide256_random_select','mse_mix_patterndivide1024_random_select', 'image_mse_decouple', 'image_mse_high_pass_rgb', 
                                           'image_mse_pretrain','image_mse_pretrain_noise', 'image_mse_add_zero_loss', 'image_mse_reg', 'image_mse_reg_pretrain', 'image_mse_invariant','image_mse_pre_contrastive_loss', 'image_boundary'
                                           ,'mse_blend_image_random_select','mse_blend_cosine_image_random_select','mse_blend_noise_pretrain_random_select','counter', 'mse_reg_motivition','mse_loss_decay','mse_mix_image_gradient_select','image_mse_pretrain_intra_independent','image_mse_pretrain_intra_independent_output',
                                           'mse_mix_cosine_noise_random_select'
                                           ,'mse_frequency_seperate','mse_gradient_selects','image_mse_gradient_loss', 'image_boundary','mse_blend_hybird_random_select'], default='mse', help='choose the loss function')
p.add_argument('--current_time',  type=str)
p.add_argument('--sh_file_path',  type=str)
p.add_argument('--save_fig', type=int, default=0)
p.add_argument('--save_res_fig', type=int, default=0)
p.add_argument('--save_first_layer_fig', type=int, default=0)
p.add_argument('--debug', type=int, default=0)
p.add_argument('--sidelength', type=int, default=256)
p.add_argument('--sidelength1', type=int, default=256)
p.add_argument('--sidelength2', type=int, default=256)
p.add_argument('--sr', type=int, default=0)
p.add_argument("--maxpoints", type=int, default=int(4e10))
p.add_argument('--save_feature', type=int, default=0)
p.add_argument('--resize_image', type=int, default=1)
p.add_argument('--norm_layer_num', type=int, default=-1)


##reg setups
p.add_argument("--reg_filter", choices=['gaussian_blur', 'average_blur', 'median_blur', 'motion_blur',None], default=None, help='method to apply for signal filter')
p.add_argument("--reg_strategy", choices=['global_average', 'patch_average', 'mask_average', 'local_average', 'global_mask_average'], default='', help='method to apply for regularization')
p.add_argument('--reg_epoch_num', type=int, default=200)
p.add_argument('--kernel_size', type=int, default=5)
p.add_argument('--blank_image', type=int, default=0)
p.add_argument('--length', type=int, default=200)
p.add_argument("--weight_decay_method", choices=['linear', 'cosine', 'sine'], default='linear', help='Weight strategy applying on the dynamict constant shift loss (reg loss)')
p.add_argument('--shift_value', type=float, default=0.0)
p.add_argument('--epoch_num_noise', type=int, default=0)


####decouple setups
p.add_argument('--decouple_epoch_num', type=int, default=200)
# p.add_argument("--weight_low", choices=['linear', 'cosine', 'sine', 'constant1', 'constant0'], default='linear', help='Weight strategy applying on the low-frequency loss')
# p.add_argument("--weight_high", choices=['linear', 'cosine', 'sine', 'constant1', 'constant0'], default='linear', help='Weight strategy applying on the high-frequency loss')
p.add_argument("--weight_low", type=float, default=1, help='Weight strategy applying on the low-frequency loss')
p.add_argument("--weight_high", type=float, default=1, help='Weight strategy applying on the high-frequency loss')

####bias_prior
p.add_argument('--bias_prior', type=int, default=0)
p.add_argument('--weight_scale', type=float, default=0)
p.add_argument('--scale_coords', type=float, default=0.0)
p.add_argument('--scale_img', type=float, default=0.0)
p.add_argument('--scale_input_method', choices=['linear', 'polynomial', 'cosine', 'gaussian', 'exponential', 'laplacian', 'None'], default=None, help='Scale method for input')
p.add_argument('--scale_output_method', choices=['linear', 'polynomial', 'cosine', 'gaussian', 'exponential', 'laplacian', 'None'], default=None, help='Scale method for input')
p.add_argument('--input_scale_factor', type=float, default=1.0)
p.add_argument('--input_shift_factor', type=float, default=0.0)
p.add_argument('--scale_output', type=float, default=0.0)
p.add_argument('--bias_prior_add', type=int, default=0)
p.add_argument('--output_scale_factor', type=float, default=1.0)
p.add_argument('--output_shift_factor', type=float, default=0.0)
p.add_argument('--norm_type', choices=['BN', 'LN', 'GN', 'IN', 'BGN', 'IBN'], default=None, help='Norm type for the model')



##pre-training setups
p.add_argument('--pretraining_epoch_num', type=int, default=0)
p.add_argument('--constant_value', type=float, default=0.0)
p.add_argument('--use_res_image', type=int, default = 0)
p.add_argument('--save_resize_gt', type=int, default = 0)
p.add_argument('--save_weight', type=int, default = 0)

###models
p.add_argument('--hidden_layers', type=int, default=3, help='hidden_layers') 
p.add_argument('--hidden_features', type=int, default=256, help='hidden_features') 
p.add_argument('--first_bias_scale', type=float, default=None, help='bias_scale of the first layer')    
p.add_argument('--scale_req_grad', action='store_true')
p.add_argument('--first_omega', type=float, default=30, help='(siren, wire, finer)')    
p.add_argument('--hidden_omega', type=float, default=30, help='(siren, wire, finer)')    
p.add_argument('--scale', type=float, default=30, help='simga (wire, guass)')    
p.add_argument('--N_freqs', type=int, default=10, help='(PEMLP)')
p.add_argument('--out_features', type=int, default=3, help='feature of output')  
p.add_argument('--noise_range', type=float, default=1, help='feature of output')  
p.add_argument('--require_ENL', type=int, default=0, help='feature of output')  
opt = p.parse_args()


rect_lenght = 128; length = opt.length; mix_value = 0.5
print('lenght is:' ,opt.length)


##uniform noise
if opt.sidelength1*opt.sidelength2 > opt.maxpoints:
    uniform_noise = torch.rand((1, opt.maxpoints))*opt.noise_range
else:
    uniform_noise = torch.rand((1, opt.sidelength1*opt.sidelength2))*opt.noise_range
# import pdb;pdb.set_trace()
uniform_noise = uniform_noise.unsqueeze(-1).repeat(1,1,3)
scaled_noise = uniform_noise
orderlist = torch.randperm(uniform_noise.shape[1])


if opt.sr:
    im = Image.open(opt.image_path)   ## load image resolution is 256
    im_resize = im.resize((opt.sidelength*2, opt.sidelength*2))
    im = np.array(im_resize).astype(np.float32)
    gt_sr = torch.tensor(((im / 255.0 ) *2.0 - 1.0)).reshape(1, im.shape[0]*im.shape[1], -1)
else:
    gt_sr = None
    
if 0:
    im = Image.open(opt.image_path)
    im_array = np.array(im)
    im_blurred = np.zeros_like(im_array)
    for i in range(3):
        im_blurred[:, :, i] = gaussian_filter(im_array[:, :, i], sigma=2)
    im_blurred = Image.fromarray(np.uint8(im_blurred))
    output_path = "/home/gpuadmin/han-0129/subsiren/siren/data/data_zs/normal_test/blurred_image.png"
    im_blurred.save(output_path)
    opt.image_path = output_path

if opt.scale==1:
    img_dataset = dataio.ImageFile(opt.image_path)
    coord_dataset = dataio.Implicit2DWrapper_scale(img_dataset, sidelength=opt.sidelength, blur=opt.blur, scale_coords=opt.scale_coords, scale_img=opt.scale_img)


import pdb;pdb.set_trace()
if opt.resize_image==1:
    if opt.batch_size > 1:
        img_dataset = dataio.ImageFolder(opt.image_path)
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=opt.sidelength, blur=opt.blur, scale_coords=opt.scale_coords, scale_img=opt.scale_img)
    else:
        img_dataset = dataio.ImageFile(opt.image_path)    
        # import pdb;pdb.set_trace()
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=opt.sidelength, blur=opt.blur, scale_coords=opt.scale_coords, scale_img=opt.scale_img)
    image_resolution = (opt.sidelength, opt.sidelength)
    # cv2.imwrite('123.jpg',np.array(img_dataset.img_downsampled)[:,:,::-1])
elif opt.resize_image==2:
    img_dataset = dataio.ImageFile_resize_han(opt.image_path, opt.sidelength1, opt.sidelength2) #resize to 512
    # import pdb;pdb.set_trace()
    # cv2.imwrite('123.jpg',np.array(img_dataset.img_downsampled)[:,:,::-1])
    coord_dataset = dataio.Implicit2DWrapper2K_han(img_dataset, sidelength=(opt.sidelength1, opt.sidelength2), scale_coords=opt.scale_coords)
    # import pdb;pdb.set_trace()
    image_resolution = (opt.sidelength2, opt.sidelength1)
elif opt.resize_image == 3:
    img_dataset = dataio.ImageFile_resize_han(opt.image_path, opt.sidelength1, opt.sidelength2,0)
    coord_dataset = dataio.Implicit2DWrapper2K_lu(img_dataset, sidelength=(opt.sidelength1, opt.sidelength2), scale_coords=opt.scale_coords)
    image_resolution = (opt.sidelength2, opt.sidelength1)
else:
    # img_dataset = dataio.ImageFile(opt.image_path) ### add for test
    img_dataset = dataio.ImageFile(opt.image_path,downsample_factor=-1)
    coord_dataset = dataio.Implicit2DWrapper2K(img_dataset, sidelength=opt.sidelength, scale_coords=opt.scale_coords)
    image_resolution = (img_dataset[0].size[1], img_dataset[0].size[0])
    

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# import pdb;pdb.set_trace()
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
        or opt.model_type == 'softplus' or opt.model_type == 'custom_sine':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp',out_features=img_dataset.img_channels, sidelength=image_resolution, norm_type=opt.norm_type, norm_layer_num=opt.norm_layer_num)
    # import pdb;pdb.set_trace()
    if opt.weight_scale:  
        ##weight normlization
        ##other layers
        model.net.net[0][0] = WeightNormLayer(model.net.net[0][0])
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    # model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution)  ##origin 
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, out_features=img_dataset.img_channels, sidelength=image_resolution)  ##revise  
elif opt.model_type == 'inr_finer':
    model = Finer(in_features=2, out_features=img_dataset.img_channels, hidden_layers=opt.hidden_layers, hidden_features=opt.hidden_features,
                    first_omega_0=opt.first_omega, hidden_omega_0=opt.hidden_omega, first_bias_scale=opt.first_bias_scale, scale_req_grad=opt.scale_req_grad, norm_type=opt.norm_type, norm_layer_num=opt.norm_layer_num)

elif opt.model_type == 'inr_siren': 
    if opt.batch_size == 1:
        data = coord_dataset[0][1]['img']
        new_bias = data.mean(dim=0).cuda()
        print('new_bias:', new_bias)
    else:
       new_bias = None 
    # output_file = '/home/gpuadmin/han-0129/subsiren/siren/logs/20img_E500_results_zs/scale/new_bias_normal.txt'
    # with open(output_file, 'a') as f:
    #    f.write(f"{new_bias}\n") 
    
    # Note: out_features=1 for CT images
    model = Siren(in_features=2, out_features=opt.out_features, hidden_layers=opt.hidden_layers, hidden_features=opt.hidden_features,
                    first_omega_0=opt.first_omega, hidden_omega_0=opt.hidden_omega, scale_input_method=opt.scale_input_method, input_scale_factor=opt.input_scale_factor, input_shift_factor=opt.input_shift_factor, scale_output=opt.scale_output, scale_output_method=opt.scale_output_method, output_scale_factor=opt.output_scale_factor, output_shift_factor=opt.output_shift_factor, bias_prior_add=new_bias if opt.bias_prior_add else None, norm_type=opt.norm_type, norm_layer_num=opt.norm_layer_num, resolution=image_resolution) 

    # if opt.bias_prior: 
    #     ####last layer
    #     print("Previous Final Linear Layer Bias:", model.net[4].bias)
    #     model.net[4].bias.data = new_bias  ##default  model.net[4].bias.data = new_bias  
    #     model.net[4].bias.requires_grad = False
    #     print("Updated Final Linear Layer Bias:", model.net[4].bias)
        
        ###other layers 
        # model.net[3].linear.bias.data = torch.zeros_like(model.net[3].linear.bias.data) + new_bias.mean().cpu()  ##default  model.net[4].bias.data = new_bias  model.net[0].linear.bias.shape
        # model.net[3].linear.bias.requires_grad = False
    if opt.weight_scale:  
        ##weight normlization
        ##other layers
        model.net[0].linear = WeightNormLayer(model.net[0].linear)
        
    if 0:
        ##load model checkpoint
        checkpoint_path = '/home/gpuadmin/han-0129/subsiren/siren/logs/20img_E500_results_zs/debug/debug0620_15-35-45mse_noise_test/checkpoints/model_final.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        ##calculate the model total parameter
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')
    
elif opt.model_type == 'inr_wire':
    model = Wire(in_features=2, out_features=opt.out_features, hidden_layers=opt.hidden_layers, hidden_features=opt.hidden_features,
                    first_omega_0=opt.first_omega, hidden_omega_0=opt.hidden_omega, scale=opt.scale, norm_type=opt.norm_type, norm_layer_num=opt.norm_layer_num)

elif opt.model_type == 'inr_gauss':
    model = Gauss(in_features=2, out_features=opt.out_features, hidden_layers=opt.hidden_layers, hidden_features=opt.hidden_features,scale=opt.scale, norm_type=opt.norm_type, norm_layer_num=opt.norm_layer_num)
elif opt.model_type == 'inr_pemlp':
    model = PEMLP(in_features=2, out_features=opt.out_features, hidden_layers=opt.hidden_layers, hidden_features=opt.hidden_features,
                    N_freqs=opt.N_freqs) 
elif opt.model_type == 'inr_siren': 
    if opt.batch_size == 1:
        data = coord_dataset[0][1]['img']
        new_bias = data.mean(dim=0).cuda()
        print('new_bias:', new_bias)
    else:
       new_bias = None 

elif opt.model_type == 'inr_siren_relu': 
    model = Siren_relu(in_features=2, out_features=3, hidden_layers=opt.hidden_layers, hidden_features=opt.hidden_features,
                    first_omega_0=opt.first_omega, hidden_omega_0=opt.hidden_omega, scale_input_method=opt.scale_input_method, input_scale_factor=opt.input_scale_factor, input_shift_factor=opt.input_shift_factor, scale_output=opt.scale_output, scale_output_method=opt.scale_output_method, output_scale_factor=opt.output_scale_factor, output_shift_factor=opt.output_shift_factor, bias_prior_add=new_bias if opt.bias_prior_add else None, norm_type=opt.norm_type, norm_layer_num=opt.norm_layer_num, resolution=image_resolution) 
    if opt.weight_scale:  
        model.net[0].linear = WeightNormLayer(model.net[0].linear)
elif opt.model_type == 'inr_relu_bias': 
    model = RELU_bias(in_features=2, out_features=opt.out_features, hidden_layers=opt.hidden_layers, hidden_features=opt.hidden_features,
                     hidden_omega_0=opt.hidden_omega, scale_input_method=opt.scale_input_method, input_scale_factor=opt.input_scale_factor, input_shift_factor=opt.input_shift_factor, scale_output=opt.scale_output, scale_output_method=opt.scale_output_method, output_scale_factor=opt.output_scale_factor, output_shift_factor=opt.output_shift_factor, bias_prior_add=new_bias if opt.bias_prior_add else None, norm_type=opt.norm_type, norm_layer_num=opt.norm_layer_num, resolution=image_resolution) 
    if opt.weight_scale:  
        model.net[0].linear = WeightNormLayer(model.net[0].linear)
elif opt.model_type == 'inr_diner':
    import pdb;pdb.set_trace()
    model = DinerSiren(hash_table_length=256*256,
                      in_features = 3,
                      hidden_features = 256,
                      hidden_layers = 3,
                      out_features = 3,
                      outermost_linear = True,
                      first_omega_0 = 30,
                      hidden_omega_0 = 30).cuda()
    # model = DinerMLP(hash_table_length = 256*256,
    #                   in_features = 2,
    #                   hidden_features = opt.hidden_features,
    #                   hidden_layers = opt.hidden_layers,
    #                   out_features = 3,).cuda()
elif opt.model_type == 'inr_miner':
    # model = DinerSiren(hash_table_length = 256*256,
    #                   in_features = 2,
    #                   hidden_features = 64,
    #                   hidden_layers = 2,
    #                   out_features = 3,
    #                   outermost_linear = True,
    #                   first_omega_0 = 30,
    #                   hidden_omega_0 = 30).cuda()
    model = AdaptiveMultiSiren(
                      in_features = 2,
                      hidden_features =opt.hidden_features,
                      hidden_layers = opt.hidden_layers,
                      out_features = 3,
                      n_channels = int(256*256/32.0)).cuda()
elif opt.model_type == 'inr_FR':
    # model = DinerSiren(hash_table_length = 256*256,
    #                   in_features = 2,
    #                   hidden_features = 64,
    #                   hidden_layers = 2,
    #                   out_features = 3,
    #                   outermost_linear = True,
    #                   first_omega_0 = 30,
    #                   hidden_omega_0 = 30).cuda()
    model=FR_INR(mode='sin+fr',in_features=2,
        hidden_features=256,
        hidden_layers=3,
        out_features=3,
        outermost_linear=True,
        high_freq_num=128,
        low_freq_num=128,
        phi_num=32,
        alpha=0.05, # for relu, alpha:0.05; for sin, alpha:0.01
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        pe=False   
        )

elif opt.model_type == 'inr_SCONE':
    model=SCONE (2,3,scone_configs=None).cuda()



else:
    raise NotImplementedError
# import pdb;pdb.set_trace()


model.cuda()


if opt.bias_prior:
    data = coord_dataset[0][1]['img']
    new_bias = data.mean(dim=0).cuda()
    if opt.model_type == 'inr_finer' or opt.model_type == 'inr_wire' or opt.model_type == 'inr_siren':
        print("Previous Final Linear Layer Bias:", model.net[4].bias.data)
        model.net[4].bias.data = new_bias
        model.net[4].bias.requires_grad = False
        print("Updated Final Linear Layer Bias:", model.net[4].bias.data)
    else:
        print("Previous Final Linear Layer Bias:", model.net.net[4][0].bias.data)
        model.net.net[4][0].bias.data = new_bias
        model.net.net[4][0].bias.requires_grad = False
        print("Updated Final Linear Layer Bias:", model.net.net[4][0].bias.data)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

if 'mix_image' in opt.loss_function or 'blend' or 'cosine' in opt.loss_function:
    # folder_path = './data/high_test/'   #1 image
    folder_path = './data/sub_bsds/'  ##480 images
    # folder_path = './data/multi_value_2image/'
    # folder_path = '/home/gpuadmin/han-0129/subsiren/siren/data_han/complex_image/'
    files = os.listdir(folder_path)
    image_list = []
    for file in files:
        image_path = os.path.join(folder_path, file)
        image_mixed =  cv2.imread (image_path)
        image_mixed = cv2.cvtColor(image_mixed, cv2.COLOR_BGR2RGB)# to change to rgb
        image_mixed = cv2.resize(image_mixed,(256,256))
        image_mixed_norm = ((image_mixed/255.0)*2.0-1)
        image_list.append(image_mixed_norm)
    mix_image = torch.from_numpy(np.array(image_list)).cuda().float()

    if opt.blank_image == 1:
        mix_image_blank = torch.zeros_like(mix_image) + 1.0
        mix_image = mix_image_blank[0].unsqueeze(0)
        # import pdb;pdb.set_trace()
    if 'cosine' in  opt.loss_function:

        x_values = np.linspace(0.5*np.pi , np.pi, int(256*256/length)+2)
        y_values = np.cos(x_values)+1.0
        y_values_scaled = (y_values / np.sum(y_values) * (256.0 * 256.0))
        cosine_lenght = y_values_scaled
        blend_weight = (y_values_scaled - np.min(y_values_scaled)) / (np.max(y_values_scaled) - np.min(y_values_scaled))
        blend_weight = blend_weight ** 0.5
        if 1:
            # import pdb;pdb.set_trace()
            data = next(iter(dataloader))
            image = data[1]['img'] 
            image_tensor = ((image+1.0)/2.0)#.cpu().numpy()
            grad_x = torch.zeros_like(image)
            grad_y = torch.zeros_like(image)

            # Compute horizontal gradient
            grad_x[:, :-1] = image_tensor[:, 1:] - image_tensor[:, :-1]

            # Compute vertical gradient
            grad_y[:-1, :] = image_tensor[1:, :] - image_tensor[:-1, :]

            # Compute gradient magnitude
            magnitude = torch.sqrt(grad_x**2 + grad_y**2)

            # Calculate the average gradient magnitude
            aaa = magnitude.mean().item()
            # import pdb;pdb.set_trace()
            if aaa > 0.15:
                # import pdb;pdb.set_trace()
                opt.loss_function =  'mse'
            
        # import pdb;pdb.set_trace()
elif 'mix_pattern'in opt.loss_function:
    mix_image = np.zeros((1, 256,256))
    for i in range(256):
        if i % 2 == 0:  
            mix_image[0, i, :] = np.where(np.indices((1, 256))[1] % 2 == 0, -1.0, 1.0)
        else:  
            mix_image[0, i, :] = np.where(np.indices((1, 256))[1] % 2 == 0, 1.0, -1.0)
    mix_image = np.repeat(mix_image[:,:,:,None], 3, axis=3)
    mix_image = torch.from_numpy(mix_image).cuda().float()
    
    if 'divide2'in opt.loss_function:
        mix_image[:,0:128,:,:]= -1
        mix_image[:,128:257,:,:]= 1
    else:
        start_index = opt.loss_function.find('divide') + len('divide')
        end_index = opt.loss_function.find('_', start_index)
        number_str = opt.loss_function[start_index:end_index]
        number = int(number_str)
        root = math.sqrt(number)
        mix_image= utils_multy.pattern_divide(mix_image, root)

    
if opt.loss_function=='counter':
    loss_fn = partial(loss_functions_multy.image_mse_counter, length)
elif opt.loss_function=='descending_order':
    loss_fn = partial(loss_functions_multy.image_descending_order, None)
elif opt.loss_function=='mse': 
    loss_fn = partial(loss_functions_multy.image_mse)
elif opt.loss_function=='mse_2stage_mix_image': 
    loss_fn = partial(loss_functions_multy.image_mse_two_stage, mix_image)
elif opt.loss_function=='mse_rect': 
    loss_fn = partial(loss_functions_multy.image_mse_rect, rect_lenght)
elif opt.loss_function=='mse_mix_rect': 
    loss_fn = partial(loss_functions_multy.image_mse_mix_rect, rect_lenght)
elif opt.loss_function=='mse_mix_value_rect': 
    loss_fn = partial(loss_functions_multy.image_mse_mix_value_rect, rect_lenght,mix_value)
elif opt.loss_function=='mse_mix_image_rect': 
    loss_fn = partial(loss_functions_multy.image_mse_mix_image_rect, rect_lenght,mix_image)
elif opt.loss_function=='mse_random_select': 
    loss_fn = partial(loss_functions_multy.image_mse_random_select, length)
elif opt.loss_function=='mse_mix_random_select': 
    loss_fn = partial(loss_functions_multy.image_mse_mix_random_select, length)
elif opt.loss_function=='mse_mix_image_random_select': 
    loss_fn = partial(loss_functions_multy.image_mse_mix_image_random_select, length, mix_image)
elif opt.loss_function=='mse_blend_image_random_select': 
    # import pdb;pdb.set_trace()
    loss_fn = partial(loss_functions_multy.mse_blend_image_random_select, length, mix_image)
elif opt.loss_function=='mse_blend_noise_random_select': 
    loss_fn = partial(loss_functions_multy.mse_blend_image_random_select, length, None)
elif opt.loss_function=='mse_blend_hybird_random_select': 
    loss_fn = partial(loss_functions_multy.mse_blend_hybird_random_select, length, None)
elif opt.loss_function=='mse_blend_noise_pretrain_random_select': 
    linear_epoch = 320
    loss_fn = partial(loss_functions_multy.mse_blend_noise_pretrain_random_select, length, linear_epoch, None)
elif opt.loss_function=='mse_blend_cosine_image_random_select': 
    loss_fn = partial(loss_functions_multy.mse_blend_cosine_image_random_select, length, blend_weight, mix_image)
elif opt.loss_function=='mse_mix_cosine_noise_random_select': 
    loss_fn = partial(loss_functions_multy.image_mse_mix_cosine_noise_random_select, length, blend_weight, mix_image)  
elif opt.loss_function=='mse_mix_noise_random_select': 
    loss_fn = partial(loss_functions_multy.image_mse_mix_noise_random_select, length, None)
elif opt.loss_function=='image_mse_pretrain_intra_independent': 
    # import pdb;pdb.set_trace()
    loss_fn = partial(loss_functions_multy.image_mse_pretrain_intra_independent,image_resolution, opt.pretraining_epoch_num)
elif opt.loss_function=='image_mse_pretrain_intra_independent_output': 
    loss_fn = partial(loss_functions_multy.image_mse_pretrain_intra_independent_output,opt.pretraining_epoch_num)
elif opt.loss_function=='mse_mix_value_random_select': 
    loss_fn = partial(loss_functions_multy.image_mse_mix_value_random_select, length, mix_value)
elif opt.loss_function=='mse_mix_pattern_random_select' or 'mse_mix_patterndivide' in opt.loss_function: 
    loss_fn = partial(loss_functions_multy.image_mse_mix_image_random_select, length, mix_image)
elif opt.loss_function=='image_mse_decouple': 
    # weight_low = loss_functions_multy.get_weight(opt.weight_low, opt.decouple_epoch_num)
    # weight_high = loss_functions_multy.get_weight(opt.weight_high, opt.decouple_epoch_num)
    weight_low = opt.weight_low; weight_high = opt.weight_high
    loss_fn = partial(loss_functions_multy.image_mse_decouple, opt.decouple_epoch_num, img_dataset[0].size[0], img_dataset[0].size[1], weight_low, weight_high)
elif opt.loss_function=='image_mse_reg': 
    gt_rg = nn.Parameter(torch.zeros(size=(uniform_noise.shape[0], uniform_noise.shape[1], 3)), requires_grad=True).cuda()
    basename = os.path.basename(opt.image_path).split('.')[0]
    loss_fn = partial(loss_functions_multy.image_mse_reg, None, opt.reg_filter, opt.reg_epoch_num, opt.kernel_size, opt.reg_strategy, basename)
elif opt.loss_function=='image_mse_reg_pretrain': 
    basename = os.path.basename(opt.image_path).split('.')[0]
    loss_fn = partial(loss_functions_multy.image_mse_reg_pretrain, None, opt.weight_decay_method, opt.reg_epoch_num, opt.pretraining_epoch_num, opt.kernel_size, opt.reg_strategy, basename)
elif opt.loss_function=='image_mse_invariant': 
    loss_fn = partial(loss_functions_multy.image_mse_invariant, rect_lenght,mix_image)
elif opt.loss_function=='image_mse_pretrain': 
    loss_fn = partial(loss_functions_multy.image_mse_pretrain, opt.pretraining_epoch_num, opt.constant_value) 
elif opt.loss_function=='image_mse_pretrain_noise': 
    loss_fn = partial(loss_functions_multy.image_mse_pretrain_noise) 
elif opt.loss_function=='image_mse_add_zero_loss': 
    loss_fn = partial(loss_functions_multy.image_mse_add_zero_loss) 
elif opt.loss_function=='image_mse_pre_contrastive_loss': 
    uniform_noise = torch.rand((1, opt.sidelength1*opt.sidelength2))*opt.noise_range
    # import pdb;pdb.set_trace()
    uniform_noise = uniform_noise.unsqueeze(-1).repeat(1,1,3)
    mix_image = uniform_noise
    loss_fn = partial(loss_functions_multy.image_mse_pre_contrastive_loss,mix_image)     
elif opt.loss_function=='image_boundary': 
    loss_fn = partial(loss_functions_multy.image_boundary)     
elif opt.loss_function=='mse_reg_motivition': 
    scaled_noise_image_norm = None
    loss_fn = partial(loss_functions_multy.image_mse_test, opt.shift_value, scaled_noise_image_norm, opt.epoch_num_noise)
elif opt.loss_function=='mse_loss_decay': 
    loss_fn = partial(loss_functions_multy.image_mse_loss_decay)
elif 'gradient_' in opt.loss_function: 
    # data = next(iter(dataloader))
    # image = data[1]['img'] 
    # image = ((image+1.0)/2.0).cpu().numpy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#.astype('uint8')
    # grad_x = cv2.Sobel(image,cv2.CV_64F, 1, 0, ksize = 5)
    # sobelx = cv2.convertScaleAbs(grad_x)
    # grad_y = cv2.Sobel(image,cv2.CV_64F, 0, 1, ksize = 5)
    # sobely = cv2.convertScaleAbs(grad_y)
    # # grad = np.sqrt(sobely **2 + sobelx **2)
    # # import pdb;pdb.set_trace()
    # grad1 = sobely*0.5 + sobelx*0.5
    
    
    # grad = torch.from_numpy(grad1).cuda()
    # grad_list = torch.argsort(grad.view(-1),descending=True) # max to min
    # grad_map = torch.from_numpy(cv2.normalize(grad1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)).cuda()

    # pixel_count = np.zeros((256, 256), dtype=np.uint32)
    # # import pdb;pdb.set_trace() 
    # image = (image * 255).astype(np.uint8)
    # for i in range(256):
    #     for j in range(256):
    #         pixel_count[i, j] = np.count_nonzero(image == image[i, j])    
    # max_value = np.max(pixel_count)
    # min_value = np.min(pixel_count)
    # normalized_pixel_count = ((pixel_count - min_value) / (max_value - min_value))
    # normalized_pixel_count = torch.from_numpy(normalized_pixel_count).cuda()
    # cv2.imwrite('grad_map.jpg',255*normalized_pixel_count.detach().cpu().numpy())
    # import pdb;pdb.set_trace()    
    # hist = cv2.calcHist([image*255], [0], None, [256], [0,255])
    # import matplotlib.pyplot as plt
    # plt.plot(hist, color='b')
    # plt.savefig('HAN2.png', bbox_inches='tight', pad_inches = 0.0)
    if opt.loss_function == 'mse_mix_noise_gradient_select': 
        loss_fn = partial(loss_functions_multy.image_mse_mix_noise_gradient_select, length, grad_list)
    if opt.loss_function == 'mse_mix_image_gradient_select': 
        loss_fn = partial(loss_functions_multy.image_mse_mix_image_gradient_select, length, grad_list, mix_image)
    if opt.loss_function == 'mse_gradient_selects': 
        loss_fn = partial(loss_functions_multy.image_mse_gradient_select, length, grad_list, mix_image)
    if opt.loss_function == 'image_mse_gradient_loss': 
        loss_fn = partial(loss_functions_multy.image_mse_gradient_loss, opt.out_features, image_resolution, None)
elif opt.loss_function=='mse_frequency_seperate': 
    loss_fn = partial(loss_functions_multy.image_mse_frequency_seperate)
# elif opt.loss_function=='mse_mix_value_random_select': 
#     loss_fn = partial(loss_functions_multy.image_mse_mix_value_random_select, length, mix_image)



summary_fn = partial(utils_multy.write_image_summary, image_resolution, opt.scale_img, scale_output=opt.scale_output,require_ENL=opt.require_ENL)
if not opt.resize_image:
   opt.sidelength = image_resolution[0]
#    if img_dataset[0].size[0]*img_dataset[0].size[1] > opt.maxpoints:
#         opt.maxpoints = min(img_dataset[0].size[0]*img_dataset[0].size[1], opt.maxpoints) 
   
training_multy.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn,loss_function=opt.loss_function,opt=opt,sr=opt.sr, gt_sr=gt_sr, scaled_noise=scaled_noise,orderlist=orderlist)
