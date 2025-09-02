'''Implements a generic training loop.
'''
# from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from torch.optim.lr_scheduler import ExponentialLR,LambdaLR,LinearLR
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from warmup_scheduler import GradualWarmupScheduler
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from datetime import datetime
import pytz
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import utils_multy
import utils
from functools import partial
from scipy.fft import fft2, fftshift
import wandb


def interpolate_coords(H, W,  coords):
    original_coords = coords.reshape(H, W, coords.shape[1])
    original_coords = original_coords.unsqueeze(0)
    target_grid = torch.meshgrid(torch.linspace(-1, 1, H*2), torch.linspace(-1, 1, W*2))
    target_grid = torch.stack(target_grid, dim=-1)
    target_grid = target_grid.unsqueeze(0).cuda()
    interpolated_coords = F.grid_sample(original_coords.permute(0,3,1,2), target_grid, mode='bilinear', align_corners=True)
    interpolated_coords = interpolated_coords.squeeze(0).permute(1,2,0)
    interpolated_coords = interpolated_coords[:,:, [1, 0]]
    interpolated_coords = interpolated_coords.reshape(-1, interpolated_coords.shape[2])
    return interpolated_coords



def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, loss_function=None,opt=None,sr=0, gt_sr=None, scaled_noise= None, orderlist=None):
    image_num = len(os.listdir(os.path.join(os.path.dirname(opt.image_path) )))

    uniform_noise_3 = torch.rand((1, opt.sidelength1*opt.sidelength2,3))#*0.9
    scaled_noise_3 = uniform_noise_3
    psnr_bset = 0
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    # optim = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0)  # SSGD with momentum set to 0
    if opt.lr_warm_up==1:
        scheduler = CosineAnnealingWarmupRestarts(optim,
                                        first_cycle_steps=int(500),
                                        cycle_mult=1.0,
                                        max_lr=lr,
                                        min_lr=1e-4,
                                        gamma=1.0)# warmup_steps=0+2500*args.warmup,
        # optim = torch.optim.Adam(model.parameters(), lr=1)
        # scheduler = LambdaLR(optim, lr_lambda=lambda step: 1e-7 + (1e-4 - 1e-7) * step / 500)
        # scheduler_steplr = StepLR(optim, step_size=300, gamma=0.8)
        # scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=300, after_scheduler=scheduler_steplr)
    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs

    # now_utc = datetime.now(pytz.utc)
    # kst = pytz.timezone('Asia/Seoul')
    # current_time = now_utc.astimezone(kst).strftime("%m%d"); 
    # directory_name = os.path.dirname(opt.image_path).split('/')[-1]
    
    if opt.debug ==1:
        model_dir = os.path.join(model_dir+'debug', 'debug')
    model_dir = model_dir + opt.current_time+opt.loss_function+'_'+ os.path.dirname(opt.image_path).split('/')[-1]
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    summaries_dir = os.path.join(model_dir, 'summaries')
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')
    if opt.first_run == 1:
        if os.path.exists(model_dir):
            val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
            if val == 'y':
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        utils.cond_mkdir(summaries_dir)
        utils.cond_mkdir(checkpoints_dir)
        if opt.save_res_fig ==1 :
            res_fig_dir = os.path.join(model_dir, 'res_fig_dir')
            utils.cond_mkdir(res_fig_dir)
        #save code
        os.makedirs( os.path.join(model_dir,'code'))
        source_file = os.path.abspath(__file__)
        sh_file_path = opt.sh_file_path
        sh_file_name = os.path.basename(sh_file_path)
        basename = os.path.basename(__file__)
        destination_file = os.path.join(model_dir, 'code', os.path.basename(__file__))
        destination_sh_file = os.path.join(model_dir, 'code', os.path.basename(sh_file_name))
        destination_loss_file = os.path.join(model_dir, 'code','loss_functions_multy.py')
        destination_train_file = os.path.join(model_dir, 'code','train_img_multy.py')
        destination_models_file = os.path.join(model_dir, 'code','models.py')
        shutil.copy2(sh_file_path, destination_sh_file)
        shutil.copy2(source_file, destination_file)
        shutil.copy2(os.path.join(os.path.dirname(__file__),'loss_functions_multy.py'), destination_loss_file)
        shutil.copy2(os.path.join(os.path.dirname(__file__),'experiment_scripts','train_img_multy.py'), destination_train_file)
        shutil.copy2(os.path.join(os.path.dirname(__file__),'models.py'), destination_models_file)
    total_steps = 0
    final_gt=0
    final_output=0
    
    
    orderlist = orderlist[torch.randperm(orderlist.size(0))]
    image_name = os.path.basename(opt.image_path)[:-4]
    results_save_path =os.path.join(checkpoints_dir,image_name)
    utils.cond_mkdir(results_save_path)
    
    # #### test the learning rate increasing
    # initial_lr = 1e-4  # 初始学习率
    # final_lr = 1e-4    # 最终目标学习率
    # # epochs = 100       # epochs数量
    # optim = torch.optim.Adam(lr=initial_lr, params=model.parameters())
    
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        # wandb.init(project="inr-test")
        for epoch in range(epochs):
            #b_indices = indices[b_idx:min(H*W*T, b_idx+maxpoints)]
        # #### test the learning rate increasing
        #     with open('weight100_parameter_gradients.txt', 'w+') as file:
        #         # 打印每个参数的梯度并写入文件
        #         for name, param in model.named_parameters():
        #             if param.grad is not None:
        #                 grad_info = f'Epoch {epoch}, Parameter: {name}, Gradient: {param.grad}\n'
        #                 print(grad_info)
        #                 file.write(grad_info)
            
        #     lr = initial_lr + (final_lr - initial_lr) * (epoch / epochs)
        #     for param_group in optim.param_groups:
        #         param_group['lr'] = lr
                
            # if not epoch % epochs_til_checkpoint and epoch:
            #     # torch.save(model.state_dict(),
            #     #            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
            #                np.array(train_losses))
            
            # for name, param in model.named_parameters():
            #     wandb.log({f"{name}": wandb.Histogram(param.detach().cpu().numpy()), "epoch": epoch})  
            for step, (model_input_origin, gt_origin) in enumerate(train_dataloader):
                gt = gt_origin
                model_input = model_input_origin
                pixel_number_total = gt['img'].shape[0]*gt['img'].shape[1]
                # import pdb;pdb.set_trace()
                indices = torch.randperm(pixel_number_total)
                im_estim = torch.zeros_like(gt['img']).cuda()

                if epoch == 1:

                    # orthogonality_v = orthogonality(model)
                    # print (orthogonality_v)
                    if opt.save_resize_gt==1:
                        imten_han = gt['img'].reshape(256,256,-1).cpu().numpy()
                        imten_han = imten_han [:,:,::-1]
                        cv2.imwrite(os.path.join('/home/gpuadmin/han-0129/subsiren/siren/data/Lu/gt_img',image_name+str(epoch))+'.jpg',255.0*(imten_han+1.0)/2.0)
                    if opt.save_weight==1:
                        # weights_grad = model.net.net[0][0].weight.grad.cpu().numpy()  ##relu
                        weights_grad = model.net[0].linear.weight.grad.cpu().numpy() ##siren
                        weights_grad = model.net[0].linear.module.weight_v.grad.cpu().numpy()  ##weight norm
                        torch.save(weights_grad, '/home/gpuadmin/han-0129/subsiren/siren/logs/20img_E500_results_zs/norm/first_layer_weight_grad_wn.pth')
                        
                start_time = time.time()
                

                
                for b_idx in range(0,pixel_number_total, opt.maxpoints):
                    
                    if pixel_number_total <= opt.maxpoints:
                        model_input = {key: value.cuda() for key, value in model_input.items()}
                        gt = {key: value.cuda() for key, value in gt.items()} 
                    else:
                        model_input = {}
                        gt = {}
                        b_indices = indices[b_idx:min(pixel_number_total, b_idx+opt.maxpoints)]
                        model_input_temp = model_input_origin['coords'].clone()
                        model_input_index_temp = model_input_origin['idx'].clone()
                        model_input['idx'] = model_input_index_temp
                        model_input['coords'] = model_input_temp[:,b_indices, ...]
                        b_indices = b_indices.cuda()
                        gt_temp = gt_origin['img'].clone()
                        gt['img'] = gt_temp[:,b_indices, ...]
                        
                        model_input = {key: value.cuda() for key, value in model_input.items()}
                        gt = {key: value.cuda() for key, value in gt.items()}
                        
                        del model_input_temp, model_input_index_temp, gt_temp  


                    if sr:
                        H = W = int(torch.sqrt(torch.tensor(model_input['coords'].shape[1])))
                        coords_sr = interpolate_coords(H, W, model_input['coords'].squeeze(0))
                        model_input_sr = model_input.copy()
                        model_input_sr['coords'] = coords_sr.unsqueeze(0)
                        model_output_sr = model(model_input_sr)

                    if double_precision:
                        model_input = {key: value.double() for key, value in model_input.items()}
                        gt = {key: value.double() for key, value in gt.items()}

                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            model_output = model(model_input)
                            losses = loss_fn(model_output, gt)
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean() 
                            train_loss.backward()
                            return train_loss
                        optim.step(closure)
                        
                    if 'inr_miner' in opt.model_type:
                        model_output_list = {} 
                        # import pdb;pdb.set_trace()
                        model_output = model(model_input['coords'],model_input['coords'])
                        model_output_list['model_in'] = model_input['coords']
                        model_output_list['model_out'] = model_output  
                        model_output = model_output_list
                    elif 'inr' in opt.model_type:
                        model_output_list = {} 
                        # import pdb;pdb.set_trace()
                        model_output = model(model_input['coords'])
                        model_output_list['model_in'] = model_input['coords']
                        model_output_list['model_out'] = model_output  
                        model_output = model_output_list
                    else:
                        model_output = model(model_input)
                    # import pdb;pdb.set_trace()
                    if pixel_number_total > opt.maxpoints:
                        with torch.no_grad():
                            pixelvalues = model_output['model_out'].clone()
                            # import pdb;pdb.set_trace()
                            im_estim[:,b_indices, :] = pixelvalues
                    # if 1:
                    #     if epoch == 499:
                    #         ft_maps = model.net.net[0](model_input['coords'])
                    #         for han in range(ft_maps.shape[2]):
                    #             ft_maps_c = ft_maps[:,:,han].reshape(256,256).detach().cpu().numpy()
                    #             plt.imshow(ft_maps_c, interpolation='nearest')  ##cmap='viridis'
                    #             plt.axis('off')
                    #             plt.savefig('/home/gpuadmin/han-0129/subsiren/siren/log_han/'+str(han)+ '.jpg', bbox_inches='tight', pad_inches = 0.0)
                    #             plt.close()
                    if opt.save_feature:
                        if epoch == 499:
                            ft_maps = model.net[0](model_input['coords'])
                            save_path_feature = os.path.join(results_save_path,'feature_map/')
                            os.makedirs(save_path_feature, exist_ok=True)
                            for ft_number in range(ft_maps.shape[2]):
                                ft_maps_c = ft_maps[:,:,ft_number].reshape(256,256).detach().cpu().numpy()
                                plt.imshow(ft_maps_c, interpolation='nearest')  ##cmap='viridis'
                                plt.axis('off')
                                plt.savefig(save_path_feature + str(ft_number)+ '.jpg', bbox_inches='tight', pad_inches = 0.0)
                                plt.close()
                                
                    if loss_function == 'counter':
                        losses = loss_fn(model_output, gt, epoch, scaled_noise, orderlist)
                    elif loss_function == 'image_mse_decouple':
                        losses = loss_fn(model_output, gt, epoch)
                    elif loss_function == 'image_mse_high_pass_rgb':
                        losses = loss_fn(model_output, gt, epoch)
                    elif loss_function == 'image_mse_pretrain':
                        losses = loss_fn(model_output, gt, epoch)
                    elif loss_function == 'image_mse_pretrain_noise':
                        # import pdb;pdb.set_trace()
                        losses = loss_fn(model_output, gt, epoch, scaled_noise)#
                    elif loss_function == 'image_mse_pretrain_intra_independent':
                        if epoch == opt.pretraining_epoch_num - 1 :
                            torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'preatrain_'+ str(opt.pretraining_epoch_num)+'.pth'))
                            # orthogonality_v = orthogonality(model)
                            # print (orthogonality_v)
                        losses = loss_fn(model_output, gt, epoch,scaled_noise)#
                    elif loss_function == 'image_mse_pretrain_intra_independent_output':
                        if epoch == opt.pretraining_epoch_num - 1 :
                            torch.save(model.state_dict(),os.path.join(checkpoints_dir, 'preatrain_'+ str(opt.pretraining_epoch_num)+'.pth'))
                        # import pdb;pdb.set_trace()
                        if epoch == 0 :
                            first_output = model_output['model_out']
                        losses = loss_fn(model_output, gt, epoch,scaled_noise,first_output)#
                    elif loss_function == 'image_mse_add_zero_loss':
                        losses = loss_fn(model_output, gt, epoch)
                    elif loss_function == 'image_mse_reg':
                        losses = loss_fn(model_output, gt, epoch)
                    elif loss_function == 'image_mse_reg_pretrain':
                        losses = loss_fn(model_output, gt, epoch)
                    elif loss_function == 'image_boundary':
                        losses = loss_fn(model_output, gt, epoch)
                    elif loss_function == 'image_mse_gradient_loss':
                        # import pdb;pdb.set_trace()
                        gradient_param = 1
                        losses = loss_fn(model_output, gt, epoch, scaled_noise, gradient_param)
                    elif 'mse' in loss_function:
                        # import pdb;pdb.set_trace()
                        # gt['sidelength'] = [opt.sidelength1, opt.sidelength2] #For HiRes Images
                        losses = loss_fn(model_output, gt, epoch, scaled_noise, orderlist)
                    elif loss_function == 'descending_order':
                        losses = loss_fn(model_output, gt, orderlist, epoch,length=150, scaled_noise = scaled_noise)
  
                        
                    final_output = model_output #hds change
                    final_gt = gt #hds change
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        if loss_schedules is not None and loss_name in loss_schedules:
                            single_loss *= loss_schedules[loss_name](total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())

                    
                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                        optim.step()

                        if opt.lr_warm_up==1 and epoch <=600:
                            scheduler.step()
                        print('lr is:',optim.param_groups[0]['lr'] )
                        
                if not total_steps % steps_til_summary:
                    model_input = {key: value.cuda() for key, value in model_input_origin.items()}
                    gt = {key: value.cuda() for key, value in gt_origin.items()}  
                    if pixel_number_total > opt.maxpoints:        
                        if 'inr' in opt.model_type:
                            model_output_list = {} 
                            # model_output = model(model_input['coords'])
                            model_output = im_estim
                            model_output_list['model_in'] = model_input['coords']
                            model_output_list['model_out'] = model_output
                            model_output = model_output_list
                        else:
                            model_output['model_out'] = im_estim
                    if sr:
                        gt_im_sr = {}
                        gt_im_sr['img'] = gt_sr
                        summary_fn_sr = partial(utils_multy.write_image_summary, (H*2,W*2))
                        summary_fn_sr(model, model_input_sr, gt_im_sr, model_output_sr, None, total_steps,txt_save_path= results_save_path, epoch=epoch)
                    else:
                        # import pdb;pdb.set_trace()
                        summary_fn(model, model_input, gt, model_output, None, total_steps,txt_save_path= results_save_path, epoch=epoch)
                        if opt.psnr_choose_best ==1:
                            psnr_bset= summary_fn(model, model_input, gt, model_output, None, total_steps,txt_save_path= results_save_path, epoch=epoch,psnr_bset=psnr_bset)

                            
                            
                    if opt.save_fig:
                        # import pdb;pdb.set_trace()
                        image = ((model_output['model_out'].reshape(1,opt.sidelength1,int(model_output['model_out'].shape[1]/opt.sidelength1),-1) +1 )/2.0).clone().detach().cpu().numpy()
                        image = cv2.cvtColor(image[0]*255,cv2.COLOR_BGR2RGB)
                        cv2.imwrite(os.path.join(results_save_path,str(epoch))+'.jpg', image)
                        
                        if opt.save_res_fig == 1:
                            # import pdb;pdb.set_trace()
                            res_map = (model_output['model_out']+ 1.0) /2.0 - (gt['img']+ 1.0) /2.0
                            # import pdb;pdb.set_trace()
        
                            # res_map = (res_map + 1.0) /2.0 

                            #res_map_jpg = ((model_output['model_out']-gt['img'])+1.0) / 2.0
                            res_map_jpg = res_map.reshape(opt.sidelength,opt.sidelength,-1)
                            res_map_jpg= res_map_jpg.detach()
                            # import pdb;pdb.set_trace()
                            # torch.save(model_output['model_out']-gt['img'],os.path.join(results_save_path,str(epoch))+'_res_.pth')
                            cv2.imwrite(os.path.join(results_save_path,str(epoch))+'_res.jpg', 255.0*res_map_jpg.cpu().numpy())
                            res_map = res_map.reshape(opt.sidelength,opt.sidelength,-1).mean(-1)
                            res_map_abs = abs(res_map)
                            res_map_abs = res_map_abs.detach().cpu().numpy()

                            # import pdb;pdb.set_trace()
                            # 显示红蓝误差图
                            
                            plt.imshow(res_map_abs, cmap='jet', interpolation='nearest')  ##cmap='viridis'
                            plt.axis('off')
                            # plt.colorbar()  # 添加颜色条
                            # plt.savefig('heatmap_circle.png')
                            plt.savefig(os.path.join(results_save_path,str(epoch))+ '_jet_res.jpg', bbox_inches='tight', pad_inches = 0.0)
                            plt.close()
                            # if epoch == 99:
                            #     learned =  model_output['model_out']
                            #     gt_torch = gt['img']
                            #     import pdb;pdb.set_trace()
                            #     torch.save(learned,os.path.join(model_dir)+'/learned100.pth')      
                            #     torch.save(gt_torch,os.path.join(model_dir)+'/gt_torch.pth')              
                            # plt.imshow(res_map,cmap='bwr')
                            # plt.colorbar()
                            # plt.show()
                            # # # 使用imshow函数可视化误差地图，选择颜色映射为反转的红蓝颜色映射
                            # # plt.imshow(res_map_clipped, cmap='RdBu_r')
                            # # plt.colorbar()
                            # # plt.show()
                            # plt.savefig('error_map.png')
                            # import pdb;pdb.set_trace()
                                
                # import pdb;pdb.set_trace()
                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    # import pdb;pdb.set_trace()
                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            # import pdb;pdb.set_trace()
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            #writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1
        # import pdb;pdb.set_trace()
        # if epoch < 201:
            # model_output['model_out'] = model_output['model_out'] * ((epoch / 201) )
                
        final_image=(opt.first_run==image_num)
        if sr:
            summary_fn_sr(model, model_input_sr, gt_im_sr, model_output_sr, None, total_steps,txt_save_path= results_save_path, epoch=epoch, final_epoch=True, final_image=final_image)
        else:
            # import pdb;pdb.set_trace()
            # model_output['model_out'] = model_output['model_out'] / 0.75 ### handongshen 11.27
            summary_fn(model, model_input, gt, model_output, None, total_steps,txt_save_path= results_save_path, epoch=epoch,final_epoch=True,final_image=final_image,use_res_image = opt.use_res_image)
            if opt.save_first_layer_fig == 1:
                first_layer_output = model.net.net[0](model_input['coords'])
                first_layer_output = first_layer_output.squeeze(0).permute(1,0).reshape(256,256,256)
                batch_size = first_layer_output.shape[0]
                fig = plt.figure()
                for i in range(batch_size):
                    sample_output = first_layer_output[i].detach().cpu().numpy()
                    f_transform = np.fft.fft2(sample_output)
                    f_transform_shifted = np.fft.fftshift(f_transform)
                    magnitude_spectrum = np.abs(20 * np.log(np.abs(f_transform_shifted)))
                    max_magnitude = np.max(magnitude_spectrum)
                    average_magnitude = np.mean(magnitude_spectrum)
                    # hist, bins = np.histogram(sample_output.ravel(), bins=256, range=(0, 255))
                    # max_freq = np.max(hist)
                    plt.imshow(sample_output)
                    plt.axis('off')
                    save_path = f'{results_save_path}/first_layer_output_{max_magnitude}_{i}.png'
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.clf()
                plt.close(fig)
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #            np.array(train_losses))
        
    # if opt.save_res_fig:
        
    # # diversity_mean = abs(final_gt['img'].mean(-1)-final_output['model_out'].mean(-1))
    #     diversity_mean = abs(final_gt['img']-final_output['model_out'])
    # # torch.save(diversity_mean, 'diversity_mean_tri.pth')
    # cv2.imwrite('gen_img.jpg',255 *diversity_mean.reshape(256,256).detach().cpu().numpy())
    # import pdb;pdb.set_trace()
    # diversity_1 = abs(final_gt['img'][:,:,0:1].mean(-1) - final_output['model_out'][:,:,0:1].mean(-1))
    # diversity_2 = abs(final_gt['img'][:,:,1:2].mean(-1) - final_output['model_out'][:,:,1:2].mean(-1))
    # diversity_3 = abs(final_gt['img'][:,:,2:3].mean(-1) - final_output['model_out'][:,:,2:3].mean(-1))
    #sorted_indices_mean = torch.argsort(diversity_mean.view(-1))
    # sorted_indices_1 = torch.argsort(diversity_1.view(-1))
    # sorted_indices_2 = torch.argsort(diversity_2.view(-1))
    # sorted_indices_3 = torch.argsort(diversity_3.view(-1))
    # torch.save(sorted_indices_mean,'sorted_indices_mean.pth')
class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

def orthogonality(model):
    W = model.state_dict()['net.net.4.0.weight']
    W_transpose_W = torch.matmul(W.T, W)
    I = torch.eye(W_transpose_W.size(0), device=W.device)
    frobenius_norm = torch.sqrt(((W_transpose_W - I)**2).sum())
    # frobenius_norm = torch.norm(W_transpose_W - I, p='fro')
    return frobenius_norm