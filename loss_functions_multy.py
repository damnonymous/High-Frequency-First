import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import diff_operators
import modules
import cv2
import numpy as np
import torch.nn as nn
from info_nce import InfoNCE
import kornia
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw


add_mask = torch.zeros((256, 256, 3), dtype=torch.float32)
for i in range(add_mask.shape[0]):
    for j in range(add_mask.shape[1]):
        if (i+j) % 2 == 1:
            add_mask[i,j,:]= 1
add_mask = add_mask.reshape(1,-1,3).cuda()


def image_mse(model_output, gt, epoch, noise, orderlist):
    loss =  ((model_output['model_out'] - gt['img']) ** 2).mean() 

    if 0:
        loss = (add_mask * (model_output['model_out'] - gt['img'])** 2).mean()

    return {'img_loss': loss}

def image_mse_pixel_diff_loss(threshold, model_output, gt, epoch, noise, orderlist):
    threshold = 0.3
    neighborhood_size = 8
    
    if epoch <= 200:
        # H = gt['sidelength'][0] #For HiRes Image
        # W = gt['sidelength'][1] #For HiRes Image
        # gt_img = gt['img'].view(H, W, 3).cuda() #For HiRes Image
        # gt_img = gt['img'].view(256, 256, 1).repeat(1, 1, 3).cuda() # For CT Images
        gt_img = gt['img'].view(256, 256, 3).cuda() # For RGB Images
        H, W, C = gt_img.shape # For RGB Images and CT Images
        pad = neighborhood_size // 2
        
        padded_gt = torch.nn.functional.pad(
            gt_img.permute(2, 0, 1).unsqueeze(0), 
            (pad, pad, pad, pad), 
            mode='reflect'
        )[0].permute(1, 2, 0)
        
        max_diff = torch.zeros_like(gt_img)
        directions = []
        for dx in range(-pad, pad + 1):
            for dy in range(-pad, pad + 1):
                if dx == 0 and dy == 0:
                    continue
                if neighborhood_size == 4 and abs(dx) + abs(dy) == 1:
                    directions.append((dx, dy))
                elif neighborhood_size == 8 and max(abs(dx), abs(dy)) == 1:
                    directions.append((dx, dy))
                elif neighborhood_size > 8:
                    directions.append((dx, dy))

        for dx, dy in directions:
            neighbor_slice = padded_gt[pad+dy:H+pad+dy, pad+dx:W+pad+dx, :]
            current_diff = torch.abs(neighbor_slice - gt_img)
            max_diff = torch.max(max_diff, current_diff)
        #soft mask
        soft_mask = torch.sigmoid((max_diff - threshold) * 10)  
        mask = soft_mask.view(1, -1, 3)

        loss = (mask * (model_output['model_out'] - gt['img']) ** 2).mean()
    else:
        loss = ((model_output['model_out'] - gt['img']) ** 2).mean()
    
    return {'img_loss': loss}