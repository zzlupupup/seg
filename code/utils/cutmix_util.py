import torch
import numpy as np
from torch.nn import functional as F



def pred_mask_back(pred, mask, labeled_bs):
    
    label = mask * pred[:labeled_bs] + (1 - mask) * pred[labeled_bs:]
    unlabel = mask * pred[labeled_bs:] + (1 - mask) * pred[:labeled_bs]
    pred_back = torch.cat((label, unlabel), dim=0)

    return pred_back


def context_mask(img, mask_ratio):
    
    mask = []
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)

    for _ in range(batch_size): 

        m = torch.ones(img_x, img_y, img_z, dtype=torch.long).cuda()

        w = np.random.randint(0, img_x - patch_pixel_x + 1)
        h = np.random.randint(0, img_y - patch_pixel_y + 1)
        z = np.random.randint(0, img_z - patch_pixel_z + 1)
        m[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
        mask.append(m)

    return torch.stack(mask).unsqueeze(1)


def grid_mask(img, block_size=4, mask_ratio=0.5):

    batch_size, channel, img_x, img_y, img_z = img.shape
    shape = (batch_size, 1, img_x // block_size, img_y // block_size, img_z // block_size)
    
    x_rand = torch.rand(shape, device=img.device)
    x = (x_rand > mask_ratio).float()
    
    mask = F.interpolate(x, scale_factor=block_size, mode='nearest')
    
    return mask

def random_keep_mask(img, keep_mask_ratio=0.5, patch_scale=16):
    batch_size, channel, img_x, img_y, img_z = img.shape
    shape = (batch_size, 1, img_x, img_y, img_z)
    
    small_x = (img_x + patch_scale - 1) // patch_scale
    small_y = (img_y + patch_scale - 1) // patch_scale
    small_z = (img_z + patch_scale - 1) // patch_scale
    small_shape = (batch_size, 1, small_x, small_y, small_z)
    
    small_rand = torch.rand(small_shape, device=img.device)
    
    interp_rand = F.interpolate(small_rand, size=(img_x, img_y, img_z), mode='trilinear', align_corners=False)
    
    num_voxels = img_x * img_y * img_z
    num_to_keep = int(num_voxels * keep_mask_ratio)
    
    flat_rand = interp_rand.view(batch_size, -1)
    
    indices = torch.argsort(flat_rand, dim=1, descending=True)
    
    keep_indices = indices[:, :num_to_keep] 
    
    mask = torch.zeros(shape, device=img.device).view(batch_size, -1)
    mask.scatter_(dim=1, index=keep_indices, value=1.0)
    
    mask = mask.view(shape)
    
    return mask

