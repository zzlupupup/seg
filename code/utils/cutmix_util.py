import torch
import numpy as np

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

