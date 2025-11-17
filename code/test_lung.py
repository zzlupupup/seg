import argparse
import torch
import numpy as np

from monai import transforms
from pathlib import Path
from networks.hn import HN
from utils.test_3d_patch import test_all_case_Lung_HN

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LUNG/test', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='HN_Fusion', help='model_name')
args = parser.parse_args()

model_path = Path(f"./model/{args.exp}")
test_image_list = [dir for dir in Path(args.root_path).iterdir() if dir.is_dir()]

model = HN().cuda()
model.load_state_dict(torch.load(model_path / 'best_model.pth'))
model.eval()

if __name__ == '__main__':
    
    r = 5
    threshold = 4/3 * np.pi * r ** 3

    cls1_trans = transforms.KeepLargestConnectedComponent()
    cls2_trans = transforms.RemoveSmallObjects(min_size=threshold, independent_channels=True, by_measure=True, pixdim=[1.0, 1.0, 1.0])

    def apply_trans(img):
        back = img[0:1]
        cls1 = img[1:2]
        cls2 = img[2:]

        transed_cls1 = cls1_trans(cls1)
        transed_cls2 = cls2_trans(cls2)
        
        return torch.cat([back, transed_cls1, transed_cls2], dim=0)

    trans = transforms.Lambda(apply_trans)

    post_trans = transforms.Compose([
        transforms.AsDiscrete(to_onehot=3),
        trans
    ])
    with torch.no_grad():
        cls1_avg_metric, cls2_avg_metric = test_all_case_Lung_HN(model, test_image_list, stride_xy=32, stride_z=32 ,metric_detail=1, metric_txt_save=1, post=post_trans)




