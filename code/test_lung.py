import argparse
import torch
from pathlib import Path
from networks.hn import HN
from utils.test_3d_patch import test_all_case_Lung_HN

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LUNG/test', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='HN_Fusion', help='model_name')
args = parser.parse_args()

model_path = Path(f"./model/{args.exp}")
test_image_list = [dir for dir in Path(args.root_path).iterdir() if dir.is_dir()]

mdoel = HN().cuda()
mdoel.load_state_dict(torch.load(model_path / 'best_model.pth'))
mdoel.eval()

with torch.no_grad():
    cls1_avg_metric, cls2_avg_metric = test_all_case_Lung_HN(mdoel, test_image_list, stride_xy=16, stride_z=16 ,metric_detail=1, metric_txt_save=1)




