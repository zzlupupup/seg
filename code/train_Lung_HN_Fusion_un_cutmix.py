import os
import sys
import pickle
import matplotlib.pyplot as plt
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from networks.hn_un import HN
from utils import ramps, losses
from dataloaders.lung import Lung, TwoStreamBatchSampler
from utils.test_3d_patch import test_all_case_Lung_HN
from utils.data_util import get_transform
from utils.cutmix_util import context_mask, pred_mask_back

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/LUNG', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='HN_Fusion_un_cutmix', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--lr', type=float,  default=0.0001, help='lr')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--unsup_weight', type=float,  default=1.0, help='unsup_weight')
parser.add_argument('--unsup_rampup', type=float,  default=40.0, help='unsup_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + "/"
fig_path = Path(snapshot_path) / 'figures'

batch_size = args.batch_size
max_iterations = args.max_iterations
labeled_bs = args.labeled_bs

lr = args.lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 3
patch_size = (64, 64, 64)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.unsup_weight * ramps.sigmoid_rampup(epoch, args.unsup_rampup)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if not fig_path.exists():
        fig_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    label_transform, unlabel_transform = get_transform()
    db_train = Lung(base_dir=train_data_path,
                    split='train',
                    label_transform= label_transform,
                    unlabel_transform= unlabel_transform
                    )
    
    test_image_list = [dir for dir in (Path(train_data_path)/'test').iterdir() if dir.is_dir()]
    
    labeled_idxs = list(range(11))
    unlabeled_idxs = list(range(11, 54))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=1, pin_memory=True,worker_init_fn=worker_init_fn)

    net = HN().cuda()
    net.train()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.00001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1

    avg_x = []
    avg_cls1 = []
    avg_cls2 = []

    best_metric_sum = 0.0
    cls1_best = 0.0
    cls2_best = 0.0

    ce_weights = torch.tensor([0.2, 1, 2], dtype=torch.float32).cuda()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            mask = context_mask(volume_batch[:labeled_bs], 0.5)
            volume_batch_mix = torch.zeros_like(volume_batch)
            volume_batch_mix[:labeled_bs] = mask * volume_batch[:labeled_bs] + (1 - mask) * volume_batch[labeled_bs:]
            volume_batch_mix[labeled_bs:] = mask * volume_batch[labeled_bs:] + (1 - mask) * volume_batch[:labeled_bs]

            #threshold
            threshold = (0.55 + 0.45 * ramps.linear_rampup(iter_num, max_iterations))

            #sup_loss
            pred_fusion_, pred_l_, pred_r_ = net(volume_batch_mix, threshold)
            
            #pred mask back

            pred_fusion = pred_mask_back(pred_fusion_, mask, labeled_bs)
            pred_l = pred_mask_back(pred_l_, mask, labeled_bs)
            pred_r = pred_mask_back(pred_r_, mask, labeled_bs)

            sup_loss_fusion = losses.ce_dice_loss(pred_fusion[:labeled_bs], label_batch[:labeled_bs], ce_weights)
            sup_loss_l = losses.ce_dice_loss(pred_l[:labeled_bs], label_batch[:labeled_bs], ce_weights)
            sup_loss_r = losses.ce_dice_loss(pred_r[:labeled_bs], label_batch[:labeled_bs], ce_weights)
            sup_loss = 0.2 * sup_loss_fusion+ 0.4 * (sup_loss_l + sup_loss_r)

            #unsup_loss MSE
            unsup_loss_l = torch.mean(losses.softmax_mse_loss(pred_l[labeled_bs:], pred_fusion[labeled_bs:]))
            unsup_loss_r = torch.mean(losses.softmax_mse_loss(pred_r[labeled_bs:], pred_fusion[labeled_bs:]))
            unsup_loss = unsup_loss_l + unsup_loss_r
            
            #all_loss 
            unsup_weight = get_current_consistency_weight(iter_num//150)
            loss = sup_loss + unsup_weight * unsup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            logging.info(f'iteration{iter_num}: loss={loss.item():.4f} || fusion={sup_loss_fusion.item():.4f}, l={sup_loss_l.item():.4f}, r={sup_loss_r.item():.4f}, sup_loss={sup_loss.item():.4f} || l={unsup_loss_l.item():.4f}, r={unsup_loss_r.item():.4f}, unsup_loss={unsup_loss.item():.4f}, weight={unsup_weight:.4f} || threshold={threshold:.4f}')
            if iter_num % 100 == 0:

                label_img = volume_batch[0, 0, :, :, 32].detach().cpu().numpy()
                label_label = label_batch[0, :, :, 32].detach().cpu().numpy()
                label_pred_l = torch.argmax(torch.softmax(pred_l[0, :, :, :, 32], dim=0), dim=0).detach().cpu().numpy()
                label_pred_r = torch.argmax(torch.softmax(pred_r[0, :, :, :, 32], dim=0), dim=0).detach().cpu().numpy()
                label_pred_fusion = torch.argmax(torch.softmax(pred_fusion[0, :, :, :, 32], dim=0), dim=0).detach().cpu().numpy()
                
                unlabel_img = volume_batch[labeled_bs, 0, :, :, 32].detach().cpu().numpy()
                unlabel_pred_l = torch.argmax(torch.softmax(pred_l[labeled_bs, :, :, :, 32], dim=0), dim=0).detach().cpu().numpy()
                unlabel_pred_r = torch.argmax(torch.softmax(pred_r[labeled_bs, :, :, :, 32], dim=0), dim=0).detach().cpu().numpy()
                unlabel_pred_fusion = torch.argmax(torch.softmax(pred_fusion[labeled_bs, :, :, :, 32], dim=0), dim=0).detach().cpu().numpy()

                fig, axes = plt.subplots(2, 5, figsize=(12, 8))
                axes[0,0].imshow(label_img, cmap='gray')
                axes[0,1].imshow(label_pred_l, cmap='gray')
                axes[0,2].imshow(label_pred_r, cmap='gray')
                axes[0,3].imshow(label_pred_fusion, cmap='gray')
                axes[0,4].imshow(label_label, cmap='gray')

                axes[1,0].imshow(unlabel_img, cmap='gray')
                axes[1,1].imshow(unlabel_pred_l, cmap='gray')
                axes[1,2].imshow(unlabel_pred_r, cmap='gray')
                axes[1,3].imshow(unlabel_pred_fusion, cmap='gray')
                axes[1,4].imshow(unlabel_pred_fusion, cmap='gray')

                for ax in axes.ravel():
                    ax.set_axis_off()
                fig.tight_layout(pad=1)
                fig.savefig(fig_path / (f'train{iter_num}.png'))
                plt.close()

            if iter_num % 400 == 0:
                logging.info("start validation")
                net.eval()
                with torch.no_grad():
                    cls1_avg_metric, cls2_avg_metric = test_all_case_Lung_HN(net, test_image_list, metric_detail=1)

                current_metric_sum = cls1_avg_metric[0] + cls2_avg_metric[0]

                if current_metric_sum >= best_metric_sum and cls2_avg_metric[0] >= cls2_best:
                    
                    best_metric_sum = current_metric_sum
                    cls1_best = cls1_avg_metric[0]
                    cls2_best = cls2_avg_metric[0]

                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(net.state_dict(), save_mode_path)
                    logging.info("=============save best model===============")

                avg_x.append(iter_num)
                avg_cls1.append(cls1_avg_metric[0])
                avg_cls2.append(cls2_avg_metric[0])

                logging.info(f'cls1_avg_metric: dice={cls1_avg_metric[0]:.4f},  cls2_avg_metric: dice={cls2_avg_metric[0]:.4f}')
                logging.info(f'cls1_best={cls1_best:.4f},  cls2_best={cls2_best:.4f}')

                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model ")

                net.train()
                logging.info("end validation")

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(net.state_dict(), save_mode_path)

    with open(snapshot_path + '/dif.pkl', 'wb') as f:
        pickle.dump([avg_x, avg_cls1, avg_cls2], f)
    logging.info("save model")
    writer.close()

