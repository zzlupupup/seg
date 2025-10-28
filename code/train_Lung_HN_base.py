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
from networks.vnet import VNet
from monai.networks.nets import SwinUNETR
from utils import ramps, losses
from dataloaders.lung import Lung, TwoStreamBatchSampler
from utils.test_3d_patch import test_all_case_Lung
from utils.data_util import get_transform

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LUNG', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='HN_base', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--lr_l', type=float,  default=0.01, help='lr_l')
parser.add_argument('--lr_r', type=float,  default=0.0001, help='lr_r')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--cps_weight', type=float,  default=0.1, help='cps_weight')
parser.add_argument('--cps_rampup', type=float,  default=40.0, help='cps_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + "/"
fig_path = Path(snapshot_path) / 'figures'

batch_size = args.batch_size
max_iterations = args.max_iterations
labeled_bs = args.labeled_bs

lr_l = args.lr_l
lr_r = args.lr_r

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
    return args.cps_weight * ramps.sigmoid_rampup(epoch, args.cps_rampup)

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

    model_l = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    model_r = SwinUNETR(in_channels=1, out_channels=num_classes).cuda()

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

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)

    model_l.train()
    model_r.train()

    optimizer_l = optim.SGD(model_l.parameters(), lr=lr_l, momentum=0.9, weight_decay=0.0001)
    optimizer_r = optim.AdamW(model_r.parameters(), lr=lr_r, weight_decay=0.00001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1

    avg_x = []
    avg_cls1 = []
    avg_cls2 = []

    cls1_best = 0.0
    cls2_best = 0.0

    ce_weights = torch.tensor([0.2, 1, 2], dtype=torch.float32).cuda()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            outputs_l = model_l(volume_batch)
            outputs_r = model_r(volume_batch)
            #sup loss
            sup_ce_l = F.cross_entropy(outputs_l[:labeled_bs], label_batch[:labeled_bs], weight=ce_weights)
            sup_ce_r = F.cross_entropy(outputs_r[:labeled_bs], label_batch[:labeled_bs], weight=ce_weights)

            outputs_soft_l = F.softmax(outputs_l, dim=1)
            outputs_soft_r = F.softmax(outputs_r, dim=1)

            sup_dice_l = 0.5*losses.dice_loss(outputs_soft_l[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1) + losses.dice_loss(outputs_soft_l[:labeled_bs, 2, :, :, :], label_batch[:labeled_bs] == 2)
            sup_dice_r = 0.5*losses.dice_loss(outputs_soft_r[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1) + losses.dice_loss(outputs_soft_r[:labeled_bs, 2, :, :, :], label_batch[:labeled_bs] == 2)
            
            sup_loss = 0.5*(sup_ce_l + sup_ce_r + sup_dice_l + sup_dice_r)

            #cps_loss
            pseudo_l = torch.argmax(outputs_soft_l, dim=1).long().detach()
            pseudo_r = torch.argmax(outputs_soft_r, dim=1).long().detach()  

            cps_ce_l = F.cross_entropy(outputs_l[labeled_bs:], pseudo_r[labeled_bs:], weight=ce_weights)
            cps_ce_r = F.cross_entropy(outputs_r[labeled_bs:], pseudo_l[labeled_bs:], weight=ce_weights)
            
            cps_dice_l = 0.5 * losses.dice_loss(outputs_soft_l[labeled_bs:, 1, ...], pseudo_r[labeled_bs:] == 1) + losses.dice_loss(outputs_soft_l[labeled_bs:, 2, ...], pseudo_r[labeled_bs:] == 2)
            cps_dice_r = 0.5 * losses.dice_loss(outputs_soft_r[labeled_bs:, 1, ...], pseudo_l[labeled_bs:] == 1) + losses.dice_loss(outputs_soft_r[labeled_bs:, 2, ...], pseudo_l[labeled_bs:] == 2)
            
            cps_loss = 0.5 * (cps_ce_l + cps_ce_r + cps_dice_l + cps_dice_r)

            #all_loss 
            cps_weight = get_current_consistency_weight(iter_num//150)
            loss = sup_loss + cps_weight * cps_loss

            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            loss.backward()
            optimizer_l.step()
            optimizer_r.step()

            iter_num = iter_num + 1
            writer.add_scalar('loss/loss', loss.item(), iter_num)

            logging.info(f'iteration{iter_num}: loss={loss.item():.4f}, sup_loss={sup_loss.item():.4f}, cps_loss={cps_loss.item():.4f}, cps_weight={cps_weight:.4f}')

            if iter_num % 100 == 0:

                l_img_show = volume_batch[0, 0, :, :, 40].detach().cpu().numpy()
                l_label_show = label_batch[0, :, :, 40].detach().cpu().numpy()
                l_output_show = pseudo_l[0, :, :, 40].cpu().numpy()

                u_img_show = volume_batch[labeled_bs, 0, :, :, 40].detach().cpu().numpy()
                u_output_l = pseudo_l[labeled_bs, :, :, 40].cpu().numpy()
                u_output_r = pseudo_r[labeled_bs, :, :, 40].cpu().numpy()

                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                axes[0,0].imshow(l_img_show, cmap='gray')
                axes[0,1].imshow(l_label_show, cmap='gray')
                axes[0,2].imshow(l_output_show, cmap='gray')

                axes[1,0].imshow(u_img_show, cmap='gray')
                axes[1,1].imshow(u_output_l, cmap='gray')
                axes[1,2].imshow(u_output_r, cmap='gray')

                for ax in axes.ravel():
                    ax.set_axis_off()
                fig.tight_layout(pad=1)
                fig.savefig(fig_path / (f'train_fig.png'))
                plt.close()

            if iter_num % 200 == 0:
                logging.info("start validation")
                model_r.eval()
                with torch.no_grad():
                    cls1_avg_metric, cls2_avg_metric = test_all_case_Lung(model_r, test_image_list, metric_detail=1)

                if cls1_avg_metric[0] >= cls1_best and cls2_avg_metric[0] >= cls2_best:

                    cls1_best = cls1_avg_metric[0]
                    cls2_best = cls2_avg_metric[0]

                    save_mode_path_l = os.path.join(snapshot_path, 'best_model_l.pth')
                    save_mode_path_r = os.path.join(snapshot_path, 'best_model_r.pth')
                    torch.save(model_l.state_dict(), save_mode_path_l)
                    torch.save(model_r.state_dict(), save_mode_path_r)
                    logging.info("=============save best model===============")

                avg_x.append(iter_num)
                avg_cls1.append(cls1_avg_metric[0])
                avg_cls2.append(cls2_avg_metric[0])

                fig, ax = plt.subplots(figsize=(8,5))
                ax.plot(avg_x, avg_cls1, label='cls1')
                ax.plot(avg_x, avg_cls2, label='cls2')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(fig_path/'cls_dif.png')
                plt.close()

                writer.add_scalar('val/cls1_dice', cls1_avg_metric[0], iter_num)
                writer.add_scalar('val/cls2_dice', cls2_avg_metric[0], iter_num)
                logging.info(f'cls1_avg_metric: dice={cls1_avg_metric[0]:.4f},  cls2_avg_metric: dice={cls2_avg_metric[0]:.4f}')
                logging.info(f'cls1_best={cls1_best:.4f},  cls2_best={cls2_best:.4f}')

                model_r.train()
                logging.info("end validation")

            if iter_num % 1000 == 0:
                save_mode_path_l = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_l.pth')
                torch.save(model_l.state_dict(), save_mode_path_l)

                save_mode_path_r = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_r.pth')
                torch.save(model_r.state_dict(), save_mode_path_r)
                logging.info("save model ")

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    save_mode_path_l = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'_l.pth')
    torch.save(model_l.state_dict(), save_mode_path_l)
    save_mode_path_r = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'_r.pth')
    torch.save(model_r.state_dict(), save_mode_path_r)

    with open(snapshot_path + '/dif.pkl', 'wb') as f:
        pickle.dump([avg_x, avg_cls1, avg_cls2], f)
    logging.info("save model")
    writer.close()

