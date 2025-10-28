import os
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import random
import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,Subset

from networks.vnet import VNet
from utils.losses import dice_loss
from dataloaders.lung import Lung
from utils.data_util import get_transform
from utils.test_3d_patch import test_all_case_Lung

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LUNG', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + "/"
fig_path = Path(snapshot_path) / 'figures'


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (64, 64, 64)
num_classes = 3

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

    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    net = net.cuda()

    label_transform, unlabel_transform = get_transform()
    db_train = Lung(base_dir=train_data_path,
                    split='train',
                    label_transform= label_transform,
                    unlabel_transform= unlabel_transform
                    )

    test_image_list = [dir for dir in (Path(train_data_path)/'test').iterdir() if dir.is_dir()]

    train_subset = Subset(db_train, range(11))
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()

    avg_x = []
    avg_cls1 = []
    avg_cls2 = []

    cls1_best = 0.0
    cls2_best = 0.0

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            outputs = net(volume_batch)

            weights = torch.tensor([0.2, 1, 2], dtype=torch.float32).cuda()
            loss_seg = F.cross_entropy(outputs, label_batch, weight=weights)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = 0.5 * dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1) + dice_loss(outputs_soft[:, 2, :, :, :], label_batch == 2)
            
            loss = 0.5*(loss_seg+loss_seg_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info(f'iteration{iter_num}: loss={loss.item():.4f}, ce_loss={loss_seg.item():.4f}, dc_loss={loss_seg_dice.item():.4f}')
            
            if iter_num % 100 == 0:

                l_img_show = volume_batch[0, 0, :, :, 40].detach().cpu().numpy()
                l_label_show = label_batch[0, :, :, 40].detach().cpu().numpy()
                l_output_show = torch.argmax(outputs_soft[0], dim=0)[:, :, 40].detach().cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 8))
                axes[0].imshow(l_img_show, cmap='gray')
                axes[1].imshow(l_label_show, cmap='gray')
                axes[2].imshow(l_output_show, cmap='gray')

                for ax in axes.ravel():
                    ax.set_axis_off()
                fig.tight_layout(pad=1)
                fig.savefig(fig_path / (f'train_fig.png'))
                plt.close()

            if iter_num % 200 == 0:
                logging.info("start validation")
                net.eval()

                with torch.no_grad():
                    cls1_avg_metric, cls2_avg_metric = test_all_case_Lung(net, test_image_list, metric_detail=1)

                if cls1_avg_metric[0] >= cls1_best and cls2_avg_metric[0] >= cls2_best:

                    cls1_best = cls1_avg_metric[0]
                    cls2_best = cls2_avg_metric[0]
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(net.state_dict(), save_mode_path)
                    logging.info("save best model")

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
                logging.info(f'cls1_best={cls1_best},  cls2_best={cls2_best}')
                net.train()
                logging.info("end validation")

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
        if iter_num > max_iterations:
            break

    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    with open(snapshot_path + '/dif.pkl', 'wb') as f:
        pickle.dump([avg_x, avg_cls1, avg_cls2], f)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
