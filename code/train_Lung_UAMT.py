import os
import sys
import pickle
import matplotlib.pyplot as plt
import argparse
import logging
import time
import random
import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from networks.vnet import VNet
from utils import ramps, losses
from dataloaders.lung import Lung, TwoStreamBatchSampler
from utils.test_3d_patch import test_all_case_Lung
from utils.data_util import get_transform

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LUNG', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/" + args.exp + "/"
fig_path = Path(snapshot_path) / 'figures'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

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
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

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

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()

    avg_x = []
    avg_cls1 = []
    avg_cls2 = []

    cls1_best = 0.0
    cls2_best = 0.0

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            outputs = model(volume_batch)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 3, 64, 64, 64]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 3, 64, 64, 64)
            preds = torch.mean(preds, dim=0)  
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) 
            ## calculate the loss

            weights = torch.tensor([0.2, 1, 2], dtype=torch.float32).cuda()
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs], weight=weights)
            
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = 0.5*losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1) + losses.dice_loss(outputs_soft[:labeled_bs, 2, :, :, :], label_batch[:labeled_bs] == 2)
            
            supervised_loss = 0.5*(loss_seg+loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output) 
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(3)
            mask = (uncertainty<threshold).float()
            consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            consistency_loss = consistency_weight * consistency_dist
            loss = supervised_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            # writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            # writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            # writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            # writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            # writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss.item(), iter_num)
            # writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            # writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            # writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info(f'iteration{iter_num}: loss={loss.item():.4f} lr={lr_} loss_weight={consistency_weight:.4f}')

            if iter_num % 100 == 0:

                l_img_show = volume_batch[0, 0, :, :, 40].detach().cpu().numpy()
                l_label_show = label_batch[0, :, :, 40].detach().cpu().numpy()
                l_output_show = torch.argmax(outputs_soft[0], dim=0)[:, :, 40].detach().cpu().numpy()

                u_img_show = volume_batch[labeled_bs, 0, :, :, 40].detach().cpu().numpy()
                u_output_s = torch.argmax(outputs_soft[labeled_bs], dim=0)[:, :, 40].detach().cpu().numpy()
                u_output_t = torch.argmax(torch.softmax(ema_output[0], dim=0), dim=0)[:, :, 40].detach().cpu().numpy()

                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                axes[0,0].imshow(l_img_show, cmap='gray')
                axes[0,1].imshow(l_label_show, cmap='gray')
                axes[0,2].imshow(l_output_show, cmap='gray')

                axes[1,0].imshow(u_img_show, cmap='gray')
                axes[1,1].imshow(u_output_s, cmap='gray')
                axes[1,2].imshow(u_output_t, cmap='gray')

                for ax in axes.ravel():
                    ax.set_axis_off()
                fig.tight_layout(pad=1)
                fig.savefig(fig_path / (f'train_fig.png'))
                plt.close()

            if iter_num % 200 == 0:
                logging.info("start validation")
                model.eval()
                with torch.no_grad():
                    cls1_avg_metric, cls2_avg_metric = test_all_case_Lung(model, test_image_list, metric_detail=1)

                if cls1_avg_metric[0] >= cls1_best and cls2_avg_metric[0] >= cls2_best:

                    cls1_best = cls1_avg_metric[0]
                    cls2_best = cls2_avg_metric[0]
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
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
                logging.info(f'cls1_best={cls1_best:.4f},  cls2_best={cls2_best:.4f}')

                model.train()
                logging.info("end validation")

            ## change lr
            # if iter_num % 2500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    with open(snapshot_path + '/dif.pkl', 'wb') as f:
        pickle.dump([avg_x, avg_cls1, avg_cls2], f)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

