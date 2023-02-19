import copy
import json
import os
import warnings

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CelebA
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T 
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

import argparse

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--train', default=False, type=bool, help='train from scratch')
parser.add_argument('--eval', default=False, type=bool, help='load ckpt.pt and evaluate FID and IS')
parser.add_argument('--ch', default=128, type=int, help='base channel of UNet')
parser.add_argument('--ch_mult', default=[1, 2, 2, 2], type=list, help='channel multiplier')
parser.add_argument('--attn', default=[1], type=list, help='add attention to these levels')
parser.add_argument('--num_res_blocks', default=2, type=int, help='# resblock in each level')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of resblock')
# Gaussian Diffusion
parser.add_argument('--beta_1', default=1e-4, type=float, help='start beta value')
parser.add_argument('--beta_T', default=0.02, type=float, help='end beta value')
parser.add_argument('--T', default=1000, type=int, help='total diffusion steps')
parser.add_argument('--mean_type', default='epsilon', type=str, choices=['xprev', 'xstart', 'epsilon'], help='predict variable')
parser.add_argument('--var_type', default='fixedlarge', type=str, choices=['fixedlarge', 'fixedsmall'], help='variance type')
# Training
parser.add_argument('--lr', default=2e-4, type=float, help='target learning rate')
parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient norm clipping')
parser.add_argument('--total_steps', default=800000, type=int, help='total training steps')
parser.add_argument('--img_size', default=32, type=int, help='image size')
parser.add_argument('--warmup', default=5000, type=int, help='learning rate warmup')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='workers of Dataloader')
parser.add_argument('--ema_decay', default=0.9999, type=float, help='ema decay rate')

# Logging & Sampling
parser.add_argument('--logdir', default='./logs/DDPM_CelebA', type=str, help='log directory')
parser.add_argument('--sample_size', default=64, type=int, help='sampling size of images')
parser.add_argument('--sample_step', default=1000, type=int, help='frequency of sampling')
# Evaluation
parser.add_argument('--save_step', default=5000, type=int, help='frequency of saving checkpoints, 0 to disable during training')
parser.add_argument('--eval_step', default=0, type=int, help='frequency of evaluating model, 0 to disable during training')
parser.add_argument('--num_images', default=50000, type=int, help='the number of generated images for evaluation')
parser.add_argument('--fid_use_torch', default=False, type=bool, help='calculate IS and FID on gpu')
parser.add_argument('--fid_cache', default='./stats/cifar10.train.npz', type=str, help='FID cache')

args = parser.parse_args()


device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, args.warmup) / args.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)
            x_T = torch.randn((batch_size, 3, args.img_size, args.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, args.fid_cache, num_images=args.num_images,
        use_torch=args.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    # dataset
    # dataset = CIFAR10(
    #     root='../../data', train=True, download=True,
    #     transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    dataset = CelebA(
        root="../../data", split='train', target_type='identity', download = True, 
        transform=T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=args.T, 
        ch=args.ch, 
        ch_mult=args.ch_mult, 
        attn=args.attn, 
        num_res_blocks=args.num_res_blocks, 
        dropout=args.dropout
    )
    ema_model = copy.deepcopy(net_model)
    optimizer = optim.Adam(net_model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, args.beta_1, args.beta_T, args.T
    ).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, args.beta_1, args.beta_T, args.T, args.img_size,
        args.mean_type, args.var_type
    ).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, args.beta_1, args.beta_T, args.T, args.img_size,
        args.mean_type, args.var_type
    ).to(device)

    # log setup
    os.makedirs(os.path.join(args.logdir, 'sample'), exist_ok=True)
    x_T = torch.randn(args.sample_size, 3, args.img_size, args.img_size).to(device)
    grid = (make_grid(next(iter(dataloader))[0][:args.sample_size]) + 1) / 2
    writer = SummaryWriter(args.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(args.logdir, "flagfile.txt"), 'w') as f:
        f.write(str(args))
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(args.total_steps, dynamic_ncols=True, ncols=100) as pbar:
        for step in pbar:
            # train
            
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema(net_model, ema_model, args.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if args.sample_step > 0 and step % args.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(args.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if args.save_step > 0 and step % args.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': scheduler.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(args.logdir, 'ckpt.pt'))

            # evaluate
            if args.eval_step > 0 and step % args.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, args.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(args.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def eval():
    # model setup
    model = UNet(
        T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
        num_res_blocks=args.num_res_blocks, dropout=args.dropout)
    sampler = GaussianDiffusionSampler(
        model, args.beta_1, args.beta_T, args.T, img_size=args.img_size,
        mean_type=args.mean_type, var_type=args.var_type).to(device)

    # load model and evaluate
    ckpt = torch.load(os.path.join(args.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(args.logdir, 'samples.png'),
        nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(args.logdir, 'samples_ema.png'),
        nrow=16)


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if args.train:
        train()
    if args.eval:
        eval()
    if not args.train and not args.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    main(args)