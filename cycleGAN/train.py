import argparse

from itertools import chain
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torchvision.utils import make_grid, save_image
from datasets import ImageDataset
from models import Generator, Discriminator
from utils import ReplayBuffer
# from utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../../datasets/horse2zebra', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
opts = parser.parse_args()
print(opts)

if torch.cuda.is_available() and not opts.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device('cuda' if torch.cuda.is_available() and opts.cuda else 'cpu')

###### Definition of variables ######
# Networks
netG_A2B = Generator(opts.input_nc, opts.output_nc).to(device)
netG_B2A = Generator(opts.output_nc, opts.input_nc).to(device)
netD_A = Discriminator(opts.input_nc).to(device)
netD_B = Discriminator(opts.output_nc).to(device)

# Lossess
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = optim.Adam(chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opts.lr, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(netD_A.parameters(), lr=opts.lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(netD_B.parameters(), lr=opts.lr, betas=(0.5, 0.999))

lr_lambda = lambda i: 1.0 - max(0, i - opts.decay_epoch)/(opts.n_epochs - opts.decay_epoch)
scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
scheduler_D_A = lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
scheduler_D_B = lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

# Inputs & targets memory allocation
# Tensor = torch.cuda.FloatTensor if opts.cuda else torch.Tensor
# input_A = Tensor(opts.batchSize, opts.input_nc, opts.size, opts.size)
# input_B = Tensor(opts.batchSize, opts.output_nc, opts.size, opts.size)
target_real = torch.ones(opts.batchSize, 1).float().to(device)
target_fake = torch.zeros(opts.batchSize, 1).float().to(device)
# target_real = Variable(Tensor(opts.batchSize).fill_(1.0), requires_grad=False)
# target_fake = Variable(Tensor(opts.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transform = T.Compose([
    T.Resize(int(opts.size*1.12), T.functional.InterpolationMode.BICUBIC), 
    T.RandomCrop(opts.size), 
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
dataset = ImageDataset(opts.dataroot, transform, unaligned=True)
dataloader = DataLoader(dataset, batch_size=opts.batchSize, shuffle=True, num_workers=opts.n_cpu, drop_last=True)

# Loss plot
# logger = Logger(opts.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opts.epoch, opts.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch['A'].clone().to(device)
        real_B = batch['B'].clone().to(device)

        ###### Generators A2B and B2A ######
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

        # Total loss
        loss_G = 5.0 * (loss_identity_A + loss_identity_B) + loss_GAN_A2B + loss_GAN_B2A + 10.0 * (loss_cycle_ABA + loss_cycle_BAB)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        ###### Discriminator A ######
        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) / 2
        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()

        ###### Discriminator B ######
        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) / 2
        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()
        
        # Progress report (http://localhost:8097)
            
            # print({'loss_G': loss_G.item(), 'loss_G_identity': (loss_identity_A + loss_identity_B).item(), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).item(), 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(), 'loss_D': (loss_D_A + loss_D_B).item()})
        # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
        #             images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    scheduler_G.step()
    scheduler_D_A.step()
    scheduler_D_B.step()
    
    save_image(make_grid(fake_A.detach().cpu(), nrow=4, normalize=True, value_range=(-1, 1)), f"outputs/fakeA/ep_{epoch}.jpg")
    save_image(make_grid(fake_B.detach().cpu(), nrow=4, normalize=True, value_range=(-1, 1)), f"outputs/fakeB/ep_{epoch}.jpg")
    print(f'L_G(I, G, C): {loss_G.item():.3f}({(loss_identity_A + loss_identity_B).item():.3f}, {(loss_GAN_A2B + loss_GAN_B2A).item():.3f}, {(loss_cycle_ABA + loss_cycle_BAB).item():.3f}). L_D: {(loss_D_A + loss_D_B).item():.3f}')
    # Save models checkpoints
    # torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    # torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    # torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    # torch.save(netD_B.state_dict(), 'output/netD_B.pth')


