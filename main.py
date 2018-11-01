#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:52:27 2018

@author: noorsajid
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import itertools
import os
import shutil
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable
os.chdir('/Users/noorsajid/Desktop/1stRotation/Code/second_pass/')
import model
import utils as logging


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/Volumes/LaCie/1stRotation/cropped/')
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--num_channels', type=int, default=3)
parser.add_argument('--num_gpus', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=2)

# logging
parser.add_argument('--clean_ckpt', type=bool, default=True)
parser.add_argument('--load_ckpt', type=bool, default=False)
parser.add_argument('--ckpt_path', type=str, default='/Users/noorsajid/Desktop/1stRotation/ckpt2')
parser.add_argument('--print_every', type=int, default=50)

# hyperparameter
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--z_dim', type=int, default=256)
parser.add_argument('--lr_adam', type=float, default=1e-4)
parser.add_argument('--lr_rmsprop', type=float, default=5e-5)
parser.add_argument('--beta1', type=float, default=0.5, help='for adam')
parser.add_argument('--slope', type=float, default=1e-2, help='for leaky ReLU')
parser.add_argument('--std', type=float, default=0.02, help='for weight')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--clamp', type=float, default=1e-2)
parser.add_argument('--wasserstein', type=bool, default=False)

opt = parser.parse_args()
if opt.clean_ckpt:
  shutil.rmtree(opt.ckpt_path)
os.makedirs(opt.ckpt_path, exist_ok=True)
logger = logging.Logger(opt.ckpt_path)
opt.seed = 1703
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True
EPS = 1e-12

transform = transforms.Compose([transforms.Resize(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
dataset = dset.ImageFolder(root=opt.dataset_path, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

D = model.Discriminator(opt) #.cuda()  # discriminator net D(x, z)
P = model.P(opt) #.cuda()  # generator net (decoder) P(x|z)
Q = model.Q(opt) #.cuda()  # inference net (encoder) Q(z|x)

if opt.load_ckpt:
  D.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'D.pth')))
  P.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'P.pth')))
  Q.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'Q.pth')))

if opt.wasserstein:
  optimizer_d = optim.RMSprop(D.parameters(), lr=opt.lr_rmsprop)
  optimizer_pq = optim.RMSprop(itertools.chain(P.parameters(), Q.parameters()), lr=opt.lr_rmsprop)
else:
  optimizer_d = optim.Adam(D.parameters(), lr=opt.lr_adam, betas=(opt.beta1, 0.999))
  optimizer_pq = optim.Adam(itertools.chain(P.parameters(), Q.parameters()), lr=opt.lr_adam, betas=(opt.beta1, 0.999))

#fixed_z = Variable(torch.randn(opt.batch_size, opt.z_dim, 1, 1)) #.type(torch.cuda.FloatTensor))
fixed_z = torch.randn((opt.batch_size, opt.z_dim, 1, 1), requires_grad=True)


for epoch in range(opt.num_epochs):
  stats = logging.Statistics(['loss_d', 'loss_pq'])

  print('one')
  for step, (images, _) in enumerate(data_loader, 0):
    batch_size = images.size(0)  # batch_size <= opt.batch_size
    print('grad')
    D.zero_grad()
    P.zero_grad()
    Q.zero_grad()
    
    print('updating P')

    # P network
    #z_p = Variable(torch.randn(batch_size, opt.z_dim, 1, 1)) #.type(torch.cuda.FloatTensor))
    z_p = torch.randn((batch_size, opt.z_dim, 1, 1), requires_grad=True)
    x_p = P(z_p)
    
    print('updating Q')
    # Q network
    #x_q = Variable(images) #.type(torch.cuda.FloatTensor))
    # torch.Size([128, 3, 1024, 1024])
    x_q = Variable(images, requires_grad = True)
    
    print('updating D')
    z_q = Q(x_q) #torch.Size([128, 3, 1024, 1024])
    # D network
    output_p = D(x_p, z_p)
    output_q = D(x_q, z_q)
    
    # x_p.size(), z_p.size()

    print('losssss')
    # loss & back propagation
    if opt.wasserstein:
      loss_d = -torch.mean(output_q)+torch.mean(output_p)
      loss_pq = -torch.mean(output_p)+torch.mean(output_q)
    else:
      loss_d = -torch.mean(torch.log(output_q+EPS) +torch.log(1-output_p+EPS))
      loss_pq = -torch.mean(torch.log(output_p+EPS)+torch.log(1-output_q+EPS))
       # torch.Size([128, 6, 1024, 1024])
       # torch.Size([128, 512, 1, 1]) .. torch.Size([128, 512, 1, 1]) 
    
    print('losssss 1')
    loss_d.backward(retain_graph=True)
    optimizer_d.step()
    print('losssss 2')
    loss_pq.backward()
    optimizer_pq.step()

    if opt.wasserstein:
      for p in D.parameters():
        p.data.clamp_(-opt.clamp, opt.clamp)

    # logging
    info = stats.update(batch_size, loss_d=loss_d.data[0], loss_pq=loss_pq.data[0])
    if opt.print_every > 0 and step%opt.print_every == 0:
      logger.log('epoch {}/{}, step {}/{}:\t{}'.format(epoch, opt.num_epochs, step, len(data_loader), info))

    if step == 0:
      torchvision.utils.save_image(images, '%s/real_samples.png'%opt.ckpt_path)
      fake = P(fixed_z[:batch_size])
      torchvision.utils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'%(opt.ckpt_path, epoch))

  logger.log('[Summary] epoch {}/{}:\t{}\n'.format(epoch, opt.num_epochs, stats.summary()))

  torch.save(D.state_dict(), os.path.join(opt.ckpt_path, 'D.pth'))
  torch.save(P.state_dict(), os.path.join(opt.ckpt_path, 'P.pth'))
  torch.save(Q.state_dict(), os.path.join(opt.ckpt_path, 'Q.pth'))