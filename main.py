# -*- coding: utf-8 -*-

import torch
import os
from model import Hide, Reveal
from utils import DatasetFromFolder
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    result_dir = 'result'
    ckpt_dir = 'ckpt'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = DatasetFromFolder('./data', crop_size=256)
    dataloader = DataLoader(dataset, 8, shuffle=True, num_workers=8)

    hide_net = Hide()
    hide_net.apply(init_weights)
    reveal_net = Reveal()
    reveal_net.apply(init_weights)

    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hide_net.to(device)
    reveal_net.to(device)
    criterion.to(device)

    optim_h = optim.Adam(hide_net.parameters(), lr=1e-3)
    optim_r = optim.Adam(reveal_net.parameters(), lr=1e-3)

    schedulee_h = MultiStepLR(optim_h, milestones=[100, 1000])
    schedulee_r = MultiStepLR(optim_r, milestones=[100, 1000])

    for epoch in range(2000):
        schedulee_h.step()
        schedulee_r.step()

        epoch_loss_h = 0
        epoch_loss_r = 0
        for i, (secret, cover) in enumerate(dataloader):
            secret = Variable(secret).to(device)
            cover = Variable(cover).to(device)

            optim_h.zero_grad()
            optim_r.zero_grad()

            output = hide_net(secret, cover)
            loss_h = criterion(output, cover)
            reveal_secret = reveal_net(output)
            loss_r = criterion(reveal_secret, secret)

            epoch_loss_h += loss_h.item()
            epoch_loss_r += loss_r.item()

            loss = loss_h + 0.75 * loss_r
            loss.backward()
            optim_h.step()
            optim_r.step()

        print('epoch', epoch)
        print('hide loss: %.3f' % epoch_loss_h)
        print('reveal loss: %.3f' % epoch_loss_r)
        print('=======' * 5 + '>>>')

        if epoch % 5 == 0:
            save_image(torch.cat([secret.cpu().data[:4], reveal_secret.cpu().data[:4], cover.cpu().data[:4], output.cpu().data[:4]], dim=0), fp='./{}/res_epoch_{}.png'.format(result_dir, epoch), nrow=4)
            torch.jit.save(torch.jit.script(hide_net), './{}/epoch_{}_hide.pkl'.format(ckpt_dir, epoch))
            torch.jit.save(torch.jit.script(reveal_net), './{}/epoch_{}_reveal.pkl'.format(ckpt_dir, epoch))
        