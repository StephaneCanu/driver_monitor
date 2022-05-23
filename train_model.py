import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import logging
import time
from tqdm import trange, tqdm
from poseidon_model import Poseidon, HeadLocModel
from torch.utils.data import DataLoader
from dataset_pandora import PandoraData

from torch.utils.data.distributed import DistributedSampler
# import local rank from sys environment
local_rank = 0


def weighted_loss(w, output, target):
    return torch.norm(torch.mul(w, output-target), p='fro')


def crop_im(im, pos, w, h):
    n_img = len(im)
    croped_im = torch.zeros((n_img, im.shape[1], h, w))
    for i in range(n_img):
        croped_im[i, :, :, :] = im[i, :, int(pos[i][1]-h/2):int(pos[i][1]+h/2), int(pos[i][0]-w/2):int(pos[i][0]+w/2)]

    return transforms.Resize((64, 64))(croped_im)


def train_loc_model():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    root = 'dataset/pandora/'
    transform = transforms.Compose([
        transforms.Resize((132, 132)),
        transforms.ToTensor(),
    ])
    trainset = PandoraData(root=root, split='train', transform=transform, modal='RGB')
    trainloder = DataLoader(trainset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)

    valset = PandoraData(root=root, split='val', transform=transform, modal='RGB')
    valloader = DataLoader(valset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)

    head_loc_model = HeadLocModel().to(device=device)

    epochs = 200
    optimise_loc = optim.AdamW(head_loc_model.parameters(), lr=0.001)
    loss_loc = nn.MSELoss(reduction='mean')
    err_list = []

    for epoch in trange(epochs):

        head_loc_model.train()

        for im, angle, pos in trainloder:

            pos = pos.to(device)
            im = im.to(device)
            pre_pos = head_loc_model(im)

            optimise_loc.zero_grad()

            loss = loss_loc(pre_pos, pos)
            loss.backward()
            optimise_loc.step()
            # print(loss.item())

        head_loc_model.eval()
        err_pos = 0
        for im, _, pos in valloader:
            pos = pos.to(device)
            im = im.to(device)

            pre_pos = head_loc_model(im)

            with torch.no_grad():
                err_pos += torch.abs(pre_pos-pos).sum()

        err_pos = err_pos/len(valset)
        err_list.append(err_pos)
        if epoch % 5 == 0:
            print(f"=========={epoch}th epoch in {epochs}=========== head loc prediction error: {err_pos}")
            if len(err_list) > 1 and err_list[-1]-err_list[-6] > 0:
                break

    torch.save(head_loc_model, 'head_loc_model.bin')
    return


def train_pose_model():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    root = 'dataset/pandora/'
    transform = transforms.Compose([
        transforms.Resize((135, 160)),
        transforms.ToTensor(),
    ])
    trainset = PandoraData(root=root, split='train', transform=transform, modal='RGB')
    trainloder = DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)

    valset = PandoraData(root=root, split='val', transform=transform, modal='RGB')
    valloader = DataLoader(valset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

    head_pose_model = Poseidon().to(device=device)

    epochs = 200
    optimise_pose = optim.AdamW(head_pose_model.parameters(), lr=0.01)
    loss_pose = weighted_loss
    w = 20
    h = 20

    err_list = []

    for epoch in trange(epochs):

        head_pose_model.train()

        for im, angle, pos in trainloder:

            pos = pos.to(device)
            angle = angle.to(device)
            pos = pos.tolist()

            im = crop_im(im, pos, w, h)
            im = im.to(device)
            pre_pose = head_pose_model(im)

            optimise_pose.zero_grad()
            loss_p = loss_pose(torch.Tensor([0.2, 0.35, 0.45]).to(device), pre_pose, angle)
            loss_p.backward()
            optimise_pose.step()

            head_pose_model.eval()
            err_pose = 0
            for im, angle, pos in valloader:
                pos = torch.Tensor(pos).to(device)
                angle = torch.Tensor(angle).to(device)
                im = im.to(device)

                im = crop_im(im, pos, w, h)
                im = im.to(device)
                pre_pose = head_pose_model(im)

                with torch.no_grad():
                    err_pose += torch.sum(torch.sum(torch.abs(pre_pose-angle)))

            err_pose = err_pose/len(valset)

        if epoch % 5 == 0:
            print(f"=========={epoch}th epoch in {epochs}=========== head pose estimation error: {err_pose}")
            err_list.append(err_pose)
            if len(err_list) > 1 and err_list[-1]-err_list[-2] < 0:
                break

    torch.save(head_pose_model, 'head_pose_model.bin')
    return


if __name__ == '__main__':
    train_loc_model()
