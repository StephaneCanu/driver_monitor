from torchattacks import *
import torch
import torch.nn as nn
from distraction import Is_Distraction


class Normalize(nn.Module):
    def __init__(self, dist, scale):
        super(Normalize, self).__init__()
        self.register_buffer('dist', torch.Tensor(dist))
        self.register_buffer('scale', torch.Tensor(scale))

    def forward(self, input):
        # Broadcasting
        dist = self.dist.repeat(1, 6, 1, 1)
        scale = self.scale.repeat(1, 6, 1, 1)
        return input*scale-dist


def attack_model(model, image, label, attack_method='FGSM'):
    model = model.eval()
    if attack_method == 'FGSM':
        attack = FGSM(model, eps=8/255)
    elif attack_method == 'BIM':
        attack = BIM(model, eps=8/255, alpha=2/255, steps=100)
    elif attack_method == 'RFGSM':
        attack = RFGSM(model, eps=8/255, alpha=2/255, steps=100)
    elif attack_method == 'CW':
        attack = CW(model, c=1, lr=0.01, steps=100, kappa=0)
    elif attack_method == 'PGD':
        attack = PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True)
    elif attack_method == 'PGDL2':
        attack = PGDL2(model, eps=1, alpha=0.2, steps=100)
    elif attack_method == 'FFGSM':
        attack = FFGSM(model, eps=8/255, alpha=10/255)
    elif attack_method == 'TPGD':
        attack = TPGD(model, eps=8/255, alpha=2/255, steps=100)
    elif attack_method == 'MIFGSM':
        attack = MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1)
    elif attack_method == 'APGD':
        attack = APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce')
    elif attack_method == 'FAB':
        attack = FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False)
    elif attack_method == 'AutoAttack':
        attack = AutoAttack(model, eps=8/255, n_classes=10, version='standard')
    elif attack_method == 'OnePixel':
        attack = OnePixel(model, pixels=5, inf_batch=50)
    elif attack_method == 'DeepFool':
        attack = DeepFool(model, steps=100)
    elif attack_method == 'DIFGSM':
        attack = DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)

    attack_img = attack(image, label)
    return attack_img


def make_attack_img(inputs):

    device = 'cuda'
    model_distraction = Is_Distraction(device)
    model = torch.load('dm_model.bin')
    inputs = (inputs+1)/2

    normalize = Normalize(dist=[1], scale=[2])
    model = torch.nn.Sequential(
        normalize,
        model,
        model_distraction
    )
    model = model.to(device)
    distracted = model(torch.from_numpy(inputs).to(device))
    # print(distracted)

    # attack_img = attack_model(model, torch.from_numpy(inputs), torch.argmax(distracted, dim=-1))
    attack_img = attack_model(model, torch.from_numpy(inputs), torch.zeros(distracted.shape[0], dtype=int))
    attack_img = attack_img.detach().cpu().numpy()

    return attack_img*2-1


