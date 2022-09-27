import numpy as np
from torchattacks import *
import torch
import torch.nn as nn
from distraction import Is_Distraction
from model_params import MODEL_WIDTH, MODEL_HEIGHT

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method as fgsm
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2 as cw


from art.attacks import evasion
from art.estimators.classification import PyTorchClassifier


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


def attack_model(model, image, label, attack_method='FGSM', eps=8/255, c=1):
    model = model.eval()
    if attack_method == 'FGSM':
        attack = FGSM(model, eps=eps)
    elif attack_method == 'BIM':
        attack = BIM(model, eps=eps, alpha=2/255, steps=100)
    elif attack_method == 'RFGSM':
        attack = RFGSM(model, eps=eps, alpha=2/255, steps=100)
    elif attack_method == 'CW':
        attack = CW(model, c=c, lr=0.01, steps=100, kappa=0)
    elif attack_method == 'PGD':
        attack = PGD(model, eps=eps, alpha=2/225, steps=50, random_start=True)
    elif attack_method == 'PGDL2':
        attack = PGDL2(model, eps=1, alpha=0.2, steps=50)
    elif attack_method == 'FFGSM':
        attack = FFGSM(model, eps=eps, alpha=10/255)
    elif attack_method == 'TPGD':
        attack = TPGD(model, eps=eps, alpha=2/255, steps=100)
    elif attack_method == 'MIFGSM':
        attack = MIFGSM(model, eps=eps, alpha=2/255, steps=100, decay=0.1)
    elif attack_method == 'APGD':
        attack = APGD(model, eps=eps, steps=50, eot_iter=1, n_restarts=1, loss='ce')
    elif attack_method == 'FAB':
        attack = FAB(model, eps=eps, steps=100, n_classes=10, n_restarts=1, targeted=False)
    elif attack_method == 'AutoAttack':
        attack = AutoAttack(model, eps=eps, n_classes=2, version='standard')
    elif attack_method == 'OnePixel':
        attack = OnePixel(model, pixels=5, inf_batch=50)
    elif attack_method == 'DeepFool':
        attack = DeepFool(model, steps=50)
    elif attack_method == 'DIFGSM':
        attack = DIFGSM(model, eps=eps, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)

    attack_img = attack(image, label)
    return attack_img


def attack_model_cleverhans(model, image, label, attack_method='FGSM', eps=8/255):
    model = model.eval()
    if attack_method == 'FGSM':
        return fgsm(model, image, eps, np.inf)
    elif attack_method == 'PGD':
        return pgd(model, image, eps, 0.01, 50, np.inf)
    elif attack_method == 'CW':
        return cw(model, image, 2, initial_const=1500, max_iterations=50, lr=0.01)


def attack_model_robustness(model, image, label, attack_method='FGSM', eps=8/255):
    model = model.eval()
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(6, MODEL_HEIGHT, MODEL_WIDTH),
        nb_classes=2,
    )
    if attack_method == 'FGSM':
        return evasion.FastGradientMethod(estimator=classifier, eps=eps, norm=np.inf).generate(image)
    elif attack_method == 'PGD':
        return evasion.ProjectedGradientDescentPyTorch(estimator=classifier, eps=eps, eps_step=0.01,
                                                       max_iter=50, norm=np.inf).generate(image)
    elif attack_method == 'CW':
        return evasion.CarliniL2Method(classifier=classifier, initial_const=1500,
                                       max_iter=10, verbose=False).generate(image)
    elif attack_method == 'APGD':
        return evasion.AutoProjectedGradientDescent(estimator=classifier, eps=eps, max_iter=50,
                                                    eps_step=1, loss_type='cross_entropy').generate(image)
    elif attack_method == 'AutoAttack':
        return evasion.AutoAttack(estimator=classifier, eps=eps).generate(image)
    elif attack_method == 'DeepFool':
        return evasion.DeepFool(classifier=classifier, max_iter=50, epsilon=eps, nb_grads=2).generate(image)


def make_attack_img(inputs, attack_method='FGSM', lib='torchattack', eps=10/255, c=1):

    device = 'cuda'
    model_distraction = Is_Distraction(device)
    model = torch.load('models/dm_model.bin')
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

    if lib == 'torchattack':
        attack_img = attack_model(model, torch.from_numpy(inputs), torch.argmax(distracted, dim=-1), eps=eps/2,
                                  attack_method=attack_method, c=c)
        attack_img = attack_img.detach().cpu().numpy()
    elif lib == 'cleverhans':
        attack_img = attack_model_cleverhans(model, torch.from_numpy(inputs).to(device), torch.argmax(distracted, dim=-1),
                                             eps=eps, attack_method=attack_method)
        attack_img = attack_img.detach().cpu().numpy()
    elif lib == 'robustness':
        attack_img = attack_model_robustness(model, inputs, torch.argmax(distracted, dim=-1),
                                             eps=eps, attack_method=attack_method)

    return attack_img*2-1


