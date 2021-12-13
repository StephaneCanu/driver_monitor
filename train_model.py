import torch
import torch.nn as nn
import torch.optim.optimizer as optimiser

import logging
import time
from model import Poseidon, HeadLocModel
from torch.utils.data import dataloader
from data_preprocessing import RGBData

from torch.utils.data.distributed import DistributedSampler
# import local rank from sys environment
local_rank=1


def train_head_loc_model(args):
    dataset = RGBData(args.root)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=local_rank)












