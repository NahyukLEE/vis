r""" Helper functions """
import random

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    batch['pcd_t_cpu'] = [v.clone() for v in batch['pcd_t']]
    for key, value in batch.items():
        if key == 'pcd_t_cpu': continue
        if isinstance(value[0], torch.Tensor):
            batch[key] = [v.cuda() for v in value]
    # batch['match_metadata'] = batch['match_metadata'][0]
    batch['filepath'] = batch['filepath'][0]
    batch['n_frac'] = batch['n_frac'][0]
    batch['order'] = batch['order'][0]

    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()