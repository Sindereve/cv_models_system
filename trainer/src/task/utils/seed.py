import os
import random
import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global random seed for reproducibility.

    :param seed: Seed value.
    :type seed: int
    :param deterministic: Enforce deterministic behavior (slower but reproducible).
    :type deterministic: bool
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)

    else:
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int):
    """
    Seed function for data loader workers.
    
    :param worker_id: ID of the worker.
    :type worker_id: int
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)