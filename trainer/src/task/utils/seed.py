import os
import random
import numpy as np
import torch
from shared.logging import get_logger

logger = get_logger(__name__)

def set_seed(seed: int, deterministic: bool = True) -> None:
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

    logger.debug('Seed deterministic function: 42')