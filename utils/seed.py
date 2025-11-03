import torch
import numpy
import random
import os

#   more info
#   https://docs.pytorch.org/docs/stable/notes/randomness.html

def set_global_seed(
        random_state: int,
        device: torch.device,
        use_deterministic_alg: bool = True
    )-> None:
    """
    Установка глобального seed и настройка детерминированных алгоритмов

    Args:
        random_state: Начальное значение для всех генераторов
        use_deterministic_alg: Включить детерминированные алгоритмы
        device: Используемое устройство
    """
    random.seed(random_state)
    numpy.random.seed(random_state)
    torch.manual_seed(random_state)

    if device.type == "cuda":
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)    # вдруг gpu пользователь поменяет
    
    if device.type == "cuda" and use_deterministic_alg:
        
        # !!! РАЗОБРАТЬ ПОЗЖЕ !!!!
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8') 
        # !!! РАЗОБРАТЬ ПОЗЖЕ !!!!

        torch.backends.cudnn.deterministic = True   # использует только детерминированные алгоритмы
        torch.backends.cudnn.benchmark = False      # отключает авто-подбор лучшего алгоритма
        torch.use_deterministic_algorithms(True)

    print("🟢 Finish setting random:")
    print(" ➖ Random seed:", random_state)
    print(" ➖ Use deterministic alg:", use_deterministic_alg)


