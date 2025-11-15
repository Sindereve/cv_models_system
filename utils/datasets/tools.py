import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_normalize_datasets(
        dataloader: DataLoader
    ):
    """
    Вычисляем значения для нормализации датасета

    Args:
        dataloader: весь известный нам датасет
    """
    print("⚪[calculate_normalize_datasets] start")
    channels_sum = torch.zeros(3)
    channels_sq_sum = torch.zeros(3)
    num_batches = 0

    for data, _ in tqdm(dataloader):
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_sq_sum +=  torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Dataloader пуст")
    
    mean = channels_sum / num_batches
    std = (channels_sq_sum / num_batches - mean**2)**0.5
    print("🟢[calculate_normalize_datasets] finish")
    return mean, std

def denormalize_image(
        tensor: torch.Tensor, 
        mean: torch.Tensor, 
        std: torch.Tensor,
    ) -> torch.Tensor:
    """
    Денормализация для отображения

    Args:
        tensor: изображение в виде тензора
        mean и std: параметры используемые при нормализации

    Return:
        изображение в виде тензора
    """
    denorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    return denorm(tensor)