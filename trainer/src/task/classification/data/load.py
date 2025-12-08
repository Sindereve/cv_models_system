from typing import List, Tuple
import torch
import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

def load_dataloader_classification(
        path_data_dir: str,
        img_w_size: int = 224,
        img_h_size: int = 224,
        total_img: int = 0,
        batch_size: int = 32,
        train_ration: float = 0.8,
        is_calculate_normalize_dataset: bool = True
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Созданиём Dataloader

    Args:
        path_data_dir: путь к папке с данными
        img_w_size: ширина изображений после преобразований
        img_h_size: высота изображений после преобразований
        total_img: количество изображений, которое нужно
        batch_size: размер батчей
        train_ration: отношение тренировочных данных к всем данным
        is_calculate_normalize_dataset: нужно ли нормализировать наши данные

    Returns:
        Dataloader: Dataloader для тренировочных данных
        Dataloader: Dataloader для валидационнх данных
        list[str]: список названий классов
    """
    print("⚪[load_dataloader_classification] start create dataloaders")

    base_transform = transforms.Compose([
        transforms.Resize((img_h_size, img_w_size)),
        transforms.ToTensor()
    ])

    temp_dataset = datasets.ImageFolder(
        root=path_data_dir,
        transform=base_transform
    )

    temp_loader = DataLoader(
        temp_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    classes = temp_loader.dataset.class_to_idx

    if is_calculate_normalize_dataset:
        print("🟣[normalize_dataset] processing")
        mean, std = calculate_normalize_datasets(temp_loader)
    
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(img_h_size, img_w_size),
                scale=(0.7, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        val_transform = transforms.Compose([
            transforms.Resize(size=(img_h_size, img_w_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(img_h_size, img_w_size),
                scale=(0.7, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(size=(img_h_size, img_w_size)),
            transforms.ToTensor(),
        ])

    if total_img == 0:
        total_img = len(temp_dataset)

    indxs = torch.randperm(len(temp_dataset))[:total_img]
    temp_dataset = Subset(temp_dataset, indxs)

    train_size = int(train_ration * total_img)
    val_size = total_img - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        temp_dataset, [train_size, val_size]
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print("🟢[load_dataloader_classification] finish create dataloaders")
    print(f" ➖ Train samples: {len(train_dataset)}")
    print(f" ➖ Val samples:   {len(val_dataset)}")
    print(f" ➖ Classes:       {classes}")

    return train_loader, val_loader, classes


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