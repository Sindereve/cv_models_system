from typing import List, Tuple
import torch
import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def load_dataloader(
        path_data_dir: str,
        img_w_size: int = 224,
        img_h_size: int = 224,
        total_img: int = 0,
        batch_size: int = 32,
        train_ratio: float = 0.75,
        val_ratio: float = 0.15,
        is_calculate_normalize_dataset: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Созданиём Dataloader

    Args:
        path_data_dir: путь к папке с данными
        img_w_size: ширина изображений после преобразований
        img_h_size: высота изображений после преобразований
        total_img: количество изображений, которое нужно
        batch_size: размер батчей
        train_ratio: отношение тренировочных данных к всем данным
        val_ratio: отношение валидационных данных к всем данным
        is_calculate_normalize_dataset: нужно ли нормализировать наши данные

    Returns:
        Dataloader: Dataloader для тренировочных данных
        Dataloader: Dataloader для валидационнх данных
        Dataloader: Dataloader для тестовых данных
        list[str]: список названий классов
    """
    print("⚪[load_dataloader_classification] start create dataloaders")

    base_transform = transforms.Compose([
        transforms.Resize((img_h_size, img_w_size)),
        transforms.ToTensor()
    ])

    # full dataset
    full_dataset = datasets.ImageFolder(
        root=path_data_dir,
        transform=base_transform
    )

    classes = list(full_dataset.class_to_idx.keys())

    if total_img == 0:
        total_img = len(full_dataset)

    # subset load
    indxs = torch.randperm(len(full_dataset))[:total_img]
    subset = Subset(full_dataset, indxs)

    loader_for_norm = DataLoader(
        dataset=subset,
        batch_size=batch_size
    )

    if is_calculate_normalize_dataset:
        mean, std = calculate_normalize_datasets(loader_for_norm)
    else:
        mean, std = None, None

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((img_h_size, img_w_size), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    val_train_transform = transforms.Compose([
        transforms.Resize((img_h_size, img_w_size)),
    ])

    if mean is not None:
        train_transform.transforms.append(transforms.Normalize(mean, std))
        val_train_transform.transforms.append(transforms.Normalize(mean, std))


    # split subset train/val/test 
    train_size = int(train_ratio * total_img)
    val_size = int(val_ratio * total_img)
    test_size = total_img - train_size - val_size

    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        subset, [train_size, val_size, test_size]
    )

    train_dataset = TransformDataset(train_subset, transform=train_transform)
    val_dataset = TransformDataset(val_subset, transform=val_train_transform)
    test_dataset = TransformDataset(val_subset, transform=val_train_transform)

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("🟢[load_dataloader_classification] finish create dataloaders")
    print(f" ➖ Train samples: {train_size}")
    print(f" ➖ Val samples:   {val_size}")
    print(f" ➖ Test samples:  {test_size}")
    print(f" ➖ Classes:       {classes}")

    return train_loader, val_loader, test_loader, classes


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