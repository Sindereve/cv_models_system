import yaml
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List

from .detection import DetectionDataset
from .tools import calculate_normalize_datasets, denormalize_image

__all__ = [denormalize_image, calculate_normalize_datasets]

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

    classes = temp_loader.dataset.classes

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

def load_dataloader_detection(
        path_data_dir: str,
        img_w_size: int = 224,
        img_h_size: int = 224,
        total_img: int = 0,
        batch_size: int = 32,
        train_ration: float = 0.8,
        verbose: bool = False
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
    
    print("⚪[load_dataloader_detection] start create dataloaders")

    with open(path_data_dir+'/data.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_path =  config['train']
    val_path =  config['val']
    classes = config['names']

    train_dataset = DetectionDataset(
        images_dir=train_path,
        global_path=path_data_dir,
        img_size=(img_h_size, img_w_size)
        verbose=verbose
    )

    val_dataset = DetectionDataset(
        images_dir=val_path,
        global_path=path_data_dir,
        img_size=(img_h_size, img_w_size),
        verbose=verbose
    )

    # Ограничиваем количество изображений
    if total_img > 0:
        train_count = int(total_img * train_ration)
        val_count = total_img - train_count
        
        train_count = min(train_count, len(train_dataset))
        val_count = min(val_count, len(val_dataset))
        
        # Создаем подмножества
        train_indices = torch.randperm(len(train_dataset))[:train_count]
        val_indices = torch.randperm(len(val_dataset))[:val_count]
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    else:
        train_dataset = train_dataset
        val_dataset = val_dataset

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        return images, labels

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    print("🟢[load_dataloader_detection] finish create dataloaders")
    print(f" ➖ Train samples: {len(train_dataset)}")
    print(f" ➖ Val samples:   {len(val_dataset)}")
    print(f" ➖ Classes:       {classes}")

    return train_dataloader, val_dataloader, classes