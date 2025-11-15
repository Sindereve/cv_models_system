import os
import yaml
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple, Any
import glob

def load_dataloader(
        data_dir: str,
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
        data_dir: путь к папке с данными
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
    print("⚪[load_dataloader] start create dataloaders")

    base_transform = transforms.Compose([
        transforms.Resize((img_h_size, img_w_size)),
        transforms.ToTensor()
    ])

    temp_dataset = datasets.ImageFolder(
        root=data_dir,
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

    print("🟢[load_dataloader] finish create dataloaders")
    print(f" ➖ Train samples: {len(train_dataset)}")
    print(f" ➖ Val samples:   {len(val_dataset)}")
    print(f" ➖ Classes:       {classes}")

    return train_loader, val_loader, classes

def load_dataloader_detection(
        path: str,
        img_w_size: int = 224,
        img_h_size: int = 224,
        total_img: int = 0,
        batch_size: int = 32,
        train_ration: float = 0.8,
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
    with open(path+'/data.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_path =  config['train']
    val_path =  config['val']

    classes = config['names']

    train_dataset = DetectionDataset(
        images_dir=train_path,
        global_path=path,
        img_size=(img_h_size, img_w_size)
    )

    val_dataset = DetectionDataset(
        images_dir=val_path,
        global_path=path,
        img_size=(img_h_size, img_w_size)
    )

    # Ограничиваем количество изображений
    if total_img > 0:
        train_count = int(total_img * train_ration)
        val_count = total_img - train_count
        
        # Но не больше чем есть в датасетах
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

    return train_dataloader, val_dataloader, classes

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


def get_images_labels_path(
        images_dir: str, 
        global_path: str,
        verbose: bool = False
    ) -> Tuple[List[str], List[str]]:
    """
    Получаем пары изображение-метка для задачи детекции

    Args: 
        images_dir: путь к директории с изображениями
        global_path: базовый путь для решения относительных путей
        verbose: выводить информацию 

    Returns:
        Кортеж (список патчей изображения, списко патчей меток)
    """
    if verbose:
        print("🔘[get_images_labels_path] start")

    base_path = images_dir.replace('/images','').replace("..", global_path)
    base_path = Path(base_path)

    path_images = base_path / 'images'
    path_labels = base_path / 'labels'
    
    if not path_images.exists():
        raise FileNotFoundError(f"Dir for images not found: {path_images}")
    if not path_labels.exists:
        raise FileNotFoundError(f"Dir for labels not found: {path_labels}")
    
    image_ext = ['.png', '.jpg', 'jpeg']
    images_paths = []
    
    for ext in image_ext:
        pattern = str(path_images / f'*{ext}')
        images_paths.extend(glob.glob(pattern))
    
    if not images_paths:
        raise ValueError(f"Not found images in {path_images}")
    
    if verbose:
        print("🟤[get_images_labels_path] path has been verified")

    valid_image_paths = []
    valid_label_paths = []
    missing_labels = []

    for img_path in images_paths:
        img_path = Path(img_path)
        img_name = img_path.stem
        labels_paths = path_labels / f"{img_name}.txt"

        if os.path.exists(labels_paths):
            valid_image_paths.append(img_path)
            valid_label_paths.append(str(labels_paths))
        else:
            missing_labels.append(img_name)

    if verbose:
        print(f"🟢[get_images_labels_path] finish")
        print(f"   - count images:{len(valid_image_paths)}")
        print(f"   - count labels:{len(valid_image_paths)}")
        if missing_labels:
            print(f"   🔴 missing labels:{len(missing_labels)}")

    return valid_image_paths, valid_label_paths


class DetectionDataset(Dataset):
    """Датасет для детекции объектов"""
    
    def __init__(
        self, 
        images_dir: str,
        global_path: str,
        img_size: Tuple[int, int] = (640, 640),
        transform: Any = None,
        verbose: bool = False
    ):
        self.img_size = img_size
        self.transform = transform
        
        self.image_paths, self.label_paths = get_images_labels_path(
            images_dir, global_path, verbose
        )
        
    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, idx):
        """
        Загрузка и препроцессинг изображения
        """
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])

        image = self.transform(image)
        return image, (orig_h, orig_w)
    
    def load_labels(self, idx, orig_size):
        """
        Загрузка и преобразование меток
        """
        label_path = self.label_paths[idx]
        orig_w, orig_h = orig_size
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    
                    # масштабирование координат к новому размеру
                    x_center = x_center * self.img_size[0] / orig_w
                    y_center = y_center * self.img_size[1] / orig_h
                    width = width * self.img_size[0] / orig_w
                    height = height * self.img_size[1] / orig_h
                    
                    labels.append([class_id, x_center, y_center, width, height])
        
        if labels:
            return torch.tensor(labels, dtype=torch.float32)
        else:
            return torch.zeros((0, 5), dtype=torch.float32)
    
    def __getitem__(self, idx):
        image, orig_size = self.load_image(idx)
        labels = self.load_labels(idx, orig_size)
        return image, labels
    