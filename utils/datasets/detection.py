import os
import glob
from pathlib import Path
from typing import Tuple, Any, List
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    """
    Датасет с данными для решения задачи детекции
    """
    
    def __init__(
        self, 
        images_dir: str,
        global_path: str,
        img_size: Tuple[int, int] = (640, 640),
        transform: Any = None,
        verbose: bool = False
    ):
        """
        Датасет с данными для решения задачи детекции

        Params:
            images_dir: путь к папке с изображениями 
            global_path: путь к папке, где хранится data.yaml 
            img_size: размер изображения (высота, ширина)
            transform: трансформер изменения изображения. Если None, то только изменяет размер изображения
            verbose: логировать процесс загрузки данных. Если False, то не логирует в консоль
        """
        self.img_height, self.img_width = img_size
        
        self.image_paths, self.label_paths = get_images_labels_path(
            images_dir, global_path, verbose
        )

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor()
            ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, idx):
        """
        Загрузка и препроцессинг изображения
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        image = self.transform(image)
        return image, (orig_h, orig_w)
    
    def _yolo_to_xyxy(self, yolo_bbox, orig_size):
        """
        Конвертирует YOLO формат (x_center, y_center, width, height) 
        в формат xyxy (x_min, y_min, x_max, y_max)
        """
        orig_h, orig_w = orig_size
        class_id, x_center, y_center, width, height = yolo_bbox
        
        # Масштабирование к исходному размеру
        x_center = x_center * orig_w
        y_center = y_center * orig_h
        width = width * orig_w
        height = height * orig_h
        
        # Конвертация в xyxy
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        return [x_min, y_min, x_max, y_max]
    
    def _load_labels(self, idx, orig_size):
        """
        Загрузка и преобразование меток в формат для torchvision
        """
        label_path = self.label_paths[idx]
        
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    yolo_bbox = list(map(float, line.split()))
                    
                    # Конвертируем YOLO → XYXY
                    xyxy_bbox = self._yolo_to_xyxy(yolo_bbox, orig_size)
                    class_id = int(yolo_bbox[0])
                    
                    boxes.append(xyxy_bbox)
                    labels.append(class_id)
        
        if boxes:
            return {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }
        else:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }
    
    def __getitem__(self, idx):
        image, orig_size = self._load_image(idx)
        target = self._load_labels(idx, orig_size)
        return image, target
    


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
