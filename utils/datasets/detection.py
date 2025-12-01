import os
import glob
from pathlib import Path
from typing import Tuple, Any, List, Dict, Union
from PIL import Image
import torch

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    """
    Датасет с данными задачи детекции
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
            img_size: размер изображения (ширина, высота)
            transform: трансформер изменения изображения. Если None, то только изменяет размер изображения
            verbose: логировать процесс загрузки данных. Если False, то не логирует в консоль
        """
        self.img_width, self.img_height = img_size
        
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
    
    def _load_image(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Загрузка изображения, трансформация и возвращение тензора изображения вместе с оригинальным размером
        (изображение конвертируется в RGB)
        
        Args:
            idx: номер изображения

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]:
                * тензор изображения (C, H, W)
                * оригинальный размер изображения (ширина, высота)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_orig_size = image.size

        image = self.transform(image)
        return image, image_orig_size
    
    def _yolo_to_xyxy(
            self, 
            yolo_box: Tuple[int, float, float, float, float], 
            orig_size: Tuple[int, int]
        ) -> Tuple[int, List[int]]:
        """
        Конвертация yolo(нормализированной) разметки в xyxy(в пиксельную)

        Args:
            yolo_box: нормализованные параметры бокса в формате yolo
            orig_size: размер оригинального изображения (ширина, высота)
        Returns:
            Tuple[int, List[int]]:
                * номер класса
                * квадрат в котором находится обьект[x_min, y_min, x_max, y_max]
        """
        orig_w, orig_h = orig_size
        class_id, x_center, y_center, width, height = yolo_box
        
        x_center *= orig_w
        y_center *= orig_h
        width *= orig_w
        height *= orig_h
        
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        # масштабирование под новый размер
        scale_x = self.img_width / orig_w
        scale_y = self.img_height / orig_h

        x_min *= scale_x
        x_max *= scale_x
        y_min *= scale_y
        y_max *= scale_y

        return int(class_id), [x_min, y_min, x_max, y_max]
    
    def _load_labels(
            self, 
            idx: int, 
            orig_size: Tuple[int, int]
        ) -> Dict[str, Union[torch.Tensor, torch.Tensor]]:
        """
        Загрузка и преобразование меток в формат совместимый с torchvision

        Args:
            idx: номер изображения
            orig_size: размер оригинального изображения (ширина, высота)

        Returns:
            Dict['boxis': torch.Tensor, 'labels': torch.Tensor]:
                * boxis: тензоры квадратов в которых находятся обьекты
                * labels: id классов обьектов
        """
        label_path = self.label_paths[idx]
        
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    yolo_bbox = list(map(float, line.split()))
                    
                    # Конвертируем YOLO → XYXY
                    class_id, xyxy_bbox = self._yolo_to_xyxy(yolo_bbox, orig_size)
                    
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image, origin_size_img  = self._load_image(idx)
        target = self._load_labels(idx, origin_size_img)
        return image, target
    

def get_images_labels_path(
        images_dir: str, 
        global_path: str,
        verbose: bool = False
    ) -> Tuple[List[str], List[str]]:
    """
    Получаем пары изображение-метка для задачи детекции

    Params: 
        images_dir: путь к директории с изображениями
        global_path: базовый путь для решения относительных путей
        verbose: выводить информацию 
    Returns:
        Кортеж из 2 списков:
            - список путей к изображению
            - список путей к соотвествующим веткам
    """
    if verbose:
        print("🔘[get_images_labels_path] start")

    base_path = images_dir.replace('/images','').replace("..", global_path)
    base_path = Path(base_path)

    path_images = base_path / 'images'
    path_labels = base_path / 'labels'
    
    if not path_images.exists():
        raise FileNotFoundError(f"Dir for images not found: {path_images}")
    if not path_labels.exists():
        raise FileNotFoundError(f"Dir for labels not found: {path_labels}")
    
    image_exts = ['*.png', '*.jpg', '*.jpeg']
    images_paths = []
    
    for ext in image_exts:
        images_paths.extend(path_images.glob(ext))
    images_paths = sorted([str(p) for p in images_paths])

    if not images_paths:
        raise ValueError(f"Not images found in {path_images}")
    
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

    valid_image_paths, valid_label_paths = search_bad_syntaxis_in_label(valid_image_paths, 
                                                                        valid_label_paths, 
                                                                        verbose=verbose)

    if verbose:
        print(f"🟢[get_images_labels_path] finish")
        print(f"   ➖ count images:{len(valid_image_paths)}")
        print(f"   ➖ count labels:{len(valid_label_paths)}")
        if missing_labels:
            print(f"   🔴 missing labels:{len(missing_labels)}")

    return valid_image_paths, valid_label_paths

def search_bad_syntaxis_in_label(
        valid_images_paths: List[str], 
        valid_label_paths: List[str],
        verbose: bool = False
    ) -> Tuple[list, list]:

    bad_patchs_labels = []
    indxs_bad_patch_label = []

    for indx_label, valid_label_path in enumerate(valid_label_paths):
        
        line_count = 0
        is_bad_file = False

        with open(valid_label_path, 'r') as f:
            for line in f:
                line_count+=1
                if line.strip():
                    yolo_bbox = line.split()
                    if len(yolo_bbox) != 5:

                        if not is_bad_file:
                            bad_patchs_labels.append(valid_label_path)
                            indxs_bad_patch_label.append(indx_label)
                            is_bad_file = True

    for indx_bad in indxs_bad_patch_label[::-1]:
        valid_images_paths.pop(indx_bad)
        valid_label_paths.pop(indx_bad)
    
    if verbose:
        print("🟡[search_bad_syntaxis_in_label] Finish")
        print(f"   ➖ count bad labels:{len(bad_patchs_labels)}")

    return valid_images_paths, valid_label_paths
                    
