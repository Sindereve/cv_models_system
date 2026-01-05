from typing import List, Tuple
import torch
import tqdm
import os
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from shared.logging import get_logger

logger = get_logger(__name__)

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
    Creates train, validation, and test DataLoaders from an image directory.

    The function loads images organized in subdirectories per class, resizes
    them to the specified dimensions, optionally calculates dataset normalization,
    and splits them into training, validation, and test sets according to the
    specified ratios.

    Args:
        path_data_dir (str):
            Path to the root directory containing subdirectories of classes.
        img_w_size (int, optional):
            Target width for image resizing. Default is 224.
        img_h_size (int, optional):
            Target height for image resizing. Default is 224.
        total_img (int, optional):
            Total number of images to use. If 0, uses all available images.
            Default is 0.
        batch_size (int, optional):
            Batch size for all DataLoaders. Default is 32.
        train_ratio (float, optional):
            Proportion of images for training. Default is 0.75.
        val_ratio (float, optional):
            Proportion of images for validation. Default is 0.15.
        is_calculate_normalize_dataset (bool, optional):
            If True, calculates mean and std for dataset normalization.
            If False, uses default normalization values. Default is False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
            A tuple containing:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - test_loader: DataLoader for test data
            - classes: List of class names (alphabetically sorted)

    Raises:
        ValueError:
            If `train_ratio + val_ratio > 1` or if directory structure is invalid.
        FileNotFoundError:
            If `path_data_dir` does not exist or contains no images.

    Behavior / Notes:
        - Images in the training set are shuffled by default.
        - Class names are automatically inferred from subdirectory names.
        - The sum of `train_ratio` and `val_ratio` must not exceed 1.0.
    """
    logger.info("⚪[load_dataloader_classification] start create dataloaders")

    _validate_dataloader_parameters(
        path_data_dir=path_data_dir,
        img_w_size=img_w_size,
        img_h_size=img_h_size,
        total_img=total_img,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    base_transform = transforms.Compose([
        transforms.Resize((img_h_size, img_w_size)),
        transforms.ToTensor()
    ])

    # full dataset
    full_dataset = datasets.ImageFolder(
        root=path_data_dir,
        transform=base_transform
    )

    classes = _validate_dataset(
        dataset=full_dataset,
        path_data_dir=path_data_dir,
    )

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
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    logger.info("🟢[load_dataloader_classification] finish create dataloaders")
    logger.info(f" ➖ Train samples: {train_size}")
    logger.info(f" ➖ Val samples:   {val_size}")
    logger.info(f" ➖ Test samples:  {test_size}")
    logger.info(f" ➖ Classes:       {classes}")

    return train_loader, val_loader, test_loader, classes


def _validate_dataloader_parameters(
    path_data_dir: str,
    img_w_size: int,
    img_h_size: int,
    total_img: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    ) -> None:
    """
    Validates input parameters for dataloader creation.

    This function checks the integrity and correctness of all
    parameters required to create training, validation, and test
    DataLoaders from an image directory.

    Args:
        path_data_dir (str):
            Path to the directory containing image data.
        img_w_size (int):
            Target width for image resizing.
        img_h_size (int):
            Target height for image resizing.
        total_img (int):
            Total number of images to use. Must be non-negative.
        batch_size (int):
            Batch size for all DataLoaders. Must be positive.
        train_ratio (float):
            Proportion of data for training. Must be between 0 and 1.
        val_ratio (float):
            Proportion of data for validation. Must be between 0 and 1.
            The sum of `train_ratio + val_ratio` must not exceed 1.0.

    Raises:
        FileNotFoundError:
            If `path_data_dir` does not exist.
        ValueError:
            If any parameter has an invalid value (e.g., negative sizes,
            invalid ratios, or batch size <= 0).

    Behavior / Notes:
        - Ensures that dataloader parameters are suitable for dataset splitting.
        - Intended for internal use before creating DataLoaders.
    """
    if not os.path.isdir(path_data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {path_data_dir}"
        )
    
    if img_w_size <= 0 or img_h_size > 5000:
        raise ValueError(
            f"Image width must be between 1 and 5000, got {img_w_size}"
        )
    
    if img_h_size <= 0 or img_h_size > 5000:
        raise ValueError(
            f"Image height must be between 1 and 5000, got {img_h_size}"
        )
    
    if total_img < 0:
        raise ValueError(
            f"total_img cannot be negative, got {total_img}"
        )
    
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}"
        )
    
    if not 0 < train_ratio <= 1:
        raise ValueError(
            f"train_ratio must be in range (0, 1], got {train_ratio}"
        )
    
    if not 0 <= val_ratio < 1:
        raise ValueError(
            f"val_ratio must be in range [0, 1), got {val_ratio}"
        )
    
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"Sum of train_ratio ({train_ratio}) and val_ratio ({val_ratio}) "
            f"must be <= 1.0, got {train_ratio + val_ratio:.2f}"
        )
    
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"Resulting test_ratio would be negative: {test_ratio:.2f}. "
            f"Check train_ratio and val_ratio."
        )

def _validate_dataset(
    dataset: datasets.ImageFolder, 
    path_data_dir: str,
    min_classes: int = 2,
) -> List[str]:
    """
    Validates an ImageFolder dataset structure and content.

    Performs a comprehensive validation of the dataset, including:
    - Existence of classes (subdirectories)
    - Minimum number of classes
    - Minimum number of images per class

    Args:
        dataset (datasets.ImageFolder):
            PyTorch ImageFolder dataset to validate.
        path_data_dir (str):
            Path to the data directory (used for error messages).
        min_classes (int, optional):
            Minimum number of required classes. Default is 2.

    Returns:
        List[str]:
            Alphabetically sorted list of validated class names.

    Raises:
        ValueError:
            If the dataset fails validation checks (e.g., insufficient
            images or classes).
        RuntimeError:
            If the dataset structure is invalid (e.g., missing directories).

    Behavior / Notes:
        - Ensures that the dataset is ready for DataLoader creation.
        - Intended for internal use before loading data into training pipeline.
    """
    classes = list(dataset.class_to_idx.keys())
    
    if len(classes) < min_classes:
        raise ValueError(
            f"Insufficient number of classes in {path_data_dir}. "
            f"Found {len(classes)} classes, expected at least {min_classes}. "
            f"Check directory structure: {path_data_dir}/class_name/images.jpg"
        )
    
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError(
            f"No images found in {path_data_dir}. "
            f"Directory should contain image files in class subdirectories."
        )

    if not hasattr(dataset, 'samples') or not dataset.samples:
        raise RuntimeError(
            f"Dataset from {path_data_dir} has invalid structure. "
            f"Missing or empty 'samples' attribute."
        )
    
    return classes

def calculate_normalize_datasets(
        dataloader: DataLoader
    ):
    """
    Computes mean and standard deviation for dataset normalization.

    This function iterates over all batches in the dataloader and computes
    the channel-wise mean and standard deviation, which can be used to
    normalize the dataset for training.

    Args:
        dataloader (DataLoader):
            DataLoader providing image batches in the form (images, labels).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - mean: Tensor of shape [C] containing channel-wise mean.
            - std: Tensor of shape [C] containing channel-wise standard deviation.

    Raises:
        ValueError:
            If the dataloader contains no batches (empty dataset).

    Behavior / Notes:
        - Logs the start and end of computation using `logger`.
        - Intended for internal use when preparing datasets for training.
        - Assumes images are in the format (batch, channels, height, width).
    """
    logger.info("⚪[calculate_normalize_datasets] start")
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
    logger.info("🟢[calculate_normalize_datasets] finish")
    return mean, std

def denormalize_image(
        tensor: torch.Tensor, 
        mean: torch.Tensor, 
        std: torch.Tensor,
    ) -> torch.Tensor:
    """
    Reverses normalization for an image tensor for visualization.

    This function applies the inverse of the normalization transform
    (using the provided mean and standard deviation) to convert a
    normalized image tensor back to its original range.

    Args:
        tensor (torch.Tensor):
            Normalized image tensor of shape (C, H, W) or (B, C, H, W).
        mean (torch.Tensor):
            Mean values used during normalization (channel-wise).
        std (torch.Tensor):
            Standard deviation values used during normalization (channel-wise).

    Returns:
        torch.Tensor:
            Denormalized image tensor, same shape as input.

    Behavior / Notes:
        - Useful for visualizing images after normalization.
        - Works for both single images and batches.
        - The operation does not modify the input tensor in place.
    """
    denorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    return denorm(tensor)