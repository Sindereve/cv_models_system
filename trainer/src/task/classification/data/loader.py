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
    Creates train, validation, and test DataLoaders from image directory.
    
    The function loads images from a directory structure where each subdirectory
    represents a class. Images are resized, optionally normalized, and split into
    train/validation/test sets according to specified ratios.
    
    Parameters
    ----------
    path_data_dir : str
        Path to the directory containing image data. Expected structure:
        path_data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                ...
    
    img_w_size : int, default=224
        Target width for image resizing.
    
    img_h_size : int, default=224
        Target height for image resizing.
    
    total_img : int, default=0
        Total number of images to use. If 0, uses all available images.
    
    batch_size : int, default=32
        Batch size for all DataLoaders.
    
    train_ratio : float, default=0.75
        Proportion of data to use for training. Should be between 0 and 1.
    
    val_ratio : float, default=0.15
        Proportion of data to use for validation. Should be between 0 and 1.
    
    is_calculate_normalize_dataset : bool, default=False
        If True, calculates mean and std for dataset normalization.
        If False, uses default normalization values.
    
    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader, List[str]]
        Returns a tuple containing:
        - train_loader : DataLoader for training data
        - val_loader : DataLoader for validation data  
        - test_loader : DataLoader for test data
        - classes : List of class names (alphabetically sorted)
    
    Raises
    ------
    ValueError
        If `train_ratio + val_ratio > 1` or if directory doesn't exist.
    
    FileNotFoundError
        If `path_data_dir` doesn't exist or contains no images.
    
    Examples
    --------
    >>> train_loader, val_loader, test_loader, classes = load_dataloader(
    ...     path_data_dir='./data/images',
    ...     img_w_size=256,
    ...     img_h_size=256,
    ...     batch_size=64
    ... )
    >>> print(f"Number of classes: {len(classes)}")
    >>> print(f"Classes: {classes}")
    
    Notes
    -----
    - Images are automatically shuffled for training set.
    - The sum of train_ratio and val_ratio must not exceed 1.0.
    - Class names are determined from subdirectory names.
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
    Validate parameters for dataloader creation.
    
    Parameters
    ----------
    path_data_dir : str
        Path to data directory
    img_w_size : int
        Image width
    img_h_size : int
        Image height
    total_img : int
        Total number of images
    batch_size : int
        Batch size
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    
    Raises
    ------
    FileNotFoundError
        If data directory doesn't exist
    ValueError
        If any parameter has invalid value
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
    Validate ImageFolder dataset structure and content.
    
    Performs comprehensive validation of the dataset including:
    - Existence of classes
    - Minimum number of images
    
    Parameters
    ----------
    dataset : datasets.ImageFolder
        PyTorch ImageFolder dataset to validate
    path_data_dir : str
        Path to the data directory (for error messages)
    min_classes : int, default=2
        Minimum number of required classes
    
    Returns
    -------
    List[str]
        List of validated class names
    
    Raises
    ------
    ValueError
        If dataset fails validation checks
    RuntimeError
        If dataset structure is invalid
    
    Examples
    --------
    >>> dataset = datasets.ImageFolder('./data/train')
    >>> classes = _validate_dataset(dataset, './data/train', min_classes=2)
    >>> print(f"Validated {len(classes)} classes")
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
    Compute mean and standard deviation for dataset normalization.
    
    Parameters
    ----------
    dataloader : DataLoader
        DataLoader with image batches (images, labels)
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (mean, std) tensors for each channel
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
    Reverse normalization transform for image visualization.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Normalized image tensor
    mean : torch.Tensor
        Mean values used for normalization
    std : torch.Tensor
        Standard deviation values used for normalization
    
    Returns
    -------
    torch.Tensor
        Denormalized image tensor
    """
    denorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    return denorm(tensor)