from torch import nn
from .registry import register
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    resnext50_32x4d, resnext101_32x8d, resnext101_64x4d,
    wide_resnet50_2, wide_resnet101_2,
    ResNet as TorchvisionResNet
)

model_mapping = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "resnext101_64x4d": resnext101_64x4d,
    "wide_resnet50_2": wide_resnet50_2, 
    "wide_resnet101_2": wide_resnet101_2
}

@register("resnet")
class ResNet(nn.Module):
    """
    ResNet model wrapper for classification tasks.

    This class provides an interface to load different ResNet variants
    from torchvision, replace the classifier for a specific number
    of output classes, and optionally use pretrained weights.

    Args:
        num_class (int):
            Number of output classes for the classification task.
        name (str, optional):
            Name of the ResNet variant to use. Default is 'resnet18'.
            Supported variants:
                - 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
                - 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d'
                - 'wide_resnet50_2', 'wide_resnet101_2'
        weights (bool, optional):
            If True, loads pretrained weights. Default is False.

    Behavior / Notes:
        - The classifier layer is automatically replaced to match `num_class`.
        - If `weights=True`, pretrained weights are frozen to prevent updating
          during training.
        - Provides methods to get model name and expected input size.
    """
    def __init__(
            self, 
            num_class: int,
            name: str = 'resnet18', 
            weights: bool = False,
        ):
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_class)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits tensor of shape (B, num_class).
        """
        return self.model(x)
    
    def get_name_model(self):
        """
        Returns the name of the ResNet variant being used.

        Returns:
            str: Model name (e.g., 'resnet18').
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
        """
        Returns the expected input image size for the selected pretrained model.

        Returns:
            tuple[int, int]: (width, height) of input images. Default is (224, 224).
        """
        return (224, 224)
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionResNet:
        """
        Loads a ResNet model from torchvision with optional pretrained weights.

        Args:
            model_name (str): Name of the ResNet variant to load.
                Supported variants:
                    - 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
                    - 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d'
                    - 'wide_resnet50_2', 'wide_resnet101_2'
            weights (bool): Whether to load pretrained weights.

        Returns:
            TorchvisionResNet: Instantiated ResNet model from torchvision.

        Raises:
            ValueError: If `model_name` is not in the supported variants.

        Behavior / Notes:
            - If `weights=True`, all model parameters are frozen.
            - The returned model includes the default torchvision classifier,
              which is replaced in the wrapper's __init__ method.
        """
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionResNet = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model