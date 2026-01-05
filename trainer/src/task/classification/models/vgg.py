from torch import nn
from .registry import register
from torchvision.models import (
    vgg11, vgg11_bn, vgg13, vgg13_bn,
    vgg16, vgg16_bn, vgg19, vgg19_bn,
    VGG as TorchvisionVGG
)

model_mapping = {
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn, 
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn
}

@register("vgg")
class VGG(nn.Module):
    """
    VGG model wrapper for classification tasks.

    This class provides an interface to load different VGG variants from 
    torchvision, replace the classifier head for a specific number of output 
    classes, and optionally use pretrained weights.

    Args:
        num_class (int):
            Number of output classes for the classification task.
        name (str, optional):
            Name of the VGG variant to use. Default is 'vgg19'.
            Supported variants: 
                'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
        weights (bool, optional):
            If True, loads pretrained weights. Default is False.

    Behavior / Notes:
        - The classifier head is automatically replaced to match `num_class`.
        - If `weights=True`, pretrained weights are frozen, making only the 
          classifier head trainable.
        - Provides methods to get model name and expected input size.
    """
    def __init__(
            self, 
            num_class: int,
            name: str = 'vgg19', 
            weights: bool = False,
        ):
        super().__init__()

        self.model: VGG = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_class)

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
        Returns the name of the VGG variant being used.

        Returns:
            str: Model name (e.g., 'vgg19').
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
        """
        Returns the expected input image size for the selected pretrained model.

        Returns:
            tuple[int, int]: (width, height) of input images, default is (224, 224).
        """
        return (224, 224)
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionVGG:
        """
        Loads a VGG model from torchvision with optional pretrained weights.

        Args:
            model_name (str): Name of the VGG variant to load.
            weights (bool): Whether to load pretrained weights.

        Returns:
            TorchvisionVGG: Instantiated VGG model from torchvision.

        Raises:
            ValueError: If `model_name` is not in the supported variants.

        Behavior / Notes:
            - If `weights=True`, all pretrained weights are frozen, leaving only 
              the classifier head trainable.
        """
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionVGG = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model