from torch import nn
from .registry import register
from torchvision.models import (
    convnext_tiny, convnext_small,
    convnext_base, convnext_large,
    ConvNeXt as TorchvisionConvNeXt,
)

model_mapping = {
    'convnext_tiny': convnext_tiny,
    'convnext_small': convnext_small,
    'convnext_base': convnext_base,
    'convnext_large': convnext_large,
}

@register("convnext")
class ConvNeXt(nn.Module):
    """
    ConvNeXt model wrapper for classification tasks.

    This class provides an interface to load different variants of
    ConvNeXt from torchvision, replace the classifier for a specific
    number of output classes, and optionally use pretrained weights.

    Args:
        num_class (int):
            Number of output classes for the classification task.
        name (str, optional):
            Name of the ConvNeXt variant to use. Options:
            'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'.
            Default is 'convnext_tiny'.
        weights (bool, optional):
            If True, loads pretrained weights. Default is False.

    Behavior / Notes:
        - The classifier layer is automatically replaced to match `num_class`.
        - If `weights=True`, pretrained weights are frozen to prevent updating
        during training.
        - Supports methods to get model name and input size required for pretrained weights.
    """
    
    def __init__(
            self, 
            num_class: int,
            name: str = 'convnext_tiny', 
            weights: bool = False,
        ):
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_class)

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
        Returns the name of the ConvNeXt variant being used.

        Returns:
            str: Model name (e.g., 'convnext_tiny').
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
        """
        Returns the expected input image size for the selected pretrained model.

        Returns:
            tuple[int, int]: (width, height) of input images.
        """
        if self.model_name in ["convnext_tiny", "convnext_small", "convnext_base"]:
            return (224, 224)
        elif self.model_name in ["convnext_large"]:
            return (384, 384)
        else:
            return (224, 224)
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionConvNeXt:
        """
        Loads a ConvNeXt model from torchvision with optional pretrained weights.

        Args:
            model_name (str): Name of the ConvNeXt variant to load.
                Options: 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'.
            weights (bool): Whether to load pretrained weights.

        Returns:
            TorchvisionConvNeXt: Instantiated ConvNeXt model from torchvision.

        Raises:
            ValueError: If `model_name` is not in the supported variants.

        Behavior / Notes:
            - If `weights=True`, all model parameters are frozen.
            - The returned model includes the default torchvision classifier,
              which will be replaced in the wrapper's __init__ method.
        """
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionConvNeXt = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model