from torch import nn
from .registry import register
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5,
    efficientnet_b6, efficientnet_b7, efficientnet_v2_s,
    efficientnet_v2_m, efficientnet_v2_l,
    EfficientNet as TorchvisionEfficientNet
)

model_mapping = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
    "efficientnet_v2_s": efficientnet_v2_s,
    "efficientnet_v2_m": efficientnet_v2_m,
    "efficientnet_v2_l": efficientnet_v2_l
}

EFFICIENTNET_SIZES = {
    "efficientnet_b0": (224, 224),
    "efficientnet_b1": (240, 240),
    "efficientnet_b2": (260, 260),
    "efficientnet_b3": (300, 300),
    "efficientnet_b4": (380, 380),
    "efficientnet_b5": (456, 456),
    "efficientnet_b6": (528, 528),
    "efficientnet_b7": (600, 600),
}

@register("efficientnet")
class EfficientNet(nn.Module):
    """
    EfficientNet model wrapper for classification tasks.

    This class provides an interface to load different variants of
    EfficientNet from torchvision, replace the classifier for a specific
    number of output classes, and optionally use pretrained weights.

    Args:
        num_class (int):
            Number of output classes for the classification task.
        name (str, optional):
            Name of the EfficientNet variant to use. Default is 'efficientnet_b0'.
            Supported variants:
                - 'efficientnet_b0' ... 'efficientnet_b7'
                - 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'
        weights (bool, optional):
            If True, loads pretrained weights. Default is False.

    Behavior / Notes:
        - The classifier layer is automatically replaced to match `num_class`.
        - If `weights=True`, pretrained weights are frozen to prevent updating
          during training.
        - Provides methods to get model name and input size for pretrained weights.
    """
    def __init__(
            self, 
            num_class: int,
            name: str = 'efficientnet_b0', 
            weights: bool = False,
        ):
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_class)

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
        Returns the name of the EfficientNet variant being used.

        Returns:
            str: Model name (e.g., 'efficientnet_b0').
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
        """
        Returns the expected input image size for the selected pretrained model.

        Returns:
            tuple[int, int]: (width, height) of input images.
        """
        return EFFICIENTNET_SIZES.get(self.model_name, (224, 224))
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionEfficientNet:
        """
        Loads an EfficientNet model from torchvision with optional pretrained weights.

        Args:
            model_name (str): Name of the EfficientNet variant to load.
                Supported variants:
                    - 'efficientnet_b0' ... 'efficientnet_b7'
                    - 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'
            weights (bool): Whether to load pretrained weights.

        Returns:
            TorchvisionEfficientNet: Instantiated EfficientNet model from torchvision.

        Raises:
            ValueError: If `model_name` is not in the supported variants.

        Behavior / Notes:
            - If `weights=True`, all model parameters are frozen.
            - The returned model includes the default torchvision classifier,
              which is replaced in the wrapper's __init__ method.
        """
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionEfficientNet = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model