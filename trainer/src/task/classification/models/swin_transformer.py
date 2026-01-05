from torch import nn
from .registry import register
from torchvision.models import (
    swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b,
    SwinTransformer as TorchvisionSwinTransformer,
)

model_mapping = {
    "swin_t": swin_t,
    "swin_s": swin_s,
    "swin_b": swin_b,
    "swin_v2_t": swin_v2_t,
    "swin_v2_s": swin_v2_s,
    "swin_v2_b": swin_v2_b,
}

SWIN_INPUT_SIZES = {
    "swin_t": (224, 224), 
    "swin_s": (224, 224),
    "swin_b": (224, 224),
    "swin_v2_t": (256, 256),
    "swin_v2_s": (256, 256),
    "swin_v2_b": (256, 256),
}

@register("swintransformer")
class SwinTransformer(nn.Module):
    """
    Swin Transformer model wrapper for classification tasks.

    This class provides an interface to load different Swin Transformer variants
    from torchvision, replace the classification head for a specific number
    of output classes, and optionally use pretrained weights.

    Args:
        num_class (int):
            Number of output classes for the classification task.
        name (str, optional):
            Name of the Swin Transformer variant to use. Default is 'swin_t'.
            Supported variants:
                - 'swin_t', 'swin_s', 'swin_b'
                - 'swin_v2_t', 'swin_v2_s', 'swin_v2_b'
        weights (bool, optional):
            If True, loads pretrained weights. Default is False.

    Behavior / Notes:
        - The classifier head is automatically replaced to match `num_class`.
        - If `weights=True`, pretrained weights are frozen except for the
          classification head, which remains trainable.
        - Provides methods to get model name and expected input size.
    """
    def __init__(
            self, 
            num_class: int,
            name: str = 'swin_t', 
            weights: bool = False,
        ):
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_class)

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
        Returns the name of the Swin Transformer variant being used.

        Returns:
            str: Model name (e.g., 'swin_t').
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> int:
        """
        Returns the expected input image size for the selected pretrained model.

        Returns:
            tuple[int, int]: (width, height) of input images.
        """
        return SWIN_INPUT_SIZES.get(self.model_name, (224, 224))
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionSwinTransformer:
        """
        Loads a Swin Transformer model from torchvision with optional pretrained weights.

        Args:
            model_name (str): Name of the Swin Transformer variant to load.
                Supported variants:
                    - 'swin_t', 'swin_s', 'swin_b'
                    - 'swin_v2_t', 'swin_v2_s', 'swin_v2_b'
            weights (bool): Whether to load pretrained weights.

        Returns:
            TorchvisionSwinTransformer: Instantiated Swin Transformer model from torchvision.

        Raises:
            ValueError: If `model_name` is not in the supported variants.

        Behavior / Notes:
            - If `weights=True`, all pretrained weights are frozen except for
              the classification head, which remains trainable.
            - The returned model includes the default torchvision head,
              which is replaced in the wrapper's __init__ method.
        """
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionSwinTransformer = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        if weights:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        return model