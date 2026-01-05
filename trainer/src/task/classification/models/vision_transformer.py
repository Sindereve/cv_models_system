from torch import nn
from .registry import register
from torchvision.models import (
    vit_b_16, vit_b_32, vit_l_16,
    vit_l_32, vit_h_14,
    VisionTransformer as TorchvisionVisionTransformer,
)

model_mapping = {
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,   
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
    "vit_h_14": vit_h_14
}

VIT_INPUT_SIZES = {
    "vit_b_16": (224, 224),
    "vit_b_32": (224, 224),
    "vit_l_16": (224, 224),
    "vit_l_32": (224, 224),
    "vit_h_14": (518, 518) 
}

@register("visiontransformer")
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model wrapper for classification tasks.

    This class provides an interface to load different ViT variants from 
    torchvision, replace the classifier head for a specific number of output 
    classes, and optionally use pretrained weights.

    Args:
        num_class (int):
            Number of output classes for the classification task.
        name (str, optional):
            Name of the ViT variant to use. Default is 'vit_b_16'.
            Supported variants:
                'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14'
        weights (bool, optional):
            If True, loads pretrained weights. Default is False.

    Behavior / Notes:
        - The classifier head is automatically replaced to match `num_class`.
        - If `weights=True`, pretrained weights are frozen, leaving only 
          the classifier head trainable.
        - Provides methods to get model name and expected input size.
    """
    def __init__(
            self, 
            num_class: int,
            name: str = 'vit_b_16', 
            weights: bool = False,
        ):
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        # Replace classifier head (supporting both 'head' and 'heads.head' conventions)
        if hasattr(self.model, 'heads'):
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, num_class)
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_class)
        else:
            raise AttributeError(f"Could not find classifier head in ViT model. "
                               f"Model attributes: {dir(self.model)}")

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
        Returns the name of the Vision Transformer variant being used.

        Returns:
            str: Model name (e.g., 'vit_b_16').
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
        """
        Returns the expected input image size for the selected pretrained model.

        Returns:
            tuple[int, int]: (width, height) of input images.
        """
        return VIT_INPUT_SIZES.get(self.model_name, (224, 224))
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionVisionTransformer:
        """
        Loads a Vision Transformer model from torchvision with optional pretrained weights.

        Args:
            model_name (str): Name of the ViT variant to load.
            weights (bool): Whether to load pretrained weights.

        Returns:
            TorchvisionVisionTransformer: Instantiated ViT model.

        Raises:
            ValueError: If `model_name` is not in the supported variants.

        Behavior / Notes:
            - If `weights=True`, all pretrained weights except the classifier head
              are frozen.
        """
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionVisionTransformer = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        if weights:
            for name, param in model.named_parameters():
                if 'head' not in name and 'heads' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        return model