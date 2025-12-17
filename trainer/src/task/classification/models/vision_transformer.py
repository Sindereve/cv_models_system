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
    def __init__(
            self, 
            num_class: int,
            name: str = 'vit_b_16', 
            weights: bool = False,
        ):
        """
        Model VisionTransformer
        
        :param num_class: Number of output classes.
        :type num_class: int
        :param name: Name of the VisionTransformer model to load.
        :type name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        """
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        if hasattr(self.model, 'heads'):
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, num_class)
        # Альтернативная проверка для совместимости
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_class)
        else:
            raise AttributeError(f"Could not find classifier head in ViT model. "
                               f"Model attributes: {dir(self.model)}")

    def forward(self, x):
        return self.model(x)
    
    def get_name_model(self):
        """
        Get model name
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
        return VIT_INPUT_SIZES.get(self.model_name, (224, 224))
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionVisionTransformer:
        """
        Load model VisionTransformer white torchvision
        
        :param model_name: Name of the VisionTransformer model to load.
        :type model_name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        
        :return: Instantiated VisionTransformer model.
        :rtype: VisionTransformer from torchvision.models
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