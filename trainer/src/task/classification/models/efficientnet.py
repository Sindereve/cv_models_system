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
    def __init__(
            self, 
            num_class: int,
            name: str = 'efficientnet_b0', 
            weights: bool = False,
        ):
        """
        Model EfficientNet
        
        :param num_class: Number of output classes.
        :type num_class: int
        :param name: Name of the EfficientNet model to load.
        :type name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        """
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)
    
    def get_name_model(self):
        """
        Get model name
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
        return EFFICIENTNET_SIZES.get(self.model_name, (224, 224))
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionEfficientNet:
        """
        Load model EfficientNet white torchvision
        
        :param model_name: Name of the EfficientNet model to load.
        :type model_name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        
        :return: Instantiated EfficientNet model.
        :rtype: TorchvisionEfficientNet
        """
        
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionEfficientNet = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        # Заморозка весов
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model