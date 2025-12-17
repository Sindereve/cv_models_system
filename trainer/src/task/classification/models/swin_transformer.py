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
    def __init__(
            self, 
            num_class: int,
            name: str = 'swin_t', 
            weights: bool = False,
        ):
        """
        Model SwinTransformer
        
        :param num_class: Number of output classes.
        :type num_class: int
        :param name: Name of the SwinTransformer model to load.
        :type name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        """
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)
    
    def get_name_model(self):
        """
        Get model name
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> int:
        return SWIN_INPUT_SIZES.get(self.model_name, (224, 224))
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionSwinTransformer:
        """
        Load model SwinTransformer with torchvision
        
        :param model_name: Name of the SwinTransformer model to load.
        :type model_name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        
        :return: Instantiated SwinTransformer model.
        :rtype: SwinTransformer from torchvision.models
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