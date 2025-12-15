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
    def __init__(
            self, 
            num_class: int,
            name: str = 'convnext_tiny', 
            weights: bool = False,
        ):
        """
        Model ConvNeXt
        
        :param num_class: Number of output classes.
        :type num_class: int
        :param name: Name of the ConvNeXt model to load.
        :type name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        """
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)
    
    def get_name_model(self):
        """
        Get model name
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
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
        Load model ConvNeXt white torchvision
        
        :param model_name: Name of the ConvNeXt model to load.
        :type model_name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        
        :return: Instantiated ConvNeXt model.
        :rtype: ConvNeXt from torchvision.models
        """
        
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionConvNeXt = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model