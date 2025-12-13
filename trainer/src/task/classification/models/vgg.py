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
    def __init__(
            self, 
            num_class: int,
            name: str = 'vgg19', 
            weights: bool = False,
        ):
        """
        Model VGG
        
        :param num_class: Number of output classes.
        :type num_class: int
        :param name: Name of the VGG model to load.
        :type name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        """
        super().__init__()

        self.model: VGG = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)
    
    def get_name_model(self):
        """
        Get model name
        """
        return self.model_name
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionVGG:
        """
        Load model VGG white torchvision
        
        :param model_name: Name of the VGG model to load.
        :type model_name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        
        :return: Instantiated VGG model.
        :rtype: TorchvisionVGG
        """
        
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionVGG = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        # Замарозка весов
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model