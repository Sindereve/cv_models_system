from torch import nn
from .registry import register
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    resnext50_32x4d, resnext101_32x8d, resnext101_64x4d,
    wide_resnet50_2, wide_resnet101_2,
    ResNet as TorchvisionResNet
)

model_mapping = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "resnext101_64x4d": resnext101_64x4d,
    "wide_resnet50_2": wide_resnet50_2, 
    "wide_resnet101_2": wide_resnet101_2
}

@register("resnet")
class ResNet(nn.Module):
    def __init__(
            self, 
            num_class: int,
            name: str = 'resnet18', 
            weights: bool = False,
        ):
        """
        Model ResNet
        
        :param num_class: Number of output classes.
        :type num_class: int
        :param name: Name of the ResNet model to load.
        :type name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        """
        super().__init__()

        self.model = self._load_model(name, weights)
        self.model_name = name

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)
    
    def get_name_model(self):
        """
        Get model name
        """
        return self.model_name
    
    def get_input_size_for_weights(self) -> tuple[int, int]:
        return (224, 224)
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> TorchvisionResNet:
        """
        Load model ResNet white torchvision
        
        :param model_name: Name of the ResNet model to load.
        :type model_name: str
        :param weights: Whether to load pretrained weights.
        :type weights: bool
        
        :return: Instantiated ResNet model.
        :rtype: TorchvisionResNet
        """
        
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: TorchvisionResNet = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        # Заморозка весов
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model