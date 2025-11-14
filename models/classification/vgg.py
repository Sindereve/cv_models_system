from torch import nn
from torchvision.models import (
    vgg11, vgg11_bn, vgg13, vgg13_bn,
    vgg16, vgg16_bn, vgg19, vgg19_bn,
    VGG
)

class VGG(nn.Module):
    def __init__(
            self, 
            num_class: int,
            model_name: str = 'resnet18', 
            weights: bool = False,
        ):
        """
        Загрузка одной из моделей VGG

        Params: 
            num_classes: количество классов
            model_name: имя модели ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
            weights: если True - загружает веса
        """
        super().__init__()

        self.model: VGG = self._load_model(model_name, weights)
        self.model_name = model_name

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)
    
    def get_name_model(self):
        """
        Получаем имя модели
        """
        return self.model_name
    
    def _load_model(
            self,
            model_name: str,
            weights: bool
        ) -> VGG:
        """
        Загрузка модели VGG с опциональными предобученными весами.

        Params:
            model_name: имя модели
            weights: загруженная модель будет с весами
        """

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
        
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: VGG = model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        # Замарозка весов
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model