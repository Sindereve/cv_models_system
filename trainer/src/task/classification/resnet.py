from torch import nn
from torchvision.models import (
    resnet18, resnet34, 
    resnet50, resnet101, 
    resnet152, ResNet
)

class ResNet(nn.Module):
    def __init__(
            self, 
            num_class: int,
            model_name: str = 'resnet18', 
            weights: bool = False,
        ):
        """
        Загрузка одной из моделей ResNet

        Params: 
            num_classes: количество классов
            model_name: имя модели ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
            weights: если True - загружает веса
        """
        super().__init__()


        self.model = self._load_model(model_name, weights)
        self.model_name = model_name

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_class)

        self.model_mapping = {
            "resnet18": resnet18,
            "resnet34": resnet34, 
            "resnet50": resnet50,
            "resnet101": resnet101,
            "resnet152": resnet152
        }

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
        ) -> ResNet:
        """
        Загрузка модели ResNet с опциональными предобученными весами.

        Params:
            model_name: имя модели
            weights: загруженная модель будет с весами
        """
        
        if model_name not in self.model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(self.model_mapping.keys())}")

        model: ResNet = self.model_mapping[model_name](weights="DEFAULT" if weights else None)
        
        # Заморозка весов
        if weights:
            for param in model.parameters():
                param.requires_grad = False

        return model