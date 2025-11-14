from torch import nn
from torchvision.models.detection import ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSD as SSD_torchvision

class SSD(nn.Module):
    def __init__(
            self, 
            num_class: int,
            model_name: str = 'ssd300_vgg16', 
            weights: bool = False,
        ):
        """
        Загрузка одной из моделей SSD

        Params: 
            num_classes: количество классов
            model_name: имя модели ['ssd300_vgg16', 'ssdlite320_mobilenet_v3_large']
            weights: если True - загружает веса
        """
        super().__init__()

        self.model:SSD_torchvision = self._load_model(model_name, weights, num_class)
        self.model_name = model_name

    def forward(self, x, targets=None):
        """
        
        Params:
            x: входные изображения [batch_size, 3, H, W]
            targets: опционально, таргеты для обучения
        """
        return self.model(x, targets)
    
    def get_name_model(self):
        """
        Получаем имя модели
        """
        return self.model_name
    
    def _load_model(
            self,
            model_name: str,
            weights: bool,
            num_classes: int
        ) -> SSD_torchvision:
        """
        Загрузка модели SSD с опциональными предобученными весами.

        Params:
            model_name: имя модели
            weights: загруженная модель будет с весами
            num_classes: количество классов
        """

        model_mapping = {
            "resnet18": ssd300_vgg16,
            "resnet34": ssdlite320_mobilenet_v3_large, 
        }
        
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model: SSD_torchvision = model_mapping[model_name](weights="DEFAULT" if weights else None, num_classes=num_classes)
        
        # Заморозка весов
        if weights:
            for param in model.backbone.parameters():
                param.requires_grad = False

        return model