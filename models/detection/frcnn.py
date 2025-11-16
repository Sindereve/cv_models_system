from torch import nn
from torchvision.models.detection.faster_rcnn import FasterRCNN as FasterRCNN_torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2
)

class FasterRCNN(nn.Module):
    def __init__(
            self,
            num_class: int,
            model_name: str = 'fasterrcnn_mobilenet_v3_large_320_fpn',
            weights: bool = False
        ):
        """
        Загрузка одной из моделей FasterRCNN

        Params:
            num_classes: количество классов
            model_name: имя модели ['fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 
                'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2']
            weights: если True - загружает веса
        """
        super().__init__()

        self.model = self._load_model(model_name, weights, num_class)
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
        ) -> FasterRCNN_torchvision:
        """
        Загрузка модели FasterRCNN с опциональными предобученными весами.

        Params:
            model_name: имя модели
            weights: загруженная модель будет с весами
            num_classes: количество классов
        """

        model_mapping = {
            "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn, 
            "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2, 
            "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn, 
            "fasterrcnn_mobilenet_v3_large_320_fpn": fasterrcnn_mobilenet_v3_large_320_fpn,
        }
        
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_mapping.keys())}")

        model_fn = model_mapping[model_name]
        model: FasterRCNN_torchvision = model_fn(
            weights_backbone="DEFAULT" if weights else None, 
            num_classes=num_classes
        )
        
        # Заморозка весов
        if weights:
            for param in model.backbone.parameters():
                param.requires_grad = False

        return model