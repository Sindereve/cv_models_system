from torch import nn
from . import vgg, resnet, efficientnet, convnext, vision_transformer, swin_transformer
from .registry import REGISTRY

from shared.logging import get_logger

logger = get_logger(__name__)

def get_model(
        type: str = 'Resnet',  
        *args,
        **kwargs
    ) -> nn.Module:
    """
    Creates a model instance from the model registry.

    This function retrieves a model class from the `REGISTRY` by its
    name and instantiates it with the provided arguments. It supports
    all models registered via the `@register` decorator.

    Args:
        type (str, optional):
            Name of the model architecture to load. Must be registered
            in `REGISTRY`. Default is 'Resnet'.
        *args:
            Positional arguments passed to the model constructor.
        **kwargs:
            Keyword arguments passed to the model constructor.

    Returns:
        torch.nn.Module:
            Instantiated model object.

    Raises:
        ValueError:
            If the specified model `type` is not found in `REGISTRY`.

    Behavior / Notes:
        - Logs the start and completion of model loading using `logger`.
        - Model classes should be registered in `REGISTRY` using the
          `@register(name)` decorator.
        - Supports any model that inherits from `torch.nn.Module`.
    """
    logger.info("Start model load")
    if type.lower() not in REGISTRY:
        logger.error(f"Unknown model type: {type}")
        raise ValueError(f"Unknown model type: {type}")
    
    cls = REGISTRY[type.lower()]
    model = cls(*args, **kwargs)
    logger.info("Model loading complite")
    return model
