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
    Create a model instance from the registry.
    
    :param type: Model architecture name registered in REGISTRY.
    :type type: str
    :param args: Positional arguments passed to the model constructor.
    :param kwargs: Keyword arguments passed to the model constructor.
    :return:  Instantiated model object.
    :rtype: torch.nn.Module
    """
    logger.info("Start model load")
    if type.lower() not in REGISTRY:
        logger.error(f"Unknown model type: {type}")
        raise ValueError(f"Unknown model type: {type}")
    
    cls = REGISTRY[type.lower()]
    model = cls(*args, **kwargs)
    logger.info("Model loading complite")
    return model
