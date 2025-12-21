import uuid
from fastapi import APIRouter
from shared.logging import get_logger
from ..models import TrainingConfig
from task import training_model_clf

router = APIRouter()
logger = get_logger(__name__)


@router.post("/clf")
async def start_training(request: TrainingConfig):
    """
    Запуск обучения модели классификации изображений
    
    :param request: Конфигурации обучения
    :type request: TrainingConfig
    """
    
    train_model(request)

    return {"status": "ok"}

def train_model(config: TrainingConfig):
    """
    Training model
    
    :param config: Config
    :type config: TrainingConfig
    """
    try:
        config = config.model_dump()
        logger.debug(f"\nCONFIG:\n {config}")
        training_model_clf(
            data_loader_params=config['data_loader_params'],
            model_params=config['model_params'],
            trainer_params=config['trainer_params']
        )
    except Exception as e:
        logger.error(f"🔴 Error: {e}")
        raise