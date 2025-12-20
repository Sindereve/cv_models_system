import uuid
from fastapi import APIRouter
from shared.logging import get_logger
from ..models import TrainingConfig
from task import training_model_clf

router = APIRouter()
logger = get_logger(__name__)


@router.post("/clf")
async def start_training(request: TrainingConfig):
    """Проверка работоспособности сервиса"""
    task_id = str(uuid.uuid4())
    
    config = request.model_dump()
    logger.debug(f"\nCONFIG:\n {config}")
    
    training_model_clf(
        data_loader_params=config['data_loader_params'],
        model_params=config['model_params'],
        trainer_params=config['trainer_params']
    )

    logger.debug(f"Create task(id {task_id})")
    return {"status": "ok", "message": f"Task register in redis. (Id {task_id})"}