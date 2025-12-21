import time
import json
import redis
from api.models import TrainingConfig
from api.routers.cv_clf import train_model
from shared.logging import get_logger

logger = get_logger(__name__)

redis_client = redis.Redis(
    host='redis',
    port=6379,
    decode_responses=True   
)

def start_work():
    while True:
        try:
            task_json = redis_client.lpop("ml:tasks")
            
            if task_json:
                task = json.loads(task_json)
                task_id = task.get('id', 'unknown')
                config = task.get('config', 'unknown')
                logger.info(f"🏍 Start task {task_id} \nConfig:\n{config}")
                training_config = TrainingConfig(**config)
                train_model(training_config)
            else:
                time.sleep(2)
                logger.info("Wait...")
        except json.JSONDecodeError:
            logger.error("🔴 Error format task")
            time.sleep(5)
        except Exception as e:
            logger.error(f"🔴 Error: {e}")
            raise
        