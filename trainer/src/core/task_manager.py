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
    logger.info('✨Searcher task✨ is RUN')
    while True:
        try:
            task_json = redis_client.lpop("ml:tasks:pending")
            
            if not task_json:
                time.sleep(2)
                continue
            
            task = json.loads(task_json)
            task_in_proc(task)

            task_id = task.get('id', 'unknown')
            config = task.get('config', 'unknown')

            logger.info(f"🏍 Start task {task_id} \nConfig:\n{config}")
            training_config = TrainingConfig(**config)
            train_model(training_config)
            task_in_completed(task)

        except Exception as e:
            logger.error(f"🔴 Error: {e}")
            task['status'] = 'failed'
            task['error'] = str(e)
            task['completed_at'] = time.time()
            task_in_failed(task)        
            raise


def task_in_proc(task):
    redis_client.lpush("ml:tasks:processing", json.dumps(task))
    redis_client.lrem("ml:tasks:pending", 0, json.dumps(task))

def task_in_completed(task):
    redis_client.lrem("ml:tasks:processing", 0, json.dumps(task))
    redis_client.lpush("ml:tasks:completed", json.dumps(task))

def task_in_failed(task):
    redis_client.lrem("ml:tasks:processing", 0, json.dumps(task))
    redis_client.lpush("ml:tasks:failed", json.dumps(task))
        