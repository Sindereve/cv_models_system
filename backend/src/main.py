from fastapi import FastAPI, HTTPException
from typing import Optional
import redis
import json
import uuid
from datetime import datetime
from shared.models import TaskLearning, TrainingConfig
from shared.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="ML Training Backend", 
    version="1.0.0"
)

redis_client = redis.Redis(
    host='redis',
    port=6379
)


@app.get("/")
async def root():
    return {"message": "ML Training Backend Service"}

@app.post("/new_task")
async def create_training_job(config: TrainingConfig):
    """Создание нового задания на обучение"""
    
    # Генерация ID задания
    task_id = str(uuid.uuid4())
    
    # Создание объекта задания
    task = TaskLearning(
        task_id=task_id,
        config=config,
    )
    
    try:
        # Добавляем в очередь заданий
        redis_client.lpush("ml:tasks:pending", json.dumps(task.dict()))
        logger.info(f"Created training task {task_id}")

        return {
            "status": "ok",
            "task_id": task_id,
            "message": f"Task {task_id} added to pending queue"
        }
        
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Получение статуса задачи по ID"""
    
    try:
        # Ищем задачу во всех очередях
        queues = [
            "ml:tasks:pending",
            "ml:tasks:processing", 
            "ml:tasks:completed",
            "ml:tasks:failed"
        ]
        
        for queue in queues:
            # Получаем все задачи из очереди
            tasks = redis_client.lrange(queue, 0, -1)
            
            for task_bytes in tasks:
                try:
                    task = json.loads(task_bytes.decode('utf-8'))
                    if task.get('task_id') == task_id:
                        # Определяем статус по имени очереди
                        status = queue.split(':')[-1]
                        task['status'] = status
                        task['queue'] = queue
                        return task
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    continue
        
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/tasks")
async def get_all_tasks(status: Optional[str] = None):
    """Получение всех задач с фильтром по статусу"""
    
    try:
        # Соответствие очередей и статусов
        queue_status_map = {
            "ml:tasks:pending": "pending",
            "ml:tasks:processing": "processing",
            "ml:tasks:completed": "completed", 
            "ml:tasks:failed": "failed"
        }
        
        all_tasks = []
        
        for queue, status_name in queue_status_map.items():
            if status and status != status_name:
                continue
                
            tasks = redis_client.lrange(queue, 0, -1)
            
            for task_bytes in tasks:
                try:
                    task = json.loads(task_bytes.decode('utf-8'))
                    task['status'] = status_name
                    task['queue'] = queue
                    all_tasks.append(task)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    continue
        
        # Сортируем по времени создания (новые первыми)
        all_tasks.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        
        return {
            "status": "ok",
            "count": len(all_tasks),
            "tasks": all_tasks
        }
        
    except Exception as e:
        logger.error(f"Error getting all tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queues/stats")
async def get_queue_stats():
    """Получение статистики по всем очередям"""
    
    try:
        queues = [
            "ml:tasks:pending",
            "ml:tasks:processing",
            "ml:tasks:completed",
            "ml:tasks:failed"
        ]
        
        stats = {}
        total_tasks = 0
        
        for queue in queues:
            count = redis_client.llen(queue)
            stats[queue] = count
            total_tasks += count
        
        return {
            "status": "ok",
            "total_tasks": total_tasks,
            "queue_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Удаление задачи из любой очереди"""
    
    try:
        queues = [
            "ml:tasks:pending",
            "ml:tasks:processing",
            "ml:tasks:completed",
            "ml:tasks:failed"
        ]
        
        deleted = False
        
        for queue in queues:
            tasks = redis_client.lrange(queue, 0, -1)
            
            for task_bytes in tasks:
                try:
                    task = json.loads(task_bytes.decode('utf-8'))
                    
                    if task.get('task_id') == task_id:
                        # Удаляем задачу из очереди
                        redis_client.lrem(queue, 0, task_bytes)
                        deleted = True
                        break
                        
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    continue
            
            if deleted:
                break
        
        if deleted:
            logger.info(f"Deleted task {task_id}")
            return {
                "status": "ok",
                "message": f"Task {task_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error deleting task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    
    try:
        # Проверяем соединение с Redis
        redis_client.ping()
        
        return {
            "status": "healthy",
            "service": "ml-backend",
            "redis": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail={
                "status": "unhealthy",
                "service": "ml-backend",
                "redis": "disconnected",
                "error": str(e)
            }
        )
