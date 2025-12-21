from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import multiprocessing
from api import router as api_router
from core.task_manager import start_work

app = FastAPI(
    title="cv_back_api",
    description="Api info for ML project"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # !!!!!!!!! В продакшене указать конкретный домен !!!!!!!!!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Service work!"}

@app.on_event("startup")
async def startup_event():
    """Запускаем воркер в отдельном процессе при старте"""
    process = multiprocessing.Process(target=start_work, daemon=True)
    process.start()
    print(f"🚀 Worker запущен в процессе PID: {process.pid}")

@app.on_event("shutdown")
async def shutdown_event():
    """Останавливаем воркер при завершении"""
    pass
    print("🛑 Останавливаем worker...")