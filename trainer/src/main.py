from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import multiprocessing
from api import router as api_router
from core.task_manager import start_work
from shared.logging import get_logger

logger = get_logger(__name__)
process: multiprocessing.Process | None = None

async def lifespan(app: FastAPI):
    global process
    process = multiprocessing.Process(target=start_work)
    process.start()
    logger.info(f"🚀 Worker started PID={process.pid}")

    yield

    logger.info("🔴 Stopping worker...")
    process.terminate()
    process.join()

app = FastAPI(
    title="cv_trainer_api",
    description="Api info for project",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
async def root():
    """Get status service"""
    return {"status": "ok", "message": "Service work!"}