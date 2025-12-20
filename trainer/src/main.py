from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router

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