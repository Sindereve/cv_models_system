from fastapi import APIRouter
from .routers.cv_clf import router as router_cv_clf

router = APIRouter()

router.include_router(router_cv_clf, prefix="/api/train")

__all__ = ['router']