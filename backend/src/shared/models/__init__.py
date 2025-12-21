from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ModelParams(BaseModel):
    type: str = Field("Resnet", description="Тип модели (Resnet)")
    name: str = Field("resnet18", description="Название модели (resnet18, resnet50 и т.д.)")
    weights: bool = Field(False, description="Использовать предобученные веса")


class LossFnConfig(BaseModel):
    type: str = Field(
        "CrossEntropyLoss", 
        description="Тип функции потерь (например, CrossEntropyLoss)"
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"reduction": "mean"},
        description="Параметры функции потерь"
    )


class OptimizerConfig(BaseModel):
    type: str = Field(
        "Adam", 
        description="Тип оптимизатора (например, Adam)"
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "amsgrad": False
        },
        description="Параметры оптимизатора"
    )


class SchedulerConfig(BaseModel):
    type: str = Field(
        "OneCycleLR", 
        description="Тип планировщика (например, OneCycleLR)"
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            'max_lr': 0.01,
            'total_steps': 100,
            'pct_start': 0.3,
        },
        description="Параметры планировщика"
    )


class DataLoaderParams(BaseModel):
    path_data_dir: str = Field(..., description="Путь к папке с данными")
    img_w_size: int = Field(1080, description="Ширина изображений после преобразований")
    img_h_size: int = Field(720, description="Высота изображений после преобразований")
    total_img: int = Field(0, description="Общее количество изображений (0 - все изображения)")
    batch_size: int = Field(32, description="Размер батча")
    train_ratio: float = Field(0.75, description="Доля тренировочных данных")
    val_ratio: float = Field(0.15, description="Доля валидационных данных")
    is_calculate_normalize_dataset: bool = Field(False, description="Вычислять ли нормализацию датасета")


class TrainerParams(BaseModel):
    loss_fn_config: LossFnConfig = Field(
        default_factory=LossFnConfig,
        description="Конфигурация функции потерь"
    )
    optimizer_config: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Конфигурация оптимизатора"
    )
    scheduler_config: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Конфигурация планировщика"
    )
    device: Optional[str] = Field(
        default=None,
        description="Устройство для обучения (cpu/cuda)"
    )
    log_mlflow: bool = Field(
        default=True,
        description="Логировать в MLflow"
    )
    mlflow_uri: str = Field(
        default="http://mlflow:5000",
        description="URI MLflow сервера"
    )
    log_artifacts: bool = Field(
        default=False,
        description="Логировать артефакты"
    )
    log_checkpoint: bool = Field(
        default=False,
        description="Логировать чекпоинты"
    )
    experiment_name: str = Field(
        default="Exp_test",
        description="Название эксперимента в MLflow"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Название запуска в MLflow"
    )
    epochs: int = Field(
        default=30,
        description="Количество эпох обучения"
    )


class TrainingConfig(BaseModel):
    data_loader_params: DataLoaderParams
    model_params: ModelParams = Field(
        default_factory=ModelParams,
        description="Параметры модели"
    )
    trainer_params: TrainerParams = Field(
        default_factory=TrainerParams,
        description="Параметры тренера"
    )

class TaskLearning(BaseModel):
    task_id: str = Field(
        ...,
        description="Уникальный индефикатор задачи"
    )
    config: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Конфигурации обучения модели"
    )