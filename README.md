# ML Models Portfolio

## Описание

Личное портфолио с реализацией моделей машинного обучения в области компьютерного зрения (классификация, детекция, сегментация). 
Каждая модель включает код, конфигурацию и результаты обучения. 
Все эксперименты логируются с помощью MLflow (параметры, метрики, артефакты), а анализ 
проводится в Jupyter-ноутбуках. 

## Структура проекта:

```
cv_models_system/
├── data/                    # ← .gitignore
│                            # Там буду храниться все датасеты
├── models/
│   ├── classification/
│   │   ├── resnet/
│   │   │   ├── train.py
│   │   │   ├── model.py
│   │   │   └── config.yaml   # параметры для обучения модели
│   │   └── vgg_11/
│   ├── detection/
│   └── segmentation/
│
├── mlflow/
│   └── ...
│
├── notebooks/
│   ├── resnet_analysis.ipynb
│   └── comparison.ipynb
│
├── utils/
│   └── data_download.py       # функции для скачивания датасетов
│
├── scripts/
│   ├── run_all.sh             # Скрипт для обучения всех моделей сразу
│   └── serve_mlflow.sh        # Запуск MlfLow
│
├── .gitignore
├── README.md
└── requirements.txt
```
