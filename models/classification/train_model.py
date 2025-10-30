import mlflow

import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

import time
from typing import Dict, Callable

class BaseTrainer:
    def __init__(
            self, 
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module = None,
            optimizer: Optimizer = None,
            scheduler: lr_scheduler._LRScheduler = None,
            device: torch.device = None,
            metrics: Dict[str, Callable] = None,
            # next arg mlflow module
            experiment_name: str = "No_name",
            run_name: str = None,
            log_mlflow: bool = True
        ):
        """
        
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_mlflow = log_mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name

        # device
        self.device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # loss and optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler

        self.history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'learning_rate': []
        }

        self.best_weights = None
        self.best_accuracy = 0.0
        if self.log_mlflow:
            self._setup_mlflow()

        print("Training on:", self.device)
        print("Train sample:", len(self.train_loader.dataset))
        print("Val sample:", len(self.val_loader.dataset))

    def _setup_mlflow(self):
        """
        Настройка MLFlow эксперемента
        """
        try:
            mlflow.set_experiment(self.experiment_name)
            
            if self.run_name is None:
                time_str = time.strftime('%Y%m%d_%H%M%S')
                self.run_name = f"{self.model.__class__.__name__}_{time_str}"

            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            self._log_model_parameters()
            
        except Exception as e:
            print("🔴 Error seting MLFlow:", e)
            self.log_mlflow = False

    def _log_model_parameters(self):
        """
        Логирование параметров модели и обучения
        """
        
        model_params = {
            'model_type': self.model.__class__.__name__,
            'model_parameters': sum(p.numel for p in self.model.parameters()),
            'device': self.device
        }

        optiizer_params = {
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        for key, value in self.optimizer.param_groups[0].items():
            if key != 'params':
                optiizer_params[f'optimizer_{key}'] = value

        data_params = {
            'train_sample': len(self.train_loader.dataset),
            'val_sample': len(self.val_loader.dataset),
            'batch_size': self.train_loader.batch_size,
            'num_classes': getattr(self.train_loader, 'num_classes', 'unknown')
        }

        all_params = {**model_params, **optiizer_params, **data_params}
        mlflow.log_params(all_params)


    def train_epoch(self):
        pass

    def validate(self):
        pass
