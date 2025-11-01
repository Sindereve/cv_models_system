import mlflow
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

import time
from typing import Optional

class BaseTrainer:
    def __init__(
            self, 
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_fn: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[lr_scheduler._LRScheduler] = None,
            device: Optional[torch.device] = None,
            # next arg mlflow module
            log_mlflow: bool = True,
            experiment_name: str = "No_name",
            run_name: Optional[str] = None,
        ):
        """
        Инициализация тренера модели
        
        Args:
            model: Нейронная сеть для обучения
            train_loader: Данные для обучения
            val_loader: Данные для валидации
            loss_fn: Функция потерь
            optimizer: Оптимизатор
            scheduler: Планировщик learning rate (optional)
            device: Устройство вычислений GPU\CPU
            log_mlflow: Флаг логирования в MLflow
            experiment_name: Имя эксперимента в MLflow
            run_name: Уникальное имя запуска в MLflow
        """
        self._validate_input()
        print("⚪ Start init")
        
        self.model = model
        self.train_loader = train_loader
        print(" ➖ Train load sample:", len(self.train_loader.dataset))
        self.val_loader = val_loader
        print(" ➖ Val load sample:  ", len(self.val_loader.dataset))

        # device
        self._setup_device(device)
        self.model.to(self.device)

        # loss and optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler or lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

        self._create_history()

        self._setup_mlflow(log_mlflow, experiment_name, run_name)

        print("🟢 Finish init")

    def _validate_input(self):
        """
        Валидация входных данных
        """
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be nn.Module")
        if not isinstance(self.train_loader, DataLoader):
            raise TypeError("train_loader must be DataLoader")
        if not isinstance(self.val_loader, DataLoader):
            raise TypeError("val_loader must be DataLoader")

    def _setup_device(self, device: Optional[torch.device] = None):
        """
        Настройка используемого памяти для обучения
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print("🟠 Внимание: ошибка использования 'CUDA', используется 'CPU'")
            self.device = torch.device('cpu')
        torch.cuda.empty_cache()
        print(" ➖ Training on:", self.device)

    def _create_history(self):
        """
        Создаём историю обучения модели
        """
        self.history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'learning_rate': []
        }
        self.best_weights = None
        self.best_accuracy = 0.0

    def _setup_mlflow(
            self,
            log_mlflow: bool,
            experiment_name: str,
            run_name: str,
        ):
        """
        Настройка MLFlow эксперемента

        Args:
            log_mlflow: MLflow вкл/выкл
            experiment_name: Имя эксперемента
            run_name: Уникальное имя запуска
        """
        if not log_mlflow:
            print(" ➖ log in Mlflow: OFF")
            return

        try:
            self.run_name = run_name
            self.experiment_name = experiment_name

            mlflow.set_experiment(self.experiment_name)

            if self.run_name is None:
                time_str = time.strftime('%Y%m%d_%H%M%S')
                self.run_name = f"{self.model.__class__.__name__}_{time_str}"

            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            self._log_model_parameters()
            print(" ➖ log in Mlflow: On")
        except Exception as e:
            print("🔴[MLFlow] Error seting:", e)
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
        print('🔵[MLFlow] parameters model add in MLFlow')

    def _train_one_epoch(
            self,
        ):
        """
        Проход по тренировочным данным и тренировка на них
        """

        self.model.train()

        runner_loss = 0.0
        correct_predictions = 0
        total_sample = 0

        for data in self._tqdm_loader(self.train_loader, "Training"):
            inputs, labels = data
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # front steps
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            
            # back steps
            loss.backward()
            self.optimizer.step()

            runner_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_sample += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # cuda opyat ushla vsya pamyat'
            del inputs, labels, outputs, loss
        
        self.scheduler.step()

        epoch_loss = runner_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_sample
        lr = self.optimizer.param_groups[0]['lr']

        self.history['train_loss'].append(epoch_loss)
        self.history['train_accuracy'].append(epoch_accuracy)
        self.history['learning_rate'].append(lr)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Epoch Result:")
        print(f" ➖ Train Loss: {epoch_loss:.4f}")
        print(f" ➖ Train Acc:  {epoch_accuracy:.4f}")
        print(f" ➖ LR:         {lr:.6f}")

    def _tqdm_loader(
            self,
            data_loader: DataLoader,
            desc: str = "process"
        ):
        """
        Быстрая настройка для красивого бара загрузки
        """
        return tqdm(
            data_loader,
            desc=desc,
            bar_format="{l_bar}{bar:20}{r_bar}",
            colour="blue",
            leave=False
        )

    def _validate_one(
            self
        ) -> None:
        """
        1 проход по валидационным данным
        """
        self.model.eval()

        runner_loss = 0.0
        correct_predictions = 0
        total_sample = 0

        with torch.no_grad():
            for data in self._tqdm_loader(self.val_loader, "Validating"):
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                runner_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_sample += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # cude opyat ushla vsya pamyat'
                del inputs, outputs, labels, loss

        avg_loss = runner_loss / (len(self.val_loader))
        accuracy = correct_predictions / total_sample

        self.history["val_loss"].append(avg_loss)
        self.history["val_accuracy"].append(accuracy)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Validat:")
        print(f" ➖ Val Loss: {avg_loss:.4f}")
        print(f" ➖ Val Acc:  {accuracy:.4f}")

    def train(
            self,
            epochs: int = 20,
        ) -> nn.Module:
        """
        Полный цикл тренировки
        
        Args:
            epoch: количество эпох для тренировки
        """

        print("🔘[train] Start")

        for epoch in range(epochs):
            print("="*50)
            print(f"🔄 Epoch[🔹{epoch+1}/{epochs}🔹] start")
            self._train_one_epoch()
            self._validate_one()
            
        print("🟢[train] Completed!!!")