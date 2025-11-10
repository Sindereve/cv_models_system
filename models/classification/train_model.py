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
            log_artifacts: bool = True,
            experiment_name: str = "Experiment_name",
            run_name : Optional[str] = None,
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
            log_artifacts: Логирование артефактов
            experiment_name: Имя эксперимента в MLflow
            run_name: Уникальное имя запуска в MLflow
        """
        self._validate_input(model, train_loader, val_loader)
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

        # metrics
        self.history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'learning_rate': []
        }
        self.best_weights = None
        self.best_accuracy = 0.0
        
        # mlflow
        self.log_mlflow = log_mlflow
        self.log_artifacts = log_artifacts
        self._setup_mlflow(log_mlflow, experiment_name, run_name)

        print("🟢 Finish init")

    def _validate_input(
            self, 
            model: nn.Module, 
            train_loader: DataLoader, 
            val_loader: DataLoader
        ):
        """
        Валидация входных данных
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module")
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader must be DataLoader")
        if not isinstance(val_loader, DataLoader):
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
        

    def _setup_mlflow(
            self,
            log_mlflow: bool,
            experiment_name: str,
            run_name: str,
        ):
        """
        Настройка MLFlow эксперимента с обработкой ошибок
        """
        if not log_mlflow:
            print(" ➖ log in Mlflow: OFF")
            return

        try:
            self.run_name = run_name
            self.experiment_name = experiment_name

            # Пытаемся установить эксперимент, если не получается - создаем новый
            try:
                mlflow.set_experiment(self.experiment_name)
            except:
                # Если эксперимент удален, создаем новый с временной меткой
                self.experiment_name = f"{experiment_name}_new_{int(time.time())}"
                mlflow.create_experiment(self.experiment_name)
                mlflow.set_experiment(self.experiment_name)
                print(f"🔵[MLFlow] Created new experiment: {self.experiment_name}")

            if self.run_name is None:
                time_str = time.strftime('%Y:%m:%d_%H:%M:%S')
                self.run_name = f"{self.model.__class__.__name__}_{time_str}"

            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            self._log_model_parameters()
            print(" ➖ log in Mlflow: On")
            
        except Exception as e:
            print("🔴[MLFlow] Error setting:", e)
            self.log_mlflow = False
            try:
                mlflow.end_run()
            except:
                pass
            print("🟠[MLFlow] Continuing without MLflow logging")

    def _log_model_parameters(self):
        """
        Логирование параметров модели и обучения
        """
        try: 
            model_params = {
                'model_type': self.model.__class__.__name__,
                'device': self.device.type,
                'total_parameters': sum([p.numel() for p in self.model.parameters()])
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
            
            all_params = {
                **model_params, 
                **data_params,
                **optiizer_params, 
                
            }
            mlflow.log_params(all_params)
            print('🔵[MLFlow] parameters model add in MLFlow')
        except Exception as e:
            print("🔴[MLFlow] Error set params model:", e)
            raise


    def _log_epoch_metric(
            self, 
            epoch: int
        ):
        """
        Логирование метрик эпохи в MLflow
        """
        if not self.log_mlflow:
            return
        try:
            metrics = {
                'train_loss': self.history['train_loss'][-1],
                'train_accuracy': self.history['train_accuracy'][-1],
                'val_loss': self.history['val_loss'][-1],
                'val_accuracy': self.history['val_accuracy'][-1],
                'learning_rate': self.history['learning_rate'][-1],
                'epoch': epoch
            }

            mlflow.log_metrics(metrics, step=epoch)

        except Exception as e:
            print("🔴[MLFlow] Error set params model:", e)


    def _log_model_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Логирование чекпоинтов модели
        """
        if not self.log_mlflow or not self.log_artifacts:
            return
            
        try:
            # Сохраняем модель
            model_info = mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name=f"{self.model.__class__.__name__}",
            )
            
            # Если это лучшая модель, логируем отдельно
            if is_best:
                mlflow.pytorch.log_model(
                    self.model,
                    "best_model",
                    registered_model_name=f"{self.model.__class__.__name__}_best",
                )
                mlflow.log_metric("best_val_accuracy", self.history['val_accuracy'][-1])
                
        except Exception as e:
            print(f"🔴[MLFlow] Error logging model: {e}")

    def _log_training_artifacts(self):
        """
        Логирование дополнительных артефактов
        """
        if not self.log_mlflow or not self.log_artifacts:
            return
            
        try:
            import matplotlib.pyplot as plt
            import os
            import tempfile
            
            # Создаем временную директорию для артефактов
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # График потерь
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(self.history['train_loss'], label='Train Loss')
                plt.plot(self.history['val_loss'], label='Val Loss')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(self.history['train_accuracy'], label='Train Accuracy')
                plt.plot(self.history['val_accuracy'], label='Val Accuracy')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                loss_plot_path = os.path.join(temp_dir, 'training_metrics.png')
                plt.savefig(loss_plot_path)
                plt.close()
                
                mlflow.log_artifact(loss_plot_path)
                
                # Логируем историю обучения в файл
                history_path = os.path.join(temp_dir, 'training_history.txt')
                with open(history_path, 'w') as f:
                    f.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\tLR\n")
                    for i in range(len(self.history['train_loss'])):
                        f.write(f"{i+1}\t{self.history['train_loss'][i]:.4f}\t"
                               f"{self.history['train_accuracy'][i]:.4f}\t"
                               f"{self.history['val_loss'][i]:.4f}\t"
                               f"{self.history['val_accuracy'][i]:.4f}\t"
                               f"{self.history['learning_rate'][i]:.6f}\n")
                
                mlflow.log_artifact(history_path)
                
        except Exception as e:
            print(f"🔴[MLFlow] Error logging artifacts: {e}")


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
            
            self._log_epoch_metric(epoch+1)
            is_best = self.history['val_accuracy'][-1] == self.best_accuracy
            if is_best:
                self._log_model_checkpoint(epoch + 1, is_best=True)

            print(f"✅ Epoch[🔹{epoch+1}/{epochs}🔹] completed")

        # Логируем все артефакты
        self._log_training_artifacts()

        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

        if self.log_mlflow:
            mlflow.end_run()

        print("🟢[train] Completed!!!")