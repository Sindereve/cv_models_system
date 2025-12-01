import mlflow
import logging
import sys
from mlflow.models.signature import infer_signature
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

import time
from typing import Optional, Dict

import os
# os.environ['MLFLOW_SUPPRESS_RUN_LOGS'] = 'true'

class Trainer:
    def __init__(
            self, 
            model: nn.Module,
            # data
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader = None,
            # settings for train model
            logger_lvl: str = 'debug',
            loss_fn: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[lr_scheduler._LRScheduler] = None,
            device: Optional[torch.device] = None,
            # mlflow tracking
            log_mlflow: bool = True,
            mlflow_uri: str = 'http://127.0.0.1:5000',
            log_artifacts: bool = True,
            experiment_name: str = "Experiment_name",
            run_name : Optional[str] = None,
            mlflow_tags: Optional[Dict[str, str]] = None,
        ):
        """
        Тренер для обучения, валидации и тестирования нейронных сетей.
        
        Args:
            model: Нейронная сеть для обучения
            
            train_loader: Данные для обучения
            val_loader: Данные для валидации
            test_loader: Данные для тестирования

            logger_lvl: Уровень логирования, один из 3 варинтов: 
                * 'info' - выводится информация об обучении модели
                * 'debug' - выводится вся информация о работе тренера.
                * 'warning' - выводятся только ошибки и предупреждения
                * 'error' - выводятся только ошибки
            loss_fn: Функция потерь
            optimizer: Оптимизатор
            scheduler: Планировщик learning rate
            device: Устройство вычислений GPU\\CPU

            log_mlflow: Флаг логирования в MLflow
            mlflow_uri: URI MLflow tracking server (локальный или удаленный)
            log_artifacts: Флаг логирование артефактов
            experiment_name: Имя эксперимента в MLflow(По умолчанию: "Experiment_name")
            run_name: Уникальное имя запуска в MLflow(По умолчанию имя задаётся вида 
                "{имя_модели}_{кол_эпох}_{скорость_схождения}_{Время}". Пример: "VGG_11_ep20_lr0.001_time(11:12_19:53:16)")
            mlflow_tags: Дополнительные теги для запуска
        """

        # logger load
        self.logger = self._setup_logger(logger_lvl)
        self.logger.debug("⚪ Start init")

        # model and setting learning
        self.model = model
        self.loss_fn = loss_fn 
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.device = device

        # data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # mlflow
        self.log_mlflow = log_mlflow
        self.mlflow_uri = mlflow_uri
        self.log_artifacts = log_artifacts
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.mlflow_tags = mlflow_tags

        self._validate_input()
        self._info_data()

        # device
        self._setup_device(device)
        self.model.to(self.device)

        # metrics
        self.history = {
            'train_loss': [], 
            'train_accuracy': [],
            'val_loss': [], 
            'val_accuracy': [],
            'learning_rate': []
        }

        self.logger.debug("🏁 Finish init")

    def _setup_logger(
            self, 
            logger_lvl: str
        ):
        """
        Настройка логера
        
        Args:
            logger_lvl: уровень логирования ('debug', 'info', 'warning', 'error')
        """
        logger = logging.getLogger(f"Trainer")
        
        logger.handlers.clear()

        if logger_lvl == 'debug':
            logger.setLevel(logging.DEBUG)
        elif logger_lvl == 'info':
            logger.setLevel(logging.INFO)
        elif logger_lvl == 'warning':
            logger.setLevel(logging.WARNING)
        elif logger_lvl == 'error':
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logger.level)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        logging.addLevelName(logging.INFO,    "💙 [ INFO  ]")
        logging.addLevelName(logging.WARNING, "💛 [WARNING]")
        logging.addLevelName(logging.ERROR,   "💔 [ ERROR ]")
        logging.addLevelName(logging.DEBUG,   "🔎 [ DEBUG ]")

        logger.debug(f"Logger build.")
        return logger

    def _info_data(self):
        self.logger.debug("|🔘 Print info data")
    
        batch, _ = next(iter(self.train_loader))
        img_shape = batch[0].size()
        self.logger.info(f" ➖ Image count color:   {img_shape[0]}")
        self.logger.info(f" ➖ Image size:          {img_shape[1:]} (H×W)")

        batch_size = len(batch)
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)

        self.logger.info(f" ➖ Batch size:          {batch_size}")
        self.logger.info(f" ➖ Train data sample:   {train_size}")
        self.logger.info(f" ➖ Validate data sample:{val_size}")
        if self.test_loader is not None:
            test_size = len(self.test_loader.dataset)
            self.logger.info(f" ➖ Test data sample:   {test_size}")
        else:
            self.logger.info(" ➖ Test data sample:    Not used")
            self.logger.warning(" Model don`t testing for test data! (test_loader is None value)")
        self.logger.debug("|🏁 Finish print info for data")

    def _validate_input(self):
        """
        Валидация входных данных
        """
        self.logger.debug("|🔘 Start input value validation")

        cheks = [
            (self.model, nn.Module, "model"),
            (self.train_loader, DataLoader, "train_loader"),
            (self.val_loader, DataLoader, "val_loader"),
        ]

        for obj, type, name in cheks:
            if not isinstance(obj, type):
                self.logger.error(f"|└🔴 {name} is not {type}. Type value is {type(obj)}")
                raise TypeError(f"{name} must be {type}")
            
            if type == DataLoader:
                try:
                    next(iter(obj.dataset))
                except StopIteration:
                    self.logger.error(f"|└🔴 {name}({type}) is empty.")
                    raise StopIteration(f"{name}({type}) is empty.")
            
            self.logger.debug(f"|├🟢 {name}: OK")

        check_and_adjust = [
            (self.test_loader, DataLoader, "test_loader", None),
            (self.loss_fn, nn.Module, "loss_fn", nn.CrossEntropyLoss()),
            (self.device, torch.device, "device", None),
        ]

        for obj, type, name, new_val in check_and_adjust :
            if not isinstance(obj, type):
                self.logger.warning(f"🟠 {name} is not {type}.")
                setattr(self, name, new_val)
                self.logger.debug(f"|├🟢 {name} change in default value. ({new_val})")
            else:
                self.logger.debug(f"|├🟢 {name}: OK")

        # optimizer
        if not isinstance(self.optimizer, Optimizer):
            self.logger.warning(f"🟠 optimizer is not {Optimizer}. Change in default value({optim.Adam})")
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.logger.debug(f"|├🟢 optimizer change in default value. (learning_rate = 0.001, {optim.Adam})")
        else:
            self.logger.debug(f"|├🟢 optimizer: OK")

        if not isinstance(self.scheduler, lr_scheduler._LRScheduler):
            self.logger.warning(f"🟠 scheduler is not {lr_scheduler._LRScheduler}. Change in default value({lr_scheduler.CosineAnnealingLR})")
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
            self.logger.debug(f"|├🟢 scheduler change in default value. ({lr_scheduler.CosineAnnealingLR})")
        else:
            self.logger.debug(f"|├🟢 scheduler: OK")

        # mlflow test connect
        self._mlflow_test_connect()

        self.logger.debug("|└🏁 finish validating params")

    def _setup_device(self, device: Optional[torch.device] = None):
        """
        Настройка используемого "аппарата" обучения
        """
        self.logger.debug("|🔘 Start setting device")

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                self.logger.warning("🟠 error load 'CUDA'. Using 'CPU'")
                self.device = torch.device('cpu')
            else:
                # clear cache in cuda
                torch.cuda.empty_cache()
                gpu_info = torch.cuda.get_device_name(self.device)
                self.logger.debug(f"||🟡 GPU: {gpu_info}")

        self.logger.info(f"Training on: {self.device}")
        self.logger.debug(f"|└🟢Training on: {self.device}")

    def _mlflow_test_connect(self):
        """
        Тестовое подключение к серверу mlflow
        """
        if not self.log_mlflow:
            self.logger.debug("|🟢 MLflow tracking: OFF")
            return
        
        try:
            self.logger.debug("||🔘 Test connection for MLflow. ")
            mlflow.set_tracking_uri(self.mlflow_uri)

            _ = mlflow.search_experiments()
            self.logger.debug(f"||└🟢 Connected to MLflow at {self.mlflow_uri}")
        except Exception as e:
            self.logger.error(f"||└🔴MLflow server at {self.mlflow_uri} not available. Using local tracking.")
            mlflow.set_tracking_uri(None)

    def _setup_mlflow(
            self,
            epoch: int,
            lr: int
        ):
        """
        Настройка MLflow с предварительной проверкой сервера
        """
        if not self.log_mlflow:
            self.logger.debug("Log in MLflow: OFF")
            return

        try:
            self.logger.debug("Log in MLflow: ON")
            
            # Настройка MLflow
            mlflow.set_tracking_uri('http://127.0.0.1:5000')
            mlflow.set_experiment(self.experiment_name)

            if self.run_name is None:
                time_str = time.strftime('%m:%d_%H:%M:%S')
                self.run_name = f"{self.model.__class__.__name__}_ep{epoch}_lr{lr}_time({time_str})"

            print(f"🔵[MLFlow] Starting run: {self.run_name}")
            try:
                self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            except Exception as e:
                mlflow.end_run()
                self.mlflow_run = mlflow.start_run(run_name=self.run_name)
                print(f"🔵[MLFlow] Stop old run_name started successfully: {self.mlflow_run.info.run_id}")

            print(f"🔵[MLFlow] Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"🔵[MLFlow] Artifact URI: {mlflow.get_artifact_uri()}")
            print(f"🟢[MLFlow] Run started successfully: {self.mlflow_run.info.run_id}")
            
        except Exception as e:
            print(f"🔴[MLFlow] Setup failed: {e}")
            self.log_mlflow = False
            try:
                mlflow.end_run()
            except:
                pass

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


    def _log_model_checkpoint(self, epoch: int):
        """
        Логирование чекпоинтов модели
        """
        if not self.log_mlflow or not self.log_artifacts:
            return
            
        try:
            name = f"checkpoint_epoch_{epoch}"
            mlflow.pytorch.log_model(
                self.model,
                name=name,
                signature= self._create_mlflow_signature()
            )
            print(f"🔵[MLFlow] log model ({name})")
        except Exception as e:
            print(f"🔴[MLFlow] Error logging model: {e}")

    def _create_mlflow_signature(self):
        sample_batch = next(iter(self.train_loader))

        sample_inputs = sample_batch[0][:5]
        sample_targets = sample_batch[1][:5]

        return infer_signature(
            model_input=sample_inputs.numpy(),
            model_output=sample_targets.numpy()
        )

    def _log_training_artifacts(self):
        """
        Логирование дополнительных артефактов
        """
        if not self.log_mlflow or not self.log_artifacts:
            return
            
        try:
            import matplotlib.pyplot as plt
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
        best_val_acc = 0.0

        if self.log_mlflow:
            self._setup_mlflow(epochs, self.optimizer.param_groups[0]['lr'])
            self._log_model_parameters()

        for epoch in range(epochs):
            print("="*50)
            print(f"🔄 Epoch[🔹{epoch+1}/{epochs}🔹] start")
            self._train_one_epoch()
            self._validate_one()
            
            self._log_epoch_metric(epoch+1)

            if best_val_acc < self.history['val_accuracy'][-1]:
                best_val_acc = self.history['val_accuracy'][-1]
                self._log_model_checkpoint(epoch + 1)

            print(f"🟢 Epoch[🔹{epoch+1}/{epochs}🔹] completed")

        # Логируем все артефакты
        self._log_training_artifacts()

        if self.log_mlflow:
            mlflow.end_run()

        print("🟢[train] Completed!!!")