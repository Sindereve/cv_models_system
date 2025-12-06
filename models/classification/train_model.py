import mlflow
import logging
import sys
from contextlib import contextmanager
from mlflow.models.signature import infer_signature
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy, Recall, Precision, F1Score
)
from torchmetrics import MeanMetric

import time
from typing import Optional, Dict

import os
os.environ['MLFLOW_SUPPRESS_RUN_LOGS'] = 'true'

class Trainer:
    def __init__(
            self, 
            model: nn.Module,
            # data
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader = None,
            classes: Optional[Dict] = None,
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
            classes: Классы в данных(словарь[значение: индекс класса])

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
            mlflow_uri: URI MLflow tracking server (локальный или удаленный, !! HTTP !!)
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
        self.classes = classes
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
        self.train_loader_size, self.val_loader_size, self.test_loader_size = self._get_size_datasets()

        # device
        self._setup_device(device)
        self.model.to(self.device)

        self.train_metrics = self.create_classification_metrics(
            preset = 'minimal',
            prefix = 'train_',
        )

        self.val_metrics = self.create_classification_metrics(
            preset = 'full',
            prefix = 'val_',
        )

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

    def _get_size_datasets(self):
        self.logger.debug("├🔘 Calculate size data")
    
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
            test_size = None
            self.logger.info(" ➖ Test data sample:    Not used")
            self.logger.warning(" Model don`t testing for test data! (test_loader is None value)")
        self.logger.debug("|└🏁 Finish calculate info for data")
        return train_size, val_size, test_size 

    def _validate_input(self):
        """
        Валидация входных данных
        """
        self.logger.debug("├🔘 Start input value validation")

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
        self.logger.debug("├🔘 Start setting device")

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
            self.logger.debug("|├🔘 Test connection for MLflow. ")
            mlflow.set_tracking_uri(self.mlflow_uri)

            _ = mlflow.search_experiments()
            self.logger.debug(f"||├🟢 Connected to MLflow at {self.mlflow_uri}")
        except Exception as e:
            self.logger.error(f"||└🔴MLflow server at {self.mlflow_uri} not available. Using local tracking.")
            mlflow.set_tracking_uri(None)

    @contextmanager
    def mlflow_run_manager(self):
        """
        Контекстный менеджер для работы с MLflow runs.
        """
        if not self.log_mlflow:
            yield
            return
        
        run = None
        try:
            run = mlflow.start_run(run_name=self.run_name) 
            self.mlflow_run = run
            
            yield
            
            if run:
                mlflow.end_run(status="FINISHED")
                self.logger.debug("└🏁 MLflow run finished successfully")
                
        except Exception as e:
            if run:
                mlflow.end_run(status="FAILED")
                self.logger.error(f"🔴 MLflow run failed: {e}")

            raise
        finally:
            self._ensure_run_closed(run)

    def _ensure_run_closed(self, run):
        """Гарантированное закрытие run"""
        try:
            self.logger.debug("🔘 Start close run in mlflow")
            active_run = mlflow.active_run()
            if active_run:
                if run and active_run.info.run_id == run.info.run_id:
                    mlflow.end_run(status="KILLED")
                elif not run:
                    mlflow.end_run(status="KILLED")
                self.logger.warning("🟠 Force-closed MLflow run")
            self.logger.debug("└🏁 Finish close run in mlflow")
        except:
            pass

    def _setup_mlflow(
            self,
            epoch: int,
            lr: int
        ):
        """
        Настройка MLflow
        """
        if not self.log_mlflow:
            self.logger.debug("🟢 Tracking in MLflow: OFF")
            return

        try:
            self.logger.debug("|🔘 START MLflow setting")
            
            mlflow.set_experiment(self.experiment_name)

            if self.run_name is None:
                time_str = time.strftime('%m:%d_%H:%M:%S')
                self.run_name = f"{self.model.__class__.__name__}_ep{epoch}_lr{lr}_time({time_str})"

            self.logger.debug(f"|├🟢 run name {self.run_name}")
            self.logger.debug(f"|├🟢 run name {self.run_name}")
            self.logger.debug("|└🏁 FINISH MLflow setting")
        except Exception as e:
            self.logger.error(f"🔴 MLflow setup failed: {e}")
            self.logger.warning("🟠 No use tracking MLflow")
            self.log_mlflow = False

    def _mlflow_log_parameters(
            self,
            epochs: int,
        ):
        """
        Логирование параметров модели и обучения
        """
        try: 
            model_params = {
                'model_type': self.model.__class__.__name__,
                'device': self.device.type,
                'model_total_parameters': sum([p.numel() for p in self.model.parameters()]),
                'model_trainable_parameters': sum([p.numel() for p in self.model.parameters() if p.requires_grad]),
            }

            # Optim
            optimizer_params = {
                'optimizer': self.optimizer.__class__.__name__,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            for key, value in self.optimizer.param_groups[0].items():
                if key != 'params':
                    optimizer_params[f'optimizer_{key}'] = value
            
            data_params = {
                'data_train_sample': self.train_loader_size,
                'data_val_sample': self.val_loader_size,
                'data_test_sample': self.test_loader_size if self.test_loader_size else "Unknown",
                'batch_size': self.train_loader.batch_size,
                'classes_count': len(self.classes),
            }

            training_params = {
                'epochs': epochs,
            }

            if hasattr(self, 'sheduler') and self.scheduler:
                scheduler_params = {
                    'scheduler': self.scheduler.__class__.__name__,
                }
                if hasattr(self.scheduler, 'step_size'):
                    scheduler_params['scheduler_step_size'] = self.scheduler.step_size
                if hasattr(self.scheduler, 'gamma'):
                    scheduler_params['scheduler_gamma'] = self.scheduler.gamma
                if hasattr(self.scheduler, 'T_0'):  # CosineAnnealingWarmRestarts
                    scheduler_params['scheduler_T_0'] = self.scheduler.T_0

            if hasattr(self, 'loss_fn'):
                loss_fn_params = {
                    'loss_fn': self.loss_fn.__class__.__name__,
                }
                for attr in ['weight', 'size_average', 'reduce', 'reduction', 'ignore_index']:
                    if hasattr(self.loss_fn, attr):
                        value = getattr(self.loss_fn, attr)
                        if torch.is_tensor(value):
                            value = value.to_list()
                        loss_fn_params[f'loss_fn_{attr}'] = str(value)
                training_params.update(loss_fn_params)
            
            all_params = {
                **model_params, 
                **data_params,
                **optimizer_params,
                **training_params
            }
            mlflow.log_params(all_params)
            self.logger.debug('|🟢Parameters model add in MLFlow')
        except Exception as e:
            self.logger.error("Error set all params in mlflow:", e)
            raise

    def create_classification_metrics(
            self,
            preset: str = 'full',
            prefix: str = '',
        ) -> MetricCollection:
        """
        Создание коллекции метрик для задачи классификации
        
        Args:
            preset: 'minimal', 'standard', 'full'
            prefix: префикс для имени метрик
        Returns:
            MetricCollection с настроенными метриками
        """

        num_classes = len(self.classes)
        
        # Почему я отключил синхранизацию?
        # Потому что в моём случае обучение происходит не распределённо
        sync_on_compute = False

        PRESETS= {
            'minimal': {
                'accuracy': Accuracy(
                    task='multiclass', 
                    num_classes=num_classes,
                    sync_on_compute=sync_on_compute
                ),
            },
            'standard': {
                'accuracy': Accuracy(
                    task='multiclass', 
                    num_classes=num_classes,
                    sync_on_compute=sync_on_compute
                ),
                'precision': Precision(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'recall': Recall(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'f1': F1Score(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
            },
            'full': {
                'accuracy': Accuracy(
                    task='multiclass', 
                    num_classes=num_classes,
                    sync_on_compute=sync_on_compute
                ),
                'precision_macro': Precision(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'precision_micro': Precision(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='micro',
                    sync_on_compute=sync_on_compute
                ),
                'recall_macro': Recall(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'recall_micro': Recall(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='micro',
                    sync_on_compute=sync_on_compute
                ),
                'f1_macro': F1Score(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'f1_micro': F1Score(
                    task='multiclass', 
                    num_classes=num_classes,
                    average='micro',
                    sync_on_compute=sync_on_compute
                ),
            },
        }
        
        if preset not in PRESETS:
            preset_available = list(PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {preset_available}")
        
        metrics_dict = PRESETS[preset]
        
        metrics_dict['loss'] = MeanMetric(
            sync_on_compute=sync_on_compute,
            nan_strategy='ignore'
        )
        
        collection = MetricCollection(
            metrics_dict,
            prefix=prefix,
        ).to(self.device)
        
        return collection

    def _log_epoch_metric(
            self, 
            epoch: int,
            train_metrics_value,
            val_metrics_value
        ):
        """
        Логирование метрик эпохи в MLflow
        """
        if not self.log_mlflow:
            return
        try:
            metrics = {
                **train_metrics_value,
                **val_metrics_value,
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

    def _train_one(self) -> None:
        """
        Проход по тренировочным данным и тренировка на них
        """
        self.logger.debug("🔘 Start epoch train")
        
        self.model.train()

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

            _, predicted = torch.max(outputs, dim=1)
            
            self.train_metrics.update(
                preds=predicted, 
                target=labels,
                value=loss.item()
            )

            # cuda opyat ushla vsya pamyat'
            del inputs, labels, outputs, loss
        
        self.scheduler.step()

        train_metrics_value = self.train_metrics.compute()
        self.logger.info(f"Loss train: {train_metrics_value['train_loss']}")
        self.train_metrics.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.debug("🔘 Finish trainning data")
        return train_metrics_value

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

        with torch.no_grad():
            for data in self._tqdm_loader(self.val_loader, "Validating"):
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                self.val_metrics.update(
                    preds=predicted, 
                    target=labels,
                    value=loss.item()
                )
                
                # cude opyat ushla vsya pamyat'
                del inputs, outputs, labels, loss

        val_metrics_value = self.val_metrics.compute()
        self.logger.info(f"Loss train: {val_metrics_value['val_loss']}")
        self.val_metrics.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return val_metrics_value

    def train_with_mlflow(
            self,
            epochs: int = 20,
        ) -> nn.Module:
        """
        Полный цикл тренировки
        
        Args:
            epoch: количество эпох для тренировки
        """
        self.logger.info("🔘 Start train")
        if not self.log_mlflow:
            self.logger.error("MLFlow - OFF in params the class")
            self.logger.error("TRAIN STOP")
            return self.model

        self._setup_mlflow(
            epochs, 
            self.optimizer.param_groups[0]['lr']
        )
        
        with self.mlflow_run_manager():
            
            self._mlflow_log_parameters(
                epochs
            )

            # надо изменить на что-то своё 
            # В планах поменять
            # best_score = 0.0

            for epoch in range(epochs):
                self.logger.info("="*20)
                self.logger.info(f"🔄 Epoch[🔹{epoch+1}/{epochs}🔹] start")
                train_metrics_value = self._train_one()
                val_metrics_value = self._validate_one()
                
                self._log_epoch_metric(
                    epoch+1,
                    train_metrics_value,
                    val_metrics_value
                )

                # if best_score < self.history['val_accuracy'][-1]:
                #     best_score = self.history['val_accuracy'][-1]
                #     self._log_model_checkpoint(epoch + 1)

                self.logger.info(f"🟢 Epoch[🔹{epoch+1}/{epochs}🔹] completed")

            # Логируем все артефакты
            # self._log_training_artifacts()

            if self.log_mlflow:
                mlflow.end_run()

            self.logger.info("🏁 Finish train")