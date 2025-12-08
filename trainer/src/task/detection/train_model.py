import mlflow
from mlflow.models.signature import infer_signature
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
import numpy as np

import time
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import os
os.environ['MLFLOW_SUPPRESS_RUN_LOGS'] = 'true'

class DetectionTrainer:
    def __init__(
            self, 
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_fn: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[lr_scheduler._LRScheduler] = None,
            device: Optional[torch.device] = None,
            # Детекция-specific параметры
            num_classes: int = 1,
            iou_threshold: float = 0.5,
            confidence_threshold: float = 0.25,
            # mlflow параметры
            log_mlflow: bool = True,
            log_artifacts: bool = True,
            experiment_name: str = "Detection_Experiment",
            run_name : Optional[str] = None,
        ):
        """
        Инициализация тренера для детекции объектов
        
        Args:
            model: Нейронная сеть для детекции (YOLO, Faster R-CNN, etc.)
            train_loader: Данные для обучения
            val_loader: Данные для валидации
            loss_fn: Функция потерь для детекции
            optimizer: Оптимизатор
            scheduler: Планировщик learning rate
            device: Устройство вычислений GPU\CPU
            num_classes: Количество классов для детекции
            iou_threshold: Порог IoU для вычисления mAP
            confidence_threshold: Порог уверенности для детекций
            log_mlflow: Флаг логирования в MLflow
            log_artifacts: Логирование артефактов
            experiment_name: Имя эксперимента в MLflow
            run_name: Уникальное имя запуска в MLflow
        """
        self._validate_input(model, train_loader, val_loader)
        print("⚪ Start init Detection Trainer")
        
        self.model = model
        self.train_loader = train_loader
        print(" ➖ Train load sample:", len(self.train_loader.dataset))
        self.val_loader = val_loader
        print(" ➖ Val load sample:  ", len(self.val_loader.dataset))

        # device
        self._setup_device(device)
        self.model.to(self.device)

        # Параметры детекции
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        # loss and optimizer
        self.loss_fn = loss_fn or nn.MSELoss() # _get_default_loss() добавим позже
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler or lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

        # metrics для детекции
        self.history = {
            'train_loss': [], 'train_loss_components': [],
            'learning_rate': [],
            'mAP': [], 'precision': [], 'recall': []
        }
        self.best_weights = None
        
        # mlflow
        self.log_mlflow = log_mlflow
        self.log_artifacts = log_artifacts
        self.experiment_name = experiment_name
        self.run_name = run_name

        print("🟢 Finish init Detection Trainer")

    def _get_default_loss(self):
        """Функция потерь по умолчанию для детекции"""
        pass

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
            epoch: int,
            lr: int
        ):
        """
        Настройка MLFlow с предварительной проверкой сервера
        """
        if not self.log_mlflow:
            print(" ➖ log in Mlflow: OFF")
            return

        try:
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
        Логирование параметров модели и обучения для детекции
        """
        try: 
            model_params = {
                'model_type': self.model.__class__.__name__,
                'device': self.device.type,
                'total_parameters': sum([p.numel() for p in self.model.parameters()]),
                'num_classes': self.num_classes,
                'iou_threshold': self.iou_threshold,
                'confidence_threshold': self.confidence_threshold
            }

            optimizer_params = {
                'optimizer': self.optimizer.__class__.__name__,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }

            for key, value in self.optimizer.param_groups[0].items():
                if key != 'params':
                    optimizer_params[f'optimizer_{key}'] = value
            
            data_params = {
                'train_sample': len(self.train_loader.dataset),
                'val_sample': len(self.val_loader.dataset),
                'batch_size': self.train_loader.batch_size,
            }
            
            all_params = {
                **model_params, 
                **data_params,
                **optimizer_params, 
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
        Логирование метрик эпохи в MLflow для детекции
        """
        if not self.log_mlflow:
            return
        try:
            metrics = {
                'train_loss': self.history['train_loss'][-1],
                'learning_rate': self.history['learning_rate'][-1],
                'epoch': epoch
            }

            # Добавляем метрики детекции
            if len(self.history['mAP']) > 0:
                metrics['mAP'] = self.history['mAP'][-1]
            if len(self.history['precision']) > 0:
                metrics['precision'] = self.history['precision'][-1]
            if len(self.history['recall']) > 0:
                metrics['recall'] = self.history['recall'][-1]

            mlflow.log_metrics(metrics, step=epoch)

        except Exception as e:
            print("🔴[MLFlow] Error set params model:", e)

    def _log_model_checkpoint(self, epoch: int):
        """
        Логирование чекпоинтов модели
        """
        if not (self.log_mlflow or self.log_artifacts):
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
        """Создание сигнатуры для модели детекции"""
        try:
            # Получаем пример батча
            sample_batch = next(iter(self.train_loader))
            images, _ = sample_batch  # images: list[tensor], _: list[dict]
            
            # ✅ Берем ПЕРВОЕ изображение из батча как пример
            # Для детекции модели ожидают list тензоров, даже если один
            sample_input = [images[0].to(self.device)]  # Важно: list с одним тензором!
            
            # Переводим модель в eval режим для инференса
            self.model.eval()
            with torch.no_grad():
                sample_output = self.model(sample_input)
            
            # ✅ Конвертируем в numpy формат для MLflow
            # Вход: один тензор в list
            input_numpy = sample_input[0].cpu().numpy()  # [C, H, W]
            
            # Выход: преобразуем list[dict] в dict с массивами
            # Берем только первый элемент выхода (у нас один вход)
            output_dict = sample_output[0]
            output_numpy = {
                'boxes': output_dict['boxes'].cpu().numpy(),    # [N, 4]
                'scores': output_dict['scores'].cpu().numpy(),  # [N]
                'labels': output_dict['labels'].cpu().numpy()   # [N]
            }
            
            signature = infer_signature(
                model_input=input_numpy,
                model_output=output_numpy
            )
            
            print(f"✅ MLflow signature created: input {input_numpy.shape}, output {output_numpy['boxes'].shape}")
            return signature
            
        except Exception as e:
            print(f"🔴 Error creating MLflow signature: {e}")
            return None

    def _log_training_artifacts(self):
        """
        Логирование дополнительных артефактов для детекции
        """
        if not self.log_mlflow or not self.log_artifacts:
            return
            
        try:
            import matplotlib.pyplot as plt
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # График потерь
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(self.history['train_loss'], label='Train Loss')
                # plt.plot(self.history['val_loss'], label='Val Loss')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # График mAP если есть
                if len(self.history['mAP']) > 0:
                    plt.subplot(1, 3, 2)
                    plt.plot(self.history['mAP'], label='mAP', color='green')
                    plt.title('Mean Average Precision')
                    plt.xlabel('Epoch')
                    plt.ylabel('mAP')
                    plt.legend()
                    plt.grid(True)
                
                # График precision/recall если есть
                if len(self.history['precision']) > 0 and len(self.history['recall']) > 0:
                    plt.subplot(1, 3, 3)
                    plt.plot(self.history['precision'], label='Precision')
                    plt.plot(self.history['recall'], label='Recall')
                    plt.title('Precision & Recall')
                    plt.xlabel('Epoch')
                    plt.ylabel('Score')
                    plt.legend()
                    plt.grid(True)
                
                plt.tight_layout()
                metrics_plot_path = os.path.join(temp_dir, 'detection_metrics.png')
                plt.savefig(metrics_plot_path)
                plt.close()
                
                mlflow.log_artifact(metrics_plot_path)
                
                # Логируем историю обучения
                history_path = os.path.join(temp_dir, 'training_history.txt')
                with open(history_path, 'w') as f:
                    f.write("Epoch\tTrain_Loss\tVal_Loss\tmAP\tPrecision\tRecall\tLR\n")
                    for i in range(len(self.history['train_loss'])):
                        f.write(f"{i+1}\t{self.history['train_loss'][i]:.4f}\t"
                               f"{'None'}\t")
                        
                        # Добавляем метрики детекции если они есть
                        if i < len(self.history.get('mAP', [])):
                            f.write(f"{self.history['mAP'][i]:.4f}\t")
                        else:
                            f.write("N/A\t")
                            
                        if i < len(self.history.get('precision', [])):
                            f.write(f"{self.history['precision'][i]:.4f}\t")
                        else:
                            f.write("N/A\t")
                            
                        if i < len(self.history.get('recall', [])):
                            f.write(f"{self.history['recall'][i]:.4f}\t")
                        else:
                            f.write("N/A\t")
                            
                        f.write(f"{self.history['learning_rate'][i]:.6f}\n")
                
                mlflow.log_artifact(history_path)
                
        except Exception as e:
            print(f"🔴[MLFlow] Error logging artifacts: {e}")

    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Вычисление IoU между двумя bounding boxes"""
        # box format: [x1, y1, x2, y2]
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)

    def _evaluate_detection_metrics(self) -> Tuple[float, float, float]:
        """
        Вычисление метрик с использованием готовых методов TorchMetrics
        """
        self.model.eval()
        
        map_metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            class_metrics=True
        )
        
        iou_metric = IntersectionOverUnion(
            box_format='xyxy'
        )
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, targets = batch
                
                images = [img.to(self.device) for img in images]
                device_targets = [{k: v.to(self.device) for k, v in tg.items()} for tg in targets]
                
                # Forward pass
                predictions = self.model(images)
                
                preds_for_metrics = self._convert_to_torchmetrics_format(predictions)
                targets_for_metrics = self._convert_to_torchmetrics_format(device_targets)

                map_metric.update(preds_for_metrics, targets_for_metrics)
                iou_metric.update(preds_for_metrics, targets_for_metrics)
        
        map_results = map_metric.compute()
        iou_results = iou_metric.compute()
        
        # Извлекаем нужные метрики
        mean_ap = map_results['map'].item()         # mAP @ IoU=0.50:0.95
        # mean_ap_50 = map_results['map_50'].item()   # mAP @ IoU=0.50
        precision = map_results['map_50'].item()    # Precision @ 50 detections
        recall = map_results['mar_100'].item()       # Recall @ 50 detections
        
        print("TorchMetrics Results:")
        for key, value in map_results.items():
            if not key.startswith('map_per_class') and value.numel() == 1:
                print(f"   ➖ {key}: {value.item():.4f}")
        
        return mean_ap, precision, recall

    def _convert_to_torchmetrics_format(self, data):
        """
        Конвертирует наши данные в формат для torchmetrics
        """
        formatted_data = []
        
        for item in data:
            formatted_item = {
                'boxes': item['boxes'].cpu(),
                'scores': item['scores'].cpu() if 'scores' in item else torch.ones(len(item['boxes'])),
                'labels': item['labels'].cpu().int()
            }
            formatted_data.append(formatted_item)

        return formatted_data

    def _train_one_epoch(self):
        """
        Проход по тренировочным данным для детекции
        """
        self.model.train()

        runner_loss = 0.0
        total_sample = 0
        loss_components = defaultdict(float)

        for batch in self._tqdm_loader(self.train_loader, "Training"):
            images, targets = batch
            images = [img.to(self.device) for img in images]
            targets = [{k:v.to(self.device) for k, v in tg.items()} for tg in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            runner_loss += losses.item()
            total_sample += len(images)

            # for loss_name, loss_value in loss_dict.items():
            #     loss_components[loss_name] += loss_value.item()

            # cuda pamyt' uschla
            del images, targets, losses

        self.scheduler.step()

        epoch_loss = runner_loss / len(self.train_loader)
        lr = self.optimizer.param_groups[0]['lr']
        # avg_loss_components = {
        #     name: value / len(self.train_loader) 
        #     for name, value in loss_components.items()
        # }

        self.history['train_loss'].append(epoch_loss)
        self.history['learning_rate'].append(lr)
        # self.history['loss_components'].append(avg_loss_components)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Epoch Result:")
        print(f" ➖ Train Loss: {epoch_loss:.4f}")
        print(f" ➖ LR:         {lr:.6f}")
        # for loss_name, loss_value in avg_loss_components.items():
        #     print(f" ➖ {loss_name}: {loss_value:.4f}")

    def _validate_one(self) -> None:
        """
        1 проход по валидационным данным для детекции
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in self._tqdm_loader(self.val_loader, "Validating"):
                images, targets = batch

                images = [img.to(self.device) for img in images]
                targets = [{k:v.to(self.device) for k, v in tg.items()} for tg in targets]

                predictions = self.model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
            
                del images, targets, predictions

        # Вычисляем метрики детекции
        mean_ap, precision, recall = self._evaluate_detection_metrics()
        self.history["mAP"].append(mean_ap)
        self.history["precision"].append(precision)
        self.history["recall"].append(recall)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Validation:")
        print(f" ➖ mAP:         {mean_ap:.4f}")
        print(f" ➖ Precision:   {precision:.4f}")
        print(f" ➖ Recall:      {recall:.4f}")

    def _tqdm_loader(self, data_loader: DataLoader, desc: str = "process"):
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

    def train(self, epochs: int = 20) -> nn.Module:
        """
        Полный цикл тренировки для детекции
        """
        print("🔘[train] Start Detection Training")
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

            # Сохраняем лучшую модель по mAP
            current_mAP = self.history['mAP'][-1] if self.history['mAP'] else 0.0
            if current_mAP > best_val_acc:
                best_val_acc = current_mAP
                self._log_model_checkpoint(epoch + 1)

            print(f"🟢 Epoch[🔹{epoch+1}/{epochs}🔹] completed")

        # Логируем все артефакты
        self._log_training_artifacts()

        if self.log_mlflow:
            mlflow.end_run()

        print("🟢[train] Detection Training Completed!!!")