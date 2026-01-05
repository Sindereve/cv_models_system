import logging
import sys
from contextlib import contextmanager
import mlflow
from mlflow.types.schema import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from tqdm import tqdm
import numpy as np
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

import asyncio
import time
from typing import Optional, Dict

from shared.logging import get_logger

class Trainer:
    """
    Trainer class for training, validation, and testing
    image classification models.

    This class encapsulates the full training lifecycle of a neural network,
    including loss function initialization, optimizer and learning rate
    scheduler configuration, execution of training and validation loops,
    model evaluation on a test dataset, and experiment tracking with MLflow.

    Key features:
        - Training deep learning models for image classification tasks
        - Validation and testing using separate datasets
        - Flexible configuration of loss functions, optimizers, and schedulers
        - Support for CPU and GPU training
        - Experiment tracking, metric logging, and artifact storage via MLflow
        - Model checkpointing and saving best/final weights

    The Trainer does not handle data preparation or model architecture
    definition. It assumes that the model and dataloaders are provided
    externally.
    """


    def __init__(
            self, 
            model: nn.Module,
            # data
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader = None,
            classes: Optional[Dict] = None,
            # settings for train model
            loss_fn_config: Dict = None,
            optimizer_config: Dict = None,
            scheduler_config: Dict = None,
            epochs: int = 10,
            device: Optional[str] = 'cpu',
            # mlflow tracking
            log_mlflow: bool = True,
            mlflow_uri: str = 'http://127.0.0.1:5000',
            log_artifacts: bool = True,
            log_checkpoint: bool = True,
            experiment_name: str = "Experiment_name",
            run_name : Optional[str] = None,
            mlflow_tags: Optional[Dict[str, str]] = None,
        ):
        """
        Initializes the Trainer for an image classification model.

        Args:
            model (nn.Module):
                Neural network model to be trained.

            train_loader (DataLoader):
                DataLoader for the training dataset.

            val_loader (DataLoader):
                DataLoader for the validation dataset.

            test_loader (DataLoader, optional):
                DataLoader for the test dataset. Used for final
                model evaluation after training.

            classes (Dict, optional):
                Dictionary mapping class names to class indices
                ({class_name: class_index}). Used for logging and
                result interpretation.

            loss_fn_config (Dict, optional):
                Configuration for the loss function (type and parameters).

            optimizer_config (Dict, optional):
                Configuration for the optimizer (type, learning rate,
                weight decay, etc.).

            scheduler_config (Dict, optional):
                Configuration for the learning rate scheduler used
                during training.

            epochs (int, optional):
                Number of training epochs. Default is 10.

            device (str, optional):
                Computation device, either 'cpu' or 'cuda'.
                Default is 'cpu'.

            log_mlflow (bool, optional):
                Enables or disables MLflow experiment tracking.

            mlflow_uri (str, optional):
                URI of the MLflow Tracking Server (local or remote, HTTP).

            log_artifacts (bool, optional):
                Enables logging of artifacts such as models, metrics,
                and training outputs.

            log_checkpoint (bool, optional):
                Enables saving model checkpoints during training.

            experiment_name (str, optional):
                Name of the MLflow experiment.
                Default is "Experiment_name".

            run_name (str, optional):
                Unique name of the MLflow run.
                If not provided, a name is generated automatically
                in the following format:
                "{model_name}_ep{epochs}_lr{learning_rate}_time({timestamp})".

            mlflow_tags (Dict[str, str], optional):
                Additional tags for the MLflow run
                (e.g., dataset name, model version, author).
        """

        # logger load
        self.logger = get_logger(__name__)
        self.logger.debug("⚪ Start init")

        # model and setting learning
        self.model = model
        self.loss_fn_config = loss_fn_config
        self.loss_fn = None
        self.optimizer_config = optimizer_config 
        self.optimizer = None
        self.scheduler_config = scheduler_config
        self.scheduler = None
        self.epochs = epochs
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
        self.log_checkpoint = log_checkpoint
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
            self.logger.info(f" ➖ Test data sample:    {test_size}")
        else:
            test_size = None
            self.logger.info(" ➖ Test data sample:    Not used")
            self.logger.warning(" Model don`t testing for test data! (test_loader is None value)")
        self.logger.debug("|└🏁 Finish calculate info for data")
        return train_size, val_size, test_size 

    def _validate_input(self):
        """
        Validates and initializes trainer input parameters.

        This method performs validation of all critical inputs required for
        training an image classification model. It checks the correctness
        of types, ensures that dataloaders are not empty, and initializes
        default components when optional configurations are missing or invalid.

        Validation and initialization steps include:
            - Verifying model and dataloader types
            - Ensuring training and validation datasets are not empty
            - Validating optional inputs (test_loader, device) and resetting
            them to default values if invalid
            - Initializing the loss function from configuration or using
            CrossEntropyLoss by default
            - Initializing the optimizer from configuration or using Adam
            with a default learning rate
            - Initializing the learning rate scheduler from configuration
            or using CosineAnnealingLR by default
            - Testing the connection to the MLflow tracking server

        Raises:
            TypeError:
                If required inputs (model, train_loader, val_loader)
                have invalid types.

            ValueError:
                If training or validation datasets are empty.

        Notes:
            - This method is intended for internal use only and should be
            called during Trainer initialization.
            - If optional parameters are invalid, they are automatically
            replaced with safe default values and a warning is logged.
            - All adjustments and validation results are logged for
            transparency and debugging purposes.
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
            
            if isinstance(obj, DataLoader):
                if len(obj.dataset) == 0:
                    self.logger.error(f"|└🔴 {name}({type}) is empty.")
                    raise ValueError(f"{name}({type}) is empty.")
            
            self.logger.debug(f"|├🟢 {name}: OK")

        check_and_adjust = [
            (self.test_loader, DataLoader, "test_loader", None),
            (self.device, str, "device", None),
        ]

        for obj, type, name, new_val in check_and_adjust :
            if not isinstance(obj, type):
                self.logger.warning(f"🟠 {name} is not {type}.")
                setattr(self, name, new_val)
                self.logger.debug(f"|├🟢 {name} change in default value. ({new_val})")
            else:
                self.logger.debug(f"|├🟢 {name}: OK")

        # loss function
        if not isinstance(self.loss_fn_config, Dict):
            self.logger.warning(f"🟠 loss_fn_config is not {Dict}. Change in default value({nn.CrossEntropyLoss})")
            self.loss_fn = nn.CrossEntropyLoss()
            self.logger.debug(f"|├🟢 loss_fn change in default value. ({nn.CrossEntropyLoss})")
        else:
            self.loss_fn = self._create_loss_fn_from_config(self.loss_fn_config)
            self.logger.debug(f"|├🟢 loss_fn: OK")

        # optimizer
        if not isinstance(self.optimizer_config, Dict):
            self.logger.warning(f"🟠 optimizer_config is not {Dict}. Change in default value({optim.Adam})")
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.logger.debug(f"|├🟢 optimizer change in default value. (learning_rate = 0.001, {optim.Adam})")
        else:
            self.optimizer = self._create_optimizer_from_config(self.optimizer_config)
            self.logger.debug(f"|├🟢 optimizer: OK")

        # lr_scheduler
        if not isinstance(self.scheduler_config, Dict):
            self.logger.warning(f"🟠 scheduler is not {Dict}. Change in default value({lr_scheduler.CosineAnnealingLR})")
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
            self.logger.debug(f"|├🟢 scheduler change in default value. ({lr_scheduler.CosineAnnealingLR})")
        else:
            self.scheduler = self._create_scheduler_from_config(self.scheduler_config)
            self.logger.debug(f"|├🟢 scheduler: OK")

        # mlflow test connect
        self._mlflow_test_connect()

        self.logger.debug("|└🏁 finish validating params")

    def _create_loss_fn_from_config(self, loss_fn_config: Dict) -> nn.Module:
        """
        Creates a loss function from a configuration dictionary.

        The method dynamically instantiates a PyTorch loss function based on
        the provided configuration. The loss function class is resolved from
        `torch.nn`, and its parameters are passed directly to the constructor.

        Supported configuration format:
            {
                "type": "CrossEntropyLoss",
                "params": {
                    "label_smoothing": 0.1
                }
            }

        Where:
            - "type" (str) specifies the name of the loss function class
            available in `torch.nn`.
            - "params" (dict) contains keyword arguments passed to the
            loss function constructor.

        Args:
            loss_fn_config (Dict):
                Configuration dictionary describing the loss function
                and its parameters.

        Returns:
            nn.Module:
                Instantiated PyTorch loss function.

        Raises:
            ValueError:
                - If the specified loss function type does not exist in `torch.nn`
                - If the configuration does not contain the required "params" key
                - If "params" is not a dictionary
                - If the model does not expose a `parameters()` method

        Notes:
            - This method is intended for internal use by the Trainer.
            - All parameters provided in "params" are passed directly to the
            loss function constructor without modification.
            - A debug log entry is created after successful loss function
            initialization.
        """
        loss_fn_type = loss_fn_config.get('type', 'CrossEntropyLoss')
        
        if not hasattr(nn, loss_fn_type):
            raise ValueError(f"Loss function {loss_fn_type} is not found in torch.nn")
        
        if 'params' not in loss_fn_config:
            raise ValueError("Loss function config must contain 'params' key with optimizer parameters")
        
        loss_fn_params = loss_fn_config['params']
        
        if not isinstance(loss_fn_params, dict):
            raise ValueError("'params' must be a dictionary with optimizer parameters")

        if not hasattr(self.model, 'parameters'):
            raise ValueError("Model has no method 'parameters' for optimizer")
        
        loss_fn_class = getattr(nn, loss_fn_type)
        loss_fn = loss_fn_class(**loss_fn_params)
        
        self.logger.debug(f"|├ Loss function {loss_fn_type} created with params: {loss_fn_params}")
        return loss_fn

    def _create_optimizer_from_config(self, optimizer_config: Dict) -> Optimizer:
        """
        Creates an optimizer from a configuration dictionary.

        This method dynamically instantiates a PyTorch optimizer based on
        the provided configuration. The optimizer class is resolved from
        `torch.optim`, and its parameters are passed directly to the
        optimizer constructor.

        Supported configuration format:
            {
                "type": "AdamW",
                "params": {
                    "lr": 0.001,
                    "weight_decay": 1e-4
                }
            }

        Where:
            - "type" (str) specifies the name of the optimizer class
            available in `torch.optim`.
            - "params" (dict) contains keyword arguments passed to the
            optimizer constructor.

        Args:
            optimizer_config (Dict):
                Configuration dictionary describing the optimizer
                and its parameters.

        Returns:
            Optimizer:
                Instantiated PyTorch optimizer.

        Raises:
            ValueError:
                - If the specified optimizer type does not exist in `torch.optim`
                - If the configuration does not contain the required "params" key
                - If "params" is not a dictionary
                - If the model does not expose a `parameters()` method

        Notes:
            - This method is intended for internal use by the Trainer.
            - The optimizer is created using the model parameters returned
            by `model.parameters()`.
            - A debug log entry is created after successful optimizer
            initialization.
        """
        optimizer_type = optimizer_config.get('type', 'AdamW')
        
        if not hasattr(optim, optimizer_type):
            raise ValueError(f"Optimizer {optimizer_type} is not found in torch.optim")
        
        if 'params' not in optimizer_config:
            raise ValueError("Optimizer config must contain 'params' key with optimizer parameters")
        
        optimizer_params = optimizer_config['params']
        
        if not isinstance(optimizer_params, dict):
            raise ValueError("'params' must be a dictionary with optimizer parameters")

        if not hasattr(self.model, 'parameters'):
            raise ValueError("Model has no method 'parameters' for optimizer")
        
        optimizer_class = getattr(optim, optimizer_type)
        optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        
        self.logger.debug(f"|├ Optimizer {optimizer_type} created with params: {optimizer_params}")
        return optimizer

    def _create_scheduler_from_config(self, scheduler_config: Dict) -> Optional[lr_scheduler._LRScheduler]:
        """
        Creates a learning rate scheduler from a configuration dictionary.

        This method dynamically instantiates a PyTorch learning rate scheduler
        based on the provided configuration. The scheduler class is resolved
        from `torch.optim.lr_scheduler` and is initialized using the Trainer's
        optimizer.

        Supported configuration format:
            {
                "type": "StepLR",
                "params": {
                    "step_size": 30,
                    "gamma": 0.1
                }
            }

        Where:
            - "type" (str) specifies the name of the scheduler class available
            in `torch.optim.lr_scheduler`.
            - "params" (dict) contains keyword arguments passed to the scheduler
            constructor.

        Args:
            scheduler_config (Dict):
                Configuration dictionary describing the learning rate scheduler
                and its parameters. If None, no scheduler is created.

        Returns:
            Optional[lr_scheduler._LRScheduler]:
                Instantiated learning rate scheduler, or None if
                `scheduler_config` is None.

        Raises:
            ValueError:
                - If the configuration does not contain the required "type" key
                - If the specified scheduler type does not exist in
                `torch.optim.lr_scheduler`
                - If the configuration does not contain the required "params" key
                - If "params" is not a dictionary

        Notes:
            - This method is intended for internal use by the Trainer.
            - The scheduler is initialized with the optimizer instance
            previously created for the model.
            - A debug log entry is created after successful scheduler
            initialization.
        """
        if scheduler_config is None:
            return None
        
        scheduler_type = scheduler_config.get('type')
        if not scheduler_type:
            raise ValueError("Scheduler config must contain 'type' key")
        
        if not hasattr(lr_scheduler, scheduler_type):
            raise ValueError(f"Scheduler {scheduler_type} is not found in torch.optim.lr_scheduler")
        
        if 'params' not in scheduler_config:
            raise ValueError("Scheduler config must contain 'params' key")
        
        scheduler_params = scheduler_config['params']
        
        if not isinstance(scheduler_params, dict):
            raise ValueError("Scheduler 'params' must be a dictionary")
        
        scheduler_class = getattr(lr_scheduler, scheduler_type)
        scheduler = scheduler_class(self.optimizer, **scheduler_params)
        
        self.logger.debug(f"|├ Scheduler {scheduler_type} created with params: {scheduler_params}")
        return scheduler

    def _setup_device(self, device_str: Optional[str] = None):
        """
        Configures the computation device for model training.

        This method determines and initializes the device used for training
        (CPU or GPU). If CUDA is requested but not available, the method
        automatically falls back to CPU and logs a warning. When CUDA is
        available, the GPU cache is cleared before training starts.

        Args:
            device_str (str, optional):
                Device identifier string. Expected values are "cpu" or "cuda".
                If None or an unsupported value is provided, the value is passed
                directly to `torch.device`.

        Notes:
            - If `device_str` is set to "cuda" and CUDA is not available,
            training will automatically continue on CPU.
            - When CUDA is available, `torch.cuda.empty_cache()` is called
            to release unused GPU memory before training.
            - Information about the selected device is logged for
            debugging and monitoring purposes.
            - This method updates the internal `self.device` attribute.
        """
        self.logger.debug("├🔘 Start setting device")

        if device_str == 'cuda':
            if not torch.cuda.is_available():
                self.logger.warning("🟠 error load 'CUDA'. Using 'CPU'")
                self.device = torch.device('cpu')
            else:
                # clear cache in cuda
                torch.cuda.empty_cache()
                gpu_info = torch.cuda.get_device_name(self.device)
                self.logger.debug(f"||🟡 GPU: {gpu_info}")
                self.device = torch.device('cuda')
        else:
            self.device = torch.device(device_str)

        self.logger.info(f"Training on: {self.device}")
        self.logger.debug(f"|└🟢Training on: {self.device}")

    def _mlflow_test_connect(self):
        """
        Tests the connection to the MLflow tracking server.

        This method attempts to connect to the MLflow tracking server
        specified by `self.mlflow_uri`. If the connection succeeds, it
        confirms that experiments can be queried. If the server is
        unavailable or an error occurs, a warning is logged and
        MLflow is switched to local tracking.

        Notes:
            - This method is intended for internal use during Trainer
            initialization.
            - MLflow tracking can be disabled by setting `self.log_mlflow`
            to False.
            - Connection status and any errors are logged for debugging
            purposes.

        Behavior:
            - If `self.log_mlflow` is False, the method does nothing and
            logs that MLflow tracking is disabled.
            - On successful connection, logs confirmation.
            - On failure, logs the error and switches MLflow to local tracking.
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
            self.logger.error(f"|||🔴MLflowe{e}")
            self.logger.warning(f"||└🟠MLflow server at {self.mlflow_uri} not available. Using local tracking.")
            mlflow.set_tracking_uri(None)

    @contextmanager
    def mlflow_run_manager(self):
        """
        Context manager for managing MLflow runs.

        This method provides a safe context for executing code within an
        MLflow run. It handles the lifecycle of the run, including starting,
        ending, and error handling, ensuring that all runs are properly
        closed even in case of exceptions.

        Usage:
            with self.mlflow_run_manager():
                # training or evaluation code
                ...

        Behavior:
            - If MLflow tracking is disabled (`self.log_mlflow=False`),
            the context manager yields immediately without starting a run.
            - Starts a new MLflow run with the name `self.run_name`.
            - Yields control to the block of code within the context.
            - On successful completion, ends the run with status "FINISHED".
            - On exception, ends the run with status "FAILED" and logs the error.
            - Ensures that the run is closed by calling `_ensure_run_closed`.

        Notes:
            - Intended for internal use in the Trainer class.
            - Provides robust handling for MLflow experiment tracking,
            preventing dangling or unclosed runs.
            - Logs run start, completion, and failures for debugging purposes.
        """
        if not self.log_mlflow:
            yield
            return
        
        run = None
        try:
            run = mlflow.start_run(run_name=self.run_name) 
            self.mlflow_run = run
            self.logger.debug(f"|🟢 MLflow run started: {run.info.run_id}")
            
            yield
                        
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
        """
        Ensures that an MLflow run is properly closed.

        This method is used internally to guarantee that MLflow runs
        are not left active, even if exceptions occur or runs are
        interrupted. It force-closes the run if it is still active,
        setting the run status to "KILLED".

        Args:
            run (mlflow.entities.Run, optional):
                The MLflow run to ensure closure for. If None, any
                currently active run will be force-closed.

        Behavior:
            - Checks if there is an active MLflow run.
            - If an active run exists and matches the provided `run`,
            ends it with status "KILLED".
            - If `run` is None, any active run is also ended with
            status "KILLED".
            - Logs a warning indicating that the run was force-closed.
            - Exceptions during closure are silently ignored to
            prevent interruption of training.

        Notes:
            - Intended for internal use within the Trainer class.
            - Helps prevent dangling MLflow runs that could interfere
            with subsequent experiment tracking.
            - Logs debug information about the start and completion
            of the closure process.
        """
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

    def _setup_mlflow(self):
        """
        Configures MLflow for experiment tracking.

        This method sets up MLflow for logging metrics, parameters,
        and artifacts during training. It ensures that the experiment
        is correctly initialized and assigns a unique run name if
        none is provided.

        Behavior:
            - If MLflow tracking is disabled (`self.log_mlflow=False`),
            the method exits immediately.
            - Sets the MLflow experiment using `self.experiment_name`.
            - Enables system metrics logging in MLflow.
            - Generates a unique run name if `self.run_name` is None,
            using the model class name, number of epochs, learning
            rate, and current timestamp.
            - Logs debug messages for setup steps.
            - If any exception occurs, disables MLflow tracking and logs
            a warning and error.

        Notes:
            - Intended for internal use during Trainer initialization.
            - The run name format is:
                "{ModelClass}_ep{epochs}_lr{learning_rate}_time({MM:DD_HH:MM:SS})"
            - This method does not start an MLflow run; it only configures
            the experiment and logging environment.
        """
        if not self.log_mlflow:
            self.logger.debug("🟢 Tracking in MLflow: OFF")
            return

        try:
            self.logger.debug("|🔘 START MLflow setting")
            
            mlflow.set_experiment(self.experiment_name)
            mlflow.enable_system_metrics_logging()

            if self.run_name is None:
                time_str = time.strftime('%m:%d_%H:%M:%S')
                lr = self.optimizer.param_groups[0]['lr']
                
                self.run_name = f"{self.model.__class__.__name__}_ep{self.epochs}_lr{lr}_time({time_str})"

            self.logger.debug(f"|├🟢 run name {self.run_name}")
            self.logger.debug("|└🏁 FINISH MLflow setting")
        except Exception as e:
            self.logger.error(f"🔴 MLflow setup failed: {e}")
            self.logger.warning("🟠 No use tracking MLflow")
            self.log_mlflow = False

    def _mlflow_log_parameters(self):
        """
        Logs model, training, optimizer, scheduler, and loss parameters to MLflow.

        This method collects key parameters of the training setup and model,
        including architecture, device, dataset sizes, optimizer configuration,
        learning rate scheduler settings, and loss function details, and logs
        them as parameters in MLflow.

        Behavior:
            - Extracts model parameters:
                - Model class name
                - Device type (CPU/GPU)
                - Total and trainable parameter counts
            - Extracts optimizer parameters:
                - Optimizer type
                - Learning rate
                - Additional optimizer-specific attributes
            - Extracts dataset and batch information:
                - Training, validation, and test dataset sizes
                - Batch size
                - Number of classes
            - Extracts training parameters:
                - Number of epochs
                - Loss function type and relevant attributes
            - Extracts scheduler parameters (if a scheduler is set):
                - Scheduler type
                - Step size, gamma, T_0 for CosineAnnealingWarmRestarts
            - Combines all parameters into a single dictionary and logs
            them using `mlflow.log_params`.

        Notes:
            - Intended for internal use within the Trainer class.
            - Handles optional attributes gracefully (e.g., scheduler, loss_fn).
            - Logs a debug message on successful logging and an error
            if any exception occurs.
            - Ensures compatibility with different loss functions and schedulers.

        Raises:
            Exception:
                - Propagates any exception raised during parameter logging
                to MLflow for visibility and debugging.
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
                'epochs': self.epochs,
            }

            if hasattr(self, 'scheduler') and self.scheduler:
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
        Creates a collection of metrics for image classification tasks.

        This method builds a configurable set of metrics based on a selected
        preset and prepares them for use during training or evaluation.
        Metrics are returned as a `MetricCollection` and automatically
        moved to the Trainer's computation device.

        Args:
            preset (str, optional):
                Metric preset to use. Available options:
                    - 'minimal': Only accuracy.
                    - 'standard': Accuracy, macro precision, macro recall, macro F1.
                    - 'full': Accuracy, macro/micro precision, recall, F1 scores.
                Default is 'full'.

            prefix (str, optional):
                Optional prefix to prepend to all metric names in the collection.

        Returns:
            MetricCollection:
                A collection of configured metrics ready to be computed
                during training or evaluation.

        Raises:
            ValueError:
                If an unknown `preset` is provided. Valid options are
                'minimal', 'standard', and 'full'.

        Notes:
            - The `sync_on_compute` option is set to False because training
            is not distributed in this implementation.
            - A 'loss' metric (MeanMetric) is always added to the collection.
            - All metrics are moved to the same device as the Trainer
            (`self.device`) for consistent computation.
            - This method is intended for internal use but can also be used
            externally to obtain metric collections for logging or evaluation.
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
        Logs epoch metrics to MLflow.

        This method takes the computed training and validation metrics
        for a given epoch, combines them, and logs them to MLflow with
        the epoch number as the step.

        Args:
            epoch (int):
                The current epoch number used as the MLflow step.

            train_metrics_value (dict):
                Dictionary of metrics computed on the training dataset
                for the current epoch. Keys are metric names, values
                are metric values.

            val_metrics_value (dict):
                Dictionary of metrics computed on the validation dataset
                for the current epoch. Keys are metric names, values
                are metric values.

        Behavior:
            - If MLflow logging is disabled (`self.log_mlflow=False`),
            the method exits immediately.
            - Combines training and validation metrics into a single
            dictionary and logs them using `mlflow.log_metrics`.

        Notes:
            - Intended for internal use within the Trainer class.
            - Ensures that metrics are logged with the correct epoch
            step for visualization in MLflow.
            - Any exceptions during logging are caught and an error
            is logged.
        
        Raises:
            Exception:
                Propagates any exception raised during MLflow logging
                for visibility in logs.
        """
        if not self.log_mlflow:
            return
        try:
            metrics = {
                **train_metrics_value,
                **val_metrics_value
            }

            mlflow.log_metrics(metrics, step=epoch)

        except Exception as e:
            self.logger.error("🔴 Error set metric in mlflow:", e)

    def _log_checkpoint(self, epoch: int):
        """
        Logs a model checkpoint to MLflow.

        This method saves the current state of the model to MLflow
        as a checkpoint. The checkpoint can be saved for every epoch
        if `self.log_checkpoint=True`, or for the final epoch if
        `self.log_artifacts=True`.

        Args:
            epoch (int):
                The current epoch number, used to label the checkpoint
                and for MLflow step tracking.

        Behavior:
            - Creates a CPU copy of the model to avoid GPU memory issues.
            - Sets the model to evaluation mode before logging.
            - Generates an MLflow model signature using `_create_mlflow_signature`.
            - Logs the model to MLflow using `mlflow.pytorch.log_model`.
            - Deletes the CPU copy to free memory.
            - Logs debug messages indicating success or skipping.

        Notes:
            - Intended for internal use within the Trainer class.
            - Checkpoint logging is skipped if neither `log_checkpoint`
            nor `log_artifacts` conditions are met.
            - Exceptions during checkpoint logging are caught and logged.

        Raises:
            Exception:
                Propagates any exceptions raised during MLflow checkpoint logging
                for visibility in logs.
        """
        try:
            if self.log_checkpoint or (epoch == self.epochs and self.log_artifacts):
                self.logger.debug(f"|🔘 Start save checkpoint(save_model)")
                name = f"checkpoint_epoch_{epoch}"
                import copy 
                model_cpu = copy.deepcopy(self.model).to('cpu')
                model_cpu.eval()

                mlflow.pytorch.log_model(
                    model_cpu,
                    name=name,
                    step=epoch,
                    signature=self._create_mlflow_signature(model_cpu),
                    await_registration_for=0
                )
                
                del model_cpu
                self.logger.debug(f"|🟢 Checkpoint(save_model)")
            else:
                self.logger.debug(f"|🟢 Checkpoint(skip)")
        except Exception as e:
            self.logger.error(f"🔴 Error logging сheckpoint: {e}")

    def _create_mlflow_signature(
            self,
            model_cpu
        ):
        """
        Creates an MLflow model signature for the given model.

        The method generates an MLflow `ModelSignature` based on a
        sample batch from the training dataloader. This signature
        defines the expected input and output tensor shapes and types
        for the model, enabling proper logging and later inference.

        Args:
            model_cpu (torch.nn.Module):
                The CPU version of the model for which the signature
                is created.

        Returns:
            mlflow.models.signature.ModelSignature:
                The input-output schema of the model suitable for
                MLflow logging.

        Behavior:
            - Retrieves a sample batch from `self.train_loader`.
            - Performs a forward pass to determine output shape.
            - Defines input and output schemas with `TensorSpec`.
            - Returns the constructed `ModelSignature`.

        Notes:
            - The model must be in evaluation mode for signature creation.
            - The signature ensures reproducibility and correct
            input-output validation in MLflow.
        """
        sample_batch = next(iter(self.train_loader))
        imgs = sample_batch[0].to('cpu')

        with torch.no_grad():
            test_output = model_cpu(imgs)

        input_schema = Schema([
            TensorSpec(
                type=np.dtype(np.float32),
                shape=(imgs.shape),
                name="input_images" 
            )
        ])

        output_schema = Schema([
            TensorSpec(
                type=np.dtype(np.float32),
                shape=(test_output.shape),
                name="out_labels" 
            )
        ])

        return ModelSignature(
            inputs=input_schema,
            outputs=output_schema
        )

    def _log_training_artifacts(self):
        """
        Логирование дополнительных артефактов
        """
        if not self.log_mlflow or not self.log_artifacts:
            return
            
        try:
            pass
        except Exception as e:
            self.logger.error(f"🔴 Error set artifacts in mlflow: {e}")

    def _train_one(self) -> None:
        """
        Performs a single training pass (epoch) over the training dataset.

        This method iterates over the training dataloader, performs forward
        and backward passes for each batch, updates model parameters using
        the optimizer, and accumulates training metrics.

        Returns:
            dict:
                Dictionary containing computed training metrics for the epoch,
                including loss and any additional metrics defined in
                `self.train_metrics`.

        Behavior:
            - Sets the model to training mode (`model.train()`).
            - Iterates over the training dataset using `_tqdm_loader`.
            - Moves inputs and labels to the configured device.
            - Computes model outputs and calculates loss using `self.loss_fn`.
            - Performs backpropagation and optimizer step.
            - Updates training metrics (`self.train_metrics`) with predictions,
            targets, and loss values.
            - Steps the learning rate scheduler (`self.scheduler.step()`).
            - Computes final metrics for the epoch and resets the metric state.
            - Cleans up GPU memory after each batch and synchronizes CUDA if available.
            - Logs progress and final training loss.

        Notes:
            - Intended for internal use within the Trainer class.
            - Uses `self.train_metrics` for metric computation.
            - Automatically handles device placement for inputs, labels, and model.
            - CUDA memory cleanup is performed after each epoch to prevent memory leaks.
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
            torch.cuda.synchronize()

        self.logger.debug("🏁 Finish trainning data")
        return train_metrics_value

    def _tqdm_loader(self, data_loader: DataLoader, desc: str = "process"):
        """
        Returns a tqdm-wrapped dataloader for progress display.
        Compatible with Docker logs.
        """
        return tqdm(
            data_loader,
            desc=desc,
            bar_format="{l_bar}{bar:20}{r_bar}",
            colour="blue",
            leave=False,
            file=sys.stdout
        )

    def _validate_one(
            self
        ) -> None:
        """
        Performs a single validation pass over the validation dataset.

        This method iterates over the validation dataloader, performs
        forward passes for each batch, computes the loss, updates
        validation metrics, and returns the aggregated metrics for
        the epoch.

        Returns:
            dict:
                Dictionary containing computed validation metrics for
                the epoch, including loss and any additional metrics
                defined in `self.val_metrics`.

        Behavior:
            - Sets the model to evaluation mode (`model.eval()`).
            - Disables gradient computation (`torch.no_grad()`).
            - Iterates over the validation dataset using `_tqdm_loader`.
            - Moves inputs and labels to the configured device.
            - Computes model outputs and calculates loss using `self.loss_fn`.
            - Updates validation metrics (`self.val_metrics`) with predictions,
            targets, and loss values.
            - Computes final metrics for the validation epoch and resets the metric state.
            - Cleans up GPU memory after each batch and synchronizes CUDA if available.
            - Logs progress and final validation loss.

        Notes:
            - Intended for internal use within the Trainer class.
            - Uses `self.val_metrics` for metric computation.
            - Automatically handles device placement for inputs, labels, and model.
            - CUDA memory cleanup is performed after the validation pass
            to prevent memory leaks.
        """
        self.logger.debug("🔘 Start val data")
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
        self.logger.info(f"Validation train: {val_metrics_value['val_loss']}")
        self.val_metrics.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.logger.debug("🏁 Finish val data")
        return val_metrics_value

    def train_with_mlflow(self) -> nn.Module:
        """
        Executes the full training cycle with MLflow tracking.

        This method performs the complete training loop for the model,
        including logging parameters, metrics, and checkpoints to MLflow.
        It iterates over the specified number of epochs, trains on the
        training dataset, validates on the validation dataset, and keeps
        track of the best model based on validation loss.

        Returns:
            torch.nn.Module:
                The trained model after completing the training loop.

        Behavior:
            - Checks if MLflow logging is enabled (`self.log_mlflow`); if not,
            logs an error and returns the untrained model.
            - Sets up MLflow experiment and run using `_setup_mlflow` and
            `mlflow_run_manager`.
            - Logs model, optimizer, data, loss, and scheduler parameters
            using `_mlflow_log_parameters`.
            - Initializes `minimal_loss` to track the best validation performance.
            - Iterates over the specified number of epochs (`self.epochs`):
                - Performs one epoch of training (`_train_one`).
                - Performs one epoch of validation (`_validate_one`).
                - Logs epoch metrics to MLflow (`_log_epoch_metric`).
                - Saves a model checkpoint if the validation loss improves
                (`_log_checkpoint`).
            - Logs all additional training artifacts at the end of training
            (`_log_training_artifacts`).
            - Returns the trained model.

        Notes:
            - Intended as the main entry point for training with MLflow tracking.
            - Uses internal Trainer methods for training, validation,
            metrics logging, and checkpointing.
            - Checkpoint logging occurs only when validation loss improves.
            - Training and validation metrics, as well as checkpoints,
            are automatically tracked in MLflow.
        """
        self.logger.info("🔘 Start train")
        if not self.log_mlflow:
            self.logger.error("🔴 MLFlow - OFF in params the class")
            self.logger.error("🔴 TRAIN STOP")
            return self.model

        self._setup_mlflow()
        
        with self.mlflow_run_manager():
            
            self._mlflow_log_parameters()
            minimal_loss = float('inf')

            for epoch in range(self.epochs):
                self.logger.info("="*20)
                self.logger.info(f"🔄 Epoch[🔹{epoch+1}/{self.epochs}🔹] start")
                train_metrics_value = self._train_one()
                val_metrics_value = self._validate_one()
                
                self._log_epoch_metric(
                    epoch+1,
                    train_metrics_value,
                    val_metrics_value
                )

                if minimal_loss > val_metrics_value['val_loss']:
                    minimal_loss = val_metrics_value['val_loss']
                    self._log_checkpoint(epoch+1)

                self.logger.info(f"🟢 Epoch[🔹{epoch+1}/{self.epochs}🔹] completed")

            self._log_training_artifacts()

            self.logger.info("🏁 Finish train")
            return self.model