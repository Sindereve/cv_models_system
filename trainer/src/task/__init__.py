from typing import Dict
from task.classification import data, train_model
from task.classification.models import get_model

def training_model_clf(
        data_loader_params: Dict,
        model_params: Dict,
        trainer_params: Dict
    ):
    """
    Runs the full training pipeline for a classification model.

    The function performs the following steps:
    1. Loads train, validation, and test dataloaders.
    2. Initializes a neural network model with the given parameters.
    3. Adjusts input image size if pretrained weights are used.
    4. Creates a Trainer instance.
    5. Trains the model and logs results to MLflow.

    Args:
        data_loader_params (Dict):
            Parameters for creating dataloaders (paths, batch size,
            image size, augmentations, etc.).

        model_params (Dict):
            Parameters for model initialization (architecture,
            pretrained weights, backbone settings, etc.).

        trainer_params (Dict):
            Parameters for the training process (optimizer,
            learning rate, epochs, callbacks, etc.).

    Returns:
        None

    Notes:
        - If `weights` is specified in `model_params`, the input image
          size will be automatically adjusted to match the model
          requirements.
        - Training metrics, artifacts, and the final model are logged
          using MLflow.
    """
    train_loader, val_loader, test_loader, classes = data.load_dataloader(**data_loader_params)

    model = get_model(
        **model_params,
        num_class = len(classes),
    )

    if model_params.get('weights', False):
        img_w, img_h = model.get_input_size_for_weights()
        data_loader_params['img_w_size'] = img_w
        data_loader_params['img_h_size'] = img_h

    trainer = train_model.Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        classes,
        **trainer_params
    )

    trainer.train_with_mlflow()