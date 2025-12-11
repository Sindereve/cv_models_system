from typing import Dict
from task.classification import data, train_model
from task.classification.models import get_model


def training_model_clf(
        data_loader_params: Dict,
        model_params: Dict,
        trainer_params: Dict
    ):
    """
    Full pipeline life model
    
    :param data_loader_params: Parameters for data
    :type data_loader_params: Dict
    :param model_params: Parameters for NN model
    :type model_params: Dict
    :param trainer_params: Parameters for train model
    :type trainer_params: Dict
    """
    train_loader, val_loader, test_loader, classes = data.load_dataloader(**data_loader_params)

    model = get_model(
        **model_params,
        num_class = len(classes),
    )

    trainer = train_model.Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        classes,
        **trainer_params
    )

    trainer.train_with_mlflow()