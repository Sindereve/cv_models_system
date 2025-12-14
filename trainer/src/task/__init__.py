from typing import Dict
from task.classification import data, train_model
from task.classification.models import get_model
import os

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
    count_classes = len(
        next(os.walk(data_loader_params.get('path_data_dir', None)))[1]
    )

    model = get_model(
        **model_params,
        num_class = count_classes,
    )

    if model_params.get('weights', False):
        img_w, img_h = model.get_input_size_for_weights()
        data_loader_params['img_w_size'] = img_w
        data_loader_params['img_h_size'] = img_h
    
    train_loader, val_loader, test_loader, classes = data.load_dataloader(**data_loader_params)

    trainer = train_model.Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        classes,
        **trainer_params
    )

    trainer.train_with_mlflow()