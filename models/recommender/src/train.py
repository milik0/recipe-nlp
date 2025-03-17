import mlflow
import hydra
import torch
from omegaconf import DictConfig
from models import bert, lstm
from preprocess import preprocess_data

@hydra.main(config_path="../configs", config_name="default")
def train(cfg: DictConfig):
    mlflow.set_experiment(cfg.model.name)
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(cfg.model)

        # Load & preprocess data
        train_data, val_data = preprocess_data(cfg.data)

        # Select model
        if cfg.model.name == "bert":
            model = bert.BERTModel(cfg.model)
        else:
            model = lstm.LSTMModel(cfg.model)
        
        # Train model
        model.train(train_data, val_data)

        # Log trained model
        mlflow.pytorch.log_model(model, "models")

if __name__ == "__main__":
    train()
