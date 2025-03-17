import mlflow
from datetime import datetime
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path

from models import Multinomial
from preprocess import SimplePipeline

models = {
    "multinomial": Multinomial,
}
pipelines = {
    "simple": SimplePipeline,
}

@hydra.main(config_path="../configs", config_name="mutlinomial_l2_lbfgs")
def train(cfg: DictConfig):
    """Train a model using the configuration provided."""
    mlflow.set_experiment(cfg.model.name)
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(cfg.model)

        # Load & preprocess data
        pipeline = pipelines[cfg.pipeline.type](cfg.preprocess)
        X, y = pipeline.load()
        X_trans, y_trans = pipeline.transform(X, y)
        X_train, y_train, _, _ = pipeline.split(X_trans, y_trans)

        # Select model
        model = models[cfg.model.type](cfg.model)
        model.fit(X_train, y_train)

        # Save model
        now = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        model_path = to_absolute_path(f"models/{cfg.model.name}/{now}")
        model.save(model_path)

        # Log trained model
        mlflow.pytorch.log_model(model, "models")

if __name__ == "__main__":
    train()
