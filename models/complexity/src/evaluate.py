import mlflow
from omegaconf import DictConfig
from models import Multinomial
from preprocess import SimplePipeline
import hydra
from hydra.utils import to_absolute_path

models = {
    "multinomial": Multinomial,
}
pipelines = {
    "simple": SimplePipeline,
}

@hydra.main(config_path="../configs", config_name="mutlinomial_l2_lbfgs")
def evaluate(cfg: DictConfig):
    """Evaluate a model using the configuration provided."""
    mlflow.set_experiment(cfg.model.name)

    with mlflow.start_run():
        # Load model
        model = models[cfg.model.type](cfg.model)
        model_path = to_absolute_path(f"models/{cfg.model.name}/{cfg.model.version}")
        model = model.load(model_path)

        # Load & preprocess data
        pipeline = pipelines[cfg.pipeline.type](cfg.preprocess)
        X, y = pipeline.load()
        X_trans, y_trans = pipeline.transform(X, y)
        _, _, X_test, y_test = pipeline.split(X_trans, y_trans)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    evaluate()
