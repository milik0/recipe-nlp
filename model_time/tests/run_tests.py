from omegaconf import DictConfig
from src.models.multinomial import Multinomial
from src.pipelines.simple import SimplePipeline
import hydra
from hydra.utils import to_absolute_path
from sklearn.metrics import mean_squared_error

models = {
    "multinomial": Multinomial,
}
pipelines = {
    "simple": SimplePipeline,
}

metrics = {
    "mse": mean_squared_error,
}


@hydra.main(config_path="../configs", config_name="mutlinomial_l2_lbfgs")
def run_tests(cfg: DictConfig):
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
    for metric in cfg.metrics:
        metric_func = metrics[metric]
        score = metric_func(y_test, y_pred)
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    run_tests()
