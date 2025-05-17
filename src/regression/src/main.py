import mlflow
import mlflow.pytorch
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
import os

from exchange.logger import get_logger

from model_time.src.baseline_model import BaseLineModel
from model_time.src.baseline_pipeline import BaselinePipeline

from sklearn.metrics import mean_squared_error, r2_score

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "root")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "rootroot")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")

models = {
    "baseline": BaseLineModel,
    "linear": BaseLineModel
}

pipelines = {
    "baseline": BaselinePipeline,
}

metrics = {
    "mse": mean_squared_error,
    "r2": r2_score
}

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

logger = get_logger("model_time")

@hydra.main(config_path="../configs", config_name="baseline.yaml")
def main(cfg: DictConfig):
    """Train and/or evaluate a model using the provided configuration."""

    task = cfg.task

    if task == "train":
        return run(cfg)
    elif task == "evaluate":
        return evaluate(cfg)
    else:
        logger.info(f"Unknown task: {task}. Please use 'train' or 'evaluate'.")
        return

def run(cfg: DictConfig):
    """Training function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_params = OmegaConf.to_container(cfg.model)
    model_specific_params = {}
    if isinstance(model_params, dict):
        model_specific_params = {
            f"model.{k}": v for k, v in model_params.items() if k not in ["version"]
        }

    pipeline_params = OmegaConf.to_container(cfg.pipeline)
    pipeline_specific_params = {}
    if isinstance(pipeline_params, dict):
        pipeline_specific_params = {
            f"pipeline.{k}": v for k, v in pipeline_params.items()
        }

    experiment_name = f"{cfg.model.name}_Experiment"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{cfg.model.name}_Run_{timestamp}") as run:
        mlflow.log_params({**pipeline_specific_params, **model_specific_params})

        mlflow.set_tag("version", timestamp)
        for k, v in model_specific_params.items():
            mlflow.set_tag(k, v)
        for k, v in pipeline_specific_params.items():
            mlflow.set_tag(k, v)

        pipeline = pipelines[cfg.pipeline.type](cfg.pipeline)
        data = pipeline.load()
        X_trans, y_trans = pipeline.transform(data)
        X_train, X_test, y_train, y_test = pipeline.split(X_trans, y_trans)

        model = models[cfg.model.type](cfg.model)
        model.fit(X_train, y_train)

        model_path = to_absolute_path(f"../models/{cfg.model.name}/{timestamp}")
        model.save(model_path)

        mlflow.sklearn.log_model(
            model, cfg.model.name, registered_model_name=cfg.model.name
        )

        y_pred = model.predict(X_test)

        for metric_name in cfg.metrics:
            metric_func = metrics.get(metric_name)
            if metric_func:
                if metric_name in ["precision", "recall", "f1"]:
                    score = metric_func(y_test, y_pred, average="weighted")
                else:
                    score = metric_func(y_test, y_pred)

                mlflow.log_metric(metric_name, score)
                logger.info(f"{metric_name.capitalize()}: {score:.4f}")

        run_id = run.info.run_id
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Model: {cfg.model.name}")
        logger.info(f"Version: {timestamp}")
        logger.info(
            "Run `mlflow ui` to view results and access the model in the Model Registry."
        )

def evaluate(cfg: DictConfig):
    """Evaluation function"""
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

        # Log metrics
        for metric in cfg.metrics:
            metric_func = metrics.get(metric)
            if metric_func:
                if metric in ["precision", "recall", "f1"]:
                    score = metric_func(y_test, y_pred, average="weighted")
                else:
                    score = metric_func(y_test, y_pred)
                mlflow.log_metric(metric, score)
                logger.info(f"{metric.capitalize()}: {metric_func(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
