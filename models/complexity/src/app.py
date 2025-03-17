import subprocess
from fastapi import FastAPI, BackgroundTasks, HTTPException
import os
import omegaconf

from models import Multinomial
from preprocess import SimplePipeline

APP_NAME = os.environ.get("APP_NAME", "complexity")
DEBUG = os.environ.get("DEBUG", "false") == "false"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mutlinomial_l2_lbfgs")
DEFAULT_MODEL_DIR = os.environ.get("DEFAULT_MODEL_DIR", "../configs")

app = FastAPI(title=APP_NAME, debug=DEBUG)

models = {
    "multinomial": Multinomial,
}
pipelines = {
    "simple": SimplePipeline,
}

@app.get("/train/{name}")
def train_model(name: str, background_tasks: BackgroundTasks):
    """Trigger model training asynchronously."""
    
    def train():
        subprocess.run(["python", "train.py", f"config_name={name}"])

    background_tasks.add_task(train)
    return {"status": "Training started!"}

@app.get("/evaluate/{name}")
def evaluate_model(name: str, background_tasks: BackgroundTasks):
    """Trigger model evaluation asynchronously."""

    def evaluate():
        subprocess.run(["python", "evaluate.py", f"config_name={name}"])

    background_tasks.add_task(evaluate)
    return {"status": "Evaluation started!"}

@app.get("/{name}")
def predict(name: str):
    """Trigger model evaluation synchronously."""
    if name == "default":
        name = DEFAULT_MODEL

    if not name in pipelines.keys():
        raise HTTPException(status_code=404, detail="Item not found")

    config =  omegaconf.OmegaConf.load(DEFAULT_MODEL_DIR + "/" + DEFAULT_MODEL)
    pipeline = pipelines[name](config.pipeline)


    return {"status": "Evaluation started!"}

@app.get("/")
def predict_default():
    """Trigger model evaluation synchronously."""

    return {"status": "Evaluation started!"}
