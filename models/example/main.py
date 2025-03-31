import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import datetime

EXPERIMENT_NAME = "Example_Experiment"
MODEL_NAME = "Example"

os.environ["AWS_ACCESS_KEY_ID"] = "root"
os.environ["AWS_SECRET_ACCESS_KEY"] = "rootroot"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

mlflow.set_tracking_uri("http://localhost:5000")

try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created new experiment: {EXPERIMENT_NAME} with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {EXPERIMENT_NAME} with ID: {experiment_id}")
except Exception as e:
    print(f"Error setting up experiment: {e}")
    experiment_id = 0

mlflow.set_experiment(EXPERIMENT_NAME)

X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
artifact_path = MODEL_NAME

with mlflow.start_run(run_name=f"{MODEL_NAME}_Run_{timestamp}") as run:
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute evaluation metric
    mse = mean_squared_error(y_test, y_pred)
    
    # Log metric manually (though autolog will do this too)
    mlflow.log_metric("mse", mse)
    
    # Log additional custom metrics
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))
    
    # Log model as a registered model named "Example"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,  # This determines the artifact folder name
        registered_model_name=MODEL_NAME  # This registers the model in the Model Registry
    )
    
    # Set tags for versioning and additional information
    mlflow.set_tag("model_name", MODEL_NAME)
    mlflow.set_tag("version", timestamp)
    mlflow.set_tag("description", "Linear regression model example")
    mlflow.set_tag("creator", "MLflow Example Script")
    
    # Get the run ID for reference
    run_id = run.info.run_id
    
    print(f"Model training complete for model: {MODEL_NAME}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Version: {timestamp}")
    print(f"Run ID: {run_id}")
    print(f"Artifact path: {artifact_path}")
    print("Run `mlflow ui` to view logs and access the model in the Model Registry.")
