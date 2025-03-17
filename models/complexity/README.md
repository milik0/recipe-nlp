# Complexity Model

## Deploy with docker

```
# Example deployment command using Docker
cd models/complexity
docker build -t complexity .
docker run -d -p 8000:8000 complexity-model
```

| Variable         | Default Value               | Description                                  |
|-----------------|---------------------------|----------------------------------------------|
| `APP_NAME`      | `complexity`               | The name of the application.                |
| `DEBUG`         | `false`                    | Enables debug mode (`true` or `false`).     |
| `DEFAULT_MODEL` | `mutlinomial_l2_lbfgs`     | The default model used for training.        |
| `DEFAULT_MODEL_DIR` | `../configs`          | Directory containing model configurations.  |


## Local

```
git clone git@gitlab.cri.epita.fr:mateo.lelong/nlp1.git
cd models/complexity

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload
```

