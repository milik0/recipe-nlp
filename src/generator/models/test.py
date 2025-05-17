import os
from gru_rnn_model import (
    GruRNNModel,
)


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model = GruRNNModel()
model.load_model(
    model_path="recipe_rnn_model.keras",
    tokenizer_path="recipe_tokenizer.pickle",
)
model.predict(
    input_text="chicken, butter, salt, pepper, garlic",
    max_length=50,
    temperature=1.0,
)