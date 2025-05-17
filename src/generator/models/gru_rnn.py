import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def format_ingredients_for_model(ingredients):
    """Format ingredients as expected by the model"""
    if isinstance(ingredients, str):
        # If comma-separated string, split into list
        ingredients_list = [ing.strip() for ing in ingredients.split(",")]
    else:
        # Already a list
        ingredients_list = ingredients

    # Format as quoted items in brackets
    return "[" + ", ".join(f"'{ing}'" for ing in ingredients_list) + "]"

def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0):
        """Generate recipe text based on ingredients"""
        print(f"\nGenerating recipe from: '{seed_text}'")
        MAX_SEQUENCE_LENGTH = 100
        # Convert seed text to tokens
        input_tokens = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Initialize text generation
        generated_text = seed_text
        
        for i in range(max_length):
            # Truncate input if too long
            if len(input_tokens) > MAX_SEQUENCE_LENGTH-1:
                input_tokens = input_tokens[-(MAX_SEQUENCE_LENGTH-1):]
                
            # Pad the input to the required length
            padded_tokens = pad_sequences([input_tokens], maxlen=MAX_SEQUENCE_LENGTH-1, padding='pre')
            
            # Create input tensor
            input_tensor = tf.convert_to_tensor(padded_tokens, dtype=tf.int32)
            
            # Generate predictions
            predictions = model(input_tensor)
            
            # Apply temperature sampling
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(
                predictions,
                num_samples=1
            )[0, 0].numpy()
            
            # Add predicted token to input
            input_tokens.append(predicted_id)
            
            # Get the corresponding word (ensure it exists in vocabulary)
            if predicted_id < len(tokenizer.index_word):
                predicted_word = tokenizer.index_word.get(predicted_id, '')
                generated_text += ' ' + predicted_word
                
                # Stop if EOS token is generated
                if predicted_word == '<EOS>':
                    break
            else:
                generated_text += ' <UNKNOWN>'
        
        return generated_text

class GruRNNModel:
    def __init__(self):
        return

    def load_model(self, model_path, tokenizer_path):
        with open(tokenizer_path, "rb") as handle:
            self.tokenizer = pickle.load(handle)

        self.model = tf.keras.models.load_model(model_path)
    


    def predict(self, input_text, max_length=50, temperature=1.0):
        """Evaluate model on test data and calculate metrics"""
        formatted_ingredients = format_ingredients_for_model(input_text)
        seed_text = f"Ingredients: {formatted_ingredients} | Recipe Steps: "

        # Generate recipe
        generated_text = generate_text(
            self.model, self.tokenizer, seed_text,
            max_length=max_length, temperature=temperature
        )

        print(f"Generated Recipe: {generated_text}")
        return generated_text


