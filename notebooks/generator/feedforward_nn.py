import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List
from collections import Counter

class BatchFeedForwardNN:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None
        self.context = None
        self.batch_size = 32

    def _build_vocabulary(self, ingredients: List[List[str]], steps: List[List[str]]):
        all_tokens = []
        for ingredient_list in ingredients:
            all_tokens.extend(ingredient_list)
        for step_list in steps:
            all_tokens.extend(step_list)

        all_tokens.extend(['<UNKNOWN>', '<s>', '</s>'])

        counter = Counter(all_tokens)
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        self.vocab = [token for token, _ in sorted_tokens]

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

        return self.vocab
    
    def _prepare_train(self, ingredients: List[List[str]], steps: List[List[str]], context: int):
        self.context = context
        
        if self.vocab is None:
            self._build_vocabulary(ingredients, steps)
        
        X_train = []
        y_train = []
        
        for i in range(len(ingredients)):
            ingredient_ids = [self.token_to_id.get(token, self.token_to_id['<UNKNOWN>']) for token in ingredients[i]]
            
            while len(ingredient_ids) < context:
                ingredient_ids.append(self.token_to_id['<UNKNOWN>'])
            
            step_ids = [self.token_to_id.get(step_token, self.token_to_id['<UNKNOWN>']) 
                        for step_token in steps[i]]
            
            for k in range(len(ingredient_ids) - context + 1):
                context_window = ingredient_ids[k:k+context]
                for step_id in step_ids:
                    X_train.append(context_window)
                    y_train.append(step_id)
        
        return np.array(X_train), np.array(y_train)
    
    def _prepare_test_batch(self, ingredients: List[List[str]]):
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Fit the model first!")
        
        all_contexts = []
        
        for ingredient_list in ingredients:
            ingredient_ids = [self.token_to_id.get(token, self.token_to_id['<UNKNOWN>']) for token in ingredient_list]

            while len(ingredient_ids) < self.context:
                ingredient_ids.append(self.token_to_id['<UNKNOWN>'])
            
            contexts = []
            for k in range(len(ingredient_ids) - self.context + 1):
                context_window = ingredient_ids[k:k+self.context]
                contexts.append(context_window)
            
            all_contexts.append(contexts)
        
        return all_contexts
    
    def fit(self, ingredients: List[List[str]], steps: List[List[str]],
            embedding_dim=256, hidden_dim=512, context=3, 
            epochs=10, batch_size=32, validation_split=0.1, dropout=None, learning_rate=1e-2):
        if len(ingredients) != len(steps):
            raise ValueError(f"dimension mismatch {len(ingredients)} vs {len(steps)}")
        
        self.batch_size = batch_size
        X_train, y_train = self._prepare_train(ingredients, steps, context)
        vocab_size = len(self.vocab)
        print(f"Vocab size is: {vocab_size}")

        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=context, name="w2v_embedding"))
        self.model.add(keras.layers.Flatten(name="flatten"))
        self.model.add(keras.layers.Dense(hidden_dim, activation='relu', name="hidden"))
        if dropout != None:
            self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(keras.layers.Dense(vocab_size, activation=keras.layers.Softmax(), name="output"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy()
            ]
        )

        print(self.model.summary())

        self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=validation_split
        )
        return self
    
    def predict(self, ingredients: List[List[str]], max_steps=20) -> List[List[str]]:
        if self.model is None:
            raise ValueError("Model not fitted!")
        
        all_contexts = self._prepare_test_batch(ingredients)
        all_predictions = []
        
        for ingredient_contexts in all_contexts:
            current_tokens = ['<s>']
            
            for _ in range(max_steps):
                if not ingredient_contexts:
                    batch_input = np.array([[self.token_to_id['<UNKNOWN>']] * self.context])
                else:
                    batch_size = min(self.batch_size, len(ingredient_contexts))
                    batch_input = np.array(ingredient_contexts[:batch_size])
                
                # Make batch prediction
                predictions = self.model.predict(batch_input, verbose=0)
                
                # Use the first prediction
                next_token_id = np.argmax(predictions[0])
                next_token = self.id_to_token[next_token_id]

                if next_token == '</s>':
                    break
                
                current_tokens.append(next_token)
                
                # Remove the contexts we just used
                if len(ingredient_contexts) > self.batch_size:
                    ingredient_contexts = ingredient_contexts[self.batch_size:]
                else:
                    ingredient_contexts = []
            
            if current_tokens[0] == '<s>':
                current_tokens = current_tokens[1:]

            all_predictions.append(current_tokens)

        return all_predictions
    
    def predict_batch(self, ingredients: List[List[str]], max_steps=20) -> List[List[str]]:
        if self.model is None:
            raise ValueError("Model not fitted!")
        
        all_contexts = self._prepare_test_batch(ingredients)
        all_predictions = []
        
        for recipe_contexts in all_contexts:
            generated_steps = []
            current_step = ['<s>']
            
            step_count = 0
            while step_count < max_steps:
                # Use the beginning contexts for initial predictions
                if recipe_contexts and step_count == 0:
                    # Take a batch of contexts at once
                    batch_size = min(self.batch_size, len(recipe_contexts))
                    batch_contexts = np.array(recipe_contexts[:batch_size])
                    
                    # Get batch predictions
                    batch_predictions = self.model.predict(batch_contexts, verbose=0)
                    
                    # Get the most likely token for each context
                    token_ids = np.argmax(batch_predictions, axis=1)
                    tokens = [self.id_to_token[tid] for tid in token_ids]
                    
                    # Add the first non-end token as our next token
                    for token in tokens:
                        if token != '</s>':
                            current_step.append(token)
                            break
                else:
                    dummy_context = np.array([[self.token_to_id['<UNKNOWN>']] * self.context])
                    prediction = self.model.predict(dummy_context, verbose=0)
                    token_id = np.argmax(prediction[0])
                    token = self.id_to_token[token_id]
                    
                    if token == '</s>' or len(current_step) > 30:  # Prevent too long steps
                        if current_step[0] == '<s>':
                            current_step = current_step[1:]
                        if current_step:  # Only add non-empty steps
                            generated_steps.append(current_step)
                        
                        current_step = ['<s>']
                        step_count += 1
                        
                        if step_count >= max_steps:
                            break
                    else:
                        current_step.append(token)
                
                if recipe_contexts and step_count == 0:
                    recipe_contexts = recipe_contexts[batch_size:]
            
            if current_step != ['<s>']:
                if current_step[0] == '<s>':
                    current_step = current_step[1:]
                if current_step:
                    generated_steps.append(current_step)
            
            all_predictions.append(generated_steps)
        
        return all_predictions
    
    def save(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save(filepath)
        
        np.savez(f"{filepath}_vocab.npz", 
                 vocab=np.array(self.vocab, dtype=object),
                 context=np.array([self.context]),
                 batch_size=np.array([self.batch_size]))

    def load(self, filepath: str):
        self.model = keras.models.load_model(filepath)

        data = np.load(f"{filepath}_vocab.npz", allow_pickle=True)
        self.vocab = data['vocab'].tolist()
        self.context = int(data['context'][0])

        if 'batch_size' in data:
            self.batch_size = int(data['batch_size'][0])
        else:
            self.batch_size = 32  # Default if loading from old model

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

        return self
