import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import time

class OptimizedBatchFeedForwardNN:
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
    
    def _prepare_train_unique_contexts(self, ingredients: List[List[str]], steps: List[List[str]], context: int) -> Tuple[np.ndarray, np.ndarray]:
        self.context = context
        
        if self.vocab is None:
            self._build_vocabulary(ingredients, steps)
        
        context_targets = defaultdict(Counter)
        
        # Process all recipes
        for i in range(len(ingredients)):
            ingredient_ids = [self.token_to_id.get(token, self.token_to_id['<UNKNOWN>']) 
                              for token in ingredients[i]]
            
            if len(ingredient_ids) < context:
                # Pad with unknown tokens
                ingredient_ids = ingredient_ids + [self.token_to_id['<UNKNOWN>']] * (context - len(ingredient_ids))
            
            step_ids = [self.token_to_id.get(step_token, self.token_to_id['<UNKNOWN>']) 
                        for step_token in steps[i]]
            
            # Generate all context windows for this recipe
            for k in range(len(ingredient_ids) - context + 1):
                context_window = tuple(ingredient_ids[k:k+context])
                
                # Count target frequencies for this context
                for step_id in step_ids:
                    context_targets[context_window][step_id] += 1
        
        print(f"Found {len(context_targets)} unique context-grams")
        
        X_data = []
        y_data = []
        sample_weights = []
        
        for context_gram, target_counts in context_targets.items():
            total_count = sum(target_counts.values())
            for target_id, count in target_counts.items():
                X_data.append(list(context_gram))
                y_data.append(target_id)
                sample_weights.append(count / total_count)  # Normalize by total count
        
        return np.array(X_data, dtype=np.int32), np.array(y_data, dtype=np.int32), np.array(sample_weights, dtype=np.float32)
    
    def _prepare_test_batch(self, ingredients: List[List[str]]):
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Fit the model first!")
        
        all_contexts = []
        
        for ingredient_list in ingredients:
            ingredient_ids = [self.token_to_id.get(token, self.token_to_id['<UNKNOWN>']) for token in ingredient_list]

            if len(ingredient_ids) < self.context:
                ingredient_ids = ingredient_ids + [self.token_to_id['<UNKNOWN>']] * (self.context - len(ingredient_ids))
            
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
        
        start_time = time.time()
        self.batch_size = batch_size
        
        print("Preparing unique context-grams for training...")
        X_train, y_train, sample_weights = self._prepare_train_unique_contexts(ingredients, steps, context)
        
        vocab_size = len(self.vocab)
        print(f"Vocab size: {vocab_size}")
        print(f"Unique training samples: {len(X_train)}")
        print(f"Data preparation took {time.time() - start_time:.2f} seconds")
        print(f"One sample: {X_train[0]} - {y_train[0]}")

        print("Building model...")
        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=context, name="embedding"))
        
        self.model.add(keras.layers.GlobalAveragePooling1D(name="pooling"))

        self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Dense(hidden_dim, activation='relu', name="hidden"))

        if dropout is not None:
            self.model.add(keras.layers.Dropout(dropout))

        self.model.add(keras.layers.Dense(vocab_size, activation='softmax', name="output"))
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        print(self.model.summary())
        
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train, sample_weights)
        ).shuffle(buffer_size=10000)
        
        val_size = int(len(X_train) * validation_split)
        train_size = len(X_train) - val_size
        
        train_ds = train_dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = train_dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Train with TF Dataset
        start_train_time = time.time()
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks= [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
            ]
        )

        print(f"Training took {time.time() - start_train_time:.2f} seconds")
        print(f"Total process took {time.time() - start_time:.2f} seconds")

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
                
                predictions = self.model.predict_on_batch(batch_input)
                
                next_token_id = np.argmax(predictions[0])
                next_token = self.id_to_token[next_token_id]

                if next_token == '</s>':
                    break
                
                current_tokens.append(next_token)
                
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
        
        @tf.function(experimental_relax_shapes=True)
        def predict_batch_tf(inputs):
            return self.model(inputs, training=False)

        all_contexts = self._prepare_test_batch(ingredients)
        all_predictions = []
        
        for recipe_contexts in all_contexts:
            generated_steps = []
            current_step = ['<s>']
            
            step_count = 0
            while step_count < max_steps:
                # Use the beginning contexts for initial predictions
                if recipe_contexts and step_count == 0:
                    # Process multiple batches at once for efficiency
                    batch_size = min(self.batch_size, len(recipe_contexts))
                    batch_contexts = np.array(recipe_contexts[:batch_size])
                    
                    # Get batch predictions using the decorated function
                    batch_predictions = predict_batch_tf(batch_contexts).numpy()
                    
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
                    prediction = predict_batch_tf(dummy_context).numpy()
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