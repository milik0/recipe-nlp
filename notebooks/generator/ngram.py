import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import nltk
from nltk.util import ngrams
import os
import sys
import pickle
import random
import ast
import math
import logging

nltk.download('punkt', quiet=True)

class RecipeNGramModel:
    def __init__(self, n=8, smoothing='laplace', alpha=0.1, random_state=None):
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha
        self.vocabulary = set()
        self.logger = logging.getLogger(__name__)

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        self.model = {}

    def _tokenize_steps(self, steps: List[str]) -> List[str]:
        tokens = []
        for step in steps:
            step_tokens = step.split(" ")
            tokens.extend(step_tokens)
            if step != steps[-1]:
                tokens.append('<STEPSEP>')
        return tokens

    def _build_model(self, tokenized_texts: List[List[str]]) -> dict:
        self.logger.info(f"Building ngram model on tokenized text: {len(tokenized_texts)}")
        all_order_ngrams = {}
        for i in range(1, self.n + 1):
            all_order_ngrams[i] = []
            for tokens in tokenized_texts:
                tokens_with_markers = ['<s>'] + tokens + ['</s>']
                sentence_ngrams = list(ngrams(tokens_with_markers, i))
                all_order_ngrams[i].extend(sentence_ngrams)
                self.vocabulary.update(tokens_with_markers)

        models = {}

        for i in range(1, self.n + 1):
            # Count n-grams
            ngram_counts = Counter(all_order_ngrams[i])
            
            if i > 1:
                context_counts = Counter(all_order_ngrams[i-1])
            else:
                context_counts = Counter([() for _ in range(len(tokenized_texts))])
            
            probability_model = defaultdict(dict)
            
            for ngram in ngram_counts:
                if i > 1:
                    context = ngram[:-1]
                    word = ngram[-1]
                else:
                    context = ()
                    word = ngram[0]
                
                count = ngram_counts[ngram]
                context_count = context_counts[context]
                
                if self.smoothing == 'laplace':
                    probability = (count + self.alpha) / (context_count + self.alpha * len(self.vocabulary))
                else:
                    probability = count / context_count if context_count > 0 else 0

                probability_model[context][word] = probability
            
            models[i] = probability_model

        return models

    def fit(self, ingredients_lists: List[List[str]], steps_lists: List[List[str]]):
        if len(ingredients_lists) != len(steps_lists):
            raise ValueError(f"Inconsistent dimensions: {len(ingredients_lists)} ingredients lists vs {len(steps_lists)} steps lists")
        
        tokenized_texts = []
        for ingredients, steps in zip(ingredients_lists, steps_lists):
            step_tokens = self._tokenize_steps(steps)
            recipe_tokens = ingredients + ['<RECIPESEP>'] + step_tokens
            tokenized_texts.append(recipe_tokens)

        self.model = self._build_model(tokenized_texts)
        return self

    def _get_probability(self, word: str, context: tuple) -> float:
        context_len = len(context)
        for order in range(min(self.n, context_len + 1), 0, -1):
            if order == 1:
                # Unigram model
                if () in self.model[1] and word in self.model[1][()]:
                    return self.model[1][()][word]
            else:
                # Higher order n-gram model
                current_context = context[-(order-1):] if context_len >= (order-1) else context
                if current_context in self.model[order] and word in self.model[order][current_context]:
                    return self.model[order][current_context][word]
        
        # Fall back to smoothed probability if word not found in any model
        if self.smoothing == 'laplace':
            return self.alpha / (len(self.vocabulary) * self.alpha)
        
        return 0
    
    def _generate_next_word(self, context: tuple):
        context_len = len(context)

        # Try to use highest order n-gram model available
        for order in range(min(self.n, context_len + 1), 0, -1):
            if order == 1:
                current_context = ()
            else:
                current_context = context[-(order-1):] if context_len >= (order-1) else context
                
            if current_context in self.model[order]:
                words_probs = self.model[order][current_context].copy()
                
                if self.smoothing == 'laplace':
                    # Add smoothed probabilities for unseen words
                    context_total = sum(words_probs.values())
                    denominator = context_total + self.alpha * len(self.vocabulary)
                    for word in self.vocabulary:
                        if word not in words_probs:
                            words_probs[word] = self.alpha / denominator
                
                # Select next word based on probabilities
                items = list(words_probs.items())
                words = [item[0] for item in items]
                probs = [item[1] for item in items]

                probs_sum = sum(probs)
                if probs_sum > 0:
                    probs = [p/probs_sum for p in probs]
                    
                    try:
                        next_word = np.random.choice(words, p=probs)
                        return next_word
                    except ValueError:
                        continue

        # If no suitable context found, fall back to random word
        if self.vocabulary:
            return random.choice(list(self.vocabulary))
        
        return None
    
    def _generate_steps(self, ingredients: List[str], max_length=100) -> List[str]:
        self.logger.info("Start with ingredients and add recipe separator")
        context = tuple(['<s>'] + ingredients + ['<RECIPESEP>'])
        if len(context) > self.n - 1:
            context = context[-(self.n-1):]

        generated_tokens = []
        step_tokens = []
        current_step = []

        self.logger.info("Generating tokens...")
        while len(generated_tokens) < max_length:
            next_word = self._generate_next_word(context)

            if next_word is None or next_word == '</s>':
                # End of generation
                if current_step:
                    step_tokens.append(current_step)
                break
            
            generated_tokens.append(next_word)

            if next_word == '<STEPSEP>':
                # End of a step
                if current_step:
                    step_tokens.append(current_step)
                    current_step = []
            else:
                current_step.append(next_word)
            
            # Update context for the next prediction
            context = tuple(list(context)[1:] + [next_word]) if len(context) >= self.n - 1 else tuple(list(context) + [next_word])
        
        # Add the last step if it exists
        if current_step:
            step_tokens.append(current_step)
        
        # Convert token lists to strings
        steps = []
        for step in step_tokens:
            step_text = ' '.join(step)
            # Basic post-processing
            step_text = step_text.capitalize()
            if not step_text.endswith('.'):
                step_text += '.'
            steps.append(step_text)
        
        return steps

    def predict(self, ingredients: List[str]) -> List[str]:
        return self._generate_steps(ingredients, max_length=100)



    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)