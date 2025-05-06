"""
This file contains the text-generation model.
- We train the model following the structure `ingredients: ... ; recipe: ...`
- We will use a transformer model to generate the recipe based on the ingredients.

Steps:
1. Word to Vector (W2V) embedding
2. Transformer model
3. Train the model
4. Save the model
5. Load the model
6. Generate the recipe
"""

import os
from gensim.models import Word2Vec
