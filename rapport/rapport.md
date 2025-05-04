# Pre traitement et Choix de donnée

- statistiques descriptives: nombre de tokens par doc, classes, vue d'ensemble des jeux de données
- choix de traitement, impact ?

Pas de redondance
Clarté
Limites, difficulté, pistes

# Start of the report

# Dataset

## Description

Brief description of the dataset used for training and evaluation.

- **Source**: [Source name and link]
- **Size**: [Number of recipes]
- **Time period**: [When the data was collected]
- **Geographic coverage**: [Regional focus of recipes if applicable]

## Features

| Feature                   | Description                         | Type        | Example Values                |
| ------------------------- | ----------------------------------- | ----------- | ----------------------------- |
| Ingredients Count         | Number of ingredients in the recipe | Integer     | 5, 12, 20                     |
| Steps Count               | Number of preparation steps         | Integer     | 3, 8, 15                      |
| Recipe Type               | Category of the recipe              | Categorical | Main Dish, Dessert, Appetizer |
| Cooking Method            | Primary cooking technique           | Categorical | Baking, Frying, Boiling       |
| Difficulty Level          | Complexity rating                   | Ordinal     | Easy, Medium, Hard            |
| ...additional features... |                                     |             |                               |

# Recipe Time Prediction Models Benchmark

## Overview

This benchmark compares multiple regression models for predicting recipe preparation time based on various features. The goal is to identify the most accurate model for estimating how long a recipe will take to prepare.

## Target Variable

- **Preparation Time**: Total time in minutes to prepare the recipe (includes both active cooking and passive waiting time)
- **Distribution**: [Brief description of the distribution of preparation times]

## Methodology

### Data Preprocessing

- Feature engineering steps
- Handling of missing values
- Normalization/standardization approach
- Train/validation/test split: [e.g., 70%/15%/15%]

### Models Evaluated

1. **Linear Regression**

   - Implementation: [e.g., scikit-learn]
   - Hyperparameters: [if applicable]
   - Feature selection method: [if used]

2. **Random Forest Regression**

   - Implementation: [e.g., scikit-learn]
   - Number of trees: [value]
   - Max depth: [value]
   - Other hyperparameters: [list]

3. **Gradient Boosting Regression**

   - Implementation: [e.g., XGBoost]
   - Number of estimators: [value]
   - Learning rate: [value]
   - Other hyperparameters: [list]

4. **Support Vector Regression**

   - Implementation: [e.g., scikit-learn]
   - Kernel: [e.g., RBF, linear]
   - C: [value]
   - Epsilon: [value]

5. **Neural Network**
   - Implementation: [e.g., TensorFlow/Keras]
   - Architecture: [layer sizes]
   - Activation functions: [list]
   - Epochs: [value]
   - Batch size: [value]

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)
- Additional metrics: [if used]

### Cross-Validation

- Method: [e.g., k-fold cross-validation]
- Number of folds: [value]

## Results

### Performance Comparison

| Model             | MAE (min) | RMSE (min) | R²  | MAPE (%) | Training Time (s) | Inference Time (ms) |
| ----------------- | --------- | ---------- | --- | -------- | ----------------- | ------------------- |
| Linear Regression |           |            |     |          |                   |                     |
| Random Forest     |           |            |     |          |                   |                     |
| Gradient Boosting |           |            |     |          |                   |                     |
| SVR               |           |            |     |          |                   |                     |
| Neural Network    |           |            |     |          |                   |                     |

### Feature Importance Analysis

[Visualization or table showing the relative importance of different features across models]

### Residual Analysis

[Description and visualization of error distribution]

### Learning Curves

[Visualization showing model performance with varying training set sizes]

## Discussion

### Model Strengths and Weaknesses

| Model             | Strengths | Weaknesses |
| ----------------- | --------- | ---------- |
| Linear Regression |           |            |
| Random Forest     |           |            |
| Gradient Boosting |           |            |
| SVR               |           |            |
| Neural Network    |           |            |

### Error Analysis

- Common patterns in prediction errors
- Specific recipe types or time ranges that were challenging to predict
- Potential reasons for systematic errors

### Recommendations

- Best overall model: [model name]
- Best model for fast recipes: [model name]
- Best model for complex recipes: [model name]
- Computation-performance tradeoffs
