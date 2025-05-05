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

- **Source**: [RecipeNLG](https://recipenlg.cs.put.poznan.pl/)
- **Size**: 
   - 231 636 (total)
   - 221 457 (regression task)
- **Geographic coverage**: All around the world but mostly USA, Europe and Asia.

## Features

| Feature        | Description                                           | Type            | Example Values                             |
|----------------|-------------------------------------------------------|------------------|--------------------------------------------|
| name           | Name of the recipe                                    | String           | "Spaghetti Bolognese", "Chicken Curry"     |
| minutes        | Preparation and cooking time in minutes               | Integer          | 30, 45, 120                                 |
| tags           | List of tags describing the recipe                    | List of Strings  | ["dinner", "easy", "vegetarian"]           |
| nutrition      | Nutritional information per serving                   | List of Floats   | [500.0, 20.0, 30.0, 5.0, 10.0, 60.0]        |
| steps          | Step-by-step cooking instructions                     | List of Strings  | ["Boil water", "Add pasta", "Stir sauce"]  |
| n_steps        | Number of steps in the recipe                         | Integer          | 3, 5, 10                                    |
| description    | Detailed description of the recipe                    | String           | "A classic Italian pasta dish..."          |
| ingredients    | List of ingredients required for the recipe           | List of Strings  | ["spaghetti", "ground beef", "tomato"]     |
| n_ingredients  | Number of ingredients in the recipe                   | Integer          | 5, 8, 12                                    |



# Recipe Time Prediction Models Benchmark

## Overview

This benchmark compares multiple regression models for predicting recipe preparation time based on various features. The goal is to identify the most accurate model for estimating how long a recipe will take to prepare.

## Target Variable

- **Preparation Time**: Total time in minutes to prepare the recipe (includes both active cooking and passive waiting time)
- **Distribution**: Brief description of the distribution of preparation times
![image](../img/distribution_recipe_time.png)
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

2. **Bayesian Ridge**


3. **Feedforward Neural Network**


4. **Gated Recurrent Unit Neural Network**

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

| Model              | MAE       | MSE         | RMSE      | R²     | Train Time (s) | Embedding Type |
|--------------------|----------|------------|----------|----------|----------------|---------------|
| Linear Regression | 15.1705   | 634.5855   | 25.1910  | 0.7006   | 10.93          | TF-IDF        |
| Bayesian Ridge    | 15.1901   | 635.0655   | 25.2005  | 0.7004   | 17.44          | TF-IDF        |
| Feedforward NN   |  9.3618   | 340.7969   | 18.4607  | 0.8392   | 253.49         | TF-IDF        |
| Linear Regression | 23.6824   | 1374.7565  | 37.0777  | 0.3514   | 4.33           | Word2Vec      |
| Bayesian Ridge    | 23.6587   | 1374.4485  | 37.0736  | 0.3515   | 11.34          | Word2Vec      |
| Feedforward NN   | 20.0696   | 1174.3258  | 34.2684  | 0.4459   | 137.41         | Word2Vec      |



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
