# Wine Classification - Supervised Learning Project 🍷

This repository contains a full supervised machine learning pipeline for classifying wine types based on chemical properties. The project was developed as part of an academic assignment in machine learning.

## Project Flow:
- Data loading and initial inspection
- Feature Engineering (Scaling, Feature Selection)
- Model experimentation (DecisionTree, SVM, RandomForest)
- Hyperparameter tuning using GridSearchCV with 5-fold cross-validation
- Evaluation based on Macro-F1 Score
- Final model training on the entire train set
- Predictions and performance evaluation on test set

## Best Model:
- SVM with StandardScaler + SelectKBest (k=8)
- Parameters: C=1, kernel='rbf'
- Achieved a Macro-F1 of 0.980 on cross-validation

## Folder Structure:
- data/ — Train and Test datasets
- notebook/ — Jupyter notebook with all experiments
- results/ — Final model performance metrics

## Technologies Used:
- Python, Pandas, Scikit-learn, Seaborn, Matplotlib
