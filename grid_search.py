import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load the data
data_path = 'data/permutations/add_win_rate_6.csv'
X = pd.read_csv(data_path)

# Load the labels
labels_df = pd.read_csv('data/processed/processed_data_labels.csv')
y = labels_df['label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare the base models
base_models = {
    'logistic_regression': LogisticRegression(penalty='l1', solver='liblinear', random_state=19, n_jobs=4),
    'gradient_boosting': GradientBoostingClassifier(random_state=19, n_iter_no_change=4),
    'adaboost': AdaBoostClassifier(random_state=19),
    'catboost': CatBoostClassifier(silent=True, thread_count=4),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=4),
}

# Define the parameter grid for the grid search
param_grid = {
    'final_estimator__learning_rate': [0.01, 0.1, 0.2],
    'final_estimator__n_estimators': [100, 200],
    'final_estimator__max_depth': [3, 4, 5],
}

# Create the stacking classifier
stacking_classifier = StackingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    final_estimator=GradientBoostingClassifier(random_state=19, n_iter_no_change=4),
    n_jobs=4,
)

# Run the grid search
grid_search = GridSearchCV(stacking_classifier, param_grid, cv=5, n_jobs=4, scoring='roc_auc', verbose=1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Best parameters:  {'final_estimator__learning_rate': 0.01,
# 'final_estimator__max_depth': 3, 
# 'final_estimator__n_estimators': 100}
#
# Best score:  0.7999913115190763
