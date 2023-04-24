import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load the labels
labels_df = pd.read_csv('processed_data_labels.csv')
y = labels_df['outcome']

# Prepare the models
models = {
    'logistic_regression': LogisticRegression(penalty='l1', solver='liblinear', random_state=19),
    'gradient_boosting': GradientBoostingClassifier(random_state=19),
    'adaboost': AdaBoostClassifier(random_state=19),
    'catboost': CatBoostClassifier(silent=True),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
}

# Add the voting classifier
voting_classifier = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()], voting='hard'
)
models['voting_classifier'] = voting_classifier

# Iterate through the CSV files in the 'permutations' folder
results = []
for file in os.listdir('permutations'):
    if file.endswith('.csv'):
        # Load the data
        data_path = os.path.join('permutations', file)
        X = pd.read_csv(data_path)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Evaluate each model
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if model_name != 'voting_classifier' else None

            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            accuracy = accuracy_score(y_test, y_pred)

            results.append({
                'file': file,
                'model': model_name,
                'auc_score': auc,
                'accuracy': accuracy,
            })

# Output the results
results_df = pd.DataFrame(results)
print(results_df)