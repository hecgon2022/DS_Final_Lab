import os
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#grid search
#roc_curve
#report

#Trains, tests, and saves data of all the models throughout all the csv files in permutations folder

# Load the labels
labels_df = pd.read_csv('data/processed/processed_data_labels.csv')
y = labels_df['label']

# Prepare the models
models = {
    'logistic_regression': LogisticRegression(penalty='l1', solver='liblinear', random_state=19, n_jobs=4),
    'gradient_boosting': GradientBoostingClassifier(random_state=19, n_iter_no_change=4),
    'adaboost': AdaBoostClassifier(random_state=19),
    'catboost': CatBoostClassifier(silent=True, thread_count=4),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=4),
}

# Add the voting classifier
voting_classifier = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()], voting='hard', n_jobs=4
)
models['voting_classifier'] = voting_classifier

stacking_classifier = StackingClassifier(
    estimators=[(name, model) for name, model in models.items() if name != 'voting_classifier'], 
    final_estimator=GradientBoostingClassifier(random_state=19, n_iter_no_change=4),
    n_jobs=4
)
models['stacking_classifier'] = stacking_classifier

# Iterate through the CSV files in the 'permutations' folder
results = []
for file in os.listdir('data/permutations'):
    if file.endswith('.csv'):
        # Load the data
        data_path = os.path.join('data/permutations', file)
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
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba) if y_pred_proba is not None else (None, None, None)

            results.append({
                'file': file,
                'model': model_name,
                'auc_score': auc,
                'accuracy': accuracy,
                'fpr': fpr,
                'tpr': tpr
            })

# Output the results
results_df = pd.DataFrame(results)
results_df.to_pickle('permutation_results.pkl')
print(results_df)
