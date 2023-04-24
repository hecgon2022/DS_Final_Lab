import pandas as pd
import matplotlib.pyplot as plt

#Shows all ROC curve graphs per file for each model one by one

# Load the results from the pickle file
results_df = pd.read_pickle('permutation_results.pkl')

# Get the unique file names
unique_files = results_df['file'].unique()

# Create a ROC curve plot for each file with each model
for file in unique_files:
    plt.figure(figsize=(10, 6))
    for _, row in results_df[results_df['file'] == file].iterrows():
        if row['fpr'] is not None and row['tpr'] is not None:
            plt.plot(row['fpr'], row['tpr'], label=f"{row['model']} (AUC = {row['auc_score']:.2f})")
    
    plt.title(f"ROC curve for {file}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
