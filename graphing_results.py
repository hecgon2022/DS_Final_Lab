import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data from the CSV file
results_df = pd.read_csv('permutation_results.csv')

# Keep only the accuracy rows in the DataFrame
accuracy_df = results_df.drop(columns=['auc_score'])

# Melt the DataFrame to have separate columns for metric type (accuracy) and metric value
melted_df = accuracy_df.melt(id_vars=['file', 'model'], value_vars=['accuracy'], var_name='metric', value_name='value')

# Create the horizontal grouped bar chart
plt.figure(figsize=(15, 7))
sns.barplot(x='value', y='file', hue='model', data=melted_df, ci=None, orient='h')

# Customize the plot
plt.title('Model Performance Comparison by Accuracy')
plt.xlabel('Metric Value')
plt.ylabel('File')
plt.legend(title='Models', loc='best')
plt.xlim(0.6, None)  # Set the x-axis starting point to 0.5

# Show the plot
plt.show()