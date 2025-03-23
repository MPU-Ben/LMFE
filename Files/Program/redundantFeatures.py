import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import training  # Ensure these modules are correctly defined
import testing
import ggcorrplot

# Constants
SPECIE_NAME = 'All_species'
SUBFOLD = 'F_1'
_ALL_DATA_PATH = f'../Files/{SUBFOLD}/AllDataset_All_speices_sequences.csv'
TRAIN_DATA_PATH_INITIAL = f'../Files/{SUBFOLD}/TrainDataset_All_speices_sequences_initial.csv'
_TEST_DATA_PATH_INITIAL = f'../Files/{SUBFOLD}/TestDataset_All_speices_sequences_initial.csv'

# Load dataset
data = pd.read_csv(_ALL_DATA_PATH)

# Separate features and labels
X = data.drop(columns=['label'])  # Assuming the label column is named 'label'
y = data['label']

# Calculate correlation matrix
x_corr = X.corr()

# Set correlation threshold
THRESHOLD = 0.8
high_corr_features = set()

# Identify and randomly select high-correlation features to remove
for i in range(len(x_corr.columns)):
    for j in range(i):
        if abs(x_corr.iloc[i, j]) > THRESHOLD:
            feature1 = x_corr.columns[i]
            feature2 = x_corr.columns[j]
            feature_to_remove = random.choice([feature1, feature2])
            high_corr_features.add(feature_to_remove)

# Remove redundant features from training and testing data
X_reduced = X.drop(columns=high_corr_features)
print(f'Initial feature count: {len(X_reduced.columns)}')
print(f'Features: {X_reduced.columns.tolist()}')
print(f'Redundant feature count: {len(high_corr_features)}')
print(f'Redundant features: {high_corr_features}')

# Metrics dictionary for initial evaluation
metrics_initial = {
    'Feature Count': [],
    'Accuracy': [],
    'F1 Score': [],
    'Recall': [],
    'Sensitivity': [],
    'Specificity': [],
    'Precision': [],
    'MCC': []
}

# Evaluate model performance after removing redundant features
print("--------- Evaluating model performance after removing redundant features ---------")
XTRAIN_initial, X_test_initial, yTRAIN_initial, y_test_initial = train_test_split(X_reduced, y, test_size=0.3, shuffle=True)

# Save initial split datasets
train_data_initial = pd.DataFrame(XTRAIN_initial)
train_data_initial['label'] = yTRAIN_initial
train_data_initial.to_csv(TRAIN_DATA_PATH_INITIAL, index=False)

test_data_initial = pd.DataFrame(X_test_initial)
test_data_initial['label'] = y_test_initial
test_data_initial.to_csv(_TEST_DATA_PATH_INITIAL, index=False)

# Train classifiers
print('------------ Training classifiers with reduced features -----------')
choices = {'XGB': 1}

argsForTrain_initial = {
    'optimunTrainDataPath': TRAIN_DATA_PATH_INITIAL,
    'optimunTestDataPath': _TEST_DATA_PATH_INITIAL,
    'model': choices,
    'specieName': SPECIE_NAME,
    'subfold': SUBFOLD
}

training.train(argsForTrain_initial)

# Evaluate classifier performance
print('--------- Evaluating classifier performance -----------')
argsForEval_initial = {
    'TrainDatasetPath': TRAIN_DATA_PATH_INITIAL,
    'testDatasetPath': _TEST_DATA_PATH_INITIAL,
    'AllDataset': _ALL_DATA_PATH,
    'splited': 1,
    'specieName': SPECIE_NAME,
    'subfold': SUBFOLD
}
metrics_result_initial = testing.evaluate(argsForEval_initial)

# Record metrics
metrics_initial['Feature Count'].append(len(X_reduced.columns))
metrics_initial['Accuracy'].append(metrics_result_initial['accuracy'])
metrics_initial['F1 Score'].append(metrics_result_initial['f1_score'])
metrics_initial['Sensitivity'].append(metrics_result_initial['sensitivity'])
metrics_initial['Specificity'].append(metrics_result_initial['specificity'])
metrics_initial['MCC'].append(metrics_result_initial['mcc'])
metrics_initial['Recall'].append(metrics_result_initial['recall'])
metrics_initial['Precision'].append(metrics_result_initial['precision'])

# Print evaluation results
print("Initial Evaluation Metrics after Removing Redundant Features:")
for key, value in metrics_result_initial.items():
    print(f"{key.capitalize()}: {value}")

# Create correlation heatmap
sorted_features = X_reduced.corr().abs().sum().sort_values(ascending=False).index
sorted_corr = X_reduced.corr().loc[sorted_features, sorted_features]
plt.figure(figsize=(25, 25))
sns.clustermap(sorted_corr, cmap='coolwarm', figsize=(25, 25), linewidths=.5, annot=True, annot_kws={"size": 6})
plt.title('Correlation Heatmap of High Correlation Features')
plt.show()

# Re-split training and testing data
TRAIN_DATA_PATH = f'../Files/{SUBFOLD}/TrainDataset_All_speices_sequences_Recur_redunt_features.csv'
_TEST_DATA_PATH = f'../Files/{SUBFOLD}/TestDataset_All_speices_sequences_Recur_redunt_features.csv'
XTRAIN, X_test, yTRAIN, y_test = train_test_split(X_reduced, y, test_size=0.3, shuffle=True)

# Save updated datasets
train_data = pd.DataFrame(XTRAIN)
train_data['label'] = yTRAIN
train_data.to_csv(TRAIN_DATA_PATH, index=False)

test_data = pd.DataFrame(X_test)
test_data['label'] = y_test
test_data.to_csv(_TEST_DATA_PATH, index=False)

# Initialize metrics for feature addition
metrics = {
    'Feature Count': [],
    'Accuracy': [],
    'F1 Score': [],
    'Recall': [],
    'Sensitivity': [],
    'Specificity': [],
    'Precision': [],
    'MCC': []
}

# Create a temporary dataset to track features
X_temp = X_reduced.copy()
feature_names = []
accuracies = []

# Add back high-correlation features and evaluate performance
for feature in high_corr_features:
    if feature not in X_temp.columns:
        X_temp[feature] = X[feature]

    print(f'Current feature count: {len(X_temp.columns)}')
    print(f'Current features: {X_temp.columns.tolist()}')

    # Re-split and save updated datasets
    XTRAIN_temp, X_test_temp, yTRAIN_temp, y_test_temp = train_test_split(X_temp, y, test_size=0.3, shuffle=True)
    train_data_temp = pd.DataFrame(XTRAIN_temp)
    train_data_temp['label'] = yTRAIN_temp
    train_data_temp.to_csv(TRAIN_DATA_PATH, index=False)

    test_data_temp = pd.DataFrame(X_test_temp)
    test_data_temp['label'] = y_test_temp
    test_data_temp.to_csv(_TEST_DATA_PATH, index=False)

    # Train classifier
    print(f'------------ Training classifier with added feature {feature} -----------')
    training.train(argsForTrain_initial)

    # Evaluate classifier performance
    print('--------- Evaluating classifier -----------')
    metrics_result = testing.evaluate(argsForEval_initial)

    # Record metrics
    metrics['Feature Count'].append(len(X_temp.columns))
    metrics['Accuracy'].append(metrics_result['accuracy'])
    metrics['F1 Score'].append(metrics_result['f1_score'])
    metrics['Sensitivity'].append(metrics_result['sensitivity'])
    metrics['Specificity'].append(metrics_result['specificity'])
    metrics['MCC'].append(metrics_result['mcc'])
    metrics['Recall'].append(metrics_result['recall'])
    metrics['Precision'].append(metrics_result['precision'])

    # Record feature name and accuracy
    feature_names.append(feature)
    accuracies.append(metrics_result['accuracy'])

# Create DataFrame for results
results_df = pd.DataFrame({
    'Feature': feature_names,
    'Accuracy': accuracies
})

# Sort and save results
sorted_results_df = results_df.sort_values(by='Accuracy', ascending=False)
results_df.to_csv('unsorted_feature_accuracies.csv', index=False)
sorted_results_df.to_csv('sorted_feature_accuracies.csv', index=False)

# Plot results
plt.figure(figsize=(20, 15))
plt.plot(feature_names, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. Added Features (Sorted by Accuracy)')
plt.xlabel('Features Added')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()