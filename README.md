# LMFE Usage Guide

LMFE (lncRNA Multi-Feature Fusion Ensemble Learning) is a Python-based tool designed to extract features from RNA sequences, train machine learning models, and evaluate their performance in classifying sequences as either long non-coding RNA (lncRNA) or messenger RNA (mRNA). The tool supports a variety of feature extraction methods, multiple machine learning algorithms, and evaluation metrics.


This guide provides instructions on how to set up, run, and interpret results from LMFE using the provided scripts.
# Prerequisites
## Dataset
* The data in the dataset folder is file data
* The benchmark folder is the benchmark dataset, which contains data of 10 species
* The independent folder is the independent test set, which contains datasets of 6 species
* The unbalanced folder is the unbalanced dataset, which contains datasets of 4 species
## Software Requirements

Python 3.x: Ensure Python is installed on your system.<br>
pip install numpy pandas scikit-learn matplotlib seaborn xgboost biopython joblib shap lime deap <br>

## Hardware: 
A standard computer with sufficient memory and processing power for handling RNA sequence datasets.

## Input Data
LMFE requires the following input files:
1. FASTA File: Contains RNA sequences in FASTA format (e.g., All_species_sequences.fa).
2. Labels File: A text file with corresponding labels (e.g., All_species_labels.txt), where each line represents a label (e.g., lncRNA or mRNA).
3. Secondary Structure File: A text file containing secondary structure predictions in dot-bracket notation with minimum free energy (MFE) values (e.g., All_species_secondary_struct.txt).

Example FASTA file:

>seq1
AUGCCGUAA
>seq2
GUACCGUAA

Example Labels file:

1
0

Example Secondary Structure file:

AUGCCGUAA  ((....)) (-5.60)
GUACCGUAA  (.......) (-3.20)

## Installation
1. Clone or Download the Scripts: Place all provided Python files in a single directory.
2. Directory Structure: Create a directory structure as follows:

LMFE/ <br>
├── Datasets/ <br>
│   ├── All_species/ <br>
│   │   ├── All_species_sequences.fa <br>
│   │   ├── All_species_labels.txt <br>
│   │   └── All_species_secondary_struct.txt <br>
├── Files/ <br>
│   ├── F_1/ <br>
└── [Python scripts] <br>
4. Verify Dependencies: Ensure all required libraries are installed.

# Usage Instructions
## Step 1: Feature Extraction <br>
The features.py script extracts various features from RNA sequences, including sequence-based and structural features.

## Command
Run the main.py script to process datasets and extract features: <br>
```python
python main.py
```
## Configuration
Edit main.py to specify input files and feature extraction options:
```python
root_path = '../Datasets/'
files = ['All_species/All_species_sequences.fa']
labels = ['All_species/All_species_labels.txt']
ss_paths = ['All_species/All_species_secondary_struct.txt']
args = {
    'SeqLength': 1, 'zCurve': 1, 'gcContent': 1, 'atgcRatio': 1, 'NAC': 1,
    'DNC': 1, 'TNC': 1, 'orf_coverage': 1, 'orf_length': 1, 'count_orfs': 1, 'MFE': 1
}
subfold = 'F_1'
process_datasets(files, labels, ss_paths, args, root_path, subfold)
```
## Available Features
* SeqLength: Sequence length.
* zCurve: Z-curve features (x, y, z axes).
* gcContent: GC content percentage.
* atgcRatio: AT/GC ratio.
* NAC: Nucleotide composition (A, C, G, U).
* DNC: Dinucleotide composition.
* TNC: Trinucleotide composition.
* orf_coverage: Open Reading Frame (ORF) coverage.
* orf_length: Total ORF length.
* count_orfs: Number of ORFs.
* MFE: Minimum Free Energy and secondary structure features (e.g., base pairs, loops).
## Output
Feature-extracted datasets are saved as CSV files in ../Files/F_1/:
* [specie_name]_all_dataset.csv
* [specie_name]_Training_dataset.csv
* [specie_name]_Testing_dataset.csv
## Step 2: Model Training
The training.py script trains a machine learning model (default: XGBoost) using the extracted features.

## Command
Run the train_evaluate.py script to train a model:
```python
python train_evaluate.py
```
### Configuration
Edit train_evaluate.py to specify training parameters:
```python
specie_name = 'All_species'
subfold = 'F_1'
TrainDataPath = f'../Files/{subfold}/TrainDataset_All_speices_sequences.csv'
TestDataPath = f'../Files/{subfold}/TestDataset_All_speices_sequences.csv'
choices = {'XGB': 1}  # Enable XGBoost
argsForTrain = {
    'optimunTrainDataPath': TrainDataPath,
    'optimunTestDataPath': TestDataPath,
    'model': choices,
    'specieName': specie_name,
    'subfold': subfold
}
training.train(argsForTrain)
```
## Output
* Trained model saved as ../Files/F_1/dumpModel.pkl.
## Step 3: Model Evaluation
The testing.py script evaluates the trained model’s performance.

## Command
Run the evaluation part of train_evaluate.py:
```python
python train_evaluate.py
```
## Configuration
```python
Edit train_evaluate.py:
```
```python
argsForEval = {
    'TrainDatasetPath': TrainDataPath,
    'testDatasetPath': TestDataPath,
    'AllDataset': AllDataPath,
    'splited': 1,
    'specieName': specie_name,
    'subfold': subfold
}
testing.evaluate(argsForEval)
```
## Output
Performance metrics printed to the console (e.g., Accuracy, Precision, Recall, auROC, F1 Score, MCC).
Predictions saved as ../Files/F_1/LMFE_[specieName]_predictions.csv.

## Step 4: Compare Multiple Classifiers (Optional)
The differentMethods.py script compares multiple classifiers using cross-validation.

## Command
Run run_different_methods.py:
```python
python run_different_methods.py
```
## Configuration
```python
Edit run_different_methods.py:
```
```python
AllDataPath = '../Files/F_1/AllDataset_All_speices_sequences.csv'
argsForClassifier = {
    'nFCV': 5,  # Number of folds for cross-validation
    'dataset': AllDataPath,
    'auROC': 1, 'boxPlot': 1, 'accPlot': 1, 'timePlot': 1
}
differentMethods.runDifferentMethods(argsForClassifier)
```
### Output
Metrics saved to ../Files/evaluationMetrics.csv.
Plots (e.g., box plot, accuracy plot) saved in ../Files/.
Step 5: Feature Selection (Optional)
The third script (feature selection) removes redundant features and evaluates their impact.

## Command
Run the feature selection script:
```python
python [feature_selection_script_name].py
```
##Output
* Updated datasets with reduced features.
* CSV files with accuracy results (unsorted_feature_accuracies.csv, sorted_feature_accuracies.csv).
* Plot of accuracy vs. added features.
# Interpreting Results
* Metrics: Key metrics include:
* Accuracy: Percentage of correctly classified sequences.
* ROC: Area under the Receiver Operating Characteristic curve.
* F1 Score: Harmonic mean of precision and recall.
* MCC: Matthews Correlation Coefficient.
* Predictions: Check the LMFE_[specieName]_predictions.csv file for true labels, predicted labels, and probabilities.
# Troubleshooting
* Missing Files: Ensure all input files are in the correct directories.
* Dependency Errors: Verify all libraries are installed.
* Data Mismatch: Check that the number of sequences matches the number of labels and secondary structures.
