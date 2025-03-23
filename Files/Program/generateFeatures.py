import time
import features
import os
import pandas as pd
from sklearn.model_selection import train_test_split
start_time=time.time()


def readFastaAndLabels(fastaFileName, labelsFileName):
    # Read FASTA file
    with open(fastaFileName, 'r') as fastaFile:
        sequences = []
        genome = ''
        for line in fastaFile:
            if line.startswith('>'):
                if genome:  # If genome is not empty, save it
                    sequences.append(genome.upper())
                    genome = ''
            else:
                genome += line.strip()
        if genome:  # Append the last genome
            sequences.append(genome.upper())

    # Read labels file
    with open(labelsFileName, 'r') as labelsFile:
        labels = [line.strip().replace(' ', '') for line in labelsFile if line.strip()]

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    encoded_labels = LabelEncoder().fit_transform(labels)

    # Ensure the number of sequences matches the number of labels
    assert len(sequences) == len(encoded_labels), 'The number of sequences and labels must be equal.'

    return sequences, encoded_labels

def process_datasets(files, labels, ss_paths, args, root_path=None, subfold=None):
    for file, label, ss_path in zip(files, labels, ss_paths):
        file_path = os.path.join(root_path, file)
        label_path = os.path.join(root_path, label)
        ss_path = os.path.join(root_path, ss_path)

        X, y = readFastaAndLabels(file_path, label_path)
        print(f'------- Starting feature extraction for {file_path} -----------')

        S = features.gF(X, y, ss_path, **args)
        print(S.shape)

        specie_name = os.path.splitext(os.path.basename(file_path))[0]
        X, y = S[:, :-1], S[:, -1]
        # Convert X and y to DataFrames
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns=['target'])  # Use an appropriate column name for labels
        # Combine the features and labels into one DataFrame
        full_data = pd.concat([X_df, y_df], axis=1)

        # Save the full dataset
        full_data.to_csv(f'../Files/{subfold}/{specie_name}_all_dataset.csv', index=False)

        # Load the saved dataset
        data = pd.read_csv(f'../Files/{subfold}/{specie_name}_alldataset.csv')
        X_ = data.iloc[:, :-1]
        y_ = data.iloc[:, -1]

        # Split into training and testing sets
        print('------- Splitting into training and testing sets ---------')
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, shuffle=True, test_size=0.3, stratify=y_)

        # Save the training and testing datasets
        # Save training and testing datasets directly
        X_train_df = pd.DataFrame(X_train)
        y_train_df = pd.DataFrame(y_train, columns=['target'])  # Assuming 'target' is the name for the labels
        train_data = pd.concat([X_train_df, y_train_df], axis=1)
        train_data.to_csv(f'../Files/{subfold}/{specie_name}_Training_dataset.csv', index=False)

        X_test_df = pd.DataFrame(X_test)
        y_test_df = pd.DataFrame(y_test, columns=['target'])  # Assuming 'target' is the name for the labels
        test_data = pd.concat([X_test_df, y_test_df], axis=1)
        test_data.to_csv(f'../Files/{subfold}/{specie_name}_Testing_dataset.csv', index=False)

def main():
    start_time = time.time()
    print('Start time:', start_time)
    root_path = '../Datasets/'
    # Example dataset configurations
    files = ['All_species/All_species_sequences.fa']
    labels = ['All_species/All_species_labels.txt']
    ss_paths = ['All_species/All_species_secondary_struct.txt']

    # Feature extraction arguments
    args = { 'SeqLength':1,'zCurve': 1, 'gcContent': 1, 'atgcRatio': 1, 'NAC': 1,
             'DNC': 1,'TNC': 1, 'orf_coverage': 1, 'orf_length': 1,'count_orfs': 1, 'MFE': 1 }

    process_datasets(files, labels, ss_paths, args,root_path,'F_1')

    end_time = time.time()
    print('End time:', end_time)
    print('Total time taken:', (end_time - start_time) / 60)

if __name__ == "__main__":
    main()