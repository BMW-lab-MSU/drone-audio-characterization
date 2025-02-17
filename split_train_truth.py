# This script loads in metadata and splits datasets for training (K-Fold CV)
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold

def stratified_kfold_split(features, labels, feature_name, num_folds=10, save_dir="kfold_splits", seed=11):
    """
    Splits the dataset into stratified K-Fold partitions and saves each fold (features & labels) as a separate subset.

    Parameters:
        features (numpy array): The features.
        labels (numpy array): The corresponding labels 
        num_folds (int): Number of folds for cross-validation.
        save_dir (str): Directory to save the split datasets.
        seed (int): Random seed for reproducibility.
    """
    # Combine propeller_type, angle, pan_angle into a single composite label
    composite_labels = np.column_stack((labels[:,0], labels[:,1], labels[:,2]))
    
    # Encode each unique (propeller_type, angle, pan_angle) combination as a single integer
    label_map = {tuple(x): idx for idx, x in enumerate(set(map(tuple, composite_labels)))}
    
    # Create a single encoded label for each sample
    encoded_labels = np.array([label_map[tuple(x)] for x in composite_labels])

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Loop over each fold
    for fold_idx, (_, fold_idx_subset) in enumerate(skf.split(features, encoded_labels)):
        fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save feature subset with fold-specific naming
        feature_filename = os.path.join(fold_dir, f"{feature_name}_fold_{fold_idx}.npy")
        metadata_filename = os.path.join(fold_dir, f"labels_fold_{fold_idx}.npy")
        indices_filename = os.path.join(fold_dir, f"indices_fold_{fold_idx}.npy")

        np.save(feature_filename, features[fold_idx_subset])
        np.save(metadata_filename, metadata[fold_idx_subset])
        np.save(indices_filename, fold_idx_subset)

metadata_filepath = r'B:\drone-audio\2024-12-14\metadata.csv'
spectrograms_20ms_filepath = r'B:\drone-audio\2024-12-14\melspectrograms_20.npy'  
spectrograms_50ms_filepath = r'B:\drone-audio\2024-12-14\melspectrograms_50.npy' 
manual_features_20ms_filepath = r'B:\drone-audio\2024-12-14\features_20ms.npy' 
manual_features_50ms_filepath = r'B:\drone-audio\2024-12-14\features_50ms.npy'  

# Load metadata from CSV
metadata = pd.read_csv(metadata_filepath)

# Extract the relevant columns
metadata = metadata[['angle', 'pan_angle', 'propeller_type']]

# Convert propeller_type to numbers (0,1,2) for classification
propeller_type_map = {"black plastic": 0, "orange plastic": 1, "quad black plastic": 2}
metadata['propeller_type'] = metadata['propeller_type'].map(propeller_type_map)
metadata = metadata.to_numpy()

# Load the spectrograms and "manual" features
spectrograms_20ms = np.load(spectrograms_20ms_filepath)
spectrograms_50ms = np.load(spectrograms_50ms_filepath)
manual_features_20ms = np.load(manual_features_20ms_filepath)
manual_features_50ms = np.load(manual_features_50ms_filepath)


# Example: Running for different feature sets
stratified_kfold_split(spectrograms_20ms, metadata, feature_name="spect_20", num_folds=10, save_dir=r"B:\drone-audio\2024-12-14\splits_20ms")
stratified_kfold_split(spectrograms_50ms, metadata, feature_name="spect_50", num_folds=10, save_dir=r"B:\drone-audio\2024-12-14\splits_50ms")
stratified_kfold_split(manual_features_20ms, metadata, feature_name="man_20", num_folds=10, save_dir=r"B:\drone-audio\2024-12-14\splits_manual_20ms")
stratified_kfold_split(manual_features_50ms, metadata, feature_name="man_50", num_folds=10, save_dir=r"B:\drone-audio\2024-12-14\splits_manual_50ms")