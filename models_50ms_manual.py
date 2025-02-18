import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define models
models = {
    "RandomForest": RandomForestClassifier(random_state=11),
    "SVM": SVC(random_state=11),
    "LogisticRegression": LogisticRegression(random_state=11, max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GradientBoosting": GradientBoostingClassifier(random_state=11),
    "DecisionTree": DecisionTreeClassifier(random_state=11),
}

# Path where the folds are stored
folds_path = r"B:\drone-audio\2024-12-14\splits_manual_50ms"

# Metrics storage
metrics_list = []

# Perform 10-fold cross-validation
for test_fold in range(10):
    print(f"\n--- Testing on Fold {test_fold} ---")

    # Initialize lists for training data
    X_train, y_train = [], []
    X_test, y_test = None, None

    # Load and combine 9 training folds, leaving one for testing
    for fold_idx in range(10):
        fold_dir = os.path.join(folds_path, f"fold_{fold_idx}")

        X = np.load(os.path.join(fold_dir, f"man_50_fold_{fold_idx}.npy"))
        y = np.load(os.path.join(fold_dir, f"labels_fold_{fold_idx}.npy"))  # Shape (samples, 3)

        if fold_idx == test_fold:
            X_test, y_test = X, y  # Test fold
        else:
            X_train.append(X)
            y_train.append(y)

    # Convert lists to numpy arrays
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{name}:")
        
        for col_idx in range(y_train.shape[1]):  # Iterate over each label
            model.fit(X_train, y_train[:, col_idx])  # Train on 9 folds
            y_pred = model.predict(X_test)  # Test on 1 fold

            # Compute metrics
            accuracy = accuracy_score(y_test[:, col_idx], y_pred)
            precision = precision_score(y_test[:, col_idx], y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test[:, col_idx], y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test[:, col_idx], y_pred, average='weighted', zero_division=0)

            print(f"  Label {col_idx+1}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

            # Store results
            metrics_list.append({
                "Model": name,
                "Test Fold": test_fold,
                "Label": col_idx + 1,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

# Convert results to a DataFrame and save to Excel
results_df = pd.DataFrame(metrics_list)
excel_filename = "results_manual_50.xlsx"
results_df.to_excel(excel_filename, index=False)

print(f"\nResults saved to {excel_filename}")
