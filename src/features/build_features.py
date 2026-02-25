import pandas as pd
from sklearn.model_selection import train_test_split
from src.utensil.handle_encodings import handle_target_encodings
from src.utensil.save_file import save_file

import os


def built_features() -> pd.DataFrame:

    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "interim", "cleaned_df.csv")

    clean_df = pd.read_csv(file_path)

    # Encoding the failure type
    clean_df["failure_type_encoded"] = handle_target_encodings(clean_df["failure_type"])

    # Splitting the data into X and y
    X = clean_df.drop(columns=["target", "failure_type_encoded", "failure_type"])
    y1_binary = clean_df["target"]
    y2_multi = clean_df["failure_type_encoded"]

    # Splitting data into train and test
    X_train, X_test, y1_binary_train, y1_binary_test, y2_multi_train, y2_multi_test = (
        train_test_split(
            X, y1_binary, y2_multi, test_size=0.2, random_state=42, stratify=y2_multi
        )
    )

    # Saving the train and test data
    train_data = X_train.copy()
    train_data['target'] = y1_binary_train
    train_data['failure_type'] = y2_multi_train
    
    file_path = os.path.join(os.getcwd(), 'data', 'processed', 'train_processed.csv')
    save_file(file_path, train_data)

    test_data = X_test.copy()
    test_data['target'] = y1_binary_test
    test_data['failure_type'] = y2_multi_test

    file_path = os.path.join(os.getcwd(), 'data', 'processed', 'test_processed.csv')
    save_file(file_path, test_data)

if __name__ == "__main__":
    built_features()
