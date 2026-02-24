import pandas as pd
from sklearn.model_selection import train_test_split
from src.utensil.handle_encodings import handle_target_encodings
from src.utensil.handle_scalability import handle_scale
from src.utensil.save_file import save_file

import os


def built_features() -> pd.DataFrame:
    
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "interim", "cleaned_df.csv")

    clean_df = pd.read_csv(file_path)

    X = clean_df.drop(columns=["target", "failure_type"])
    y1 = clean_df["target"]
    y2 = clean_df["failure_type"]

    # Y2 encoded
    y2_encoded = handle_target_encodings(y2)

    # Splitting the data into X and y
    random_state = 42
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, y1, y2_encoded, test_size=0.2, random_state=random_state, stratify=y2_encoded
    )

    # Required StandardScaling as we are using Lasso regression
    print("Scaling procedure started")
    X_train_scale = handle_scale(X_train, is_train=True)
    X_test_scale = handle_scale(X_test, is_train=False)
    print("Scaling procedure ende successfully")


    train_data = pd.DataFrame(X_train_scale, columns=X_train.columns)
    train_data['target'] = y1_train
    train_data['failure_type'] = y2_train

    test_data = pd.DataFrame(X_test_scale, columns=X_test.columns)
    test_data['target'] = y1_test
    test_data['failure_type'] = y2_test

    # Save the above 2 data as train_processed.csv and test_processed.csv
    train_path = os.path.join(os.getcwd(), 'data', 'processed', 'train_data_processed.csv')
    test_path = os.path.join(os.getcwd(), 'data', 'processed', 'test_data_processed.csv')

    save_file(train_path, train_data)
    save_file(test_path, test_data)
    print("Saved processed files successfully")

    return train_data, test_data

if __name__ == "__main__":
    built_features()
