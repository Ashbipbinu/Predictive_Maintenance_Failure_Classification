import pandas as pd
import os
import yaml

from src.utensil.handle_encodings import handle_target_encodings
from src.utensil.save_file import save_file
from src.utensil.handle_data_split import handle_data_split
from src.utensil.save_location_config import save_location_config

def built_features() -> pd.DataFrame:

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file) 
        clean_df_file_path = config['data']['cleaned_df']
    
    clean_df = pd.read_csv(clean_df_file_path)

    # Encoding the failure type
    clean_df["failure_type"] = handle_target_encodings(clean_df["failure_type"])

    # Temperature difference
    clean_df['temp_diff_k'] = abs(clean_df['process_temperature_k'] - clean_df['air_temperature_k'])
    
    # Getting the correlation of target and failure_type with features
    target_corr = clean_df.corr()['target'].sort_values(ascending=False)
    failure_type_corr = clean_df.corr()['failure_type'].sort_values(ascending=False)

    print(target_corr, '\n', failure_type_corr)
     
    # Splitting the data into X and y
    columns_to_drop = ["target", "failure_type", 'process_temperature_k']

    X_train, X_test, y1_binary_train, y1_binary_test, y2_multi_train, y2_multi_test = (
        handle_data_split(
            data=clean_df,
            test_size=0.2,
            columns_to_drop= columns_to_drop,
        )
    )

    # Saving the train and test data
    train_data = X_train.copy()
    train_data["target"] = y1_binary_train
    train_data["failure_type"] = y2_multi_train

    train_file_path = os.path.join(os.getcwd(), "data", "processed", "train_processed.csv")
    save_file(train_file_path, train_data)

    test_data = X_test.copy()
    test_data["target"] = y1_binary_test
    test_data["failure_type"] = y2_multi_test

    test_file_path = os.path.join(os.getcwd(), "data", "processed", "test_processed.csv")
    save_file(test_file_path, test_data)

    # Saving the data location - train data and test data
    save_location_config(target_loc='data', key_name='train_data', file_path=train_file_path)
    save_location_config(target_loc='data', key_name='test_data', file_path=test_file_path)



if __name__ == "__main__":
    built_features()
