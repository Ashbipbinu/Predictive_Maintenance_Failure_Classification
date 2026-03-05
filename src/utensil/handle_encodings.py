import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def handle_target_encodings(y2_data: pd.Series) -> pd.Series:
    le = LabelEncoder()
    y2_labeled = le.fit_transform(y2_data)

    
    script_dir = os.getcwd()
    folder_path = os.path.join(script_dir, "models")
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "target_encoding.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(le, f)
            
        # Verify it exists
        if os.path.exists(file_path):
            print(f"Successfully saved to: {file_path}")
        else:
            print("File was not created.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

    return y2_labeled