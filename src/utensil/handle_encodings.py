import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def handle_target_encodings(y2_data: pd.Series) -> pd.Series:
    le = LabelEncoder()
    y2_labeled = le.fit_transform(y2_data)

    # Saving the scale
    root_dir = os.getcwd()
    folder = os.path.join(root_dir, "models")
    os.makedirs(folder, exist_ok=True)
    file_name = os.path.join(folder, "target_encodings.pkl")

    try:
        with open(file_name, "wb") as f:
            pickle.dump(le, f)
        print(f"✅ Success! Saved to: {file_name}")
    except Exception as e:
        print(f"❌ Error saving: {e}")

    return y2_labeled