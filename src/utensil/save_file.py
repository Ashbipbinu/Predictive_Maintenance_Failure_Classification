import os
import pandas as pd

def save_file(file_path: str, df: pd.DataFrame) -> None:
    
    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(file_path, index=False)

    return 
