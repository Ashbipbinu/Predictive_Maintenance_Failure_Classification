import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

import os


def handle_scale(data: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    
    if is_train:
        scale = StandardScaler()
        train_scale = scale.fit_transform(data)

        # Saving the scale
        dir = os.getcwd()
        folder = os.path.join(dir, "models")
        os.makedirs(folder, exist_ok=True)
        file_name = os.path.join(folder, "scale.pkl")

        with open(file_name, "wb") as file:
            pickle.dump(scale, file)

        return pd.DataFrame(
            train_scale,
            columns=data.columns,
            index=data.index
        )
    
    with open("models/scale.pkl", "rb") as file:
        scale = pickle.load(file)
        test_scale = scale.transform(data)
        
        return pd.DataFrame(
            test_scale,
            columns=data.columns,
            index=data.index
        )
    

if  __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data', 'interim', 'cleaned_df.csv')
    df = pd.read_csv(data_path)

    print(handle_scale(df, is_train=True))
