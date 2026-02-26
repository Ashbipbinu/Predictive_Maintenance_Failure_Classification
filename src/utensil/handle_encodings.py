import pandas as pd
from sklearn.preprocessing import LabelEncoder

import os 
import pickle

def handle_target_encodings(y2_data: pd.DataFrame) -> pd.DataFrame:
   le = LabelEncoder()
   y2_labeled = le.fit_transform(y2_data)

   # Saving the label encoder
   dir = os.getcwd()
   folder = os.path.join(dir, 'models')
   os.makedirs(folder, exist_ok=True)
   file = os.path.join(folder, 'target_encoder.pkl')
   
   with open(file, 'wb') as file:
      pickle.dump(le, file)

   return y2_labeled