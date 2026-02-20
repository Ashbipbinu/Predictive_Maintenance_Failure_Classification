import pandas as pd
import numpy as np

import os

def load_data():
    
    folder = os.path.join('data', 'raw')
    os.makedirs(folder, exist_ok=True)
    
    file_path = os.path.join(folder, 'dataset')

load_data()