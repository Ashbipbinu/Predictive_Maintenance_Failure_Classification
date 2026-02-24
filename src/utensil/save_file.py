import os

def save_file(base_dir, df):
    
    file_path = os.path.join(base_dir, 'data', 'interim', 'cleaned_df.csv')
    
    # Extract ONLY the folder path ('data/interim')
    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(file_path, index=False)

    return 
