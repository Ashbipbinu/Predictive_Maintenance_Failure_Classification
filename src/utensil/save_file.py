import os

def save_file(file_path, df):
    
    # Extract ONLY the folder path ('data/interim')
    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(file_path, index=False)

    return 
