import os 

def get_filename(file_path) -> str:
    return os.path.basename(file_path)