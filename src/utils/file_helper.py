import os 

def get_filename(file_path) -> str:
    return os.path.basename(file_path)


def validate_file_extension(file: str, allowed_types: list[str]) -> bool:
    _, ext = os.path.splitext(file)
    ext = ext.lower()[1:]  # remove dot and convert to lower case
    return ext in allowed_types