import os

BASE_PATH = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_PATH, 'dataset')


def get_file_path(directory: list, filename: str) -> str:
    """
    Get file in `dataset` directory
    """
    return os.path.join(DATA_DIR, filename)
