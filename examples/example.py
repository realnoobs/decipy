import os
import decipy.executors as exe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')


def get_file_path(filename: str) -> str:
    """
    Get file in `dataset` directory
    """
    return os.path.join(DATA_DIR, filename)


def run():
    matrix = exe.DataMatrix(get_file_path('SampleData_5x5.csv'))
    beneficial = [True, True, True, True, True]
    weights = [0.1, 0.2, 0.3, 0.3, 0.2]
    saw = exe.WSM(matrix.data, beneficial, weights)
    print(saw.dataframe)


if __name__ == '__main__':
    run()
