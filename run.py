import pandas as pd
import numpy as np
from decipy.executors import *

matrix = np.array([
    [4, 3, 2, 4],
    [5, 4, 3, 7],
    [6, 5, 5, 3],
])
alts = ['A1', 'A2', 'A3']
crits = ['C1', 'C2', 'C3', 'C4']
beneficial = [True, True, True, True]
weights = [0.10, 0.20, 0.30, 0.40]
xij = pd.DataFrame(matrix, index=alts, columns=crits)


class RankSimilarityApp:

    def __init__(self):
        kwargs = {
            'data': xij,
            'beneficial': beneficial,
            'weights': weights,
            'rank_reverse': True,
            'rank_method': "ordinal"
        }
        self.wsm = WSM(**kwargs)
        self.wpm = WPM(**kwargs)
        self.moora = Moora(**kwargs)

    def analysis(self):
        analizer = RankSimilarityAnalyzer()
        analizer.add_executor(self.wsm)
        analizer.add_executor(self.wpm)
        analizer.add_executor(self.moora)
        analizer.analyze()
        return analizer


if __name__ == '__main__':
    app = RankSimilarityApp()
    analizer = app.analysis()
    print(analizer.get_results())
