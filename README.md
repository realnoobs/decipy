# Decipy
Multi-Criteria Decision Making Methods library 

## Installation
```
$ pip install decipy
```
or
```
$ pip install git+https://github.com/sasriawesome/decipy.git#egg=decipy
```

## MCDM Ranking
```
import os
import numpy as np
from decipy import executors as exe

# define matrix
matrix = np.array([
    [4, 3, 2, 4],
    [5, 4, 3, 7],
    [6, 5, 5, 3],
])

# alternatives
alts = ['A1', 'A2', 'A3']

# criterias
crits = ['C1', 'C2', 'C3', 'C4']

# criteria's beneficial values, True for benefit or False for cost
beneficial = [True, True, True, True]

# criteria's weights
weights = [0.10, 0.20, 0.30, 0.40]

# define DataFrame
xij = pd.DataFrame(matrix, index=alts, columns=crits)

# create Executor (MCDM Method implementation)

kwargs = {
    'data': xij,
    'beneficial': beneficial,
    'weights': weights,
    'rank_reverse': True,
    'rank_method': "ordinal"
}

# Build MCDM Executor
wsm = exe.WSM(**kwargs) # Weighted Sum Method
topsis = exe.Topsis(**kwargs) # Topsis 
vikor = exe.Vikor(**kwargs) # Vikor 

# show results
print("WSM Ranks")
print(wsm.dataframe)

print("TOPSIS Ranks")
print(topsis.dataframe)

print("Vikor Ranks")
print(vikor.dataframe)

```

## How to choose best MCDM Method ?
```

# Instantiate Rank Analizer
analizer = exe.RankSimilarityAnalyzer()

# Add MCDMs to anlizer
analizer.add_executor(self.wsm)
analizer.add_executor(self.wpm)
analizer.add_executor(self.moora)

# run analizer
results = analizer.analyze()
print(results)

```