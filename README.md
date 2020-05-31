# Decipy
Multi-Criteria Decision Making Methods library 

## Installation
```shell script
$ pip install decipy
```
or
```shell script
$ pip install git+https://github.com/justsasri/decipy.git#egg=decipy
```

## MCDM Ranking
```python
import numpy as np
import pandas as pd
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


# How to choose best MCDM Method ?

# Instantiate Rank Analizer
analizer = exe.RankSimilarityAnalyzer()

# Add MCDMs to anlizer
analizer.add_executor(wsm)
analizer.add_executor(topsis)
analizer.add_executor(vikor)

# run analizer
results = analizer.analyze()
print(results)
```

## references
- Triantaphyllou, E., Mann, S.H. 1989. "An Examination of The Effectiveness of Multi-dimensional Decision-making Methods: A Decision Making Paradox." Decision Support Systems (5(3)): 303–312.
- Chakraborty, S., and C.H. Yeh. 2012. "Rank Similarity based MADM Method Selection." International Conference on Statistics in Science, Business and Engineering (ICSSBE2012)
- Brauers, Willem K., and Edmundas K. Zavadskas. 2009. "Robustness of the multi‐objective MOORA method with a test for the facilities sector." Ukio Technologinisir Ekonominis (15:2): 352-375.
- Hwang, C.L., and K. Yoon. 1981. "Multiple attribute decision making, methods and applications." Lecture Notes in Economics and Mathematical Systems(Springer-Verlag) 186
- Yoon, K.P. and Hwang, C.L., “Multiple Attribute Decision Making: An Introduction”, SAGE publications, London, 1995.
- ÇELEN, Aydın. 2014. "Comparative Analysis of Normalization Procedures in TOPSIS Method: With an Application to Turkish Deposit Banking Market." INFORMATICA 25 (2): 185–208
- “Ranking”, http://en.wikipedia.org/wiki/Ranking