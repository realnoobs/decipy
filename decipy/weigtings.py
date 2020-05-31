import numpy as np
import pandas as pd


class Weight:
    def __init__(self, normalized, weights):
        self.data = normalized
        self.alts = self.data.index
        self.crits = self.data.columns
        self.weights = weights
        self.vij = np.zeros((len(self.weights), len(self.data)))
        self.rij = self.data.values.transpose()

    @property
    def dataframe(self):
        result = pd.DataFrame(self.get_weighted().transpose(), index=self.alts, columns=self.crits)
        return np.round(result, 4)

    def get_weighted(self):
        raise NotImplementedError


class Multi(Weight):
    def get_weighted(self):
        for i in range(len(self.weights)):
            self.vij[i] = self.rij[i] * self.weights[i]
        return self.vij


class Power(Weight):
    def get_weighted(self):
        for i in range(len(self.weights)):
            self.vij[i] = np.power(self.rij[i], self.weights[i])
        return self.vij


class MinMax(Weight):
    def get_weighted(self):
        for i in range(len(self.weights)):
            nomin = np.max(self.rij[i]) - self.rij[i]
            denom = np.max(self.rij[i]) - np.min(self.rij[i])
            self.vij[i] = self.weights[i] * (nomin / denom)
        return self.vij
