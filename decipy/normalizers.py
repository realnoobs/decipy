import numpy as np
import pandas as pd
from scipy import stats as sps


class NormalizerBase:
    """ Abstract parent class for data normalization """

    def __init__(self, xij, beneficial, *args, **kwargs):
        self.xij = xij.values.transpose()
        self.alts = xij.index
        self.crits = xij.columns
        self.rij = np.zeros((len(self.crits), len(self.xij[0])))
        self.beneficial = beneficial

    @property
    def dataframe(self):
        result = pd.DataFrame(self.get_normal(), index=self.alts, columns=self.crits)
        return np.round(result, 4)

    def get_normal(self):
        """ Override this method with custom normalization logic """
        raise NotImplementedError


class Vector(NormalizerBase):
    """
    Vector normalization function
    -----------------------------
        In this method, each performance rating of the
        decision matrix is divided by its norm. This method
        has the advantage of converting all attributes into
        dimensionless measurement unit, thus making inter-attribute
        comparison easier. But it has the drawback of having non-equal
        scale length leading to difficulties in straightforward
        comparison [3][4].

    Parameter
    ---------
    xij : Pandas DataFrame object
        Alternative performance rating matrix.
    beneficial : Array like
        List of criteria beneficial status (benefit = True) or
        (cost = False)

    See also
    --------
    Max, MinMax, NinMax2, Sum

    References
    ---------
    [1] ÇELEN, Aydın. 2014. "Comparative Analysis of
        Normalization Procedures in TOPSIS Method:
        With an Application to Turkish Deposit Banking Market."
        INFORMATICA 25 (2): 185–208
    [2] Chakraborty, S., and C.H. Yeh. 2012.
        "Rank Similarity based MADM Method Selection."
        International Conference on Statistics in Science,
        Business and Engineering (ICSSBE2012)
    [3] Yoon, K.P. and Hwang, C.L., “Multiple Attribute
        Decision Making: An Introduction”, SAGE publications,
        London, 1995.
    [4] Hwang, C.L. and K. Yoon, “Multiple Attribute Decision
        Making: Methods and Applications”, Springer-Verlag,
        New York, 1981.
    """

    def get_normal(self):
        """
        Return
        ------
        result : Pandas DataFrame object
            Return normalized performance rating, range 0.0 to 1.0.
        """
        for c in range(len(self.crits)):
            if self.beneficial[c]:
                # Benefit Criteria
                denom = np.power(self.xij[c], 2)
                sumd = np.sum(denom)
                self.rij[c] = 0 if sumd == 0 else self.xij[c] / np.sqrt(sumd)
            else:
                # Cost Criteria
                denom = 1 / (np.power(self.xij[c], 2))
                sumd = np.sum(denom)
                self.rij[c] = 0 if sumd == 0 else (1 / self.xij[c]) / np.sqrt(sumd)
        return self.rij.transpose()


class MinMax(NormalizerBase):
    """
    Linear Minmax normalization function
    ----------------------------------
        This method considers both the maximum and minimum
        performance ratings of criterias during calculation.

    Parameter
    ----------
    xij : Pandas DataFrame object
        Alternative performance rating matrix.
    beneficial : Array like
        List of criteria beneficial status (benefit = True) or
        (cost = False)

    See also
    --------
    Vector, Max, MinMax2, Sum

    References
    ---------
    [1] ÇELEN, Aydın. 2014. "Comparative Analysis of
        Normalization Procedures in TOPSIS Method:
        With an Application to Turkish Deposit Banking Market."
        INFORMATICA 25 (2): 185–208
    [2] Chakraborty, S., and C.H. Yeh. 2012.
        "Rank Similarity based MADM Method Selection."
        International Conference on Statistics in Science,
        Business and Engineering (ICSSBE2012)
    """

    def get_normal(self):
        """
        Return
        ------
            result : Pandas DataFrame object
                Return normalized performance rating, range 0.0 to 1.0.
        """
        for c in range(len(self.beneficial)):
            if self.beneficial[c]:
                # Benefit Criteria
                nomin = self.xij[c] - np.min(self.xij[c])
                denom = np.max(self.xij[c]) - np.min(self.xij[c])
                self.rij[c] = 0 if denom == 0 else (nomin / denom)
            else:
                # Cost Criteria
                nomin = (np.max(self.xij[c]) - self.xij[c])
                denom = (np.max(self.xij[c]) - np.min(self.xij[c]))
                self.rij[c] = 0 if denom == 0 else (nomin / denom)
        return self.rij.transpose()


class MinMax2(NormalizerBase):
    """
    Linear Minmax normalization function
    (with new_min and new_max values)
    ------------------------------------
        This method considers both the maximum and minimum
        performance ratings of criterias during calculation.
        In this method, we can sets both new minimum and new maximum
        value.

    Parameter
    ----------
    xij : Pandas DataFrame object
        Alternative performance rating matrix.
    beneficial : Array like
        List of criteria beneficial status (benefit = True) or
        (cost = False)
    new_min : integer or floating point number (optional)
        New criteria normalized minimum number, default is 0.
    new_max : integer or floating point number (optional)
        New criteria normalized maximum number, default is 1.

    See also
    --------
    norm_vector, norm_max, norm_minmax, norm_sum

    References
    ---------
    [1] Han, Jiawei, Micheline Kamber, and Jian Pei. 2012.
        Data Mining Concepts and Techniques Third Edition.
        Waltham: Elsevier Inc
    """

    def __init__(self, xij, beneficial, new_min=0, new_max=1, *args, **kwargs):
        self.new_min = new_min
        self.new_max = new_max
        super(MinMax2, self).__init__(xij, beneficial, *args, **kwargs)

    def get_normal(self):
        for c in range(len(self.beneficial)):
            nomination = (self.xij[c] - np.min(self.xij[c])) * (self.new_max - self.new_min)
            denomination = (np.max(self.xij[c]) - np.min(self.xij[c]))
            if self.beneficial[c]:
                # Benefit Criteria
                self.rij[c] = (nomination / denomination) + self.new_min
            else:
                # Cost Criteria
                self.rij[c] = self.new_max - (nomination / denomination)
        return self.rij.transpose()


class Max(NormalizerBase):
    """
    Linear Max normalization function.
    ----------------------------------
        This method divides the performance ratings of each criteria
        by the maximum performance rating for that criteria.

    Parameter
    ----------
    xij : Pandas DataFrame object
        Alternative performance rating matrix.
    beneficial : Array like
        List of criteria beneficial status (benefit = True) or
        (cost = False)

    See also
    --------
    Vector, MinMax, MinMax2, Sum

    References
    ---------
    [1] ÇELEN, Aydın. 2014. "Comparative Analysis of
        Normalization Procedures in TOPSIS Method:
        With an Application to Turkish Deposit Banking Market."
        INFORMATICA 25 (2): 185–208
    [2] Chakraborty, S., and C.H. Yeh. 2012.
        "Rank Similarity based MADM Method Selection."
        International Conference on Statistics in Science,
        Business and Engineering (ICSSBE2012)
    """

    def get_normal(self):
        """
            Return
            ------
                result : Pandas DataFrame object
                    Return normalized performance rating, range 0.0 to 1.0.
        """
        for c in range(len(self.beneficial)):
            if self.beneficial[c]:
                # Benefit Criteria
                self.rij[c] = self.xij[c] / np.max(self.xij[c])
            else:
                # Cost Criteria
                self.rij[c] = 1 - (self.xij[c] / np.max(self.xij[c]))
        return self.rij.transpose()


class Sum(NormalizerBase):
    """
    Linear Sum normalization function.
    ----------------------------------
        This method divides the performance ratings of each
        attribute by the sum of performance ratings for that
        attribute.

    Parameter
    ----------
    xij : Pandas DataFrame object
        Alternative performance rating matrix.
    beneficial : Array like
        List of criteria beneficial status (benefit = True) or
        (cost = False)

    See also
    --------
    Vector, MinMax, MinMax2, Max

    References
    ---------
    [1] ÇELEN, Aydın. 2014. "Comparative Analysis of
        Normalization Procedures in TOPSIS Method:
        With an Application to Turkish Deposit Banking Market."
        INFORMATICA 25 (2): 185–208
    [2] Chakraborty, S., and C.H. Yeh. 2012.
        "Rank Similarity based MADM Method Selection."
        International Conference on Statistics in Science,
        Business and Engineering (ICSSBE2012)
    """

    def get_normal(self):
        """
        Return
        ------
            result : Pandas DataFrame object
                Return normalized performance rating, range 0.0 to 1.0.
        """
        for c in range(len(self.beneficial)):
            if self.beneficial[c]:
                # Benefit Criteria
                self.rij[c] = self.xij[c] / np.sum(self.xij[c])
            else:
                # Cost Criteria
                self.rij[c] = (1 / self.xij[c]) / np.sum((1 / self.xij[c]))
        return self.rij.transpose()


class ZScore(NormalizerBase):
    """
    Z-Score normalization function.
    -------------------------------
        In z-score normalization (or zero-mean normalization),
        the values for an attribute, A, are normalized based on
        the mean (i.e., average) and standard deviation of A. This
        method of normalization is useful when the actual minimum
        and maximum of attribute A are unknown, or when there are
        outliers that dominate the min-max normalization.

    Parameter
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix.
        beneficial : Array like
            List of criteria beneficial status (benefit = True) or
            (cost = False)

    See also
    --------
    Gaussian, Sigmoid, Softmax

    References
    ---------
    [1] Han, Jiawei, Micheline Kamber, and Jian Pei. 2012.
        Data Mining Concepts and Techniques Third Edition.
        Waltham: Elsevier Inc
    [2] Spatz, Chris. 2008. Basic Statistics: Tales of
        Distributions, Ninth Edition. Belmont: Thomson
        Learning
    """

    def get_normal(self):
        """
        Return
        ------
            result : Pandas DataFrame object
                Return normalized performance rating, range -3.0 to 3.0.
        """
        for c in range(len(self.beneficial)):
            zscore = sps.zscore(self.xij[c])
            if self.beneficial[c]:
                # Benefit Criteria
                self.rij[c] = zscore
            else:
                # Cost Criteria
                self.rij[c] = -zscore
        return self.rij.transpose()


class Gaussian(NormalizerBase):
    """
    Gaussian normalization function
    -------------------------------
        This normalization method applies gaussian probability
        function.

    Parameter
    ----------
    xij : Pandas DataFrame object
        Alternative performance rating matrix.
    beneficial : Array like
        List of criteria beneficial status (benefit = True) or
        (cost = False)

    See also
    --------
    ZScore, Sigmoid, Softmax

    References
    ---------
    [1] Barrow, Michael. 2017. Statistics for Economics,
        Accounting and Business Studies Seventh Edition.
        Pearson Pearson Education Limited. United Kingdom
    [2] Spatz, Chris. 2008. Basic Statistics: Tales of
        Distributions, Ninth Edition. Belmont: Thomson
        Learning
    """

    def get_normal(self):
        """
        Return
        ------
            result : Pandas DataFrame object
                Return normalized performance rating, range 0.0 to 1.0.
        """
        for c in range(len(self.beneficial)):
            zscore = sps.zscore(self.xij[c])
            if self.beneficial[c]:
                # Benefit Criteria
                self.rij[c] = sps.norm.cdf(zscore)
            else:
                # Benefit Criteria
                self.rij[c] = sps.norm.cdf(-zscore)
        return self.rij.transpose()


class SoftMax(NormalizerBase):
    """
    Softmax normalization function
    ------------------------------
        The hyperbolic tangent function, tanh, limits the range
        of the normalized data to values between −1 and 1.
        The hyperbolic tangent function is almost linear near the
        mean, but has a slope of half that of the sigmoid function.
        Like sigmoid, it has smooth, monotonic nonlinearity at both
        extremes.

    Parameter
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix.
        beneficial : Array like
            List of criteria beneficial status (benefit = True) or
            (cost = False)

    See also
    --------
    ZScore, Gaussian, Sigmoid

    References
    ---------
        [1] "Sigmoid Function"
            https://en.wikipedia.org/wiki/Sigmoid_function
    """

    def get_normal(self):
        """
        Return
        ------
            result : Pandas DataFrame object
                Return normalized performance rating, range -1 to 1.
        """
        for c in range(len(self.beneficial)):
            if self.beneficial[c]:
                # Benefit Criteria
                zscore = sps.zscore(self.xij[c])
                self.rij[c] = ((1 - np.exp(-zscore)) / (1 + np.exp(-zscore)))
            else:
                # Cost Criteria
                zscore = -(sps.zscore(self.xij[c]))
                self.rij[c] = ((1 - np.exp(-zscore)) / (1 + np.exp(-zscore)))
        return self.rij.transpose()


class Sigmoid(NormalizerBase):
    """
    Sigmoid normalization function
    ------------------------------
        The sigmoid function limits the range of the normalized data
        to values between 0 and 1. The sigmoid function is almost
        linear near the mean and has smooth nonlinearity at both
        extremes, ensuring that all data points are within a limited
        range. This maintains the resolution of most values within
        a standard deviation of the mean.

    Parameter
    ----------
    xij : Pandas DataFrame object
        Alternative performance rating matrix.
    beneficial : Array like
        List of criteria beneficial status (benefit = True) or
        (cost = False)

    See also
    --------
    ZScore, Gaussian, SoftMax

    References
    ---------
        [1] "Sigmoid Function"
            https://en.wikipedia.org/wiki/Sigmoid_function
    """

    def get_normal(self):
        for c in range(len(self.beneficial)):
            if self.beneficial[c]:
                # Benefit Criteria
                z_score = sps.zscore(self.xij[c])
                self.rij[c] = 1 / (1 + np.exp(-z_score))
            else:
                # Cost Criteria
                z_score = -(sps.zscore(self.xij[c]))
                self.rij[c] = 1 / (1 + np.exp(-z_score))
        return self.rij.transpose()
