import numpy as np
import pandas as pd
from scipy import stats as sps
from . import normalizers as norm
from . import weigtings as weight


class DataMatrix:
    """ Load and Prepare data matrix """

    def __init__(self, path, delimiter=",", idx_col=0):
        self.data = pd.read_csv(path, delimiter=delimiter, index_col=idx_col)
        self.describe = self.data.describe()
        self.min = self.describe.loc["min"].values
        self.max = self.describe.loc["max"].values
        self.mean = self.describe.loc["mean"].values
        self.alts = self.data.index
        self.alts_count = len(self.alts)
        self.crits = self.data.columns
        self.crits_count = len(self.crits)
        self.crits_sum = np.sum(self.data.values, axis=0)

    @property
    def random(self):
        x = np.zeros((self.crits_count, self.alts_count))
        for i in range(self.crits_count):
            x[i] = np.random.randint(self.min[i], self.max[i], size=self.alts_count)
        df = pd.DataFrame(x.transpose(), index=self.alts, columns=self.crits)
        return df


class MCDMBase:
    """
    Multicriteria Decision Making Method Base Class
    -----------------------------------------------
        Foundation of multicriteria decision making
        calculation process.

    Parameter
    ----------
    data : Pandas DataFrame object
        Normalized (Linear Max) performance rating matrix.
    cweight : Array like
        Weight of each criterias.
    rank_method : string
        Ranking method ('max', 'min', 'average').

    """
    data = None
    weights = None
    weighting_class = None
    normalization_class = None

    def __init__(self,
                 data,
                 beneficial,
                 weights,
                 rank_reverse=True,
                 rank_method="ordinal"):
        self.data = data
        self.alts = data.index
        self.beneficial = beneficial
        self.weights = weights
        self.rank_method = rank_method
        self.rank_reverse = rank_reverse

    @property
    def dataframe(self):
        result = pd.DataFrame(
            self.get_results().transpose(),
            index=self.alts,
            columns=['RATE', 'RANK']
        )
        return np.round(result, 4)

    def get_weighted(self):
        normalizer = self.get_normalized()
        weights = self.weights
        return self.weighting_class(normalizer, weights).dataframe

    def get_normalized(self):
        xij = self.data
        return self.normalization_class(xij, self.beneficial).dataframe

    def set_weighting_class(self, weighting_class):
        self.weighting_class = weighting_class

    def set_normalization_class(self, normalization_class):
        self.normalization_class = normalization_class

    def get_rank(self, rate):
        rank = sps.rankdata(rate, method=self.rank_method).astype(int)
        if self.rank_reverse:
            rank = sps.rankdata([-1 * i for i in rate], method=self.rank_method).astype(int)
        return rank

    def get_results(self):
        raise NotImplementedError


class WSM(MCDMBase):
    """
    Weighted Sum Model / Simple Additive Weighting
    ----------------------------------------------
        The assumption that governs this model is
        the additive utility assumption.  The basic logic
        of the WSM/SAW method is to obtain a weighted sum of
        the performance ratings of each alternative over
        all attributes.

    Return
    ------
    result : Pandas DataFrame object
        Return alternatives overal rating and ranking .

    See also
    --------
    WPM, WASPAS, Moora, Topsis, Vikor

    references
    ----------
    [1] Triantaphyllou, E., Mann, S.H. 1989.
        "An Examination of The Effectiveness of Multi-dimensional
        Decision-making Methods: A Decision Making Paradox."
        Decision Support Systems (5(3)): 303–312.
    [2] Chakraborty, S., and C.H. Yeh. 2012.
        "Rank Similarity based MADM Method Selection."
        International Conference on Statistics in Science,
        Business and Engineering (ICSSBE2012)
    [3] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    weighting_class = weight.Multi
    normalization_class = norm.MinMax

    def get_results(self):
        vij = self.get_weighted()
        rate = np.sum(vij.values, axis=1)
        rank = self.get_rank(rate)
        rest = np.array([rate, rank])
        return rest


class WPM(MCDMBase):
    """
    Weighted Product Model
    ----------------------
        The weighted product model (WPM) is very
        similar to the WSM. The main difference is that
        instead of addition in the model there is
        multiplication. Each alternative is compared
        with the others by multiplying a number of ratios,
        one for each criterion. Each ratio is raised to the
        power equivalent of the relative weight of the
        corresponding criterion.

    Return
    ------
    result : Pandas DataFrame object
        Return alternatives overal rating and ranking.

    See also
    --------
    WPM, WASPAS, Moora, Topsis, Vikor

    references
    ----------
    [1] Triantaphyllou, E., Mann, S.H. 1989.
        "An Examination of The Effectiveness of Multi-dimensional
        Decision-making Methods: A Decision Making Paradox."
        Decision Support Systems (5(3)): 303–312.
    [2] Chakraborty, S., and C.H. Yeh. 2012.
        "Rank Similarity based MADM Method Selection."
        International Conference on Statistics in Science,
        Business and Engineering (ICSSBE2012)
    [3] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    weighting_class = weight.Power
    normalization_class = None

    def get_weighted(self):
        rij = self.data
        wj = self.weights
        return self.weighting_class(rij, wj).dataframe

    def get_results(self):
        vij = self.get_weighted()
        rate = np.prod(vij.values, axis=1)
        rank = self.get_rank(rate)
        rest = np.array([rate, rank])
        return rest


class Moora(MCDMBase):
    """
    Multi-Objective Optimization on the Basis of
    Ratio Analysis (MOORA) Ratio System
    --------------------------------------------
        The MOORA method consists of 2 parts: the ratio system
        and the reference point approach. This function is based on
        MOORA Ratio System.

    Return
    ------
    result : Pandas DataFrame object
        Return alternatives overal rating and ranking .

    See also
    --------
    WSM, WPM, WASPAS, Topsis, Vikor

    references
    ----------
    [1] Brauers, Willem K., and Edmundas K. Zavadskas. 2009.
        "Robustness of the multi‐objective MOORA method with
        a test for the facilities sector." Ukio Technologinis
        ir Ekonominis (15:2): 352-375.
    [2] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    weighting_class = weight.Multi
    normalization_class = norm.Vector

    def get_weighted(self):
        rij = self.data
        wj = self.weights
        return self.weighting_class(rij, wj).dataframe

    @property
    def dataframe(self):
        result = pd.DataFrame(
            self.get_results().transpose(),
            index=self.alts,
            columns=['BEN', 'COS', 'RATE', 'RANK']
        )
        return np.round(result, 4)

    def get_results(self):
        vij = self.get_weighted().values
        vij = vij.transpose()
        ben = np.zeros(len(self.data.values))
        cos = np.zeros(len(self.data.values))
        y = np.zeros(len(self.data.values))
        for i in range(len(self.beneficial)):
            if self.beneficial:
                ben = ben + vij[i]
                y = y + vij[i]
            else:
                cos = cos + vij[i]
                y = y - vij[i]
        rate = y
        rank = self.get_rank(rate)
        rest = np.array([ben, cos, rate, rank])
        return rest


class Topsis(MCDMBase):
    """
    Technique for Order Preferences by Similarity
    to an Ideal Solution (TOPSIS)
    ---------------------------------------------
        TOPSIS applies a simple concept of maximizing distance
        from the negative-ideal solution and minimizing the
        distance from the positive ideal solution.  The chosen
        alternative must be as close as possible to the ideal
        solution and as far as possible from the negative-ideal
        solution.

    Return
    ------
    result : Pandas DataFrame object
        Return alternatives overal rating and ranking .

    See also
    --------
    WPS, WPS, WASPAS, Moora, Vikor

    references
    ----------
    [1] Hwang, C.L., and K. Yoon. 1981. "Multiple attribute
        decision making, methods and applications." Lecture
        Notes in Economics and Mathematical Systems
        (Springer-Verlag) 186
    [2] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    weighting_class = weight.Multi
    normalization_class = norm.Vector

    @property
    def dataframe(self):
        result = pd.DataFrame(
            self.get_results().transpose(),
            index=self.alts,
            columns=['D+', 'D-', 'RATE', 'RANK'])
        return np.round(result, 4)

    def get_results(self):
        vij = self.get_weighted()
        pis = np.max(vij, axis=0)
        nis = np.min(vij, axis=0)
        dmax = np.sqrt(np.sum(np.power(vij - pis, 2), axis=1))
        dmin = np.sqrt(np.sum(np.power(vij - nis, 2), axis=1))
        rate = dmin / (dmax + dmin)
        rank = self.get_rank(rate)
        rest = np.array([dmax, dmin, rate, rank])
        return rest


class Vikor(MCDMBase):
    """
    VlseKriterijumska Optimizacija I Kompromisno Resenje (VIKOR)
    ------------------------------------------------------------
        This method focuses on ranking and selecting from a set of
        alternatives in the presence of conflicting criteria.
        It introduces the multicriteria ranking index based on
        the particular measure of “closeness” to the
        “ideal” solution (Opricovic 1998).

    Return
    ------
    result : Pandas DataFrame object
        Return alternatives overal rating and ranking .

    See also
    --------
    WSM, WPM, WASPAS, Moora, Topsis

    references
    ----------
    [1] Hwang, C.L., and K. Yoon. 1981. "Multiple attribute
        decision making, methods and applications." Lecture
        Notes in Economics and Mathematical Systems
        (Springer-Verlag) 186
    [2] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    weighting_class = weight.MinMax
    normalization_class = norm.Vector

    def __init__(self,
                 data, beneficial, weights,
                 new_min=0, new_max=1, rank_reverse=True,
                 rank_method="ordinal"):
        self.new_min = new_min
        self.new_max = new_max
        super(Vikor, self).__init__(
            data, beneficial, weights,
            rank_reverse=rank_reverse,
            rank_method=rank_method)

    @property
    def dataframe(self):
        result = pd.DataFrame(
            self.get_results().transpose(),
            index=self.alts,
            columns=['S', 'P', 'RATE', 'RANK'])
        return np.round(result, 4)

    def get_results(self):
        vij = self.get_weighted()
        s = np.sum(vij, axis=1)
        p = np.max(vij, axis=1)
        q1 = (
                0.5 *
                (s - np.min(s) * (self.new_max - self.new_min)) /
                ((np.max(s) - np.min(s)) + self.new_min)
        )
        q2 = (
                (1 - 0.5) *
                (p - np.min(p) * (self.new_max - self.new_min)) /
                ((np.max(p) - np.min(p)) + self.new_min)
        )
        q = q1 + q2
        best = 1 - q
        rank = self.get_rank(best)
        rest = np.array([s, p, q, rank])
        return rest


class RankSimilarityAnalyzer:

    def __init__(self, rank_method='ordinal', rank_reverse=True):
        self.rho = None
        self.rsi = None
        self.rsr = None
        self.pval = None
        self.results = {}
        self.executors = []
        self.rate_matrix = []
        self.rank_matrix = []
        self.rank_method = rank_method
        self.rank_reverse = rank_reverse

    def reset(self, hard=False):
        self.rho = None
        self.rsi = None
        self.pval = None
        self.results = {}
        self.rate_matrix = []
        self.rank_matrix = []
        if hard:
            self.executors = []

    @property
    def alternatives_index(self):
        return self.executors[0].dataframe.index

    @property
    def executor_labels(self):
        return [x.__class__.__name__ for x in self.executors]

    def add_executor(self, executor):
        """ add executors """
        if not isinstance(executor, MCDMBase):
            raise TypeError("Executor type should be MCDMBase sub class")
        self.executors.append(executor)
        self.rate_matrix.append(executor.dataframe['RATE'])
        self.rank_matrix.append(executor.dataframe['RANK'])

    def get_results(self):
        correlation_df = self.get_correlations()
        correlation_df['RSI'] = np.round(self.rsi, 4)
        correlation_df['RSR'] = np.round(self.rsr, 4)
        return correlation_df

    def get_rates(self):
        rate_df = pd.DataFrame(
            np.array(self.rate_matrix).transpose(),
            index=self.alternatives_index,
            columns=self.executor_labels)
        return np.round(rate_df, 4)

    def _get_rank(self, rsi):
        rank = sps.rankdata(rsi, method=self.rank_method).astype(int)
        if self.rank_reverse:
            rank = sps.rankdata([-1 * i for i in rsi], method=self.rank_method).astype(int)
        return rank

    def get_ranks(self):
        rank_df = pd.DataFrame(
            np.array(self.rank_matrix).transpose(),
            index=self.alternatives_index,
            columns=self.executor_labels)
        return np.round(rank_df, 4)

    def get_correlations(self):
        correlations = pd.DataFrame(
            self.rho,
            index=self.executor_labels,
            columns=self.executor_labels)
        return np.round(correlations, 4)

    def analyze(self):
        if len(self.executors) < 2:
            raise IndexError("Please add at least 2 executors")
        self.rho, self.pval = sps.spearmanr(self.rank_matrix, axis=1)
        self.rsi = np.average(self.rho, axis=0)
        self.rsr = self._get_rank(self.rsi)
        return self.get_results()


class RankSimulator:
    analizer_class = RankSimilarityAnalyzer

    def __init__(self,
                 matrix,
                 beneficial,
                 weights,
                 executor_classes,
                 events=1000,
                 trials=100,
                 rank_reverse=True,
                 rank_method='ordinal'):

        # Validation
        if not isinstance(matrix, DataMatrix):
            raise TypeError("'data' argument should be Decipy DataMatrix instance")
        if not isinstance(beneficial, (list, tuple)):
            raise TypeError("'beneficial' argument should be List or Tuple instance")
        if matrix.crits_count != len(beneficial):
            msg = "'beneficial' length does not match beneficial %s columns %s"
            raise ValueError(msg % (matrix.alts_count, len(beneficial)))
        if not isinstance(weights, (list, tuple)):
            raise TypeError("'weights' argument should be List or Tuple instance")

        if matrix.crits_count != len(weights):
            msg = "'weights' length does not match weights %s columns %s"
            raise ValueError(msg % (matrix.crits_count, len(weights)))

        if not isinstance(executor_classes, (list, tuple)):
            raise TypeError("'executor_classes' argument should be List or Tuple instance")
        if len(executor_classes) < 2:
            raise TypeError("'executor_classes' need 2 or more items")
        for executor in executor_classes:
            if not issubclass(executor, MCDMBase):
                raise TypeError("'%s' is should be subclass of MCDMBase" % executor.__name__)

        self.executors = []
        self.analizer = None
        self.trials = trials
        self.events = events
        self.matrix = matrix
        self.beneficial = beneficial
        self.weights = weights
        self.executor_classes = executor_classes
        self.executor_labels = [e.__name__ for e in self.executor_classes]
        self.rank_reverse = rank_reverse
        self.rank_method = rank_method
        self.executor_count = len(self.executor_classes)

        self.event_indexes = None
        self.event_ranks = None
        self.top_ranks = None
        self.top_ranks_all = None
        self.top_ranks_probability = None
        self.global_similarity_index = None
        self.top_similarity_index = None

    def get_executors(self):
        self.executors = []
        for exec_class in self.executor_classes:
            self.executors.append(
                exec_class(
                    data=self.matrix.random,
                    beneficial=self.beneficial,
                    weights=self.weights,
                    rank_reverse=self.rank_reverse,
                    rank_method=self.rank_method
                )
            )
        return self.executors

    def get_steps(self):
        steps = self.events / self.trials
        return [(s * self.trials) for s in range(1, int(steps) + 1)]

    def run_analizer(self):
        executors = self.get_executors()
        self.analizer = self.analizer_class()
        for executor in executors:
            self.analizer.add_executor(executor)
        self.analizer.analyze()
        return self.analizer

    def run(self):
        steps = self.get_steps()
        self.event_indexes = np.zeros((np.max(steps), self.executor_count))
        self.event_ranks = np.zeros((np.max(steps), self.executor_count))

        self.top_ranks = np.zeros((len(steps), self.executor_count))
        self.top_ranks_probability = np.zeros((len(steps), self.executor_count))
        self.global_similarity_index = np.zeros((len(steps), self.executor_count))
        self.top_similarity_index = np.zeros((len(steps), self.executor_count))

        for trial in range(len(steps)):
            if trial == 0:
                prev_event = 0
                current_event = steps[trial]
            else:
                prev_event = steps[trial - 1]
                current_event = steps[trial] - steps[trial - 1]

            for event in range(current_event):
                analizer = self.run_analizer()
                self.event_indexes[event + prev_event] = analizer.get_results()['RSI']
                self.event_ranks[event + prev_event] = analizer.get_results()['RSR']

            self.top_ranks_all = np.count_nonzero(self.event_ranks[:steps[trial]] == self.executor_count, axis=0)
            self.top_ranks[trial] = self.top_ranks_all
            self.top_ranks_probability[trial] = self.top_ranks_all / steps[trial]
            self.global_similarity_index[trial] = np.average(self.event_ranks[:steps[trial]], axis=0)
            self.top_similarity_index[trial] = np.sum(self.event_ranks[:steps[trial]], axis=0)
        return 'Simulation complete ...'

    def get_event_ranks(self):
        return pd.DataFrame(np.array(self.event_ranks).transpose(), index=self.executor_labels)

    def get_top_ranks(self):
        return pd.DataFrame(
            np.array(self.top_ranks, dtype="int").transpose(),
            index=self.executor_labels,
            columns=self.get_steps()
        )

    def get_top_rank_probabilities(self):
        return pd.DataFrame(
            np.array(self.top_ranks_probability).transpose(),
            index=self.executor_labels,
            columns=self.get_steps()
        )

    def get_event_index(self):
        return pd.DataFrame(
            np.array(self.event_indexes).transpose(),
            index=self.executor_labels
        )

    def get_global_similarity_index(self):
        return pd.DataFrame(
            np.array(self.global_similarity_index).transpose(),
            index=self.executor_labels,
            columns=self.get_steps()
        )

    def get_top_similarity_index(self):
        return pd.DataFrame(
            np.array(self.top_similarity_index).transpose(),
            index=self.executor_labels,
            columns=self.get_steps()
        )
