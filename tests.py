import pandas as pd
import numpy as np
import unittest
import decipy.executors as exe
import decipy.normalizers as norm
import decipy.weigtings as wgt

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


class NormalizerTestCase(unittest.TestCase):

    def setUp(self):
        self.vector = norm.Vector(xij=xij, beneficial=beneficial)
        self.minmax = norm.MinMax(xij=xij, beneficial=beneficial)
        self.minmax2 = norm.MinMax2(xij=xij, beneficial=beneficial)
        self.max = norm.Max(xij=xij, beneficial=beneficial)
        self.sum = norm.Sum(xij=xij, beneficial=beneficial)
        self.zscore = norm.ZScore(xij=xij, beneficial=beneficial)
        self.gaussian = norm.Gaussian(xij=xij, beneficial=beneficial)
        self.softmax = norm.SoftMax(xij=xij, beneficial=beneficial)
        self.sigmoid = norm.Sigmoid(xij=xij, beneficial=beneficial)

    def test_dataframe(self):
        self.assertIsInstance(
            self.vector.dataframe, pd.DataFrame,
            msg="Normalizer dataframe method should return pandas DataFrame instance")
        self.assertIsInstance(
            self.vector.dataframe, pd.DataFrame,
            msg="Normalizer dataframe method should return pandas DataFrame instance")

    def test_vector_values(self):
        results = np.array([[0.4558, 0.4243, 0.3244, 0.4650],
                            [0.5698, 0.5657, 0.4867, 0.8137],
                            [0.6838, 0.7071, 0.8111, 0.3487]])
        np.testing.assert_array_equal(self.vector.dataframe.values, results)

    def test_minmax_values(self):
        results = np.array([[0.0000, 0.0000, 0.0000, 0.2500],
                            [0.5000, 0.5000, 0.3333, 1.0000],
                            [1.0000, 1.0000, 1.0000, 0.0000]])
        np.testing.assert_array_equal(self.minmax.dataframe.values, results)

    def test_minmax2_values(self):
        results = np.array([[0.0000, 0.0000, 0.0000, 0.2500],
                            [0.5000, 0.5000, 0.3333, 1.0000],
                            [1.0000, 1.0000, 1.0000, 0.0000]])
        np.testing.assert_array_equal(self.minmax2.dataframe.values, results)

    def test_max_values(self):
        results = np.array([[0.6667, 0.6000, 0.4000, 0.5714],
                            [0.8333, 0.8000, 0.6000, 1.0000],
                            [1.0000, 1.0000, 1.0000, 0.4286]])
        np.testing.assert_array_equal(self.max.dataframe.values, results)

    def test_sum_values(self):
        results = np.array([[0.2667, 0.2500, 0.2000, 0.2857],
                            [0.3333, 0.3333, 0.3000, 0.5000],
                            [0.4000, 0.4167, 0.5000, 0.2143]])
        np.testing.assert_array_equal(self.sum.dataframe.values, results)

    def test_zscore_values(self):
        results = np.array([[-1.2247, -1.2247, -1.069, -0.3922],
                            [0.0000, 0.0000, -0.2673, 1.3728],
                            [1.2247, 1.2247, 1.3363, -0.9806]])
        np.testing.assert_array_equal(self.zscore.dataframe.values, results)

    def test_gaussian_values(self):
        results = np.array([[0.1103, 0.1103, 0.1425, 0.3474],
                            [0.5000, 0.5000, 0.3946, 0.9151],
                            [0.8897, 0.8897, 0.9093, 0.1634]])
        np.testing.assert_array_equal(self.gaussian.dataframe.values, results)

    def test_softmax_values(self):
        results = np.array([[-0.5458, -0.5458, -0.4888, -0.1936],
                            [0.0000, 0.0000, -0.1328, 0.5957],
                            [0.5458, 0.5458, 0.5838, -0.4544]])
        np.testing.assert_array_equal(self.softmax.dataframe.values, results)

    def test_sigmoid_values(self):
        results = np.array([[0.2271, 0.2271, 0.2556, 0.4032],
                            [0.5000, 0.5000, 0.4336, 0.7978],
                            [0.7729, 0.7729, 0.7919, 0.2728]])
        np.testing.assert_array_equal(self.sigmoid.dataframe.values, results)


class WeightingTestCase(unittest.TestCase):
    def setUp(self):
        self.rij = norm.MinMax(xij=xij, beneficial=beneficial).dataframe
        self.power = wgt.Power(self.rij, weights=weights)
        self.multi = wgt.Multi(self.rij, weights=weights)
        self.minmax = wgt.MinMax(self.rij, weights=weights)

    def test_dataframe(self):
        self.assertIsInstance(
            self.power.dataframe, pd.DataFrame,
            msg="Normalizer dataframe method should return pandas DataFrame instance")
        self.assertIsInstance(
            self.power.dataframe, pd.DataFrame,
            msg="Normalizer dataframe method should return pandas DataFrame instance")

    def test_power_values(self):
        results = np.array([[0.000, 0.0000, 0.0000, 0.5743],
                            [0.933, 0.8706, 0.7192, 1.0000],
                            [1.000, 1.0000, 1.0000, 0.0000]])
        np.testing.assert_array_equal(self.power.dataframe.values, results)

    def test_multi_values(self):
        results = np.array([[0.0000, 0.0000, 0.0000, 0.1000],
                            [0.0500, 0.1000, 0.1000, 0.4000],
                            [0.1000, 0.2000, 0.3000, 0.0000]])
        np.testing.assert_array_equal(self.multi.dataframe.values, results)

    def test_minmax_values(self):
        results = np.array([[0.1000, 0.2000, 0.3000, 0.3000],
                            [0.0500, 0.1000, 0.2000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.4000]])
        np.testing.assert_array_equal(self.minmax.dataframe.values, results)


class ExecutorTestCase(unittest.TestCase):
    def setUp(self):
        kwargs = {
            'data': xij,
            'beneficial': beneficial,
            'weights': weights,
            'rank_reverse': True,
            'rank_method': "ordinal"
        }
        self.wsm = exe.WSM(**kwargs)
        self.wpm = exe.WPM(**kwargs)
        self.moora = exe.Moora(**kwargs)
        self.topsis = exe.Topsis(**kwargs)
        self.vikor = exe.Vikor(**kwargs)

    def test_wsm_rank(self):
        results = np.array([[0.1000, 3.],
                            [0.6500, 1.],
                            [0.6000, 2.]])
        np.testing.assert_array_equal(self.wsm.dataframe.values, results)

    def test_wpm_rank(self):
        results = np.array([[3.0672, 3.],
                            [4.6933, 1.],
                            [4.1508, 2.]])
        np.testing.assert_array_equal(self.wpm.dataframe.values, results)

    def test_moora_rank(self):
        results = np.array([[3.2000, 0.0000, 3.2000, 3.],
                            [5.0000, 0.0000, 5.0000, 1.],
                            [4.3000, 0.0000, 4.3000, 2.]])
        np.testing.assert_array_equal(self.moora.dataframe.values, results)

    def test_topsis_rank(self):
        results = np.array([[0.2109, 0.0465, 0.1806, 3.],
                            [0.1020, 0.1947, 0.6562, 1.],
                            [0.1860, 0.1582, 0.4596, 2.]])
        np.testing.assert_array_equal(self.topsis.dataframe.values, results)

    def test_vikor_rank(self):
        results = np.array([[0.9000, 0.3000, 0.7500, 3.],
                            [0.3500, 0.2000, 0.0000, 1.],
                            [0.4000, 0.4000, 0.5455, 2.]])
        np.testing.assert_array_equal(self.vikor.dataframe.values, results)


class RankSimilarityTestCase(unittest.TestCase):

    def setUp(self):
        kwargs = {
            'data': xij,
            'beneficial': beneficial,
            'weights': weights,
            'rank_reverse': True,
            'rank_method': "ordinal"
        }
        self.wsm = exe.WSM(**kwargs)
        self.wpm = exe.WPM(**kwargs)
        self.moora = exe.Moora(**kwargs)

    def test_rank_similarity_analysis(self):
        analizer = exe.RankSimilarityAnalyzer()
        analizer.add_executor(self.wsm)
        analizer.add_executor(self.wpm)
        analizer.add_executor(self.moora)
        results = analizer.analyze()


if __name__ == '__main__':
    unittest.main()
