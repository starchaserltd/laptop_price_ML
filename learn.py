import pdb

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
)

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
    read_csv, 
)

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)


def load_data() -> DataFrame:
    return read_csv('data/pricing.csv', sep='|')


def load_fold(split: str, fold: int) -> List[int]:
    PATH = 'filelists/{:s}_{:02d}.txt'
    with open(PATH.format(split, fold), 'r') as f:
        return [int(i) for i in f.readlines()]


def select_data(data_frame: DataFrame, ids: List[int]) -> DataFrame:
    return data_frame[data_frame['id'].isin(ids)]


def rel_error(x, y):
    return np.abs(x - y) / x * 100


def mean_rel_error(true_values, estimated_values):
    xs = np.array(true_values)
    ys = np.array(estimated_values)
    return np.mean(rel_error(xs, ys))


def evaluate_fold(classifier, data: DataFrame, i: int, verbose: int=0) -> float:
    # Train
    tr_ids = load_fold('train', i)
    tr_data = select_data(data, tr_ids)
    classifier.fit(tr_data)
    # Test
    te_ids = load_fold('test', i)
    te_data = select_data(data, te_ids)
    # Predict
    tr_preds = classifier.predict(tr_data)
    te_preds = classifier.predict(te_data)
    if verbose == 0:
        print('\n'.join('{:10d} {:10.2f} {:10.2f} {:10.2f}'.format(i, t, p, rel_error(t, p))
                        for i, t, p in zip(np.array(te_data.id), te_data.realprice, te_preds)))
    tr_error = mean_rel_error(tr_data.realprice, tr_preds)
    te_error = mean_rel_error(te_data.realprice, te_preds)
    return tr_error, te_error


def evaluate(classifier, data: DataFrame):
    return zip(*[evaluate_fold(classifier, data, i) for i in range(10)])


def print_results(results: List[float]):
    print('{:.2f} ± {:.2f}'.format(
        np.mean(results),
        np.std(results) / len(results)), end=' | ')
    for r in results:
        print('{:.1f}'.format(r), end=' ') 
    print()


class DBPricePredictor:

    def fit(self, data_frame: DataFrame):
        return self

    def predict(self, data_frame):
        return data_frame.price


class SilviuNumericFeaturesPredictor:

    def __init__(self):
        self.feature_names_ = [
            "CPU_rating",
            "CPU_tdp",
            "GPU_rating",
            "GPU_power",
            "ACUM_rating",
            "CHASSIS_thic",
            "CHASSIS_weight",
            "CHASSIS_rating",
            "DISPLAY_rating",
            "HDD_rating",
            "MDB_rating",
            "MEM_rating",
            "ODD_price",
            "SIST_price",
            "WAR_rating",
            "WNET_rating",
        ]
        self.estimator_ = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.05)
        # self.estimator_ = KernelRidge(alpha=1, kernel='polynomial', degree=4)
        self.scaler_ = StandardScaler()
        # self.poly_ = PolynomialFeatures()

    def fit(self, data_frame):
        X = np.array(data_frame[self.feature_names_])
        y = np.array(data_frame.realprice).astype(np.float)
        X = self.scaler_.fit_transform(X)
        # X = self.poly_.fit_transform(X)
        self.estimator_.fit(X, y)
        return self

    def predict(self, data_frame):
        X = np.array(data_frame[self.feature_names_])
        X = self.scaler_.transform(X)
        # X = self.poly_.transform(X)
        return self.estimator_.predict(X)


def main():
    # classifier = DBPricePredictor()
    classifier = SilviuNumericFeaturesPredictor()
    data = load_data()
    tr_errors, te_errors = evaluate(classifier, data)

    print('Tr:', end=' ')
    print_results(tr_errors)

    print('Te:', end=' ')
    print_results(te_errors)


if __name__ == '__main__':
    main()
