"""Testing for ForgetDimension and Scaler."""
# License: GNU AGPLv3

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from gtda.diagrams import ForgetDimension, Scaler

X_1 = np.array([[[0., 0.36905774, 0],
                 [0., 0.37293977, 0],
                 [0., 0.38995215, 0],
                 [0., 3.00457644, 0],
                 [0., 3.04772496, 0],
                 [0., 3.32096624, 0],
                 [7.97852135, 8.00382805, 1],
                 [1.79289687, 1.8224113, 1],
                 [1.69005811, 2.32093406, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 0.36905774, 0],
                 [0., 0.37293977, 0],
                 [0., 0.38995215, 0],
                 [0., 3.00457644, 0],
                 [0., 3.04772496, 0],
                 [0., 3.32096624, 0],
                 [15.27686119, 24.32133484, 1],
                 [1.79289687, 1.8224113, 1],
                 [1.69005811, 2.32093406, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [25.24895287, 25.36620903, 2],
                 [25.15629959, 25.18988037, 2],
                 [25.06381798, 25.23542404, 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 1.32215321, 0],
                 [0., 1.48069561, 0],
                 [0., 1.62762213, 0],
                 [0., 3.18582344, 0],
                 [0., 3.25349188, 0],
                 [0., 3.28288555, 0],
                 [3.32096624, 23.92891693, 1],
                 [2.93474603, 3.07139683, 1],
                 [2.83503842, 2.94497037, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [24.84643745, 25.06381798, 2],
                 [24.77123451, 25.04314995, 2],
                 [24.67935562, 24.93212509, 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 0.13842253, 0],
                 [0., 0.13912651, 0],
                 [0., 0.15125643, 0],
                 [0., 3.30339432, 0],
                 [0., 3.3078723, 0],
                 [0., 3.33029914, 0],
                 [16.82829285, 16.84351158, 1],
                 [16.8180275, 16.84162521, 1],
                 [16.80234337, 16.80629158, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 0.13842253, 0],
                 [0., 0.13912651, 0],
                 [0., 0.15125643, 0],
                 [0., 2.91534185, 0],
                 [0., 2.93620634, 0],
                 [0., 3.00776553, 0],
                 [2.85910106, 3.39503384, 1],
                 [1.25564897, 1.25871313, 1],
                 [1.24251938, 1.27403092, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [3.66717434, 3.69763446, 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 0.13842253, 0],
                 [0., 0.13912651, 0],
                 [0., 0.15125643, 0],
                 [0., 2.91534185, 0],
                 [0., 2.93620634, 0],
                 [0., 3.00776553, 0],
                 [21.34438705, 24.6866188, 1],
                 [2.85910106, 2.88541412, 1],
                 [1.52559161, 3.39503384, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [25.05523109, 25.27045441, 2],
                 [24.93939018, 25.25673294, 2],
                 [24.89836693, 25.20828819, 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]]])

X_2 = np.array([[[0., 0.36905774, 0],
                 [0., 0.37293977, 0],
                 [0., 0.38995215, 0],
                 [0., 3.00457644, 0],
                 [0., 3.32096624, 0],
                 [7.97852135, 8.00382805, 1],
                 [1.79289687, 1.8224113, 1],
                 [1.69005811, 2.32093406, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 0.36905774, 0],
                 [0., 0.37293977, 0],
                 [0., 3.00457644, 0],
                 [0., 3.04772496, 0],
                 [0., 3.32096624, 0],
                 [15.27686119, 24.32133484, 1],
                 [1.79289687, 1.8224113, 1],
                 [1.69005811, 2.32093406, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [25.24895287, 25.36620903, 2],
                 [25.15629959, 25.18988037, 2],
                 [25.06381798, 25.23542404, 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 1.32215321, 0],
                 [0., 1.48069561, 0],
                 [0., 3.18582344, 0],
                 [0., 3.25349188, 0],
                 [0., 3.28288555, 0],
                 [3.32096624, 23.92891693, 1],
                 [2.93474603, 3.07139683, 1],
                 [2.83503842, 2.94497037, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [24.84643745, 25.06381798, 2],
                 [24.77123451, 25.04314995, 2],
                 [24.67935562, 24.93212509, 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 0.13842253, 0],
                 [0., 0.13912651, 0],
                 [0., 0.15125643, 0],
                 [0., 3.3078723, 0],
                 [0., 3.33029914, 0],
                 [16.82829285, 16.84351158, 1],
                 [16.8180275, 16.84162521, 1],
                 [16.80234337, 16.80629158, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 0.13842253, 0],
                 [0., 0.13912651, 0],
                 [0., 0.15125643, 0],
                 [0., 2.93620634, 0],
                 [0., 3.00776553, 0],
                 [2.85910106, 3.39503384, 1],
                 [1.25564897, 1.25871313, 1],
                 [1.24251938, 1.27403092, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [3.66717434, 3.69763446, 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2],
                 [0., 0., 2]],
                [[0., 0.13842253, 0],
                 [0., 0.13912651, 0],
                 [0., 0.15125643, 0],
                 [0., 2.93620634, 0],
                 [0., 3.00776553, 0],
                 [21.34438705, 24.6866188, 1],
                 [2.85910106, 2.88541412, 1],
                 [1.52559161, 3.39503384, 1],
                 [0., 0., 1],
                 [0., 0., 1],
                 [25.05523109, 25.27045441, 2],
                 [24.93939018, 25.25673294, 2],
                 [24.89836693, 26.20828819, 2],
                 [0., 0., 2],
                 [0., 0., 2]]])


def test_not_fitted():
    dst = ForgetDimension()
    dsc = Scaler()

    with pytest.raises(NotFittedError):
        dst.transform(X_1)

    with pytest.raises(NotFittedError):
        dsc.transform(X_1)

    with pytest.raises(NotFittedError):
        dsc.inverse_transform(X_1)


@pytest.mark.parametrize('X', [X_1, X_2])
def test_dst_transform(X):
    dst = ForgetDimension()
    X_res = dst.fit_transform(X)
    assert X_res.shape == X.shape


parameters = [('wasserstein', {'p': 2}),
              ('betti', {'n_bins': 10}),
              ('bottleneck', None)]


@pytest.mark.parametrize(('metric', 'metric_params'), parameters)
@pytest.mark.parametrize('X', [X_1, X_2])
def test_dd_transform(X, metric, metric_params):
    dsc = Scaler(metric=metric, metric_params=metric_params, n_jobs=1)
    X_res = dsc.fit_transform(X)
    assert X_res.shape == X.shape

    dsc = Scaler(metric=metric, metric_params=metric_params, n_jobs=1)
    X_inv_res = dsc.fit(X_res).inverse_transform(X_res)
    assert X_inv_res.shape == X.shape
