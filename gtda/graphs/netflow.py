"""Net-flow calculations."""
# License: GNU AGPLv3

from functools import reduce
from operator import and_

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.csgraph import laplacian
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_symmetric

from ..base import PlotterMixin
from ..plotting import plot_heatmap
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import check_graph


def _check_connected(G):
    raise NotImplementedError


def net_flow(G, efficiency="speed"):
    _check_connected(G)

    # TODO remove if directed impl added
    check_symmetric(G)
    L = laplacian(G)
    C = np.zeros(L.shape)
    C[1:, 1:] = np.linalg.inv(L[1:, 1:])

    N = max(G.shape)
    E = G.number_of_edges()
    B = nx.incidence_matrix(G, oriented=True).T  # shape=(nodes,edges)

    if efficiency == "speed":
        F = B @ C
        F_ranks = np.apply_along_axis(rankdata, arr=F, axis=1)
        values = np.sum((2 * F_ranks - 1 - N) * F, axis=1)
    elif efficiency == "memory":
        values = np.zeros(G.number_of_edges())
        for idx, B_row in enumerate(B):
            F_row = B_row@C
            rank = rankdata(F_row)
            values[idx] = np.sum((2 * rank - 1 - N) * F_row)
    else:
        raise Exception("Efficiency unknown.")

    edge_dict = dict(zip(G.edges, values))
    return edge_dict


@adapt_fit_transform_docs
class NetFlow(BaseEstimator, TransformerMixin, PlotterMixin):
    """Weighted graphs constructed ...

    For each (possibly weighted and/or directed) graph in a collection, this
    transformer calculates ???.

    The graphs are represented by their adjacency matrices which can be dense
    arrays, sparse matrices or masked arrays. The following rules apply:

    - In dense arrays of Boolean type, entries which are ``False`` represent
      absent edges.
    - In dense arrays of integer or float type, zero entries represent edges
      of length 0. Absent edges must be indicated by ``numpy.inf``.
    - In sparse matrices, non-stored values represent absent edges. Explicitly
      stored zero or ``False`` edges represent edges of length 0.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    directed : bool, optional, default: ``True``
        If ``True`` (default), then find the shortest path on a directed graph.
        If ``False``, then find the shortest path on an undirected graph.

    weighted : bool

    method : ``"speed"`` | ``"memory"``, optional, default: ``"speed"``
        Algorithm to use. See ?.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.graphs import TransitionGraph, GraphGeodesicDistance
    >>> X = np.arange(4).reshape(1, -1, 1)
    >>> X_tg = NetFlow().fit_transform(X)
    >>> print(X_tg[0].toarray())
    [[0 1 0 0]
     [0 0 1 0]
     [0 0 0 1]
     [0 0 0 0]]

    See also
    --------
    GraphGeodesicDistance, TransitionGraph, KNeighborsGraph

    """

    def __init__(self, n_jobs=None, directed=False, method="speed"):
        self.n_jobs = n_jobs
        self.directed = directed
        self.method = method

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Input data: a collection of adjacency matrices of graphs. Each
            adjacency matrix may be a dense or a sparse array.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_graph(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute ?.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Input data: a collection of ``n_samples`` adjacency matrices of
            graphs. Each adjacency matrix may be a dense array, a sparse
            matrix, or a masked array.

        y : None
            Ignored.

        Returns
        -------
        Xt : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Output collection of dense distance matrices. If the distance
            matrices all have the same shape, a single 3D ndarray is returned.

        """
        check_is_fitted(self, '_is_fitted')
        X = check_graph(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(net_flow)(x) for x in X)

        x0_shape = Xt[0].shape
        if reduce(and_, (x.shape == x0_shape for x in Xt), True):
            Xt = np.asarray(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='blues', plotly_params=None):
        """Plot a sample from a collection of distance matrices.

        Parameters
        ----------
        Xt : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Collection of distance matrices, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample to be plotted.

        colorscale : str, optional, default: ``'blues'``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_heatmap(
            Xt[sample], colorscale=colorscale,
            title=f"{sample}-th geodesic distance matrix",
            plotly_params=plotly_params
            )