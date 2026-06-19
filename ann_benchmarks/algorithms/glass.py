from __future__ import absolute_import
import os
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR

_GLASS_PATH = '/home/kpango/go/src/github.com/zilliztech/pyglass/python/build/lib.linux-x86_64-cpython-314'


class Glass(BaseANN):
    """Glass: high-performance HNSW/NSG variant from Zilliz (zilliztech/pyglass)."""

    def __init__(self, metric, params):
        self._metric = {'euclidean': 'L2', 'angular': 'IP'}[metric]
        self._M      = int(params.get('M',              32))
        self._ef_con = int(params.get('ef_construction', 200))
        # SQ8U for angular (IP) captures more precision for cosine similarity;
        # SQ4U for L2 is fine since euclidean ranges compress well at 4 bits.
        default_quant = 'SQ8U' if self._metric == 'IP' else 'SQ4U'
        self._quant  = str(params.get('quantizer', default_quant))
        self._ef     = self._ef_con
        self._graph   = None
        self._data    = None
        self._searcher = None
        self.name = 'Glass'

    def _graph_cache_path(self, n, d):
        # Primary cache: canonical name based on shape + build params
        fname = 'glass-N%d-D%d-M%d-ef%d-%s.graph' % (n, d, self._M, self._ef_con, self._metric)
        return os.path.join(INDEX_DIR, fname)

    def _find_legacy_graph(self, d):
        """Look for a pre-built graph matching (d, M, metric) in INDEX_DIR by filename."""
        import glob
        pattern = os.path.join(INDEX_DIR, 'glass-*-M%d.graph' % self._M)
        metric_tag = 'angular' if self._metric == 'IP' else 'euclidean'
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            if str(d) in fname and metric_tag in fname:
                return path
        return None

    def fit(self, X):
        import sys as _sys
        if _GLASS_PATH not in _sys.path:
            _sys.path.insert(0, _GLASS_PATH)
        import glass

        data = np.ascontiguousarray(X, dtype=np.float32)
        if self._metric == 'IP':
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            data = data / norms
        self._data = data
        n, d = data.shape

        cache_path = self._graph_cache_path(n, d)
        if os.path.exists(cache_path):
            print('Glass: loading cached graph from %s' % cache_path)
            self._graph = glass.Graph(cache_path)
        else:
            legacy = self._find_legacy_graph(d)
            if legacy:
                print('Glass: loading legacy graph from %s' % legacy)
                self._graph = glass.Graph(legacy)
            else:
                print('Glass: building  N=%d D=%d M=%d ef_con=%d metric=%s' % (
                    n, d, self._M, self._ef_con, self._metric))
                index = glass.Index(index_type='HNSW', metric=self._metric,
                                    R=self._M, L=self._ef_con)
                self._graph = index.build(data)
                os.makedirs(INDEX_DIR, exist_ok=True)
                self._graph.save(cache_path)
                print('Glass: graph saved to %s' % cache_path)

        self._searcher = glass.Searcher(
            graph=self._graph, data=self._data,
            metric=self._metric, quantizer=self._quant)
        self._searcher.set_ef(self._ef)
        self._searcher.optimize()
        print('Glass: index ready')

    def set_query_arguments(self, ef):
        self._ef = int(ef)
        self.name = 'Glass(M={},ef_con={},ef={})'.format(
            self._M, self._ef_con, self._ef)
        self._searcher.set_ef(self._ef)

    def query(self, v, n):
        q = np.asarray(v, dtype=np.float32)
        if self._metric == 'IP':
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
        ids, _ = self._searcher.search(query=q, k=n)
        return ids

    def batch_query(self, X, n):
        data = np.ascontiguousarray(X, dtype=np.float32)
        if self._metric == 'IP':
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            data = data / norms
        # batch_search returns (2, n_queries, k): [0]=ids, [1]=distances
        result = self._searcher.batch_search(data, n, 80)
        self._res = np.array(result[0])

    def get_batch_results(self):
        return self._res

    def __str__(self):
        return self.name
