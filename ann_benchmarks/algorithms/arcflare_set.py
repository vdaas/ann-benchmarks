from __future__ import absolute_import
import os
import zlib
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.arcflare import _load_ext

def _ext():
    """SetIndex lives in the same nanobind module as the dense index. With ARCFLARE_EXT_DIR set,
    that dir must hold an arcflare_ext*.so (strict, via _load_ext); otherwise use the installed module."""
    if os.environ.get('ARCFLARE_EXT_DIR'):
        return _load_ext(None, '')
    import arcflare_ext
    return arcflare_ext


class ArcFlareSet(BaseANN):
    """ArcFlare SetIndex: Jaccard ANN over sparse sets (NN-descent + alpha-RNG graph, exact beam).

    sig_bits > 0 enables 1-bit MinHash routing; it is dominated by the exact beam on kosarak and
    movielens10m (docs/superpowers/specs/2026-11-03-set-index-minhash-routing-negative.md).
    """

    def __init__(self, metric, params):
        if metric != 'jaccard':
            raise ValueError('ArcFlareSet supports jaccard only, got %r' % metric)
        self._R = int(params.get('R', 48))
        self._nn_k = int(params.get('nn_k', 64))
        self._alpha = float(params.get('alpha', 1.2))
        self._sig_bits = int(params.get('sig_bits', 0))
        self._n_landmark = int(params.get('n_landmark', 64))
        self._seed = int(params.get('seed', 42))
        self._nt = int(params.get('nt', 16))
        self._ef, self._rerank, self._n_entry = 64, 0, 0
        self.name = 'ArcFlareSet'

    def fit(self, X):
        ext = _ext()
        if not hasattr(ext, 'SetIndex'):
            raise RuntimeError('arcflare_ext has no SetIndex; point ARCFLARE_EXT_DIR at a build with SetIndex')
        n = len(X)
        lens = np.fromiter((len(x) for x in X), dtype=np.uint64, count=n)
        offsets = np.zeros(n + 1, dtype=np.uint64)
        np.cumsum(lens, out=offsets[1:])
        items64 = np.concatenate([np.asarray(x, dtype=np.int64) for x in X])
        if items64.size and (items64.min() < 0 or items64.max() >= 2 ** 32):
            raise ValueError('set element ids must fit in uint32')
        if n >= 2 ** 32 - 1:
            raise ValueError('too many rows')
        items = np.ascontiguousarray(items64, dtype=np.uint32)
        # cache tag: every parameter that affects the encoded data, plus a checksum of the data itself
        crc = zlib.crc32(items.tobytes(), zlib.crc32(offsets.tobytes()))
        tag = 'arcflareset-n{}-nnz{}-crc{:08x}-R{}-nnk{}-a{:.2f}-sig{}-lm{}-seed{}'.format(
            n, items.size, crc, self._R, self._nn_k, self._alpha, self._sig_bits, self._n_landmark, self._seed)
        index_dir = os.path.join(os.environ.get('ARCFLARE_SET_INDEX_DIR', INDEX_DIR), tag)
        self._index = ext.SetIndex(R=self._R, nn_k=self._nn_k, alpha=self._alpha, sig_bits=self._sig_bits,
                                   n_landmark=self._n_landmark, seed=self._seed)
        loaded = False
        if os.path.exists(os.path.join(index_dir, 'setindex.bin')):
            try:
                print('ArcFlareSet: loading cached index from', index_dir)
                self._index.load(index_dir)
                loaded = True
            except (RuntimeError, ValueError) as e:  # corrupt/stale cache: rebuild instead of failing the run
                print('ArcFlareSet: cache unusable (%s); rebuilding' % e)
        if not loaded:
            print('ArcFlareSet: building n=%d nnz=%d R=%d nn_k=%d alpha=%.2f sig=%d' % (
                n, items.size, self._R, self._nn_k, self._alpha, self._sig_bits))
            self._index.build(offsets, items, self._nt)
            os.makedirs(index_dir, exist_ok=True)
            self._index.save(index_dir)  # atomic (tmp + rename) in C++

    def set_query_arguments(self, ef, rerank=0, n_entry=0):
        self._ef, self._rerank, self._n_entry = int(ef), int(rerank), int(n_entry)
        self.name = 'ArcFlareSet(R={},nnk={},a={:.2f},sig={},lm={},ef={},rerank={},nentry={})'.format(
            self._R, self._nn_k, self._alpha, self._sig_bits, self._n_landmark,
            self._ef, self._rerank, self._n_entry)

    def query(self, v, n):
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.int64)
        return self._index.search(v, n, self._ef, self._rerank, self._n_entry)

    def batch_query(self, X, n):
        self._res = [self.query(v, n) for v in X]

    def get_batch_results(self):
        return self._res

    def __str__(self):
        return self.name
