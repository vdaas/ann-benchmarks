from __future__ import absolute_import
import os
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR

# Enable exact fp16 L2 beam scoring by default; overridable via environment.
# AF_FP16_BEAM=1 replaces RoutingPQ K=16 distance with true fp16 L2² per neighbor,
# giving HNSW-quality beam navigation — set small ef (40-200) for competitive QPS.
os.environ.setdefault('AF_FP16_BEAM', '1')


class ArcFlare(BaseANN):
    """ArcFlare: Vamana graph + multi-tier ADC (RoutingPQ + SQ8)."""

    def __init__(self, metric, params):
        self._metric      = metric          # 'euclidean' or 'angular'
        self._alpha       = float(params.get('alpha',            1.2))
        self._max_edges   = int(params.get('max_edges',          64))
        self._k_clusters  = int(params.get('k_clusters',        256))
        self._n_entry_pts = int(params.get('n_entry_points',      8))
        self._n_seeds     = int(params.get('n_cluster_seeds',    32))
        self._reorder     = str(params.get('reorder',        'gorder'))
        self._soar        = bool(params.get('soar',           False))
        self._rebuild     = bool(params.get('rebuild',        False))
        self._routing_pq_m = int(params.get('routing_pq_m',    0))  # 0 = default (auto)
        # sq8_beam: use AF_SQ8=ON, AF_FP16_BEAM=OFF build (SQ8 int8 beam, 2x memory bandwidth vs fp16)
        self._sq8_beam    = bool(params.get('sq8_beam',       False))
        # sq4_beam: use AF_SQ4=ON, AF_SQ8=ON, AF_FP16_BEAM=OFF build (~4x smaller beam footprint)
        self._sq4_beam    = bool(params.get('sq4_beam',       False))
        # sq2_beam: use AF_SQ2=ON, AF_SQ8=ON, AF_FP16_BEAM=OFF build (~8x smaller beam footprint)
        self._sq2_beam    = bool(params.get('sq2_beam',       False))
        # bapq: variance-sorted RoutingPQ subspace assignment (AF_BAPQ=ON)
        self._bapq        = bool(params.get('bapq',           False))
        # Query-time defaults
        self._gamma_enq           = 0.20
        self._gamma_term          = 0.40
        self._n_probe             = 4
        self._max_visits          = 0
        self._rerank_factor       = 0
        self._ef                  = 0
        self._cap_cluster_members = 0
        self._term_eps_fp16       = -1.0
        self._rerank_sq8          = 0
        self._pool_extra          = 0
        self._angular_expand_n    = 0
        self.name = 'ArcFlare'

    def fit(self, X):
        import sys as _sys
        import glob as _glob
        import importlib.util as _util
        # Python 3.14 find_spec() prefers the editable-install dist-info metadata over
        # sys.path entries, so a normal 'import arcflare_ext' always picks the
        # dist-packages copy.  Use spec_from_file_location to force-load the freshly
        # built .so from the build directory regardless of installed metadata.
        # worktree fp16:     AF_FP16_BEAM=ON + AF_SQ8=ON (default for L2 and angular fp16 beam)
        # worktree fp32_sq8: AF_FP16_BEAM=OFF + AF_SQ8=ON (SQ8 int8 beam; 2x bandwidth vs fp16)
        # main soar:         AF_SOAR=ON + AF_SQ8=ON (soar not yet in worktree build)
        _WORKTREE = '/home/kpango/go/src/github.com/vdaas/ArcFlare/.claude/worktrees/routing-pq-improve'
        _MAIN     = '/home/kpango/go/src/github.com/vdaas/ArcFlare'
        if self._soar:
            _BUILD = _MAIN + '/build/python_sdk_soar'
        elif self._bapq:
            _BUILD = _WORKTREE + '/build/python_sdk_bapq'
        elif self._sq2_beam:
            _BUILD = _WORKTREE + '/build/python_sdk_sq2'
        elif self._sq4_beam:
            _BUILD = _WORKTREE + '/build/python_sdk_sq4'
        elif self._sq8_beam:
            _BUILD = _WORKTREE + '/build/python_sdk_fp32_sq8'
        else:
            _BUILD = _WORKTREE + '/build/python_sdk_fp16'
        _loaded = _sys.modules.get('arcflare_ext')
        if _loaded is None or not getattr(_loaded, '__file__', '').startswith(_BUILD):
            _sos = sorted(_glob.glob(_BUILD + '/arcflare_ext*.so'))
            if _sos:
                _spec = _util.spec_from_file_location('arcflare_ext', _sos[0])
                _mod = _util.module_from_spec(_spec)
                _sys.modules['arcflare_ext'] = _mod
                _spec.loader.exec_module(_mod)
        import arcflare_ext  # noqa: E402 (delayed import by design)

        n, d = X.shape
        angular = self._metric == 'angular'
        tag = 'arcflare-D{}-M{}-K{}-a{:.2f}-ep{}-cs{}-{}{}-{}{}{}{}'.format(
            d, self._max_edges, self._k_clusters, self._alpha,
            self._n_entry_pts, self._n_seeds,
            self._reorder,
            '-rebuild' if self._rebuild else '',
            'ang' if angular else 'l2',
            '-soar' if self._soar else '',
            '-rpq{}'.format(self._routing_pq_m) if self._routing_pq_m > 0 else '',
            '-bapq' if self._bapq else ('-sq2' if self._sq2_beam else ('-sq4' if self._sq4_beam else '')))
        index_dir = os.path.join(INDEX_DIR, tag)

        self._index = arcflare_ext.ArcFlare(
            dim              = d,
            alpha            = self._alpha,
            max_edges        = self._max_edges,
            angular          = angular,
            k_clusters       = self._k_clusters,
            n_entry_points   = self._n_entry_pts,
            n_cluster_seeds  = self._n_seeds,
        )

        if os.path.exists(os.path.join(index_dir, 'aqindex')):
            print('ArcFlare: loading cached index from', index_dir)
            self._index.load(index_dir)
        else:
            n_threads = 0  # omp_get_max_threads() at call time
            print('ArcFlare: building N=%d D=%d metric=%s M=%d alpha=%.2f' % (
                n, d, self._metric, self._max_edges, self._alpha))
            self._index.build(
                np.ascontiguousarray(X, dtype=np.float32),
                n_threads=n_threads,
                batch_size=50000,
            )
            print('ArcFlare: rebuilding graph (brute_force)...')
            self._index.rebuild_brute_force(
                k_neighbors=self._max_edges, n_threads=n_threads,
                new_alpha=self._alpha, new_max_edges=self._max_edges,
            )
            print('ArcFlare: building multi-tier index (RoutingPQ + ADC)...')
            self._index.build_multi_tier(n_threads=n_threads)
            print('ArcFlare: reordering for cache locality (%s)...' % self._reorder)
            self._index.reorder_locality(self._reorder)
            if self._routing_pq_m > 0:
                print('ArcFlare: rebuilding RoutingPQ (m=%d)...' % self._routing_pq_m)
                self._index.rebuild_routing_pq(self._routing_pq_m)
            os.makedirs(index_dir, exist_ok=True)
            self._index.save(index_dir)
            print('ArcFlare: index saved to', index_dir)

    def set_query_arguments(self, *args):
        # runner.py calls set_query_arguments(*query_arguments).
        # query_arguments comes from algos.yaml query-args via _generate_combinations.
        # Depending on nesting, args may be a flat 5-tuple or a 1-tuple whose element is a list.
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            params = list(args[0])
        else:
            params = list(args)
        self._gamma_enq, self._gamma_term, self._n_probe, \
            self._max_visits, self._rerank_factor = params[:5]
        self._ef                  = int(params[5]) if len(params) > 5 else 0
        self._cap_cluster_members = int(params[6]) if len(params) > 6 else 0
        self._term_eps_fp16       = float(params[7]) if len(params) > 7 else -1.0
        self._rerank_sq8          = int(params[8])  if len(params) > 8  else 0
        self._pool_extra          = int(params[9])  if len(params) > 9  else 0
        self._angular_expand_n    = int(params[10]) if len(params) > 10 else 0
        self._n_cluster_seeds_s      = int(params[11])   if len(params) > 11 else 0
        self._adaptive_probe_ratio   = float(params[12]) if len(params) > 12 else 0.0
        cap_str = ',cap={}'.format(self._cap_cluster_members) if self._cap_cluster_members > 0 else ''
        eps_str = ',eps={:.3f}'.format(self._term_eps_fp16) if self._term_eps_fp16 >= 0 else ''
        rr8_str = ',rr8={}'.format(self._rerank_sq8) if self._rerank_sq8 > 0 else ''
        pe_str  = ',pe={}'.format(self._pool_extra) if self._pool_extra > 0 else ''
        en_str  = ',en={}'.format(self._angular_expand_n) if self._angular_expand_n > 0 else ''
        apr_str = ',apr={:.1f}'.format(self._adaptive_probe_ratio) if self._adaptive_probe_ratio > 0 else ''
        self.name = 'ArcFlare(ge={:.2f},gt={:.2f},np={},mv={},rf={},ef={}{}{}{}{}{}{})'.format(
            self._gamma_enq, self._gamma_term, self._n_probe,
            self._max_visits, self._rerank_factor, self._ef,
            cap_str, eps_str, rr8_str, pe_str, en_str, apr_str)

    def query(self, v, n):
        return self._index.search(
            np.asarray(v, dtype=np.float32),
            n,
            gamma_enq             = float(self._gamma_enq),
            gamma_term            = float(self._gamma_term),
            n_probe               = int(self._n_probe),
            max_visits            = int(self._max_visits),
            rerank_factor         = int(self._rerank_factor),
            ef                    = int(self._ef),
            cap_cluster_members   = int(self._cap_cluster_members),
            term_eps_fp16         = float(self._term_eps_fp16),
            pool_extra            = int(self._pool_extra),
            angular_expand_n      = int(self._angular_expand_n),
            n_cluster_seeds       = int(self._n_cluster_seeds_s),
            adaptive_probe_ratio  = float(self._adaptive_probe_ratio),
        )

    def batch_query(self, X, n):
        self._res = self._index.search_batch(
            np.ascontiguousarray(X, dtype=np.float32),
            n,
            gamma_enq             = float(self._gamma_enq),
            gamma_term            = float(self._gamma_term),
            n_probe               = int(self._n_probe),
            max_visits            = int(self._max_visits),
            rerank_factor         = int(self._rerank_factor),
            n_threads             = 80,
            ef                    = int(self._ef),
            cap_cluster_members   = int(self._cap_cluster_members),
            term_eps_fp16         = float(self._term_eps_fp16),
            rerank_sq8            = int(self._rerank_sq8),
            pool_extra            = int(self._pool_extra),
            angular_expand_n      = int(self._angular_expand_n),
            n_cluster_seeds       = int(self._n_cluster_seeds_s),
            adaptive_probe_ratio  = float(self._adaptive_probe_ratio),
        )

    def get_batch_results(self):
        return self._res

    def __str__(self):
        return self.name
