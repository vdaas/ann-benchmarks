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
        # sq4_asym: asymmetric SQ4 encoding (per-node min/range, unsigned nibbles [0,15]) — adds '-asym' index suffix
        self._sq4_asym    = bool(params.get('sq4_asym',       False))
        # sq2_beam: use AF_SQ2=ON, AF_SQ8=ON, AF_FP16_BEAM=OFF build (~8x smaller beam footprint)
        self._sq2_beam    = bool(params.get('sq2_beam',       False))
        # bapq: variance-sorted RoutingPQ subspace assignment (AF_BAPQ=ON)
        self._bapq        = bool(params.get('bapq',           False))
        # saq: per-subspace OPQ/PCA rotation for RoutingPQ (AF_SAQ=ON + AF_SQ8=ON + AF_SOAR=ON)
        self._saq         = bool(params.get('saq',            False))
        # saq_fp16: SAQ + fp16 beam (AF_SAQ=ON + AF_FP16_BEAM=ON + AF_SQ8=ON + AF_SOAR=OFF)
        self._saq_fp16    = bool(params.get('saq_fp16',       False))
        # beam_mode: "auto" (fp16 if available), "fp16", "sq8", "sq4", "sq2"
        self._beam_mode   = str(params.get('beam_mode',       'auto'))
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
        self._adaptive_ef_threshold = -1.0
        self._adaptive_ef_extended  = 0
        self._presort_seed_mult     = 0
        self._term_eps              = -1.0
        self._crinn_no_improve_limit = 0
        self._beam_rpq_filter_k     = 0
        self._beam_rpq_gate_margin  = -1.0
        self._seed_rpq_gate          = -1.0
        self._fp16_early_exit_factor = -1.0
        self._weighted_seed_budget   = False
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
        elif self._saq_fp16:
            _BUILD = _WORKTREE + '/build_saq_fp16'
        elif self._saq:
            _BUILD = _WORKTREE + '/build_soar_saq'
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
            '-saqfp16' if self._saq_fp16 else ('-saq' if self._saq else ('-bapq' if self._bapq else ('-sq2' if self._sq2_beam else ('-sq4-asym' if (self._sq4_beam and self._sq4_asym) else ('-sq4' if self._sq4_beam else ''))))))
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
        # Compact format: [np, ef, ncs=128, psm=24, crinn=0, rpqgm=auto, eps=auto, srg=-1, feef=-1, wsb=0]
        #   np    - n_probe: cluster probes (required)
        #   ef    - beam size (required)
        #   ncs   - n_cluster_seeds: seeds per probe (default 128)
        #   psm   - presort_seed_mult: RPQ-ranked seed cap before fp16 rescore (default 24)
        #   crinn - crinn_no_improve_limit: 0 means C++ auto (ef*2/3) (default 0)
        #   rpqgm - beam_rpq_gate_margin: -1 = off, auto = 0.3 (ef<80) / 0.2 (ef>=80)
        #   eps   - term_eps_fp16: -1 = off, auto = 0.3 (ef<80, lossless) / -1 (ef>=80)
        #   srg   - seed_rpq_gate: RPQ score threshold to early-filter seeds (-1 = off)
        #   feef  - fp16_early_exit_factor: fp16 beam early-exit factor (-1 = off)
        #   wsb   - weighted_seed_budget: allocate more seeds to nearby clusters (0=off, 1=on)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            params = list(args[0])
        else:
            params = list(args)
        self._n_probe = int(params[0])
        self._ef      = int(params[1])
        self._n_cluster_seeds_s      = int(params[2])   if len(params) > 2 else 128
        self._presort_seed_mult      = int(params[3])   if len(params) > 3 else 24
        self._crinn_no_improve_limit = int(params[4])   if len(params) > 4 else 0
        self._beam_rpq_gate_margin   = float(params[5]) if len(params) > 5 \
                                       else (0.3 if self._ef < 80 else 0.2)
        self._term_eps_fp16          = float(params[6]) if len(params) > 6 \
                                       else (0.3 if self._ef < 80 else -1.0)
        self._seed_rpq_gate          = float(params[7]) if len(params) > 7 else -1.0
        self._fp16_early_exit_factor = float(params[8]) if len(params) > 8 else -1.0
        self._weighted_seed_budget   = bool(int(params[9])) if len(params) > 9 else False
        crinn_disp = self._crinn_no_improve_limit if self._crinn_no_improve_limit > 0 else 'auto'
        rpqgm_disp = '{:.2f}'.format(self._beam_rpq_gate_margin) if self._beam_rpq_gate_margin >= 0 else 'off'
        eps_disp   = '{:.2f}'.format(self._term_eps_fp16) if self._term_eps_fp16 >= 0 else 'off'
        srg_disp   = '{:.2f}'.format(self._seed_rpq_gate) if self._seed_rpq_gate >= 0 else 'off'
        feef_disp  = '{:.2f}'.format(self._fp16_early_exit_factor) if self._fp16_early_exit_factor >= 0 else 'off'
        self.name = 'ArcFlare(np={},ef={},ncs={},psm={},crinn={},rpqgm={},eps={},srg={},feef={},wsb={})'.format(
            self._n_probe, self._ef, self._n_cluster_seeds_s,
            self._presort_seed_mult, crinn_disp, rpqgm_disp, eps_disp, srg_disp, feef_disp,
            1 if self._weighted_seed_budget else 0)

    def query(self, v, n):
        return self._index.search(
            np.asarray(v, dtype=np.float32),
            n,
            n_probe                  = int(self._n_probe),
            ef                       = int(self._ef),
            n_cluster_seeds          = int(self._n_cluster_seeds_s),
            presort_seed_mult        = int(self._presort_seed_mult),
            crinn_no_improve_limit   = int(self._crinn_no_improve_limit),
            beam_rpq_gate_margin     = float(self._beam_rpq_gate_margin),
            term_eps_fp16            = float(self._term_eps_fp16),
            seed_rpq_gate            = float(self._seed_rpq_gate),
            fp16_early_exit_factor   = float(self._fp16_early_exit_factor),
            weighted_seed_budget     = bool(self._weighted_seed_budget),
            beam_mode                = str(self._beam_mode),
        )

    def batch_query(self, X, n):
        self._res = self._index.search_batch(
            np.ascontiguousarray(X, dtype=np.float32),
            n,
            n_threads                = 80,
            n_probe                  = int(self._n_probe),
            ef                       = int(self._ef),
            n_cluster_seeds          = int(self._n_cluster_seeds_s),
            presort_seed_mult        = int(self._presort_seed_mult),
            crinn_no_improve_limit   = int(self._crinn_no_improve_limit),
            beam_rpq_gate_margin     = float(self._beam_rpq_gate_margin),
            term_eps_fp16            = float(self._term_eps_fp16),
            seed_rpq_gate            = float(self._seed_rpq_gate),
            fp16_early_exit_factor   = float(self._fp16_early_exit_factor),
            weighted_seed_budget     = bool(self._weighted_seed_budget),
            beam_mode                = str(self._beam_mode),
        )

    def get_batch_results(self):
        return self._res

    def __str__(self):
        return self.name
