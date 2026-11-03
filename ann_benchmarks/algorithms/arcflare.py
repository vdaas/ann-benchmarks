from __future__ import absolute_import
import os
import warnings
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR

# Enable exact fp16 L2 beam scoring by default; overridable via environment.
# AF_FP16_BEAM=1 replaces RoutingPQ K=16 distance with true fp16 L2² per neighbor,
# giving HNSW-quality beam navigation — set small ef (40-200) for competitive QPS.
os.environ.setdefault('AF_FP16_BEAM', '1')


_AUTOFILL_WARNED = False  # warn once per process when rpqgm/eps are auto-filled
_PIN = None  # (realpath of the arcflare_ext file loaded in this process, came from plain import?)


def _load_ext(var, legacy):
    """Load arcflare_ext. var: variant suffix (None = default fp16 build).

    Dir resolution: ARCFLARE_EXT_DIR[_<VAR>] (strict: no .so there is an error) ->
    legacy hard-coded dir (lenient: no .so there falls back to an installed module).
    A non-default variant never falls back to the generic ARCFLARE_EXT_DIR, and if any
    ARCFLARE_EXT_DIR* is set its own variant var is required.
    The first successful load pins the extension file for the process (hot-swapping
    nanobind modules under one name is unsafe); a later call that resolves to a different
    file raises. The engine exposes no build-flag introspection, so the variant can't be
    verified from the loaded module itself.
    """
    global _PIN
    import sys
    import glob
    import importlib.machinery
    import importlib.util
    name = 'ARCFLARE_EXT_DIR' + ('_' + var if var else '')
    given = sorted(k for k, v in os.environ.items() if k.startswith('ARCFLARE_EXT_DIR') and v)
    if os.environ.get(name):
        build, strict = os.environ[name], True
    elif var and given:
        raise FileNotFoundError('ArcFlare variant %s requested but %s is unset (set: %s)' % (
            var, name, ', '.join(given)))
    else:
        build, strict = legacy, False
    sos = sorted(glob.glob(build + '/arcflare_ext*.so'))
    if not sos and strict:
        raise FileNotFoundError('ArcFlare: no arcflare_ext*.so in %s (from %s)' % (build, name))
    target = os.path.realpath(sos[0]) if sos else None  # None: use the installed module
    loaded = sys.modules.get('arcflare_ext')
    if _PIN is None and loaded is not None:  # imported by someone else
        _PIN = (os.path.realpath(loaded.__file__), False)  # origin unknown: verify via PathFinder
    if _PIN is not None:
        if target is None:
            spec = importlib.machinery.PathFinder.find_spec('arcflare_ext')
            ok = _PIN[1] or (spec is not None and spec.origin is not None
                             and os.path.realpath(spec.origin) == _PIN[0])
        else:
            ok = target == _PIN[0]
        if not ok:
            raise RuntimeError(
                'ArcFlare: a different arcflare_ext (%s) is already loaded in this process; '
                'run variant %s in a separate process' % (_PIN[0], var or 'default'))
        return sys.modules['arcflare_ext']
    if target:
        spec = importlib.util.spec_from_file_location('arcflare_ext', target)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['arcflare_ext'] = mod
        spec.loader.exec_module(mod)
        _PIN = (target, False)
        print('ArcFlare: loaded extension', target)
        return mod
    import arcflare_ext  # noqa: E402 (delayed import by design; legacy dir has no .so)
    _PIN = (os.path.realpath(arcflare_ext.__file__), True)
    print('ArcFlare: using importable extension', arcflare_ext.__file__)
    return arcflare_ext


def parse_query_args(args):
    """Resolve set_query_arguments(*args) into effective parameters (pure, no side effects).

    Returns (eff, autofilled): eff maps attribute-name suffix -> value; autofilled lists
    the names ('rpqgm', 'eps') whose value was not given explicitly by the row.
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        params = list(args[0])
    else:
        params = list(args)
    # Backward-compat: old 6/10/12-element format [ge, gt, np, mv, rf, ef, ...]
    # detected by ge==0.0 and gt<0 as first two elements.
    if (len(params) in {6, 10, 12} and
            isinstance(params[0], float) and params[0] == 0.0 and
            isinstance(params[1], float) and params[1] < 0):
        np_v  = int(params[2])
        ef_v  = int(params[5])
        ncs_v = int(params[11]) if len(params) >= 12 and int(params[11]) > 0 else 128
        params = [np_v, ef_v, ncs_v]
    ef = int(params[1])
    eff = dict(
        n_probe               = int(params[0]),
        ef                    = ef,
        n_cluster_seeds_s     = int(params[2])   if len(params) > 2 else 128,
        presort_seed_mult     = int(params[3])   if len(params) > 3 else 24,
        crinn_no_improve_limit = int(params[4])  if len(params) > 4 else 0,
        beam_rpq_gate_margin  = float(params[5]) if len(params) > 5 else (0.3 if ef < 80 else 0.2),
        term_eps_fp16         = float(params[6]) if len(params) > 6 else (0.3 if ef < 80 else -1.0),
        seed_rpq_gate         = float(params[7]) if len(params) > 7 else -1.0,
        fp16_early_exit_factor = float(params[8]) if len(params) > 8 else -1.0,
        weighted_seed_budget  = bool(int(params[9])) if len(params) > 9 else False,
        finger_gate_factor    = float(params[10]) if len(params) > 10 else -1.0,
        sq4_pfah              = int(params[11])   if len(params) > 11 else 0,
        term_eps              = float(params[12]) if len(params) > 12 else -1.0,
        adaptive_probe_ratio  = float(params[13]) if len(params) > 13 else 0.0,
        scan_reorder          = int(params[14])   if len(params) > 14 else -1,
    )
    autofilled = [n for n, i in (('rpqgm', 5), ('eps', 6)) if len(params) <= i]
    return eff, autofilled


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
        # graph: "hnsw" grafts an hnswlib level-0 graph onto the built index
        # (edge quality of HNSW + ArcFlare's ADC/beam scoring). Adds '-hnswgraft'.
        self._graph_mode  = str(params.get('graph',           ''))
        self._hnsw_m      = int(params.get('hnsw_m',          32))
        self._hnsw_efc    = int(params.get('hnsw_efc',       500))
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
        self._adaptive_probe_ratio  = 0.0
        self._crinn_no_improve_limit = 0
        self._beam_rpq_filter_k     = 0
        self._beam_rpq_gate_margin  = -1.0
        self._seed_rpq_gate          = -1.0
        self._fp16_early_exit_factor = -1.0
        self._weighted_seed_budget   = False
        self._scan_reorder          = -1
        self.name = 'ArcFlare'

    def fit(self, X):
        # Python 3.14 find_spec() prefers the editable-install dist-info metadata over
        # sys.path entries, so a normal 'import arcflare_ext' always picks the
        # dist-packages copy.  Use spec_from_file_location to force-load the freshly
        # built .so from the build directory regardless of installed metadata.
        _MAIN     = '/home/kpango/go/src/github.com/vdaas/ArcFlare'
        if self._soar:
            _BUILD, _VAR = _MAIN + '/build/python_sdk_soar', 'SOAR'
        elif self._saq_fp16:
            _BUILD, _VAR = _MAIN + '/build_saq_fp16', 'SAQ_FP16'
        elif self._saq:
            _BUILD, _VAR = _MAIN + '/build_soar_saq', 'SAQ'
        elif self._bapq:
            _BUILD, _VAR = _MAIN + '/build/python_sdk_bapq', 'BAPQ'
        elif self._sq2_beam:
            _BUILD, _VAR = _MAIN + '/build/python_sdk_sq2', 'SQ2'
        elif self._sq4_beam:
            _BUILD, _VAR = _MAIN + '/build/python_sdk_sq4', 'SQ4'
        elif self._sq8_beam:
            _BUILD, _VAR = _MAIN + '/build/python_sdk_fp32_sq8', 'SQ8'
        else:
            _BUILD, _VAR = _MAIN + '/build/python_sdk_fp16', None
        arcflare_ext = _load_ext(_VAR, _BUILD)

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
        if self._graph_mode == 'hnsw':
            tag += '-hnswgraft'
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
            if self._graph_mode == 'hnsw':
                import hnswlib  # delayed: only needed for graph=hnsw
                import tempfile
                space = 'cosine' if angular else 'l2'
                print('ArcFlare: building hnswlib graph (space=%s M=%d efc=%d) for graft...' % (
                    space, self._hnsw_m, self._hnsw_efc))
                hp = hnswlib.Index(space=space, dim=d)
                hp.init_index(max_elements=n, ef_construction=self._hnsw_efc, M=self._hnsw_m)
                hp.set_num_threads(os.cpu_count() or 1)
                hp.add_items(np.ascontiguousarray(X, dtype=np.float32),
                             np.arange(n, dtype=np.int64))
                fd, tmp = tempfile.mkstemp(suffix='.hnsw')
                os.close(fd)
                try:
                    hp.save_index(tmp)
                    print('ArcFlare: grafting hnswlib level-0 graph...')
                    self._index.graft_hnsw(tmp)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
            os.makedirs(index_dir, exist_ok=True)
            self._index.save(index_dir)
            print('ArcFlare: index saved to', index_dir)

    def set_query_arguments(self, *args):
        # Compact format: [np, ef, ncs=128, psm=24, crinn=0, rpqgm=auto, eps=auto, srg=-1, feef=-1, wsb=0, fgf=-1, sq4pfah=0, teps=-1, apr=0, scan_reorder=-1]
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
        #   fgf   - finger_gate_factor: FINGER triangle-inequality gate (-1 = off)
        #   teps  - term_eps: early-term coefficient on the pool's already-computed approximate
        #           distance (SearchParameters::term_eps). Structurally identical to NGT/ONNG's
        #           `distance > explorationRadius` cutoff; reuses the existing pool distance so it
        #           adds zero extra distance computation (unlike term_eps_fp16). -1 = off (default).
        #   apr   - adaptive_probe_ratio: IVF-style adaptive n_probe. Coarse-quantizer probe
        #           clusters are truncated to those within centroid_dist <= dist[0]*(1+apr) of the
        #           nearest cluster (KMeansCentering.hpp / Index.cpp:2174-2217). Orthogonal to teps
        #           (different pipeline stage: coarse-quantizer probe selection vs beam-loop early
        #           termination) — effects compound rather than overlap. 0 = off (default, matches
        #           SearchContext.hpp's adaptive_probe_ratio=0.0f).
        #   scan_reorder - scan_reorder: enables searchScanTier() (src/ScanTier.cpp), an IVF-style
        #           partition-scan search that bypasses the graph beam entirely. The value is BOTH
        #           the dynamic-eps top-N capacity and the number of candidates handed to the fp16
        #           exact rerank pass; n_probe controls how many centroid partitions are scanned.
        #           adaptive_probe_ratio (apr) applies to this path too (wired in scan-tier-apr-wiring).
        #           -1 = off (default, uses the graph beam path via Index.cpp).
        eff, autofilled = parse_query_args(args)
        global _AUTOFILL_WARNED
        if autofilled and not _AUTOFILL_WARNED:
            _AUTOFILL_WARNED = True
            warnings.warn(
                'ArcFlare: query args %r left %s to auto-fill (rpqgm 0.3/0.2 and eps 0.3/-1 by ef); '
                'auto-fill can cost recall (sift np16/ef200: 0.9838 -> 0.8319). Pass rpqgm/eps explicitly.'
                % (list(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else list(args),
                   '/'.join(autofilled)), stacklevel=2)
        self._n_probe                = eff['n_probe']
        self._ef                     = eff['ef']
        self._n_cluster_seeds_s      = eff['n_cluster_seeds_s']
        self._presort_seed_mult      = eff['presort_seed_mult']
        self._crinn_no_improve_limit = eff['crinn_no_improve_limit']
        self._beam_rpq_gate_margin   = eff['beam_rpq_gate_margin']
        self._term_eps_fp16          = eff['term_eps_fp16']
        self._seed_rpq_gate          = eff['seed_rpq_gate']
        self._fp16_early_exit_factor = eff['fp16_early_exit_factor']
        self._weighted_seed_budget   = eff['weighted_seed_budget']
        self._finger_gate_factor     = eff['finger_gate_factor']
        self._sq4_pfah               = eff['sq4_pfah']
        self._term_eps               = eff['term_eps']
        self._adaptive_probe_ratio   = eff['adaptive_probe_ratio']
        self._scan_reorder           = eff['scan_reorder']
        crinn_disp = self._crinn_no_improve_limit if self._crinn_no_improve_limit > 0 else 'auto'
        rpqgm_disp = '{:.2f}'.format(self._beam_rpq_gate_margin) if self._beam_rpq_gate_margin >= 0 else 'off'
        eps_disp   = '{:.2f}'.format(self._term_eps_fp16) if self._term_eps_fp16 >= 0 else 'off'
        srg_disp   = '{:.2f}'.format(self._seed_rpq_gate) if self._seed_rpq_gate >= 0 else 'off'
        feef_disp  = '{:.2f}'.format(self._fp16_early_exit_factor) if self._fp16_early_exit_factor >= 0 else 'off'
        fgf_disp   = '{:.2f}'.format(self._finger_gate_factor) if self._finger_gate_factor >= 0 else 'off'
        pfah_disp  = self._sq4_pfah if self._sq4_pfah > 0 else 'def'
        teps_disp  = '{:.2f}'.format(self._term_eps) if self._term_eps >= 0 else 'off'
        apr_disp   = '{:.2f}'.format(self._adaptive_probe_ratio) if self._adaptive_probe_ratio > 0 else 'off'
        scan_disp  = self._scan_reorder if self._scan_reorder > 0 else 'off'
        self.name = 'ArcFlare(np={},ef={},ncs={},psm={},crinn={},rpqgm={},eps={},srg={},feef={},wsb={},fgf={},sq4pfah={},teps={},apr={},scan={})'.format(
            self._n_probe, self._ef, self._n_cluster_seeds_s,
            self._presort_seed_mult, crinn_disp, rpqgm_disp, eps_disp, srg_disp, feef_disp,
            1 if self._weighted_seed_budget else 0, fgf_disp, pfah_disp, teps_disp, apr_disp, scan_disp)

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
            term_eps                 = float(self._term_eps),
            adaptive_probe_ratio     = float(self._adaptive_probe_ratio),
            seed_rpq_gate            = float(self._seed_rpq_gate),
            fp16_early_exit_factor   = float(self._fp16_early_exit_factor),
            weighted_seed_budget     = bool(self._weighted_seed_budget),
            beam_mode                = str(self._beam_mode),
            finger_gate_factor       = float(self._finger_gate_factor),
            sq4_pfah                 = int(self._sq4_pfah),
            scan_reorder             = int(self._scan_reorder),
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
            term_eps                 = float(self._term_eps),
            adaptive_probe_ratio     = float(self._adaptive_probe_ratio),
            seed_rpq_gate            = float(self._seed_rpq_gate),
            fp16_early_exit_factor   = float(self._fp16_early_exit_factor),
            weighted_seed_budget     = bool(self._weighted_seed_budget),
            beam_mode                = str(self._beam_mode),
            finger_gate_factor       = float(self._finger_gate_factor),
            sq4_pfah                 = int(self._sq4_pfah),
            scan_reorder             = int(self._scan_reorder),
        )

    def get_batch_results(self):
        return self._res

    def __str__(self):
        return self.name
