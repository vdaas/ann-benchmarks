"""Smoke test: ArcFlare harness end-to-end on sift-128 against a cached index.

Mirrors ann_benchmarks/runner.py (definitions -> instantiate_algorithm -> fit ->
set_query_arguments -> query()) and scores with ann_benchmarks' own recall code.
Never builds an index: aborts if the cached index dir is missing.

  ARCFLARE_EXT_DIR=<dir with arcflare_ext*.so> venv/bin/python -I tools_arcflare/smoke_arcflare.py [--row NP:EF ...] [--qargs JSON ...]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # INDEX_DIR / data/ are relative paths, as when running run.py
sys.path.insert(0, ROOT)

import numpy  # noqa: E402

from ann_benchmarks.algorithms.definitions import get_definitions, instantiate_algorithm  # noqa: E402
from ann_benchmarks.constants import INDEX_DIR  # noqa: E402
from ann_benchmarks.datasets import get_dataset  # noqa: E402
from ann_benchmarks.distance import dataset_transform, metrics  # noqa: E402
from ann_benchmarks.plotting.metrics import get_recall_values, knn_threshold  # noqa: E402

DATASET = 'sift-128-euclidean'
ALGO = 'ArcFlare-A140-L2'  # algos.yaml; plain fp16 variant, cached index arcflare-D128-M64-K256-a1.40-ep32-cs128-gorder-l2

ap = argparse.ArgumentParser()
ap.add_argument('--row', action='append', default=None, metavar='NP:EF',
                help='legacy algos.yaml query row with this n_probe:ef (default 8:100)')
ap.add_argument('--qargs', action='append', default=[], metavar='JSON',
                help='explicit set_query_arguments list, e.g. [16,200,128,0,0,-1,-1] (no auto rpqgm/eps)')
ap.add_argument('--nq', type=int, default=1000)
ap.add_argument('--count', type=int, default=10)
args = ap.parse_args()
wanted = [tuple(int(x) for x in r.split(':')) for r in (args.row or ([] if args.qargs else ['8:100']))]

D, dimension = get_dataset(DATASET)
distance = D.attrs['distance']
X_train, X_test = dataset_transform(D)
X_test = X_test[:args.nq]
true_dists = numpy.array(D['distances'])[:args.nq]

defs = [d for d in get_definitions('algos.yaml', dimension, 'float', 'euclidean', args.count)
        if d.algorithm == ALGO and not d.disabled]
assert defs, 'no %s definitions in algos.yaml' % ALGO
definition = defs[0]  # run-group a140-l2-fp16
assert not any(k.endswith('_beam') for k in definition.arguments[1]), definition.arguments

tag = 'arcflare-D{}-M{}-K{}-a{:.2f}-ep{}-cs{}-{}-l2'.format(
    dimension, definition.arguments[1]['max_edges'], definition.arguments[1]['k_clusters'],
    definition.arguments[1]['alpha'], definition.arguments[1]['n_entry_points'],
    definition.arguments[1]['n_cluster_seeds'], definition.arguments[1]['reorder'])
cache = os.path.join(INDEX_DIR, tag, 'aqindex')
if not os.path.exists(cache):
    sys.exit('refusing to build: cached index missing: ' + cache)

algo = instantiate_algorithm(definition)
t0 = time.time()
algo.fit(X_train)  # hits the cache (checked above)
print('fit (cache load) %.1fs' % (time.time() - t0))

runs = []
for np_v, ef_v in wanted:
    qa = [g for g in definition.query_argument_groups
          if g[0][2] == np_v and g[0][5] == ef_v]
    assert qa, 'no legacy row with np=%d ef=%d in %s' % (np_v, ef_v, ALGO)
    runs.append(qa[0])
runs += [[json.loads(q)] for q in args.qargs]

for qarg in runs:
    algo.set_query_arguments(*qarg)  # exactly as runner.run()
    for v in X_test[:20]:  # warm-up, untimed
        algo.query(v, args.count)
    run_dists, total = [], 0.0
    for v in X_test:
        start = time.time()
        cand = algo.query(v, args.count)
        total += time.time() - start
        run_dists.append([float(metrics[distance]['distance'](v, X_train[int(i)])) for i in cand])
    mean, std, _ = get_recall_values(true_dists, run_dists, args.count, knn_threshold, 1e-3)
    print('%s\n  recall@%d=%.4f  QPS=%.0f  (nq=%d, single thread, query() only)' % (
        algo, args.count, mean, len(X_test) / total, len(X_test)))
