"""Every ArcFlare query-arg combination, expanded by ann_benchmarks.definitions.get_definitions
(the loader run.py uses), must be accepted by set_query_arguments(*group) and resolve to
(n_probe > 0 or == -1 graph_entry) and ef > 0."""
import os
import sys
import unittest

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ann_benchmarks.algorithms.arcflare import parse_query_args  # noqa: E402
from ann_benchmarks.algorithms.definitions import get_definitions  # noqa: E402

# n_probe < 0 selects graph_entry seeding in src/Index.cpp (documented in algos.yaml a140-l2-graph-entry)
GRAPH_ENTRY = -1
YAML = os.path.join(ROOT, 'algos.yaml')


def resolved_runs(path=YAML):
    """Yield (point, metric, algo, build args, query group, eff|None, error|None)."""
    with open(path) as f:
        data = yaml.load(f, yaml.SafeLoader)
    for point, metrics in data.items():
        for metric in metrics:
            if metric == 'any':
                continue
            for d in get_definitions(path, 128, point, metric):
                if d.constructor != 'ArcFlare':
                    continue
                for group in d.query_argument_groups or [[]]:
                    try:
                        eff, _ = parse_query_args(tuple(group))
                        err = None
                    except TypeError as e:
                        eff, err = None, repr(e)
                    yield point, metric, d.algorithm, d.arguments, group, eff, err


class TestQueryArgsValid(unittest.TestCase):
    def test_all_rows_valid(self):
        runs = list(resolved_runs())
        self.assertTrue(runs)
        type_err = [r for r in runs if r[6]]
        if type_err:
            self.fail('%d TypeError runs in %s' % (len(type_err), sorted({r[2] for r in type_err})))
        bad = [r for r in runs if r[5] is None or (r[5]['n_probe'] <= 0 and r[5]['n_probe'] != GRAPH_ENTRY) or r[5]['ef'] <= 0]
        if bad:
            self.fail('%d of %d runs have n_probe not in (>0, -1) or ef<=0; first: %s'
                      % (len(bad), len(runs), [(b[2], b[4]) for b in bad[:3]]))


if __name__ == '__main__':
    unittest.main()
