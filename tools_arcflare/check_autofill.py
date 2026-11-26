"""Report/dump ArcFlare algos.yaml query-arg rows whose rpqgm/eps come from auto-fill.

Rows are resolved with ann_benchmarks.algorithms.arcflare.parse_query_args (the same
code set_query_arguments uses).

  check_autofill.py [--yaml algos.yaml] report   # rows relying on auto-fill (exit 1 if any)
  check_autofill.py [--yaml algos.yaml] dump     # JSON {row-key: effective params}

A run-group whose query-args holds several lists is a cross product in
definitions._generate_combinations: set_query_arguments would receive 2 rows and fail.
Such groups are still checked row by row (key suffix "d<j>") so the guard sees them.
"""
import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ann_benchmarks.algorithms.arcflare import parse_query_args  # noqa: E402
from ann_benchmarks.algorithms.definitions import _generate_combinations  # noqa: E402

ARCFLARE_MODULE = 'ann_benchmarks.algorithms.arcflare'


def _get(mapping_node, key):
    return next((v for k, v in mapping_node.value if k.value == key), None)


def _rows(rg, rg_node):
    """Yield (label, args-tuple for set_query_arguments, 1-based line|None)."""
    if 'query-args' in rg:
        qa = rg['query-args']
        qa_node = _get(rg_node, 'query-args')
        for j, el in enumerate(qa):
            dim = el if isinstance(el, list) else [el]
            nodes = qa_node.value[j].value if isinstance(el, list) else [qa_node.value[j]]
            for i, row in enumerate(dim):
                yield ('[%d]' % i if len(qa) == 1 else 'd%d[%d]' % (j, i)), (row,), nodes[i].start_mark.line + 1
    elif 'query-arg-groups' in rg:
        groups = [_generate_combinations(g) if isinstance(g, dict) else g for g in rg['query-arg-groups']]
        for i, combo in enumerate(_generate_combinations(groups)):
            yield '[%d]' % i, tuple(combo), None


def raw_rows(path):
    """Yield (key, line|None, args-tuple for set_query_arguments) for every ArcFlare row."""
    with open(path) as f:
        text = f.read()
    data = yaml.load(text, yaml.SafeLoader)
    root = yaml.compose(text, yaml.SafeLoader)
    for (pk, pn) in root.value:
        for (mk, mn) in pn.value:
            for (ak, an) in mn.value:
                algo = data[pk.value][mk.value][ak.value]
                if algo.get('module') != ARCFLARE_MODULE or algo.get('constructor') != 'ArcFlare':
                    continue
                for gk, gn in _get(an, 'run-groups').value:
                    for label, args, line in _rows(algo['run-groups'][gk.value], gn):
                        yield '%s/%s/%s/%s%s' % (pk.value, mk.value, ak.value, gk.value, label), line, args


def resolve(path):
    """Return list of (key, line|None, eff, autofilled) for every ArcFlare row."""
    res = []
    for key, line, args in raw_rows(path):
        eff, auto = parse_query_args(args)
        res.append((key, line, eff, auto))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yaml', default=os.path.join(ROOT, 'algos.yaml'))
    ap.add_argument('cmd', choices=['report', 'dump'])
    a = ap.parse_args()
    rows = resolve(a.yaml)
    if a.cmd == 'dump':
        json.dump({k: eff for k, _, eff, _ in rows}, sys.stdout, indent=1, sort_keys=True)
        print()
        return 0
    bad = [(k, ln, auto) for k, ln, _, auto in rows if auto]
    for k, ln, auto in bad:
        print('line=%s auto=%s %s' % (ln, ','.join(auto), k))
    print('%d of %d rows rely on auto-fill' % (len(bad), len(rows)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
