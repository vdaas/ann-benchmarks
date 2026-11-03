"""Guard: no ArcFlare row in algos.yaml may leave rpqgm/eps to auto-fill.

Run: python tools_arcflare/test_no_autofill.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_autofill  # noqa: E402
from ann_benchmarks.algorithms.arcflare import parse_query_args  # noqa: E402


class ParseQueryArgs(unittest.TestCase):
    def test_short_rows_autofill(self):
        eff, auto = parse_query_args(([16, 200],))
        self.assertEqual(auto, ['rpqgm', 'eps'])
        self.assertEqual((eff['beam_rpq_gate_margin'], eff['term_eps_fp16']), (0.2, -1.0))
        eff, auto = parse_query_args(([16, 64],))
        self.assertEqual((eff['beam_rpq_gate_margin'], eff['term_eps_fp16']), (0.3, 0.3))

    def test_eps_only_autofill(self):
        _, auto = parse_query_args(([16, 200, 128, 24, 0, 0.2],))
        self.assertEqual(auto, ['eps'])

    def test_explicit_row(self):
        eff, auto = parse_query_args(([16, 200, 128, 24, 0, 0.2, -1],))
        self.assertEqual(auto, [])
        self.assertEqual((eff['beam_rpq_gate_margin'], eff['term_eps_fp16']), (0.2, -1.0))

    def test_legacy_row_autofills(self):
        _, auto = parse_query_args(([0.0, -1.0, 16, 0, 0, 200],))
        self.assertEqual(auto, ['rpqgm', 'eps'])


class AlgosYaml(unittest.TestCase):
    def test_no_row_relies_on_autofill(self):
        rows = check_autofill.resolve(os.path.join(check_autofill.ROOT, 'algos.yaml'))
        self.assertTrue(rows)
        bad = ['line=%s auto=%s %s' % (ln, ','.join(a), k) for k, ln, _, a in rows if a]
        self.assertFalse(bad, '%d of %d rows rely on auto-fill; first: %s' % (len(bad), len(rows), bad[:3]))


if __name__ == '__main__':
    unittest.main()
