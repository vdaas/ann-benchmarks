"""Checks for arcflare._load_ext dir resolution. Each case runs in a fresh interpreter.

  venv/bin/python -I tools_arcflare/test_ext_resolve.py [FP16_DIR SQ4_DIR]
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T02 = ('/home/kpango/go/src/github.com/vdaas/ArcFlare/.claude/worktrees/'
       'mission-arcflare-annb-top-20261103-110228/.claude/worktrees/t02-ctx-reuse-20261103-113855/')
FP16 = sys.argv[1] if len(sys.argv) > 2 else T02 + 'build-py-fp16'
SQ4 = sys.argv[2] if len(sys.argv) > 2 else T02 + 'build-py-sq4'
EMPTY = tempfile.mkdtemp()
NOLEGACY = os.path.join(EMPTY, 'no-such-legacy-dir')

# Body runs in the child: load each (var) in order, print the resulting module file.
CHILD = """
import sys
sys.path.insert(0, %r)
from ann_benchmarks.algorithms.arcflare import _load_ext
for var in %%r:
    m = _load_ext(var, %r)
    print('RESULT', var, m.__file__)
""" % (ROOT, NOLEGACY)


def run(env, vars_):
    clean = {k: v for k, v in os.environ.items() if not k.startswith('ARCFLARE_EXT_DIR')}
    clean.update(env)
    p = subprocess.run([sys.executable, '-I', '-c', CHILD % (vars_,)], env=clean,
                       capture_output=True, text=True, cwd=EMPTY)
    return p.returncode, p.stdout, p.stderr


fails = []


def check(name, ok, detail):
    print(('PASS ' if ok else 'FAIL ') + name)
    if not ok:
        fails.append(name)
        print('   ', detail)


rc, out, err = run({}, [None, 'SQ4'])
check('no env: missing legacy dir falls back to installed module (default + SQ4)',
      rc == 0 and out.count('RESULT') == 2 and 'using importable extension' in out, (rc, out, err[-400:]))

rc, out, err = run({'ARCFLARE_EXT_DIR': EMPTY}, [None])
check('generic env set but dir has no .so: error', rc != 0 and 'no arcflare_ext*.so' in err, (rc, out, err[-400:]))

rc, out, err = run({'ARCFLARE_EXT_DIR': FP16}, ['SQ4'])
check('SQ4 without ARCFLARE_EXT_DIR_SQ4 while generic set: error, no fp16 reuse',
      rc != 0 and 'ARCFLARE_EXT_DIR_SQ4 is unset' in err and 'RESULT' not in out, (rc, out, err[-400:]))

rc, out, err = run({'ARCFLARE_EXT_DIR_SQ4': EMPTY}, [None])
check('default variant with only a variant env set: legacy/installed path, no error',
      rc == 0 and 'RESULT None' in out, (rc, out, err[-400:]))

rc, out, err = run({'ARCFLARE_EXT_DIR_SQ4': SQ4}, ['SQ4'])
check('SQ4 env only: loads SQ4 dir', rc == 0 and ('RESULT SQ4 ' + SQ4) in out, (rc, out, err[-400:]))

rc, out, err = run({'ARCFLARE_EXT_DIR': FP16, 'ARCFLARE_EXT_DIR_SQ4': SQ4}, ['SQ4', None, 'SQ4'])
res = [l for l in out.splitlines() if l.startswith('RESULT')]
check('generic=fp16 + SQ4 env: sq4 -> fp16 -> sq4 in one process each load their own dir',
      rc == 0 and len(res) == 3 and SQ4 in res[0] and FP16 in res[1] and SQ4 in res[2], (rc, out, err[-400:]))

print('%d failed' % len(fails))
sys.exit(1 if fails else 0)
