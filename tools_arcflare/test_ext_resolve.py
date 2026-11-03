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

# Body runs in the child: load each (var) in order, print the resulting module file
# (an exception stops the child; stdout keeps the results printed before it).
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

rc, out, err = run({'ARCFLARE_EXT_DIR_SQ4': SQ4}, ['SQ4', 'SQ4'])
check('same file requested twice: reuse OK',
      rc == 0 and out.count('RESULT SQ4 ' + SQ4) == 2, (rc, out, err[-400:]))

rc, out, err = run({'ARCFLARE_EXT_DIR': FP16, 'ARCFLARE_EXT_DIR_SQ4': SQ4}, ['SQ4', None])
check('different file after first load (sq4 then fp16): error',
      rc != 0 and out.count('RESULT') == 1 and 'already loaded in this process' in err, (rc, out, err[-400:]))

rc, out, err = run({'ARCFLARE_EXT_DIR_SQ4': SQ4}, ['SQ4', None])
check('checker scenario: SQ4 env only, then default with empty legacy dir: error, no silent reuse',
      rc != 0 and out.count('RESULT') == 1 and 'already loaded in this process' in err, (rc, out, err[-400:]))

# Checker repro: something put the SQ4 .so into sys.modules before our first _load_ext.
PRELOAD = """
import importlib.util, sys
sys.path.insert(0, %r)
spec = importlib.util.spec_from_file_location('arcflare_ext', __import__('glob').glob(%r + '/arcflare_ext*.so')[0])
mod = importlib.util.module_from_spec(spec); sys.modules['arcflare_ext'] = mod; spec.loader.exec_module(mod)
from ann_benchmarks.algorithms.arcflare import _load_ext
print('RESULT', _load_ext(None, %r).__file__)
""" % (ROOT, SQ4, NOLEGACY)
clean = {k: v for k, v in os.environ.items() if not k.startswith('ARCFLARE_EXT_DIR')}
p = subprocess.run([sys.executable, '-I', '-c', PRELOAD], env=clean, capture_output=True, text=True, cwd=EMPTY)
check('sq4 .so pre-loaded into sys.modules, no env, default with empty legacy dir: error',
      p.returncode != 0 and 'RESULT' not in p.stdout and 'already loaded in this process' in p.stderr,
      (p.returncode, p.stdout, p.stderr[-400:]))

print('%d failed' % len(fails))
sys.exit(1 if fails else 0)
