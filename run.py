from ann_benchmarks.main import main
from multiprocessing import freeze_support
import multiprocessing

if __name__ == "__main__":
    # Python 3.14 defaults to 'forkserver'.  'spawn' is safer for C extensions that
    # use OpenMP (avoids fork-after-openmp-init crashes) while still propagating
    # stdout/stderr to worker processes via the normal fd inheritance of os.fork().
    multiprocessing.set_start_method('spawn', force=True)
    freeze_support()
    main()
