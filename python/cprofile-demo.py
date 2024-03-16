import cProfile, pstats, io
from pstats import SortKey
import time


def sleep_2(seconds):
    time.sleep(seconds)


def sleep(seconds):
    return sleep_2(seconds)


def main():
    pr = cProfile.Profile()
    pr.enable()
    for i in range(10):
        sleep(0.1)
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    ps.print_callees()
    print(s.getvalue())


if __name__ == "__main__":
    main()
