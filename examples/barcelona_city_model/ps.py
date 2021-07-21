
import pstats
import sys

from pstats import SortKey

p = pstats.Stats(sys.argv[1])
p.sort_stats(SortKey.CUMULATIVE).print_stats(30)