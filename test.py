from collections import Counter
from bisect import *

a = [i for i in range(10)]
def trig_split(idx1, idx2, values):
    return values[idx1:idx2 + 1], values[idx2:] + values[:idx1 + 1]

print(trig_split(7,9, a))