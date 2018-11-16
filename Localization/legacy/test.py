
import sys
import math
import copy
import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def low_variance_sampler_test():
    probability = np.array([0.1, 0.3, 0.15, 0.3, 0.15])
    print(probability)
    print(np.sum(probability))

    prob_cumsum = np.zeros((1, len(probability) + 1))
    prob_cumsum[0, 1:] = np.cumsum(probability)
    print(prob_cumsum)

    loop_num = 1000
    index_count = np.zeros([1, len(probability)])
    for i in range(loop_num):
        r = np.random.uniform(0.0, 1.0, len(probability))
        idx = np.digitize(r, prob_cumsum[0])
        idx -= 1
        for item in idx:
            index_count[0, item] += 1.0
    index_count /= np.sum(index_count)
    print(index_count)

def main():
    low_variance_sampler_test()

if __name__ == '__main__':
    main()