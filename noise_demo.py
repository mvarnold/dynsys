#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import noise
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if __name__ == "__main__":

    dim = 2
    const_vol = lambda x, t: 1.
    T = 10
    dt = 0.0001
    x0 = [0, 0]
    reruns = 100

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    for run in range(reruns):
        print('On rerun {} of {}\n'.format(run + 1, reruns))
        g_c = noise.gaussian(dim, const_vol, dt, T, x0)
        e_c = noise.double_exponential(dim, const_vol, dt, T, x0)

        ax.plot(g_c[:, 0], g_c[:, 1], 'k-', alpha=0.6)
        ax.plot(e_c[:, 0], e_c[:, 1], 'b-', alpha=0.6)

        del g_c
        del e_c

    ax.set_xlabel('$x$', fontsize=25)
    ax.set_ylabel('$y$', fontsize=25)

    brownian = mpatches.Patch(color='black', label='Brownian motion')
    expon = mpatches.Patch(color='blue', label='Exponential motion')
    plt.legend(handles=[brownian, expon])

    plt.savefig('../brownian_vs_exponential.pdf')
    plt.savefig('../brownian_vs_exponential.png')
