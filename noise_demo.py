#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import noise
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

if __name__ == "__main__":

    if sys.argv[1] == '2d':

        dim = 2
        const_vol = lambda x, t: 1.
        T = 5
        dt = 0.0001
        x0 = [0, 0]
        reruns = 10

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        for run in range(reruns):
            print('On rerun {} of {}\n'.format(run + 1, reruns))
            g_c = noise.gaussian(dim, const_vol, dt, T, x0)
            e_c = noise.double_exponential(dim, const_vol, dt, T, x0)

            ax.plot(g_c[:, 0], g_c[:, 1], '-', color='orange', alpha=0.5,
                    linewidth=0.5)
            ax.plot(e_c[:, 0], e_c[:, 1], '-', color='blue', alpha=0.5,
                    linewidth=0.5)

            del g_c
            del e_c

        ax.set_xlabel('$x$', fontsize=25)
        ax.set_ylabel('$y$', fontsize=25)

        brownian = mpatches.Patch(color='orange', label='Brownian motion')
        expon = mpatches.Patch(color='blue', label='Exponential motion')
        plt.legend(handles=[brownian, expon])

        #plt.savefig('../brownian_vs_exponential.pdf')
        #plt.savefig('../brownian_vs_exponential.png')
        plt.show()

    elif sys.argv[1] == '3d':

        dim = 3
        const_vol = lambda x, t: 1.
        T = 5
        dt = 0.0001
        x0 = [0, 0, 0]
        reruns = 10

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for run in range(reruns):
            print('On rerun {} of {}\n'.format(run + 1, reruns))
            g_c = noise.gaussian(dim, const_vol, dt, T, x0)
            e_c = noise.double_exponential(dim, const_vol, dt, T, x0)

            ax.plot(g_c[:, 0], g_c[:, 1], g_c[:, 2], '-', color='orange',
                    alpha=0.5, linewidth=0.5)
            ax.plot(e_c[:, 0], e_c[:, 1], e_c[:, 2], '-', color='blue',
                    alpha=0.5, linewidth=0.5)

            del g_c
            del e_c

        brownian = mpatches.Patch(color='orange', label='Brownian motion')
        expon = mpatches.Patch(color='blue', label='Exponential motion')
        plt.legend(handles=[brownian, expon])

        plt.show()
