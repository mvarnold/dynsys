#!/usr/bin/env python
# -*- coding: utf-8 -*-

import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import sys

if __name__ == "__main__":

    # walk around
    # the problem is fundamentally two-dimensional since on the surface of T^2
    dim = 2
    const_vol = lambda x, t: 1.
    T = 5
    dt = 0.0001
    x0 = [0, 0]
    reruns = 3

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111, projection='3d')

    # recall one way to construct the torus is just
    # T^2 = R^2 / Z^2, so use periodic coordinates:
    # (x, y) = (x + 1, y + 1)

    R = 5.
    r = 4.
    npts = 500
    theta = np.linspace(0., 2. * np.pi, npts)
    phi = np.linspace(0., 2. * np.pi, npts)

    # actually make a grid
    theta, phi = np.meshgrid(theta, phi)

    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    # plot the torus
    ax.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='green',
            alpha=0.25)

    for run in range(reruns):
        print('On rerun {} of {}\n'.format(run + 1, reruns))

        # treat these as theta and phi
        g_c = noise.gaussian(dim, const_vol, dt, T, x0)
        e_c = noise.double_exponential(dim, const_vol, dt, T, x0)

        g_x = (R + r * np.cos(g_c[:, 0])) * np.cos(g_c[:, 1])
        g_y = (R + r * np.cos(g_c[:, 0])) * np.sin(g_c[:, 1])
        g_z = r * np.sin(g_c[:, 0])

        e_x = (R + r * np.cos(e_c[:, 0])) * np.cos(e_c[:, 1])
        e_y = (R + r * np.cos(e_c[:, 0])) * np.sin(e_c[:, 1])
        e_z = r * np.sin(e_c[:, 0])

        ax.plot(g_x, g_y, g_z, '-', color='orange', alpha=0.5, linewidth=0.25)
        ax.plot(e_x, e_y, e_z, '-', color='blue', alpha=0.5, linewidth=0.25)

    ax.set_xlabel('$x$', fontsize=25)
    ax.set_ylabel('$y$', fontsize=25)
    ax.set_zlabel('$z$', fontsize=25)

    brownian = mpatches.Patch(color='orange', label='Brownian motion')
    expon = mpatches.Patch(color='blue', label='Exponential motion')
    plt.legend(handles=[brownian, expon])

    plt.show()


