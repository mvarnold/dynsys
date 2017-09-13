#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
from euler.stochastic import *

if __name__ == "__main__":

    mu = lambda x, t: x
    sigma = lambda x, t: 0.5 * x
    lambda_ = lambda x, t: 10.
    t0 = 0.
    T = 2.
    dt = 0.001
    x0 = 1.
    dist = 'laplace'
    dist_loc = lambda x, t: 0.
    dist_scale = lambda x, t: 0.5
    dist_mult = lambda x, t: 0.1 * x

    g_sys = GaussianSDE(mu=mu,
            sigma=sigma,
            t0=t0,
            T=T,
            dt=dt,
            x0=x0)

    p_sys = PoissonSDE(mu=mu,
            lambda_=lambda_,
            t0=t0,
            T=T,
            dt=dt,
            x0=x0,
            dist=dist,
            dist_loc=dist_loc,
            dist_scale=dist_scale,
            dist_mult=dist_mult)

    gp_sys = GaussianPoissonSDE(mu=mu,
            sigma=sigma,
            lambda_=lambda_,
            t0=t0,
            T=T,
            dt=dt,
            x0=x0,
            dist=dist,
            dist_loc=dist_loc,
            dist_scale=dist_scale,
            dist_mult=dist_mult)

    # generate sample paths
    reruns=50
    g_paths, time = g_sys.run(reruns=reruns, result=True)
    p_paths, _ = p_sys.run(reruns=reruns, result=True)
    gp_paths, _ = gp_sys.run(reruns=reruns, result=True)

    # plot things
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    for r in range(len(g_paths)):
        axes[0].plot(time, g_paths[r])
        axes[1].plot(time, p_paths[r])
        axes[2].plot(time, gp_paths[r])

    axes[0].set_title('GBM')
    axes[1].set_title('Poisson with lognormal drift, laplace RV')
    axes[2].set_title('GBM with Poisson noise')

    _ = [axes[i].set_xlabel('$t$') for i in range(2 + 1)]

    plt.show()
