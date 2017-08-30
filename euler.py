#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from scipy import stats


def euler_forward(f, t0, T, dt, x0, fargs=None, get_time=True):
    time = np.linspace(t0, T, (T - t0 ) / dt + 1)
    if type(x0) is float:
        x = np.empty(len(time))
    else:
        x = np.empty( (len(time), len(x0)) )
    x[0] = x0

    for n in range(len(time)-1):
        t = time[n]

        if fargs is None:
            x[n+1] = x[n] + f(x[n], t) * dt

        elif fargs is not None:
            x[n+1] = x[n] + f(x[n], t, **fargs) * dt

    if get_time is False:
        return x
    return x, time


def euler_ito_sde(mu, sigma, t0, T, dt, x0, reruns=100):
    time = np.linspace(t0, T, (T - t0 ) / dt + 1)
    paths = []
    dims = 'one'

    if type(x0) is not float:
        dims = 'many'

    for run in range(reruns):
        if dims == 'one':
            x = np.empty(len(time))
        elif dims == 'many':
            x =  np.empty( (len(time), len(x0)) )

        x[0] = x0

        if dims == 'one':
            for n in range(len(time)-1):
                t = time[n]
                x[n+1] = x[n] + mu(x[n], t) * dt + sigma(x[n], t) * np.random.normal(loc=0,
                    scale=np.sqrt(dt), size=1)
            paths.append(x)

        elif dims == 'many':
            for n in range(len(time)-1):
                t = time[n]
                x[n+1] = x[n] + mu(x[n], t) * dt + sigma(x[n], t) * np.random.normal(loc=0,
                    scale=np.sqrt(dt), size=len(x0))
            paths.append(x)


    return np.asarray(paths), time


def euler_forward_randomICs(f, t0, T, dt, distname, distparams):
    ics = getattr(stats, distname).rvs(**distparams)
    paths = []

    for ic in ics:
        x = euler_forward(f, t0, T, dt, ic, get_time=False)
        paths.append(x)

    return paths, np.linspace(t0, T, (T - t0 ) / dt + 1)


def euler_ito_sde_poisson(mu, lambda_, t0, T, dt, x0, dist, distparams,
        reruns=100):
    time = np.linspace(t0, T, (T - t0 ) / dt + 1)
    paths = []

    if type(x0) is not float:
        raise TypeError('Poisson is implemented in only one variable (for now)')

    for run in range(reruns):
        x = np.empty(len(time))

        x[0] = x0

        for n in range(len(time)-1):
            t = time[n]

            # calculate number of jumps
            N = stats.poisson.rvs(mu=lambda_(x[n], t) * dt, loc=0, size=1)[0]
            rv_dist = getattr(stats, dist)
            rvs = rv_dist.rvs(size=N, **distparams)
            pois_sum = sum(rvs)

            x[n+1] = x[n] + mu(x[n], t) * dt + pois_sum
        paths.append(x)

    return np.asarray(paths), time


def euler_ito_sde_gaussian_poisson(mu, sigma, lambda_, t0, T, dt, x0, dist,
        dist_loc, dist_scale, dist_mult, reruns=100):
    time = np.linspace(t0, T, (T - t0 ) / dt + 1)
    paths = []

    if type(x0) is not float:
        raise TypeError('Poisson is implemented in only one variable (for now)')

    for run in range(reruns):
        x = np.empty(len(time))

        x[0] = x0

        for n in range(len(time)-1):
            t = time[n]

            # calculate number of jumps
            N = stats.poisson.rvs(mu=lambda_(x[n], t) * dt, loc=0, size=1)[0]
            rv_dist = getattr(stats, dist)
            rvs = rv_dist.rvs(size=N, loc=dist_loc(x[n], t),
                    scale=dist_scale(x[n], t))
            pois_sum = dist_mult(x[n], t) * sum(rvs)

            x[n+1] = x[n] + mu(x[n], t) * dt + sigma(x[n], t) * \
            np.random.normal(loc=0, scale=np.sqrt(dt), size=1) + pois_sum
        paths.append(x)

    return np.asarray(paths), time
