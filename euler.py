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


def heston_stochastic_vol(mu, sigma, t0, T, dt, x0, reruns=100,
        alpha=1., sigmabar=1., corr=0., vol_vol=0.):

    time = np.linspace(t0, T, (T - t0 ) / dt + 1)
    paths = []
    x =  np.empty( (len(time), len(x0)) )

    for run in range(reruns):
        x[0] = x0


