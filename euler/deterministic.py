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


def euler_forward_randomICs(f, t0, T, dt, distname, distparams):
    ics = getattr(stats, distname).rvs(**distparams)
    paths = []

    for ic in ics:
        x = euler_forward(f, t0, T, dt, ic, get_time=False)
        paths.append(x)

    return paths, np.linspace(t0, T, (T - t0 ) / dt + 1)
