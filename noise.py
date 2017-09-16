#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from scipy import stats

def gaussian(dim, vol, dt, T, x0):
    time = np.linspace(0, T, T / dt + 1)

    if type(x0) is float:
        x = [x0]
        for n in range(len(time) - 1):
            t = time[n]
            x[n+1] = x[n] + vol(x[n], t) * np.random.normal(loc=0, scale=np.sqrt(dt),
                size=1)
    else:
        x =  np.empty( (len(time), len(x0)) )
        x[0] = x0
        for n in range(len(time) - 1):
            t = time[n]
            x[n+1] = x[n] + vol(x[n], t) * np.random.normal(loc=0,
                    scale=np.sqrt(dt), size=len(x0))
    return x


def cauchy(dim, vol, dt, T, x0):
    time = np.linspace(0, T, T / dt + 1)

    if type(x0) is float:
        x = [x0]
        for n in range(len(time) - 1):
            t = time[n]
            x[n+1] = x[n] + vol(x[n], t) * stats.cauchy.rvs(loc=[0], scale=np.sqrt(dt),
                size=1)
    else:
        x =  np.empty( (len(time), len(x0)) )
        x[0] = x0
        for n in range(len(time) - 1):
            t = time[n]
            x[n+1] = x[n] + vol(x[n], t) * stats.cauchy.rvs(loc=np.zeros(len(x0)),
                    scale=[np.sqrt(dt) for _ in range(len(x0))], size=len(x0))
    return x


def double_exponential(dim, vol, dt, T, x0):
    time = np.linspace(0, T, T / dt + 1)

    if type(x0) is float:
        x = [x0]
        for n in range(len(time) - 1):
            lr = np.random.choice([-1., 1.], size=1)
            t = time[n]
            x[n+1] = x[n] + vol(x[n], t) * lr * stats.expon.rvs(loc=0, scale=np.sqrt(dt),
                size=1)
    else:
        x =  np.empty( (len(time), len(x0)) )
        x[0] = x0
        for n in range(len(time) - 1):
            lr = np.random.choice([-1., 1.], size=len(x0))
            t = time[n]
            x[n+1] = x[n] + vol(x[n], t) * lr * stats.expon.rvs(loc=np.zeros(len(x0)),
                    scale=[np.sqrt(dt) for _ in range(len(x0))], size=len(x0))

    return x


class Noise(object):

    def __init__(self, dim, kind=None):
        self.dim = dim
