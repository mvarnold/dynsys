#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import numpy as np


def heston_model(mu, t0, T, dt, x0, reruns=100,
        alpha=1., sigmabar=1., corr=0.5, vol_vol=0.25):

    time = np.linspace(t0, T, (T - t0 ) / dt + 1)
    paths = []

    for run in range(reruns):
        x =  np.empty( (len(time), len(x0)) )
        x[0] = x0

        for n in range(len(time) - 1):
            t = time[n]
            dW_x = np.random.normal(loc=0, scale=np.sqrt(dt), size=1)
            dZ = np.random.normal(loc=0, scale=np.sqrt(dt), size=1)
            dW_s = np.sqrt(1. - corr**2) * dZ + corr * dW_x

            # volatility equation
            x[n + 1, 1] = x[n, 1] + alpha * (sigmabar - x[n, 1]) * dt + \
                    vol_vol * np.sqrt(x[n, 1]) * dW_s

            # asset equation
            x[n + 1, 0] = x[n, 0] + mu(x[n, 0], t) * dt + np.sqrt( x[n, 1] ) * \
                    x[n, 0] * dW_x

        paths.append(x)

    return np.asarray(paths), time

