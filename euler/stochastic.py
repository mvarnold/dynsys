#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def _euler_ito_sde(mu, sigma, t0, T, dt, x0, reruns=100):
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


def _euler_ito_sde_poisson(mu, lambda_, t0, T, dt, x0, dist, dist_loc,
        dist_scale, dist_mult, reruns=100):
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

            x[n+1] = x[n] + mu(x[n], t) * dt + pois_sum
        paths.append(x)

    return np.asarray(paths), time


def _euler_ito_sde_gaussian_poisson(mu, sigma, lambda_, t0, T, dt, x0, dist,
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


class SDE(object):

    def __init__(self, **kwargs):
        self.mu = kwargs.get('mu', None)
        self.sigma = kwargs.get('sigma', None)
        self.lambda_ = kwargs.get('lambda_', None)
        self.t0 = kwargs.get('t0', None)
        self.T = kwargs.get('T', None)
        self.dt = kwargs.get('dt', None)
        self.x0 = kwargs.get('x0', None)
        self.dist = kwargs.get('dist', None)
        self.dist_loc = kwargs.get('dist_loc', None)
        self.dist_scale = kwargs.get('dist_scale', None)
        self.dist_mult = kwargs.get('dist_scale', None)

        self.sample_paths = None
        self.time = None


class GaussianSDE(SDE):

    def run(self, reruns=100, result=False):
        paths, time = _euler_ito_sde(self.mu, self.sigma, self.t0,
                self.T, self.dt, self.x0, reruns)
        self.sample_paths = paths
        self.time = time

        if result:
            return paths, time

    def distribution(self, timearr=None):
        if (self.sample_paths is not None) and (self.time is not None):
            if timearr is None:
                return self.sample_paths[:, -1]
            return self.sample_paths[:, timearr]
        else:
            raise TypeError('One or both of time and sample paths is None. ' \
                    'You probably have not run a simulation yet.')
            return None

    def dist_hist(self, ax, timearr=None, **kwargs):
        path = self.distribution(timearr=timearr)
        n, bins, patches = ax.hist(path, **kwargs)
        return n, bins, patches


class PoissonSDE(SDE):

    def run(self, reruns=100, result=False):
        paths, time = _euler_ito_sde_poisson(self.mu, self.lambda_, self.t0,
                self.T, self.dt, self.x0, self.dist, self.dist_loc,
                self.dist_scale, self.dist_mult, reruns)
        self.sample_paths = paths
        self.time = time

        if result:
            return paths, time

    def distribution(self, timearr=None):
        if (self.sample_paths is not None) and (self.time is not None):
            if timearr is None:
                return self.sample_paths[:, -1]
            return self.sample_paths[:, timearr]
        else:
            raise TypeError('One or both of time and sample paths is None. ' \
                    'You probably have not run a simulation yet.')
            return None

    def dist_hist(self, ax, timearr=None, **kwargs):
        path = self.distribution(timearr=timearr)
        n, bins, patches = ax.hist(path, **kwargs)
        return n, bins, patches


class GaussianPoissonSDE(SDE):

    def run(self, reruns=100, result=False):
        paths, time = _euler_ito_sde_gaussian_poisson(self.mu, self.sigma, self.lambda_,
                self.t0, self.T, self.dt, self.x0, self.dist, self.dist_loc,
                self.dist_scale, self.dist_mult, reruns)
        self.sample_paths = paths
        self.time = time

        if result:
            return paths, time

    def distribution(self, timearr=None):
        if (self.sample_paths is not None) and (self.time is not None):
            if timearr is None:
                return self.sample_paths[:, -1]
            return self.sample_paths[:, timearr]
        else:
            raise TypeError('One or both of time and sample paths is None. ' \
                    'You probably have not run a simulation yet.')
            return None

    def dist_hist(self, ax, timearr=None, **kwargs):
        path = self.distribution(timearr=timearr)
        n, bins, patches = ax.hist(path, **kwargs)
        return n, bins, patches
