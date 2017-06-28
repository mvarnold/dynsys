#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt


def quickplot(paths, time, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for path in paths:
        ax.plot(time, path)
    plt.show()


def plot_paths(paths, time, ax, labels=('$t$', '$x(t)$')):

    for path in paths:
        ax.plot(time, path)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    return ax
