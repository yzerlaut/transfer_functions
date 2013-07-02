#!/usr/bin/env python
#coding=utf-8

import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
from matplotlib import mlab
from mpl_toolkits.mplot3d import Axes3D

import glob
import cPickle
import os

def collect_data(path, pattern="*.pickle"):

    pge_mean = []
    pgi_mean = []
    firing_rate = []
    for fname in glob.glob(os.path.join(path, pattern)):
        with file(fname) as fid:
            data = cPickle.load(fid)
        stats = data['stats']
        firing_rate.append(stats['firing_rate'])
        pge_mean.append(stats['ge_mean'])
        pgi_mean.append(stats['gi_mean'])

    pge_mean = np.array(pge_mean)
    pgi_mean = np.array(pgi_mean)
    firing_rate = np.array(firing_rate)

    return pge_mean, pgi_mean, firing_rate

def construct_grid(x, y, z, x_range=None, y_range=None, n=40):
    
    if x_range is None:
        x_range = x.min(), x.max()
    if y_range is None:
        y_range = y.min(), y.max()

    xmax, xmin = x_range
    ymax, ymin = y_range

    xi = np.linspace(xmin, xmax, n)
    yi = np.linspace(ymin, ymax, n)

    #zi = mlab.griddata(x, y, z, xi, yi)
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata(x, y, z, xx, yy)
    return xx, yy, zz

def griddata(x, y, z, xx, yy):
    points = np.vstack((x, y)).T
    zz = interpolate.griddata(points, z, (xx, yy), 'linear')

    return zz


def plot_transfer_function(x, y, tf):

    fig = plt.gcf()
    x_range = np.percentile(x, [1, 99])
    y_range = np.percentile(y, [1, 99])
    xi, yi, zi = construct_grid(x, y, tf, x_range, y_range)
    ax1 = fig.add_subplot(221, projection='3d', axisbg='none')
    ax1.scatter(x, y, tf, c=tf)
    ax2 = fig.add_subplot(224)
    ax2.contourf(xi, yi, zi)

    ax3 = fig.add_subplot(222)
    ax3.scatter(x, y, c=tf)
    ax3.set_xlabel('Pge')
    ax3.set_ylabel('Pgi')
    ax3.set_xlim(x_range)
    ax3.set_ylim(y_range)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--pattern', '-p', default='*.pickle')

    args = parser.parse_args()
    pge, pgi, firing_rate = collect_data(args.path, args.pattern)

    plt.figure()
    plot_transfer_function(pge, pgi, firing_rate)
    plt.show()

