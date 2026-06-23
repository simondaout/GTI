#!/usr/bin/env python3

import numpy as np
import math


class fault2d:
    """2D antiplane and edge dislocation models."""

    def __init__(self, name, x, y, w, dip, ssini, dsini, L):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.dip = dip
        self.dipr = (dip * math.pi) / 180.
        self.ss = ssini
        self.ds = dsini
        self.L = L

    def edge(self, yp, shift0, r):
        """Edge dislocation displacement."""
        shift = -shift0 * np.ones(yp.shape)
        w1 = self.w
        w2 = self.w + self.L * math.sin(self.dipr)
        zeta1 = (yp + shift) / w1
        zeta2 = (yp + shift - self.L * math.cos(self.dipr)) / w2
        denom1 = 1 + zeta1**2
        denom2 = 1 + zeta2**2

        uv = (1. * math.sin(r) / math.pi) * (
            math.sin(self.dipr) * np.arctan(zeta1)
            + (math.cos(self.dipr) + math.sin(self.dipr) * zeta1) / denom1
            - math.sin(self.dipr) * np.arctan(zeta2)
            - (math.cos(self.dipr) + math.sin(self.dipr) * zeta2) / denom2
        )

        uh = -(1. * math.sin(r) / math.pi) * (
            math.cos(self.dipr) * np.arctan(zeta1)
            + (math.sin(self.dipr) - math.cos(self.dipr) * zeta1) / denom1
            - math.cos(self.dipr) * np.arctan(zeta2)
            - (math.sin(self.dipr) - math.cos(self.dipr) * zeta2) / denom2
        )

        return self.ds * uh, self.ds * uv

    def antiplane(self, yp, shift0, z):
        """Antiplane (screw dislocation) displacement."""
        shift = -shift0 * np.ones(yp.shape)

        uy = (
            (1. / (2 * math.pi)) * (
                np.arctan2(
                    math.sin(self.dipr) * (yp + shift) + math.cos(self.dipr) * (self.w + z),
                    math.sin(self.dipr) * (self.w + z) - math.cos(self.dipr) * (yp + shift)
                )
                + np.arctan2(
                    math.sin(self.dipr) * (yp + shift) - math.cos(self.dipr) * (z - self.w),
                    -math.cos(self.dipr) * (yp + shift) - math.sin(self.dipr) * (z - self.w)
                )
            )
            - (1. / (2 * math.pi)) * (
                np.arctan2(
                    math.sin(self.dipr) * (yp + shift + self.L * math.cos(self.dipr))
                    + math.cos(self.dipr) * (self.w + z + self.L * math.sin(self.dipr)),
                    math.sin(self.dipr) * (self.w + z + self.L * math.sin(self.dipr))
                    - math.cos(self.dipr) * (yp + shift + self.L * math.cos(self.dipr))
                )
                + np.arctan2(
                    math.sin(self.dipr) * (yp + shift + self.L * math.cos(self.dipr))
                    - math.cos(self.dipr) * (z - (self.w + self.L * math.sin(self.dipr))),
                    -math.cos(self.dipr) * (yp + shift + self.L * math.cos(self.dipr))
                    - math.sin(self.dipr) * (z - (self.w + self.L * math.sin(self.dipr)))
                )
            )
        )

        return self.ss * uy


class prof:
    def __init__(self, name, x, y, l, w):
        self.name = name
        self.x = x
        self.y = y
        self.l = l
        self.w = w


class topo:
    def __init__(self, name, wdir, filename, color, width):
        self.name = name
        self.wdir = wdir
        self.filename = filename
        self.color = color
        self.width = width
        self.yp = []
        self.xp = []

    def load(self, xlim, ylim):
        fname = self.wdir + self.filename
        x, y, z = np.loadtxt(fname, comments='#', unpack=True, dtype='f,f,f')
        index = np.nonzero((x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1]))
        self.x = np.delete(x, index)
        self.y = np.delete(y, index)
        self.z = np.delete(z, index)


class seismi:
    def __init__(self, name, wdir, filename, color, width):
        self.name = name
        self.wdir = wdir
        self.filename = filename
        self.color = color
        self.width = width
        self.yp = []
        self.xp = []

    def load(self, xlim, ylim):
        fname = self.wdir + self.filename
        x, y, z, mw = np.loadtxt(fname, comments='#', unpack=True, dtype='f,f,f,f')
        index = np.nonzero((x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1]))
        self.x = np.delete(x, index)
        self.y = np.delete(y, index)
        self.z = np.delete(z, index)
        self.mw = np.delete(mw, index)


class moho:
    def __init__(self, name, wdir, filename, color, width):
        self.name = name
        self.wdir = wdir
        self.filename = filename
        self.color = color
        self.width = width
        self.yp = []
        self.xp = []

    def load(self, xlim, ylim):
        fname = self.wdir + self.filename
        x, y, z = np.loadtxt(fname, comments='#', unpack=True, dtype='f,f,f')
        index = np.nonzero((x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1]))
        self.x = np.delete(x, index)
        self.y = np.delete(y, index)
        self.z = np.delete(z, index)
