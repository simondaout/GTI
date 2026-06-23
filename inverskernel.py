#!/usr/bin/env python3

"""
Time-series basis functions (kernels) for geodetic inversion.

@author: simon daout
"""

import numpy as np
import math
import sys
from matplotlib import pyplot as plt
from flatten import flatten


# =============================================================================
# Helper step functions
# =============================================================================

def Heaviside(t):
    h = np.zeros(len(t))
    h[t >= 0] = 1.0
    return h


def Box(t):
    return Heaviside(t + 0.5) - Heaviside(t - 0.5)


# =============================================================================
# Base class
# =============================================================================

class pattern:
    def __init__(self, name, date, inversion, sigmam):
        self.name = name
        self.date = date
        self.inversion = inversion
        self.sigmam = sigmam

    def gp(self, t):
        """Numerical derivative of the basis function."""
        dt = 0.001
        return (self.g(t) - self.g(t + dt)) / dt


# =============================================================================
# Kernel container
# =============================================================================

class kernel:
    def __init__(self, kernels):
        self.kernels = flatten(kernels)

    def info(self):
        print()
        print('Kernels:')
        print('#index  #inversion  #name  #date')
        for i, k in enumerate(self.kernels):
            print('{:d}  {:s}  {:s}  {:4.1f}'.format(i, k.inversion, k.name, k.date))


# =============================================================================
# Kernel classes
# =============================================================================

class coseismic(pattern):
    """Heaviside step at earthquake time."""

    def __init__(self, name, date, inversion, sigmam=1.):
        super().__init__(name, date, inversion, sigmam)
        self.to = date

    def g(self, t):
        return Heaviside(t - self.to) / self.sigmam


def generatepost(Mpp, to, tp, inversion, sigmam=1.):
    """Generate a list of postseismic basis functions (piecewise polynomial)."""
    T = 2 * tp / Mpp
    tl = to + (np.arange(Mpp) + 1) * T / 2

    postseismics = [transienti('initial transient', to, inversion, T, sigmam)]
    for j in range(len(tl) - 1):
        postseismics.append(postseismic('postseismic', tl[j], inversion, T, sigmam))
    postseismics.append(transientf('final transient', to + tp, inversion, T, sigmam))

    return postseismics


class postseismic(pattern):
    """Piecewise polynomial postseismic transient (Box-car in time)."""

    def __init__(self, name, date, inversion, T, sigmam=1.):
        super().__init__(name, date, inversion, sigmam)
        self.to = date
        self.T = T

    def g(self, tp):
        t = (tp - self.to) / self.T
        return (2 * (t - np.sign(t) * t**2 + 0.25) * Box(t) + Heaviside(t - 0.5)) / self.sigmam


class transienti(pattern):
    """Initial transient ramp-up."""

    def __init__(self, name, date, inversion, T, sigmam=1.):
        super().__init__(name, date, inversion, sigmam)
        self.to = date
        self.T = T

    def g(self, tp):
        t = (tp - self.to) / self.T
        return (4 * (t - t**2) * Box(2 * t - 0.5) + Heaviside(t - 0.5)) / self.sigmam


class transientf(pattern):
    """Final transient ramp-down."""

    def __init__(self, name, date, inversion, T, sigmam=1.):
        super().__init__(name, date, inversion, sigmam)
        self.to = date
        self.T = T

    def g(self, tp):
        t = (tp - self.to) / self.T
        return (4 * (t + t**2) + 1) * Box(2 * t + 0.5) / self.sigmam + Heaviside(t) / self.sigmam


class interseismic(pattern):
    """Linear ramp (interseismic loading)."""

    def __init__(self, name, date, inversion, sigmam=1.):
        super().__init__(name, date, inversion, sigmam)
        self.to = date

    def g(self, t):
        func = (t - self.to) * Heaviside(t - self.to)
        return func / self.sigmam


class reference(pattern):
    """Constant offset (reference frame)."""

    def __init__(self, name, date, inversion, sigmam=1.):
        super().__init__(name, date, inversion, sigmam)

    def g(self, t):
        return np.ones(len(t)) / self.sigmam


class seasonalvar(pattern):
    """Annual cosine term (seasonal variation)."""

    def __init__(self, name, date, inversion, sigmam=1.):
        super().__init__(name, date, inversion, sigmam)

    def g(self, t):
        return np.cos(2 * np.pi * t) / self.sigmam


class annualvar(pattern):
    """Semi-annual cosine term."""

    def __init__(self, name, date, inversion, sigmam=1.):
        super().__init__(name, date, inversion, sigmam)

    def g(self, t):
        return np.cos(4 * np.pi * t) / self.sigmam


class laterror(pattern):
    """Latitude-dependent orbital error."""

    def __init__(self, name):
        self.name = name

    def g(self, x):
        return x


class lonerror(pattern):
    """Longitude-dependent orbital error."""

    def __init__(self, name):
        self.name = name

    def g(self, y):
        return y
