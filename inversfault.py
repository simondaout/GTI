#!/usr/bin/env python3

"""
Fault patch model and Okada elastic dislocation Green's function builder.

@author: simon daout
"""

import numpy as np
import math
import sys
import os
from numpy.lib.stride_tricks import as_strided
from flatten import flatten

# okada4py is a sibling package in the GTI directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import okada4py as ok4

# Default elastic parameters
_MU = 30.0e9   # shear modulus (Pa)
_NU = 0.25     # Poisson's ratio


def _okada_displacement(x_obs, y_obs, dip_rad, fault_E, fault_N, fault_depth,
                        L, W, strike_rad, slip_type, mu=_MU, nu=_NU):
    """Compute Okada (1992) elastic displacement for a single observation point
    and a single fault patch using okada4py.

    okada4py / disloc3d expects:
      - depth  : depth of the CENTRE of the patch (positive downward)
      - xc, yc : horizontal position of the CENTRE of the patch
      - al1=al2=L/2, aw1=aw2=W/2 (symmetric around centre)

    Parameters
    ----------
    x_obs, y_obs : float
        Observer position (Easting, Northing).
    dip_rad : float
        Fault dip angle in radians.
    fault_E, fault_N : float
        TOP-LEFT corner of the patch (Easting, Northing).
    fault_depth : float
        Depth of the top edge of the patch (negative in z-up convention).
    L, W : float
        Along-strike length and down-dip width.
    strike_rad : float
        Fault strike in radians.
    slip_type : int
        1 = unit strike-slip, 2 = unit dip-slip.
    mu, nu : float
        Elastic shear modulus and Poisson's ratio.

    Returns
    -------
    u : ndarray, shape (3,)
        Displacement [East, North, Down] at the observation point.
    """
    # ── Convert top-left corner → patch centre ───────────────────────────────
    # Unit along-strike vector (North, East): Sv = [cos(s), sin(s)]
    # Unit down-dip vector (North, East):     Dv = [-cos(d)*sin(s), cos(d)*cos(s)]
    cs, ss = math.cos(strike_rad), math.sin(strike_rad)
    cd     = math.cos(dip_rad)
    sd     = abs(math.sin(dip_rad))   # positive downward

    fault_E_ctr = fault_E + 0.5 * L * ss + 0.5 * W * cd * cs
    fault_N_ctr = fault_N + 0.5 * L * cs - 0.5 * W * cd * ss
    depth_ctr   = abs(float(fault_depth)) + 0.5 * W * sd   # positive depth to centre

    xs = np.array([float(x_obs)])
    ys = np.array([float(y_obs)])
    zs = np.zeros(1)
    xc = np.array([float(fault_E_ctr)])
    yc = np.array([float(fault_N_ctr)])
    depth = np.array([depth_ctr])
    length = np.array([float(L)])
    width = np.array([float(W)])
    dip_deg = np.array([np.degrees(dip_rad)])
    strike_deg = np.array([np.degrees(strike_rad)])

    if slip_type == 1:  # strike-slip
        ss, ds = np.array([1.0]), np.array([0.0])
    else:               # dip-slip
        ss, ds = np.array([0.0]), np.array([1.0])
    ts = np.zeros(1)

    u, d, s, flag, flag2 = ok4.okada92(
        xs, ys, zs, xc, yc, depth, length, width,
        dip_deg, strike_deg, ss, ds, ts, mu, nu
    )
    return u.reshape(-1, 3)[0]   # [East, North, Down]


# =============================================================================
# Fault geometry classes
# =============================================================================

class patch:
    """A single rectangular fault patch."""

    def __init__(self, i, x1, x2, x3, length, width, strike, dip, rake):
        self.i = i
        self.x1 = x1   # Northing of top-left corner
        self.x2 = x2   # Easting of top-left corner
        self.x3 = x3   # Depth of top edge
        self.length = length
        self.width = width
        self.strike = strike
        self.dip = dip
        self.rake = rake


class segment:
    """A fault segment made of rectangular patches."""

    def __init__(self, name):
        self.name = name
        self.patches = []
        self.npatches = 0

    def tolist(self):
        return [self.i, self.x1, self.x2, self.x3,
                self.length, self.width, self.strike, self.dip, self.rake]

    def getpatches(self, src):
        for t in src:
            self.patches.append(patch(*(t.tolist())))
        self.npatches += len(src)


class fault:
    """A 3-D fault model composed of one or more segments."""

    def __init__(self, name, wdir, model):
        self.wdir = wdir
        self.name = name
        self.model = model
        self.segments = []
        self.nsegments = 0
        self.npatches = 0

    def getseg(self):
        print()
        for i, seg_file in enumerate(self.model):
            print('Load segment:', seg_file)
            newseg = segment(seg_file)
            fname = self.wdir + seg_file
            source = np.atleast_2d(np.loadtxt(fname, comments='#', ndmin=2))
            newseg.getpatches(source)
            self.segments.append(newseg)
            print('Number of patches:', self.segments[i].npatches)
            self.npatches += self.segments[i].npatches

        self.nsegments += len(self.model)


# =============================================================================
# Okada Green's function matrix builder
# =============================================================================

class okada85:
    """Build the elastic Green's function matrix G (Okada 1985/1992).

    Each column of G is the surface displacement caused by unit slip
    on one fault patch, projected onto the observation geometry.
    """

    def __init__(self, flt, network):
        self.flt = flt
        self.network = network

        self.M = flt.npatches
        self.N = sum(net.Npoint * net.dim for net in network)

        self.G = self.buildG()

    def buildG(self):
        print('Building Okada Green function matrix...')
        print('  Number of observations:', self.N)
        print('  Number of patches:     ', self.M)

        G = np.zeros((self.N, self.M), dtype=float)
        Npt = 0

        for network in self.network:
            dim = network.dim
            Gt = as_strided(G[Npt:Npt + network.Npoint * dim, :])

            for j in range(network.Npoint):
                Mt = 0
                for seg in self.flt.segments:
                    Gtt = as_strided(Gt[:, Mt:Mt + seg.npatches])

                    for i, p in enumerate(seg.patches):
                        point = network.points[j]
                        x, y = point.x, point.y
                        proj = [point.proj[0], point.proj[1], point.proj[2]]

                        dip_rad    = (p.dip    * math.pi) / 180.
                        strike_rad = (p.strike * math.pi) / 180.
                        rake_rad   = (p.rake   * math.pi) / 180.

                        us = _okada_displacement(x, y, dip_rad,
                                                 p.x2, p.x1, p.x3,
                                                 p.length, p.width, strike_rad,
                                                 slip_type=1)
                        ud = _okada_displacement(x, y, dip_rad,
                                                 p.x2, p.x1, p.x3,
                                                 p.length, p.width, strike_rad,
                                                 slip_type=2)

                        u = math.cos(rake_rad) * us + math.sin(rake_rad) * ud

                        if dim == 1:
                            Gtt[j, i] = proj[0] * u[0] + proj[1] * u[1] + proj[2] * u[2]
                        else:
                            for k in range(dim):  # East, North, Down
                                Gtt[dim * j + k, i] = u[k]

                    Mt += seg.npatches

            Npt += network.Npoint * dim

        return G
