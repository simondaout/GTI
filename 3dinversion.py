#!/usr/bin/env python3

"""
3-D geodetic inversion: GPS time series + InSAR → elastic dislocation slip on fault patches.

Usage:
    python3 GTI/3dinversion.py work/gti/input_3dinv.py

@author: simon daout
"""

import sys
import os
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy
import math
import getopt
from os import path

# Ensure the GTI directory is in sys.path so local modules import correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inverskernel import *
from inversfault import *
from inversnetwork import *
from compute_laplacian import *
from transform4patch_general import *
from consInv import consInvert
from readgmt import *

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def usage():
    print('3dinversion.py - Invert GPS time series and InSAR with multiple fault patches')
    print('')
    print('Usage: 3dinversion.py <input_file.py>')
    print('')


# =============================================================================
# Inversion class
# =============================================================================

class inversion:
    def __init__(self, flt, kernels, timeseries, insar, Lambda1,
                 outputdir=None, normodel=[], orb=None, base=None,
                 mogi=None, mogisource=[], xini=[], bndmin=[], bndmax=[],
                 xlim=[-90, 90], ylim=[-90, 90], zlim=[0, 90], gmtfiles=None):

        self.flt = flt
        self.kernels = kernels.kernels
        self.timeseries = timeseries
        self.insar = insar
        self.network = timeseries + insar
        self.Lambda1 = Lambda1
        self.outputdir = outputdir
        self.normodel = normodel
        self.orb = orb
        self.base = base
        self.mogi = mogi
        self.mogisource = mogisource
        # Default bounds: -inf/+inf per model block if not specified
        if len(bndmin) == 0:
            self.bndmin = np.full(7, -np.inf)
        else:
            self.bndmin = np.array(bndmin, dtype=float)
            for i in range(min(len(bndmin), len(normodel))):
                self.bndmin[i] = bndmin[i] * normodel[i]
        if len(bndmax) == 0:
            self.bndmax = np.full(7, np.inf)
        else:
            self.bndmax = np.array(bndmax, dtype=float)
            for i in range(min(len(bndmax), len(normodel))):
                self.bndmax[i] = bndmax[i] * normodel[i]
        self.xini = xini
        self.xmin, self.xmax = xlim[0], xlim[1]
        self.ymin, self.ymax = ylim[0], ylim[1]
        self.zmin, self.zmax = zlim[0], zlim[1]
        self.gmtfiles = gmtfiles

        self.Mpatch = self.flt.npatches
        self.Mker = len(self.kernels)
        self.Mfault = self.Mpatch * self.Mker

        self.ntimeseries = len(self.timeseries)
        self.ninsar = len(self.insar)
        self.Nnetwork = self.ntimeseries + self.ninsar

        self.M, self.N, self.Nts, self.Ni = 0, 0, 0, 0
        self.Msurface = 0
        for n in range(self.ntimeseries):
            ts = self.timeseries[n]
            self.Msurface = ts.Mbasis * ts.dim * ts.Npoint + self.Msurface
            self.Nts = ts.Ndata + self.Nts

        for n in range(self.ninsar):
            ins = self.insar[n]
            self.Ni = ins.Ndata + self.Ni

        self.N = self.Nts + self.Ni
        self.M = self.Msurface + self.Mfault

        print('Size of data vector:', self.N)

        self.d, self.sigmad = self.buildd()

        self.ok = okada85(self.flt, self.network)
        print('Size of Okada matrix:', self.ok.G.shape)

        self.Mbase = 0
        if (base == 'yes') and (self.Nts > 0):
            for i in range(self.ntimeseries):
                self.Mbase += self.network[i].dim

        self.orbital = [
            reference(name='reference', date=0., inversion='basis'),
            lonerror(name='longitudinal errors'),
            laterror(name='latitudinal errors'),
        ]
        if (orb == 'yes') and (self.ninsar > 0):
            self.Morb = len(self.orbital) * self.ninsar
        else:
            self.Morb = 0

        self.Mmogi = 1 if mogi == 'yes' else 0

        self.Manti = len(fmodel2d)
        self.Medge = len(fmodel2d)

        total_M = self.M + self.Morb + self.Manti + self.Mbase + self.Mmogi + self.Medge
        print('Size of model vector:', total_M)

        self.m = np.zeros(total_M, dtype=float)
        self.sigmam = np.ones(total_M, dtype=float)

        self.maxok = 1.
        self.ok.Gnorm = self.ok.G / (self.maxok * self.normodel[1])

        self.G = self.buildG()
        self.buildsigmam()

        print()
        print('Model uncertainties:', self.sigmam)

        self.E = np.zeros((total_M, total_M), dtype=float)
        self.f = np.zeros(self.E.shape[0], dtype=float)

        self.nfigure = 0

    # =========================================================================

    def buildd(self):
        print()
        print('Initialising inversion...')
        Nt = 0
        d      = np.zeros(self.N + self.Mfault, dtype=float)
        sigmad = np.ones( self.N + self.Mfault, dtype=float)

        for n in range(self.Nnetwork):
            network = self.network[n]
            dim = network.dim
            for i in range(network.Npoint):
                point = network.points[i]
                for j in range(dim):
                    d[Nt + point.Nt * j: Nt + point.Nt * (j + 1)]      = point.d[j][:]
                    sigmad[Nt + point.Nt * j: Nt + point.Nt * (j + 1)] = point.sigmad[j][:]
                Nt += point.Nt * network.dim

        return d, sigmad

    def buildsigmam(self):
        for n in range(self.ntimeseries):
            ts     = self.timeseries[n]
            Mbasis = ts.Mbasis
            dim    = ts.dim
            Npoint = ts.Npoint
            Mt     = 0
            sigmam = as_strided(self.sigmam[Mt: Mt + Mbasis * dim * Npoint])
            for l in range(Mbasis):
                sigmam[l: Mbasis * dim * Npoint: Mbasis] = ts.basis[l].sigmam
            Mt = Mbasis * dim * Npoint + Mt

        for l in range(self.Mker):
            self.sigmam[self.Msurface:: self.Mker] = (
                self.kernels[l].sigmam * self.maxok * self.normodel[1]
            )

        if (self.orb == 'yes') and (self.ninsar > 0):
            for n in range(self.ninsar):
                self.sigmam[self.M: self.M + self.Morb] = (
                    self.normodel[2] * np.ones(self.Morb, dtype=float)
                )

        self.sigmam[self.M + self.Morb: self.M + self.Morb + self.Manti] = self.normodel[3]
        self.sigmam[
            self.M + self.Morb + self.Manti + self.Mbase + self.Mmogi:
            self.M + self.Morb + self.Manti + self.Mbase + self.Mmogi + self.Medge
        ] = self.normodel[6]

        if (self.base == 'yes') and (self.Nts > 0):
            self.sigmam[
                self.M + self.Morb + self.Manti:
                self.M + self.Morb + self.Manti + self.Mbase
            ] = self.normodel[4] * np.ones(self.Mbase, dtype=float)

        if self.mogi == 'yes':
            self.sigmam[self.M + self.Morb + self.Manti + self.Mbase] = self.normodel[5]

    def buildG(self):
        G = np.zeros(
            (self.N + self.Mfault,
             self.M + self.Morb + self.Manti + self.Mbase + self.Mmogi + self.Medge),
            dtype=float
        )

        print()
        print('Size of design matrix:', G.shape)
        Npt, Nt, Mt = 0, 0, 0

        # ── Time-series block ──────────────────────────────────────────────────
        for n in range(self.ntimeseries):
            ts     = self.timeseries[n]
            dim    = ts.dim
            Npoint = ts.Npoint
            Mbasis = ts.Mbasis

            Gok = as_strided(self.ok.Gnorm[Npt: Npt + Npoint * dim, :])

            for i in range(Npoint):
                point = ts.points[i]
                t     = point.t

                T = as_strided(G[Nt: Nt + point.Nt * dim,
                                  Mt: Mt + Mbasis * dim * Npoint])
                F = as_strided(G[Nt: Nt + point.Nt * dim, self.Msurface:])

                Mfamily = Mbasis + self.Mker
                Gfamily = np.zeros((len(t), Mfamily), dtype=float)

                Gbasis = as_strided(Gfamily[:, :Mbasis])
                for l in range(Mbasis):
                    Gbasis[:, l] = ts.basis[l].g(t)

                Gker = as_strided(Gfamily[:, Mbasis:])
                for l in range(self.Mker):
                    Gker[:, l] = self.kernels[l].g(t)

                for j in range(dim):
                    T[point.Nt * j: point.Nt * (j + 1),
                      i * Mbasis * dim + j * Mbasis: i * Mbasis * dim + (j + 1) * Mbasis
                     ] = Gfamily[:, :Mbasis]
                    for k in range(self.Mpatch):
                        F[point.Nt * j: point.Nt * (j + 1),
                          self.Mker * k: self.Mker * (k + 1)
                         ] = Gfamily[:, Mbasis:] * Gok[i * dim + j, k]
                Nt += point.Nt * dim

            Npt += Npoint * dim
            Mt  += Mbasis * dim * Npoint

        # ── InSAR block ────────────────────────────────────────────────────────
        Gt  = as_strided(G[self.Nts: self.N, self.Msurface:])
        Nt  = 0
        Npt_insar = 0
        for n in range(self.ninsar):
            ins    = self.insar[n]
            Npoint = ins.Npoint
            dim    = ins.dim
            tmin, tmax = ins.tmin, ins.tmax
            Gok = as_strided(self.ok.Gnorm[Npt_insar: Npt_insar + Npoint * dim, :])

            Gker = np.zeros(self.Mker, dtype=float)
            for l in range(self.Mker):
                Gker[l] = float(np.squeeze(self.kernels[l].g(tmax) - self.kernels[l].g(tmin)))

            for i in range(Npoint):
                point = ins.points[i]
                if (self.orb == 'yes') and (self.ninsar > 0):
                    if len(self.orbital) == 1:
                        Gt[Nt: Nt + point.Nt, self.Mfault + n] = (
                            self.orbital[0].g(np.atleast_1d(point.x))
                            * np.ones(point.Nt, dtype=float) / self.normodel[2]
                        )
                    else:
                        Gt[Nt: Nt + point.Nt, self.Mfault + len(self.orbital) * n + 0] = (
                            self.orbital[0].g(np.atleast_1d(point.x))
                            * np.ones(point.Nt, dtype=float) / self.normodel[2]
                        )
                        Gt[Nt: Nt + point.Nt, self.Mfault + len(self.orbital) * n + 1] = (
                            self.orbital[1].g(np.atleast_1d(point.x))
                            * np.ones(point.Nt, dtype=float) / self.normodel[2]
                        )
                        Gt[Nt: Nt + point.Nt, self.Mfault + len(self.orbital) * n + 2] = (
                            self.orbital[2].g(np.atleast_1d(point.y))
                            * np.ones(point.Nt, dtype=float) / self.normodel[2]
                        )
                for k in range(self.Mpatch):
                    Gt[Nt: Nt + point.Nt, self.Mker * k: self.Mker * (k + 1)] = (
                        Gker[:] * Gok[i, k]
                    )
                Nt += point.Nt
            Npt_insar += Npoint

        # ── 2-D fault model (antiplane + edge) ─────────────────────────────────
        fperp = np.zeros(len(fmodel2d))
        s = [math.sin(strike), math.cos(strike), 0]
        n = [math.cos(strike), -math.sin(strike), 0]

        for j in range(len(fmodel2d)):
            fperp[j] = ((fmodel2d[j].x - fmodel2d[0].x) * n[0]
                        + (fmodel2d[j].y - fmodel2d[0].y) * n[1])
            Nt = 0
            for i in range(len(self.network)):
                network = self.network[i]
                dim = network.dim
                for ii in range(network.Npoint):
                    point = network.points[ii]
                    t     = point.t
                    tmin  = point.tmin
                    proj  = [point.proj[0], point.proj[1], point.proj[2]]
                    yp    = ((point.x - fmodel2d[0].x) * n[0]
                             + (point.y - fmodel2d[0].y) * n[1])

                    # Antiplane
                    col0 = self.Msurface + self.Mfault + self.Morb + j
                    Gt_anti = as_strided(G[Nt: Nt + point.Nt * dim, col0: col0 + 1])
                    u = fmodel2d[j].antiplane(yp, fperp[j], 0)
                    uanti = [u * s[0] * (t - tmin) / self.normodel[3],
                             u * s[1] * (t - tmin) / self.normodel[3],
                             u * 0.]
                    if dim == 1:
                        Gt_anti[:point.Nt, :] = (proj[0] * uanti[0]
                                                  + proj[1] * uanti[1]
                                                  + proj[2] * uanti[2])
                    else:
                        for jj in range(dim):
                            Gt_anti[point.Nt * jj: point.Nt * (jj + 1), 0] = uanti[jj]

                    # Edge
                    col1 = (self.Msurface + self.Mfault + self.Morb
                            + self.Manti + self.Mbase + self.Mmogi + j)
                    Gt_edge = as_strided(G[Nt: Nt + point.Nt * dim, col1: col1 + 1])
                    uh = fmodel2d[j].edge(yp, fperp[j], rake)[0]
                    uv = fmodel2d[j].edge(yp, fperp[j], rake)[1]
                    uedge = [uh * n[0], uh * n[1], uv]
                    if dim == 1:
                        Gt_edge[:point.Nt, :] = (proj[0] * uedge[0]
                                                  + proj[1] * uedge[1]
                                                  + proj[2] * uedge[2])
                    else:
                        for jj in range(dim):
                            Gt_edge[point.Nt * jj: point.Nt * (jj + 1), 0] = uedge[jj]

                    Nt += point.Nt * dim

        # ── Baseline (reference frame) block ───────────────────────────────────
        if (self.base == 'yes') and (self.Nts > 0):
            Nt   = 0
            Mdim = 0
            for n in range(self.ntimeseries):
                network = self.network[n]
                dim     = network.dim
                col0 = (self.Msurface + self.Mfault + self.Morb + self.Manti)
                for i in range(network.Npoint):
                    point = network.points[i]
                    t     = point.t
                    tmin  = point.tmin
                    Gt_bl = as_strided(G[Nt: Nt + point.Nt * dim,
                                         col0: col0 + self.Mbase])
                    for j in range(dim):
                        Gt_bl[point.Nt * j: point.Nt * (j + 1), Mdim + j] = (
                            (t - tmin) / self.normodel[4]
                        )
                    Nt += point.Nt * dim
                Mdim += dim

        # ── Mogi source block ───────────────────────────────────────────────────
        if self.mogi == 'yes':
            nu = 1. / 4
            Nt = 0
            col0 = (self.Msurface + self.Mfault + self.Morb
                    + self.Manti + self.Mbase)
            Gt_mogi = as_strided(G[self.Nts: self.N,
                                   col0: col0 + self.Mmogi])
            for n in range(self.ninsar):
                network = self.insar[n]
                dim     = network.dim
                for i in range(network.Npoint):
                    point = network.points[i]
                    proj  = [point.proj[0], point.proj[1], point.proj[2]]
                    x = point.x - self.mogisource[0]
                    y = point.y - self.mogisource[1]
                    r  = x**2 + y**2 + self.mogisource[2]**2
                    r3 = r ** (3. / 2)
                    u = [((1 + nu) / math.pi * x) / r3 / self.normodel[5],
                         ((1 + nu) / math.pi * y) / r3 / self.normodel[5],
                         ((1 + nu) / math.pi * self.mogisource[2]) / r3 / self.normodel[5]]
                    if dim == 1:
                        Gt_mogi[Nt: Nt + point.Nt, :] = (proj[0] * u[0]
                                                           + proj[1] * u[1]
                                                           + proj[2] * u[2])
                    else:
                        for j in range(dim):
                            Gt_mogi[Nt + point.Nt * j: point.Nt * (j + 1), :] = u[j]
                    Nt += point.Nt * dim

        print('G (GPS):', G[:self.Nts, :])
        print()
        print('G (InSAR):', G[self.Nts: self.N, :])
        return G

    def buildcons(self):
        print()
        print('Using positivity constraint...')
        self.E = np.diag(np.hstack([
            np.zeros(self.Msurface, dtype=float),
            1. * np.ones(self.Mfault, dtype=float),
            np.zeros(self.Morb, dtype=float),
            -1 * np.ones(self.Manti, dtype=float),
            np.zeros(self.Mbase),
            np.zeros(self.Mmogi, dtype=float),
            np.ones(self.Medge, dtype=float),
        ]), k=0)
        rowsnul = np.nonzero(self.E.sum(axis=1) == 0)
        self.E = np.delete(self.E, rowsnul, axis=0)
        self.f = np.zeros(self.E.shape[0], dtype=float)

    def buildsmooth(self):
        if self.Mker > 0:
            mineignv = 1. / self.M
            U, s, V = np.linalg.svd(self.G, full_matrices=False)
            eignv = np.diag(s)
            pos   = np.nonzero((eignv / np.max(eignv)) > mineignv)
            Gt    = np.zeros(eignv.shape)
            Gt[pos] = eignv[pos]
            Gt = np.dot(Gt, V)
            self.Res = np.diag(
                np.dot(np.dot(Gt.T, np.linalg.pinv(np.dot(Gt, Gt.T))), Gt)
            )

            print()
            print('Smoothing poorly-resolved patches with neighbors')

            neighbors = 6
            Mt  = 0
            Lap = np.zeros((self.Mpatch, self.Mpatch), dtype=float)
            for i in range(self.flt.nsegments):
                seg = self.flt.segments[i]
                r   = self.Res[Mt: Mt + seg.npatches]
                self.Lambda = np.diag(
                    np.diag(self.Lambda1 + 0.5 * np.cos(math.pi * r / 2) ** 10)
                )
                if seg.npatches > 5 * neighbors:
                    print('Using Laplacian for segment %s' % seg.name)
                    xpt = as_strided(xp[Mt: Mt + seg.npatches, :])
                    ypt = as_strided(yp[Mt: Mt + seg.npatches, :])
                    zpt = as_strided(zp[Mt: Mt + seg.npatches, :])
                    Lap[Mt: Mt + seg.npatches, Mt: Mt + seg.npatches] = (
                        self.Lambda
                        * compute_laplacian(xpt[:, 0], ypt[:, 0], zpt[:, 0], neighbors)
                    )
                else:
                    print('Using differential for segment %s' % seg.name)
                    Lap[Mt: Mt + seg.npatches, Mt: Mt + seg.npatches] = (
                        self.Lambda * (
                            np.diag(np.hstack([np.ones(seg.npatches - 1, dtype=float), 0]), k=0)
                            - np.diag(np.ones(seg.npatches - 1, dtype=float).T, k=1)
                        )
                    )
                Mt += seg.npatches

            for i in range(self.Mpatch):
                for j in range(self.Mpatch):
                    LapM = np.diag(np.ones(self.Mker, dtype=float) * Lap[i, j], k=0)
                    self.G[self.N + self.Mker * i: self.N + self.Mker * (i + 1),
                           self.Msurface + self.Mker * j: self.Msurface + self.Mker * (j + 1)
                          ] = LapM[:, :]

    def solve(self):
        print()
        print('Rank of design matrix:', np.linalg.matrix_rank(self.G))
        if np.linalg.matrix_rank(self.G) < int(0.5 * self.G.shape[1]):
            print('Design matrix not constrained')
            sys.exit()

        if self.xini is None:
            x0 = None
        else:
            x0 = np.hstack([
                self.xini[0] * self.normodel[0] * np.ones(self.Msurface, dtype=float),
                self.xini[1] * self.normodel[1] * np.ones(self.Mfault,   dtype=float),
                self.xini[2] * self.normodel[2] * np.ones(self.Morb,     dtype=float),
                self.xini[3] * self.normodel[3] * np.ones(self.Manti,    dtype=float),
                self.xini[4] * self.normodel[4] * np.ones(self.Mbase,    dtype=float),
                self.xini[5] * self.normodel[5] * np.ones(self.Mmogi,    dtype=float),
                self.xini[6] * self.normodel[7] * np.ones(self.Medge,    dtype=float),
            ])

        xmin = np.hstack([
            self.bndmin[0] * np.ones(self.Msurface, dtype=float),
            self.bndmin[1] * np.ones(self.Mfault,   dtype=float),
            self.bndmin[2] * np.ones(self.Morb,     dtype=float),
            self.bndmin[3] * np.ones(self.Manti,    dtype=float),
            self.bndmin[4] * np.ones(self.Mbase,    dtype=float),
            self.bndmin[5] * np.ones(self.Mmogi,    dtype=float),
            self.bndmin[6] * np.ones(self.Medge,    dtype=float),
        ])
        xmax = np.hstack([
            self.bndmax[0] * np.ones(self.Msurface, dtype=float),
            self.bndmax[1] * np.ones(self.Mfault,   dtype=float),
            self.bndmax[2] * np.ones(self.Morb,     dtype=float),
            self.bndmax[3] * np.ones(self.Manti,    dtype=float),
            self.bndmax[4] * np.ones(self.Mbase,    dtype=float),
            self.bndmax[5] * np.ones(self.Mmogi,    dtype=float),
            self.bndmax[6] * np.ones(self.Medge,    dtype=float),
        ])

        bounds = list(zip(xmin, xmax))

        self.m = consInvert(self.G, self.d, ineq=[self.E, self.f], bnd=bounds, iter=5000)

        self.dmodel = self.meval(self.m, self.G)
        self.mf = self.m / self.sigmam

    def meval(self, m, G):
        return np.dot(G, m)

    def residual(self, d, m, G, sigmad):
        return d - self.meval(m, G)

    def info(self):
        for n in range(self.ntimeseries):
            ts = self.timeseries[n]
            print('Time series:', ts.network)
            print()
            print('Number of basis functions:', ts.Mbasis)
            print('Number of points:', ts.Npoint)
            print()
            for i in range(ts.Npoint):
                for k in range(ts.Mbasis):
                    print('GPS %s:' % ts.basis[k].name)
                    print(self.mf[
                        i * (ts.Mbasis * ts.dim) + k * ts.dim:
                        i * (ts.Mbasis * ts.dim) + (k + 1) * ts.dim
                    ])

        print()
        print('Number of kernels:', self.Mker)
        for i in range(self.Mker):
            print('Kernel %s:' % self.kernels[i].name)
            print(self.mf[self.Msurface + i:: self.Mker])

        if (self.orb == 'yes') and (self.ninsar > 0):
            print()
            print('Number of orbital corrections:', self.ninsar)
            for n in range(self.ninsar):
                print('ax+by+c: %f x + %f y + %f' % (
                    self.mf[self.M + 3 * n + 1],
                    self.mf[self.M + 3 * n + 2],
                    self.mf[self.M + 3 * n],
                ))

        for j in range(len(fmodel2d)):
            print()
            print('Strike-slip fault %s:' % fmodel2d[j].name, self.mf[self.M + self.Morb + j])
            print('Dip-slip fault %s:' % fmodel2d[j].name,
                  self.mf[self.M + self.Morb + self.Manti + self.Mbase + self.Mmogi + j])

        if (self.base == 'yes') and (self.Nts > 0):
            print()
            print('Reference frame correction for GPS:')
            N = 0
            for i in range(self.ntimeseries):
                if self.network[i].dim == 2:
                    print('%s: %f N + %f E' % (
                        self.network[i].network,
                        self.mf[self.M + self.Morb + self.Manti + N],
                        self.mf[self.M + self.Morb + self.Manti + N + 1],
                    ))
                elif self.network[i].dim == 3:
                    print('%f N + %f E + %f D' % (
                        self.mf[self.M + self.Morb + self.Manti + N],
                        self.mf[self.M + self.Morb + self.Manti + N + 1],
                        self.mf[self.M + self.Morb + self.Manti + N + 2],
                    ))
                N += self.network[i].dim

        if self.mogi == 'yes':
            print()
            print('Mogi source amplitude:', self.mf[self.M + self.Morb + self.Manti + self.Mbase])

    def variance(self):
        r2 = (self.residual(self.d[:self.N], self.m, self.G[:self.N, :],
                            self.sigmad[:self.N]) ** 2).sum()
        d2 = (self.d[:self.N] ** 2).sum()
        self.var = (1 - r2 / d2) * 100
        print('Variance reduction: %+6.2f%%\n' % self.var)
        print()
        Nt = 0
        for n in range(len(self.network)):
            network = self.network[n]
            Ndata   = network.Ndata
            r2  = (self.residual(self.d[Nt: Nt + Ndata], self.m,
                                 self.G[Nt: Nt + Ndata, :],
                                 self.sigmad[Nt: Nt + Ndata]) ** 2).sum()
            d2  = (self.d[Nt: Nt + Ndata] ** 2).sum()
            var = (1 - r2 / d2) * 100
            print('Variance reduction %s: %+6.2f%%' % (network.network, var))
            Nt += Ndata

    def plotxy(self):
        Ndata = 0
        Mt    = 0
        pp    = PdfPages(self.outputdir + 'timeseries.pdf')
        for n in range(self.ntimeseries):
            ts     = self.timeseries[n]
            dim    = ts.dim
            Mbasis = ts.Mbasis
            Npoint = min(4, ts.Npoint)

            d = as_strided(self.d[Ndata: Ndata + ts.Ndata])
            G = as_strided(self.G[Ndata: Ndata + ts.Ndata, :])

            Nt = 0
            for i in range(Npoint):
                point = ts.points[i]
                tmin, tmax = point.tmin, point.tmax
                x = point.t

                fig = plt.figure(self.nfigure, figsize=(18, 9))
                self.nfigure += 1
                fig.subplots_adjust(hspace=0.7)

                for j in range(dim):
                    ax1 = fig.add_subplot(3, dim, j + 1)
                    ax1.plot(x, d[Nt + point.Nt * j: Nt + point.Nt * (j + 1)],
                             'o', label=ts.plot[j])
                    ax1.errorbar(x, d[Nt + point.Nt * j: Nt + point.Nt * (j + 1)],
                                 yerr=self.sigmad[Nt + point.Nt * j: Nt + point.Nt * (j + 1)],
                                 label='uncertainties')
                    ax1.plot(x, self.meval(self.m,
                                           G[Nt + point.Nt * j: Nt + point.Nt * (j + 1), :]),
                             color='black', linewidth=4.0, label='Fit')
                    ax1.legend(bbox_to_anchor=(0., 1.02, 1, 0.102), loc=3,
                               ncol=2, mode='expand', borderaxespad=0.)
                    ax1.set_ylabel('Displacements (m)')
                    ax1.set_xlabel('Time')
                    ax1.grid(True)
                    locs, labels = plt.xticks()
                    plt.xticks(locs, ['%g' % v for v in locs])
                    ax1.set_xlim([tmin, tmax])

                    ax2 = fig.add_subplot(3, dim, dim + j + 1)
                    ax2.plot(x, self.residual(
                        d[Nt + point.Nt * j: Nt + point.Nt * (j + 1)],
                        self.m,
                        G[Nt + point.Nt * j: Nt + point.Nt * (j + 1), :],
                        self.sigmad[Nt + point.Nt * j: Nt + point.Nt * (j + 1)],
                    ))
                    ax2.legend(['residuals'])
                    ax2.set_ylabel('Error (m)')
                    ax2.set_xlabel('Time')
                    ax2.grid(True)
                    locs, labels = plt.xticks()
                    plt.xticks(locs, ['%g' % v for v in locs])
                    ax2.set_xlim([tmin, tmax])

                    ax3 = fig.add_subplot(3, dim, 2 * dim + j + 1)
                    for k in range(1, Mbasis):
                        ax3.plot(x,
                                 self.m[Mt + k + Mbasis * dim * i + j * Mbasis]
                                 * G[Nt + point.Nt * j: Nt + point.Nt * (j + 1),
                                     Mt + k + Mbasis * dim * i + j * Mbasis],
                                 label=ts.basis[k].name)
                    ax3.set_ylabel('Displacements (m)')
                    ax3.set_xlabel('Time')
                    ax3.grid(True)
                    ax3.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3,
                               ncol=2, mode='expand', borderaxespad=0.)
                    locs, labels = plt.xticks()
                    plt.xticks(locs, ['%g' % v for v in locs])
                    ax3.set_xlim([tmin, tmax])

                Nt += point.Nt * dim
                plt.suptitle('%s forward model' % point.name)
                fig.savefig(pp, format='pdf')
            Ndata += ts.Ndata
            Mt    += Mbasis * dim * ts.Npoint
        pp.close()

    def plotdisp(self, x0, y0):
        fig = plt.figure(self.nfigure, figsize=(16, 8))
        self.nfigure += 1
        plt.axis('equal')
        plt.xlim(x0 + self.xmin, x0 + self.xmax)
        plt.ylim(y0 + self.ymin, y0 + self.ymax)
        colors  = ['b', 'm', 'black']
        markers = ['^', 'v', '^', 'v', '^', 'v']

        Nt = 0
        Mt = 0
        for n in range(self.ntimeseries):
            network = self.timeseries[n]
            dim     = network.dim
            Npoint  = network.Npoint
            plt.scatter(network.x, network.y, c=colors[n], s=70,
                        marker=markers[n], label=network.network)
            for i in range(Npoint):
                x, y  = network.x[i], network.y[i]
                point = network.points[i]
                dt    = point.tmax - point.tmin
                t     = point.t
                plt.text(x, y + 1.5, point.name, fontsize=10, fontweight='bold',
                         ha='center', va='center', color='black')
                Ndata = point.Nt * dim
                scale = 4. * max(abs(self.d))
                baseline = np.zeros((dim, point.Nt))
                if self.base == 'yes':
                    for j in range(dim):
                        baseline[j, :] = (self.mf[self.M + self.Morb + self.Manti + j + Mt]
                                          * np.ones(point.Nt))
                datax  = point.veast  / scale
                datay  = point.vnorth / scale
                modelx = (self.dmodel[Nt + len(t) - 1] - self.dmodel[Nt]) / dt * scale
                modely = (self.dmodel[Nt + 2 * len(t) - 1] - self.dmodel[Nt + len(t)]) / dt * scale
                if point.Nt > 1:
                    gps   = plt.quiver(x, y, datax,  datay,  scale=None, width=0.002)
                    model = plt.quiver(x, y, modelx, modely, scale=None, width=0.002, color='r')
                else:
                    gps   = plt.quiver(x, y, datax,  datay,  scale=None, color='black', width=0.002)
                    model = plt.quiver(x, y, modelx, modely, scale=None, color='r', width=0.002)
                Nt += Ndata
            Mt += dim

        plt.quiverkey(gps,   0.1, 1.015, 0.001, 'GPS displacements', coordinates='axes', color='black')
        plt.quiverkey(model, 0.1, 1.075, 0.001, 'Model',             coordinates='axes', color='red')
        plt.scatter(x0, y0, c='red', s=300, marker='*')
        plt.title('Stack displacement')
        plt.xlabel('Distance (km)')

        for ii in range(len(self.gmtfiles)):
            gf = self.gmtfiles[ii]
            fx, fy = gf.load()
            plt.plot(fx[0], fy[0], color=gf.color, lw=gf.width, label=gf.name)
            for i in range(1, len(fx)):
                plt.plot(fx[i], fy[i], color=gf.color, lw=gf.width)

        if self.Mpatch > 0:
            plt.plot(xp[0, :], yp[0, :], c='grey', label='Fault model')
        for k in range(self.Mpatch):
            plt.plot(xp[k, :], yp[k, :], c='grey', lw=1.)

        plt.scatter(x0, y0, c='red', s=300, marker='*', label=name0)
        plt.legend()
        plt.title('Forward model for GPS')
        plt.xlabel('Distance (km)')
        fig.savefig(self.outputdir + 'coseismic_surface.png',
                    facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), format='PNG')

    def plotco(self, x0, y0):
        fig = plt.figure(self.nfigure, figsize=(16, 8))
        self.nfigure += 1
        plt.axis('equal')
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        colors  = ['b', 'm', 'black']
        markers = ['^', 'v', '^', 'v', '^', 'v']
        scale   = max(abs(self.d))
        Mt, Nt  = 0, 0

        for n in range(self.ntimeseries):
            data  = self.timeseries[n]
            x, y  = data.x, data.y
            Ndata = data.Ndata
            dim   = data.dim
            baseline = np.zeros(dim, dtype=float)
            if self.base == 'yes':
                for i in range(dim):
                    baseline[i] = self.mf[self.M + self.Morb + self.Manti + i + Mt]

            if dim != 1:
                datax  = (self.d[Nt: Nt + Ndata: dim]     - baseline[0]) / scale
                datay  = (self.d[Nt + 1: Nt + Ndata: dim] - baseline[1]) / scale
                modelx = (self.dmodel[Nt: Nt + Ndata: dim]     - baseline[0]) / scale
                modely = (self.dmodel[Nt + 1: Nt + Ndata: dim] - baseline[1]) / scale
            else:
                datax  = (self.d[Nt: Nt + Ndata]      - baseline[0]) * data.proj[0] / scale
                datay  = (self.d[Nt: Nt + Ndata]      - baseline[0]) * data.proj[1] / scale
                modelx = (self.dmodel[Nt: Nt + Ndata] - baseline[0]) * data.proj[0] / scale
                modely = (self.dmodel[Nt: Nt + Ndata] - baseline[0]) * data.proj[1] / scale

            gps   = plt.quiver(x, y, datax,  datay,  scale=12., width=0.002, color='black')
            model = plt.quiver(x, y, modelx, modely, scale=12., width=0.002, color='r')
            plt.quiverkey(gps,   0.1, 1.015, 0.4, 'GPS displacements', coordinates='axes', color='black')
            plt.quiverkey(model, 0.1, 1.075, 0.4, 'Model',             coordinates='axes', color='red')
            network = plt.scatter(x, y, c=colors[n], s=70, marker=markers[n], label=data.network)
            Nt += Ndata
            Mt += dim

        for ii in range(len(self.gmtfiles)):
            gf = self.gmtfiles[ii]
            fx, fy = gf.load()
            plt.plot(fx[0], fy[0], color=gf.color, lw=gf.width, label=gf.name)
            for i in range(1, len(fx)):
                plt.plot(fx[i], fy[i], color=gf.color, lw=gf.width)

        if self.Mpatch > 0:
            plt.plot(xp[0, :], yp[0, :], c='grey', label='Fault model')
        for k in range(self.Mpatch):
            plt.plot(xp[k, :], yp[k, :], c='grey')

        plt.legend()
        plt.title('GPS Data')
        plt.xlabel('Distance (km)')
        fig.savefig(self.outputdir + 'coseismic_surface.png',
                    facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), format='PNG')

    def plotInSAR(self, x0, y0, name0):
        if self.ninsar == 0:
            return

        n_rows = self.ninsar
        fig, axes = plt.subplots(n_rows, 3,
                                 figsize=(15, 4.5 * n_rows),
                                 constrained_layout=True)
        self.nfigure += 1
        if n_rows == 1:
            axes = [axes]  # make iterable

        Nt = 0
        for n_idx in range(self.ninsar):
            data  = self.insar[n_idx]
            x, y  = np.array(data.x), np.array(data.y)
            Ndata = data.Ndata
            label = data.network.replace('.xylos', '')

            if self.orb == 'yes':
                orb = (x * self.mf[self.M + len(self.orbital) * n_idx + 1]
                       + y * self.mf[self.M + len(self.orbital) * n_idx + 2]
                       + self.mf[self.M + len(self.orbital) * n_idx])
            else:
                orb = np.zeros(Ndata)

            los_data  = self.d[self.Nts + Nt: self.Nts + Nt + Ndata] - orb
            los_model = self.meval(self.m, self.G[self.Nts + Nt: self.Nts + Nt + Ndata, :]) - orb
            los_res   = los_data - los_model

            vmax = np.nanpercentile(np.abs(los_data), 98)
            vmax_res = np.nanpercentile(np.abs(los_res), 98)

            ax1, ax2, ax3 = axes[n_idx]
            kw = dict(s=2, rasterized=True, linewidths=0)

            sc1 = ax1.scatter(x, y, c=los_data,  cmap='RdBu_r', vmin=-vmax,     vmax=vmax,     **kw)
            sc2 = ax2.scatter(x, y, c=los_model, cmap='RdBu_r', vmin=-vmax,     vmax=vmax,     **kw)
            sc3 = ax3.scatter(x, y, c=los_res,   cmap='RdBu_r', vmin=-vmax_res, vmax=vmax_res, **kw)

            # ── data extent for this network (fixes xlim/ylim) ────────────────
            _pad = max(x.max() - x.min(), y.max() - y.min()) * 0.03
            _xlim = (x.min() - _pad, x.max() + _pad)
            _ylim = (y.min() - _pad, y.max() + _pad)

            for sc, ax, title in [(sc1, ax1, f'Data — {label}'),
                                  (sc2, ax2, f'Model — {label}'),
                                  (sc3, ax3, f'Residual — {label}')]:
                plt.colorbar(sc, ax=ax, label='LOS (mm)', shrink=0.8)
                ax.set_xlabel('Easting (km)')
                ax.set_ylabel('Northing (km)')
                ax.set_title(title, fontsize=9)
                ax.grid(True, alpha=0.3)
                # fault patches
                for k in range(self.Mpatch):
                    ax.plot(xp[k, :], yp[k, :], c='k', lw=0.5)
                # gmt overlays
                for gf in (self.gmtfiles or []):
                    fx, fy = gf.load()
                    for i_g in range(len(fx)):
                        ax.plot(fx[i_g], fy[i_g], color=gf.color, lw=gf.width)
                # set limits from data AFTER plotting overlays
                ax.set_xlim(_xlim)
                ax.set_ylim(_ylim)
                ax.set_aspect('equal')
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune='both'))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune='both'))
                ax.tick_params(axis='x', rotation=30, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)

            Nt += Ndata

        fig.suptitle('InSAR inversion results', fontsize=12)
        fig.savefig(os.path.join(self.outputdir, 'insar_results.pdf'), bbox_inches='tight')

    def graphr(self):
        fig = plt.figure(self.nfigure, figsize=(8.5, 4.5))
        self.nfigure += 1
        Ndata   = 0
        markers = ['^', '+', '+', '+', '+', '+', '+', '+']
        mews    = [1, 0.4, 0.4, 0.4, 0.4, 0.4]

        for n in range(self.ninsar):
            network = self.insar[n]
            x, y   = network.x, network.y
            if self.orb == 'yes':
                orb = (x * inv.mf[self.M + len(self.orbital) * n + 1]
                       + y * inv.mf[self.M + len(self.orbital) * n + 2]
                       + inv.mf[self.M + len(self.orbital) * n])
            else:
                orb = 0
            data  = self.d[self.Nts + Ndata: self.Nts + Ndata + network.Ndata] - orb
            model = self.meval(self.m, self.G[self.Nts + Ndata: self.Nts + Ndata + network.Ndata, :]) - orb
            plt.plot(data, model, 'o', marker='.', label=network.network, mew=mews[1])
            Ndata += network.Ndata

        Ndata = 0
        for n in range(self.ntimeseries):
            network  = self.timeseries[n]
            baseline = np.zeros(network.dim, dtype=float)
            if self.base == 'yes':
                for l in range(network.dim):
                    baseline[l] = self.mf[self.M + self.Morb + self.Manti + network.dim * n + l]
                if self.ninsar > 0:
                    proj = self.insar[0].proj
                    x = sum((self.d[Ndata + l: Ndata + network.Ndata: network.dim] - baseline[l]) * proj[l]
                            for l in range(network.dim))
                    y = sum((self.dmodel[Ndata + l: Ndata + network.Ndata: network.dim] - baseline[l]) * proj[l]
                            for l in range(network.dim))
                else:
                    x1 = self.d[Ndata: Ndata + network.Ndata: network.dim] - baseline[0]
                    x2 = self.d[Ndata + 1: Ndata + network.Ndata: network.dim] - baseline[1]
                    x  = np.hstack([x1, x2])
                    y1 = self.dmodel[Ndata: Ndata + network.Ndata: network.dim] - baseline[0]
                    y2 = self.dmodel[Ndata + 1: Ndata + network.Ndata: network.dim] - baseline[1]
                    y  = np.hstack([y1, y2])
            else:
                if self.ninsar > 0:
                    proj = self.insar[0].proj
                    x = (self.d[Ndata: Ndata + network.Ndata] * proj[0]
                         + self.d[Ndata: Ndata + network.Ndata] * proj[1])
                    y = (self.dmodel[Ndata: Ndata + network.Ndata] * proj[0]
                         + self.dmodel[Ndata: Ndata + network.Ndata] * proj[1])
                else:
                    x = np.hstack([self.d[Ndata: Ndata + network.Ndata]] * 2)
                    y = np.hstack([self.dmodel[Ndata: Ndata + network.Ndata]] * 2)
            plt.plot(x, y, 'o', marker='^', label=network.network, mew=mews[0])
            Ndata += network.Ndata

        plt.title('Error estimation')
        plt.xlabel('Data')
        plt.ylabel('Forward model')
        ax = plt.gca()
        ax.grid(True)
        ax.text(0.6, 0.1, 'variance reduction= %+6.2f%%\n' % self.var,
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(facecolor='blue', alpha=0.5))
        xmin_ax, xmax_ax = ax.get_xlim()
        plt.plot([xmin_ax, xmax_ax], [xmin_ax, xmax_ax])
        plt.legend(loc=2)
        plt.axis('equal')
        fig.savefig(self.outputdir + 'residual.png',
                    facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), format='PNG')

    def plot3D(self):
        import matplotlib.patches as _mpatch
        from matplotlib.collections import PatchCollection as _PC

        # ── fault extent from patch corners ───────────────────────────────────
        _xlo, _xhi = np.min(xp), np.max(xp)
        _ylo, _yhi = np.min(yp), np.max(yp)
        _zhi       = np.max(zp)   # positive = deep; plotted as -zp
        _pad       = max((_xhi - _xlo), (_yhi - _ylo)) * 0.08

        for i in range(self.Mker):
            # ── collect slip per patch ────────────────────────────────────────
            mker = np.zeros(self.Mpatch, dtype=float)
            j = 0
            for kk in range(self.flt.nsegments):
                seg = self.flt.segments[kk]
                for k in range(seg.npatches):
                    mker[j + k] = self.mf[self.Msurface + i + self.Mker * (j + k)]
                j += seg.npatches

            vmax = max(abs(mker)) if mker.any() else 1.
            norm  = mpl.colors.Normalize(vmin=0., vmax=vmax)
            cmap  = cm.hot_r

            # ── figure: manual subplots (no constrained_layout) ───────────────
            fig = plt.figure(figsize=(14, 6))
            self.nfigure += 1
            ax_map = fig.add_subplot(1, 2, 1)
            ax_3d  = fig.add_subplot(1, 2, 2, projection='3d')

            # ── map view ──────────────────────────────────────────────────────
            polys = []
            colors_patch = []
            for k in range(self.Mpatch):
                corners = list(zip(xp[k, :], yp[k, :]))
                polys.append(_mpatch.Polygon(corners, closed=True))
                colors_patch.append(cmap(norm(mker[k])))

            pc = _PC(polys, facecolors=colors_patch, edgecolor='k', linewidth=0.4)
            ax_map.add_collection(pc)
            ax_map.set_xlim(_xlo - _pad, _xhi + _pad)
            ax_map.set_ylim(_ylo - _pad, _yhi + _pad)
            ax_map.set_xlabel('Easting (km)')
            ax_map.set_ylabel('Northing (km)')
            ax_map.set_aspect('equal')
            ax_map.set_title('Slip — map view')
            ax_map.grid(True, alpha=0.3)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, ax=ax_map, label='Slip (mm/yr)', shrink=0.8)

            # ── 3-D view ──────────────────────────────────────────────────────
            for k in range(self.Mpatch):
                verts = [list(zip(xp[k, :], yp[k, :], -zp[k, :]))]
                ax_3d.add_collection3d(
                    Poly3DCollection(verts, facecolors=cmap(norm(mker[k])),
                                     edgecolor='k', linewidth=0.3))
            ax_3d.set_xlim(_xlo - _pad, _xhi + _pad)
            ax_3d.set_ylim(_ylo - _pad, _yhi + _pad)
            ax_3d.set_zlim(-_zhi, 0.)
            ax_3d.set_xlabel('East (km)',  fontsize=8)
            ax_3d.set_ylabel('North (km)', fontsize=8)
            ax_3d.set_zlabel('Depth (km)', fontsize=8)
            ax_3d.set_title('Slip — 3-D view')

            fig.suptitle(f'Kernel {i}: {self.kernels[i].name}', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(self.outputdir, f'fault_kernel{i}.pdf'),
                        bbox_inches='tight')

    def plotRes(self):
        import matplotlib.patches as _mpatch
        from matplotlib.collections import PatchCollection as _PC

        if self.Mker > 0:
            # ── fault extent from patch corners ───────────────────────────────
            _xlo, _xhi = np.min(xp), np.max(xp)
            _ylo, _yhi = np.min(yp), np.max(yp)
            _zhi       = np.max(zp)
            _pad       = max((_xhi - _xlo), (_yhi - _ylo)) * 0.08

            for i in range(self.Mker):
                mker = np.zeros(self.Mpatch, dtype=float)
                j = 0
                for kk in range(self.flt.nsegments):
                    seg = self.flt.segments[kk]
                    for k in range(seg.npatches):
                        mker[j + k] = self.Res[self.Msurface + i + self.Mker * (j + k)]
                    j += seg.npatches

                mker_max = np.max(np.abs(mker))
                if mker_max > 0:
                    mker = mker / mker_max
                norm  = mpl.colors.Normalize(vmin=0., vmax=1.)
                cmap  = cm.Blues_r

                # ── figure ────────────────────────────────────────────────────
                fig = plt.figure(figsize=(14, 6))
                self.nfigure += 1
                ax_map = fig.add_subplot(1, 2, 1)
                ax_3d  = fig.add_subplot(1, 2, 2, projection='3d')

                # ── map view ──────────────────────────────────────────────────
                polys = []
                colors_patch = []
                for k in range(self.Mpatch):
                    corners = list(zip(xp[k, :], yp[k, :]))
                    polys.append(_mpatch.Polygon(corners, closed=True))
                    colors_patch.append(cmap(norm(mker[k])))

                pc = _PC(polys, facecolors=colors_patch, edgecolor='k', linewidth=0.4)
                ax_map.add_collection(pc)
                ax_map.set_xlim(_xlo - _pad, _xhi + _pad)
                ax_map.set_ylim(_ylo - _pad, _yhi + _pad)
                ax_map.set_xlabel('Easting (km)')
                ax_map.set_ylabel('Northing (km)')
                ax_map.set_aspect('equal')
                ax_map.set_title('Resolution — map view')
                ax_map.grid(True, alpha=0.3)
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                fig.colorbar(sm, ax=ax_map, label='Normalised resolution', shrink=0.8)

                # ── 3-D view ──────────────────────────────────────────────────
                for k in range(self.Mpatch):
                    verts = [list(zip(xp[k, :], yp[k, :], -zp[k, :]))]
                    ax_3d.add_collection3d(
                        Poly3DCollection(verts, facecolors=cmap(norm(mker[k])),
                                         edgecolor='k', linewidth=0.3))
                ax_3d.set_xlim(_xlo - _pad, _xhi + _pad)
                ax_3d.set_ylim(_ylo - _pad, _yhi + _pad)
                ax_3d.set_zlim(-_zhi, 0.)
                ax_3d.set_xlabel('East (km)',  fontsize=8)
                ax_3d.set_ylabel('North (km)', fontsize=8)
                ax_3d.set_zlabel('Depth (km)', fontsize=8)
                ax_3d.set_title('Resolution — 3-D view')

                fig.suptitle(f'Kernel {i}: {self.kernels[i].name} — Resolution', fontsize=11)
                fig.tight_layout()
                fig.savefig(os.path.join(self.outputdir, f'resolution_kernel{i}.pdf'),
                            bbox_inches='tight')

    def plotlos(self):
        fig = plt.figure(self.nfigure, figsize=(17, 9))
        self.nfigure += 1
        xmin_los, xmax_los = -150, 150
        plt.xlim(xmin_los, xmax_los)
        Nt   = 0
        proj = self.insar[0].proj
        s = [math.sin(strike), math.cos(strike), 0]
        n = [math.cos(strike), -math.sin(strike), 0]

        for m_idx in range(len(self.timeseries)):
            network = self.timeseries[m_idx]
            dim     = network.dim
            model_vals = np.zeros(network.Npoint, dtype=float)
            data_vals  = np.zeros(network.Npoint, dtype=float)
            y_vals     = np.zeros(network.Npoint, dtype=float)
            Mt = 0
            for i in range(network.Npoint):
                point    = network.points[i]
                baseline = np.zeros((point.Nt, dim), dtype=float)
                y_vals[i] = ((point.x - self.antisource[0]) * n[0]
                             + (point.y - self.antisource[1]) * n[1])
                if (self.base == 'yes') and (self.Nts > 0):
                    for j in range(dim):
                        baseline[:, j] = (self.mf[self.M + self.Morb + self.Manti + j + Mt]
                                          * np.ones(point.Nt, dtype=float))
                        data_vals[i] += ((self.d[Nt + point.Nt * j: Nt + point.Nt * (j + 1)]
                                          - baseline[:, j]) * proj[j]).sum() / point.Nt
                Nt += point.Nt * dim
            Mt += dim
            plt.plot(y_vals, data_vals, '^', label=network.network, markersize=10)

        Nt = 0
        for m_idx in range(self.ninsar):
            data  = self.insar[m_idx]
            proj  = data.proj
            xx, yy = data.x, data.y
            Ndata = data.Ndata
            if self.orb == 'yes':
                orb = (inv.mf[self.M + 3 * m_idx + 1] * np.ones(len(xx))
                       * xx + inv.mf[self.M + 3 * m_idx + 2] * np.ones(len(xx)) * yy
                       + inv.mf[self.M + 3 * m_idx] * np.ones(len(xx)))
            else:
                orb = 0
            y_vals = ((xx - self.antisource[0]) * n[0]
                      + (yy - self.antisource[1]) * n[1])
            los = inv.d[self.Nts + Nt: self.Nts + Nt + Ndata] - orb
            plt.plot(y_vals, los, 'o', label=data.network, markersize=1.)
            Nt += Ndata

        xx = np.arange(-100, 100, 0.1)
        yy = np.arange(-100, 100, 0.1)

        if self.anti == 'yes':
            strike_anti = (self.antisource[2] * math.pi) / 180.
            phi_anti    = (self.antisource[4] * math.pi) / 180.
            s_anti = [math.sin(strike_anti), math.cos(strike_anti), 0]
            n_anti = [math.cos(strike_anti), -math.sin(strike_anti), 0]
            w      = self.antisource[3]
            y_anti = (xx - self.antisource[0]) * n_anti[0] + (yy - self.antisource[1]) * n_anti[1]

            u = ((1. / (2 * math.pi))
                 * (np.arctan2(math.sin(phi_anti) * y_anti + math.cos(phi_anti) * w,
                               math.sin(phi_anti) * w - math.cos(phi_anti) * y_anti)
                    + np.arctan2(math.sin(phi_anti) * y_anti + math.cos(phi_anti) * w,
                                 -math.cos(phi_anti) * y_anti + math.sin(phi_anti) * w)))

            uanti = [u * s_anti[0], u * s_anti[1], u * 0.]
            model1 = self.mf[self.M + self.Morb] * (
                uanti[0] * proj[0] + uanti[1] * proj[1] + uanti[2] * proj[2]
            )
            plt.plot(y_anti, model1, '--', c='red', label='antiplane model', lw=2.)

        if self.edge == 'yes':
            strike_edge = (self.edgesource[2] * math.pi) / 180.
            s_edge = [math.sin(strike_edge), math.cos(strike_edge), 0]
            n_edge = [math.cos(strike_edge), -math.sin(strike_edge), 0]
            yperp  = ((self.edgesource[0] - self.antisource[0]) * n_edge[0]
                      + (self.edgesource[1] - self.antisource[1]) * n_edge[1])
            dip_edge  = (self.edgesource[4] * math.pi) / 180.
            L_edge    = self.edgesource[5]
            rake_edge = (self.edgesource[6] * math.pi) / 180.
            w1 = self.edgesource[3]
            w2 = self.edgesource[3] + L_edge * math.sin(dip_edge)
            y_edge = (xx - self.antisource[0]) * n_edge[0] + (yy - self.antisource[1]) * n_edge[1]
            zeta1  = (y_edge - yperp) / w1
            zeta2  = ((y_edge - yperp) - L_edge * math.cos(dip_edge)) / w2
            denom1 = 1 + zeta1**2
            denom2 = 1 + zeta2**2
            uv = ((1. * math.sin(rake_edge) / math.pi)
                  * (math.sin(dip_edge) * np.arctan(zeta1)
                     + (math.cos(dip_edge) + math.sin(dip_edge) * zeta1) / denom1
                     - math.sin(dip_edge) * np.arctan(zeta2)
                     - (math.cos(dip_edge) + math.sin(dip_edge) * zeta2) / denom2))
            uh = (-(1. * math.sin(rake_edge) / math.pi)
                  * (math.cos(dip_edge) * np.arctan(zeta1)
                     + (math.sin(dip_edge) - math.cos(dip_edge) * zeta1) / denom1
                     - math.cos(dip_edge) * np.arctan(zeta2)
                     - (math.sin(dip_edge) - math.cos(dip_edge) * zeta2) / denom2))
            u_edge = [uh * n_edge[0], uh * n_edge[1], uv]
            model2 = self.mf[self.M + self.Morb + self.Manti + self.Mbase + self.Mmogi] * (
                u_edge[0] * proj[0] + u_edge[1] * proj[1] + u_edge[2] * proj[2]
            )
            plt.plot(y_edge, model2, '--', c='green', label='edge model', lw=2.)

        if (self.edge == 'yes') and (self.anti == 'yes'):
            plt.plot(y_edge, model1 + model2, '--', c='b', label='edge+anti', lw=4.)

        plt.legend()
        plt.title('Antiplane displacements')
        plt.xlabel('Distance to the fault')
        plt.ylabel('LOS Displacements')
        ax = plt.gca()
        if self.anti == 'yes':
            ax.text(0.02, 0.95,
                    'antiplane slip: %+6.3f' % abs(self.mf[self.M + self.Morb]),
                    verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(facecolor='blue', alpha=0.5))
        if self.edge == 'yes':
            ax.text(0.02, 0.85,
                    'edge slip: %+6.3f' % abs(
                        self.mf[self.M + self.Morb + self.Manti + self.Mbase + self.Mmogi]),
                    verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(facecolor='blue', alpha=0.5))
        ax.grid(True)

    def plotG(self):
        fig = plt.figure(21)
        plt.title('G options (last 200 rows)')
        plt.imshow(self.G[self.N - 200: self.N, self.M:])
        fig = plt.figure(22)
        plt.title('G time series')
        plt.imshow(self.G[0: self.Nts, :self.M])

    def export(self):
        pass


# =============================================================================
# Main script
# =============================================================================

print()
print('-' * 75)
print('Initialisation')
print('-' * 75)
print()

try:
    opts, args = getopt.getopt(sys.argv[1:], 'h', ['help'])
except getopt.GetoptError as err:
    print(str(err))
    print('For help use --help')
    sys.exit()

for o, a in opts:
    if o in ('-h', '--help'):
        usage()
        sys.exit()
    else:
        assert False, 'Unhandled option'

if len(sys.argv) == 1:
    usage()
    assert False, 'No input file provided'

if len(sys.argv) == 2:
    fname = sys.argv[1]
    print()
    print('Input file:', fname)
    sys.path.append(path.dirname(path.abspath(fname)))
    modname = path.splitext(path.basename(fname))[0]
    exec('from %s import *' % modname)
else:
    assert False, 'Too many arguments'

# ── Defaults (overridden by values in the input file) ────────────────────────
_g = globals()
smoothing   = _g.get('smoothing',   'yes')
constraint  = _g.get('constraint',  'yes')
lambda1     = _g.get('lambda1',     1.)
base        = _g.get('base',        'yes')
orb         = _g.get('orb',         'yes')
plot_data   = _g.get('plot_data',   'no')    # plot data + fault before inversion
mogi        = _g.get('mogi',        'no')
mogi_source = _g.get('mogi_source', [0., 0., 5.])
xini        = _g.get('xini',        None)
bndmin      = _g.get('bndmin',      [])
bndmax      = _g.get('bndmax',      [])
# normodel: [surface, fault, orbital, antiplane, ref_frame, mogi, edge]
normodel    = _g.get('normodel',    [1., 1., 500., 1., 1., 1., 1.])
fmodel2d    = _g.get('fmodel2d',    [])
rake2d      = _g.get('rake2d',      90.)
strike2d    = _g.get('strike2d',    0.)
plotfiles   = _g.get('plotfiles',   [])
gmtfiles    = _g.get('gmtfiles',    [])
timeseries  = _g.get('timeseries',  [])
coplot      = _g.get('coplot',      'no')
displot     = _g.get('displot',     'no')
timeplot    = _g.get('timeplot',    'no')
residual    = _g.get('residual',    'no')
insarplot   = _g.get('insarplot',   'yes')
faultplot   = _g.get('faultplot',   'yes')
plotlos     = _g.get('plotlos',     'no')
x0          = _g.get('x0',         0.)
y0          = _g.get('y0',         0.)
name0       = _g.get('name0',      'Origin')
# xlim/ylim/zlim: wide defaults so all data is loaded; refined below from data
# None in input file is treated the same as "not set"
xlim        = _g.get('xlim') or [-9999., 9999.]
ylim        = _g.get('ylim') or [-9999., 9999.]
zlim        = _g.get('zlim') or [-500.,  0.]
_xlim_user  = isinstance(_g.get('xlim'), (list, tuple))
_ylim_user  = isinstance(_g.get('ylim'), (list, tuple))

# Load time series
for i in range(len(timeseries)):
    timeseries[i].load(xlim, ylim)
    print('Number of points:', timeseries[i].Npoint)
    print('Dimension:', timeseries[i].dim)
    print('Number of data:', timeseries[i].Ndata)

# Load interferograms
print()
for i in range(len(insar)):
    insar[i].load(xlim, ylim)
    print('Number of InSAR points:', insar[i].Npoint)

# Auto-derive xlim/ylim from data if not set in input file
if not _xlim_user or not _ylim_user:
    _all_x, _all_y = [], []
    for _net in list(insar) + list(timeseries):
        if hasattr(_net, 'x') and len(_net.x) > 0:
            _all_x.append(_net.x)
            _all_y.append(_net.y)
    if _all_x:
        _ax = np.concatenate(_all_x)
        _ay = np.concatenate(_all_y)
        _pad = max((_ax.max() - _ax.min()), (_ay.max() - _ay.min())) * 0.05 + 10.
        if not _xlim_user:
            xlim = [float(_ax.min()) - _pad, float(_ax.max()) + _pad]
            print(f'Auto xlim: {xlim}')
        if not _ylim_user:
            ylim = [float(_ay.min()) - _pad, float(_ay.max()) + _pad]
            print(f'Auto ylim: {ylim}')

# Load kernels
kernels = kernel(kernels)
print(kernels.info())

# Load plot options
for i in range(len(plotfiles)):
    plotfiles[i].load(xlim, ylim)

# Load fault model
d2r    = np.pi / 180
rake   = rake2d * d2r
strike = strike2d * d2r

flt.getseg()
print()
print('Fault model:', flt.model)
print('Total segments:', flt.nsegments)
print('Total patches:', flt.npatches)

fault_arr = np.zeros((flt.npatches, 7), dtype=float)
Nt = 0
for i in range(flt.nsegments):
    seg = flt.segments[i]
    ftemp = as_strided(fault_arr[Nt: Nt + seg.npatches, :])
    for j in range(seg.npatches):
        p = seg.patches[j]
        ftemp[j, :] = [p.x1, p.x2, p.x3, p.length, p.width, p.dip, p.strike]
    Nt += seg.npatches

xp, yp, zp = transform4patch_general(
    fault_arr[:, 1], fault_arr[:, 0], fault_arr[:, 2],
    fault_arr[:, 3], fault_arr[:, 4], fault_arr[:, 5], fault_arr[:, 6]
)

if (len(kernels.kernels) > 0) and (flt.nsegments == 0):
    raise ValueError('No fault model')

# ── Preview: data map + fault model ──────────────────────────────────────────
if plot_data == 'yes':
    import matplotlib.patches as _mpatches
    from matplotlib.collections import PatchCollection as _PC

    def _patch_corners(x1, x2, x3, length, width, strike_deg, dip_deg):
        """Return (4,2) Easting/Northing corners of a fault patch."""
        _d2r = np.pi / 180.
        _s, _d = strike_deg * _d2r, dip_deg * _d2r
        _sv = np.array([np.sin(_s),  np.cos(_s)])           # E, N along-strike
        _dv = np.array([np.cos(_d) * np.cos(_s),
                        -np.cos(_d) * np.sin(_s)])           # E, N down-dip
        _orig = np.array([x2, x1])
        _c0 = _orig
        _c1 = _orig + length * _sv
        _c2 = _orig + length * _sv + width * _dv
        _c3 = _orig               + width * _dv
        return np.array([_c0, _c1, _c2, _c3])

    _n = len(insar)
    _fig, _axes = plt.subplots(1, _n, figsize=(7 * _n, 6), constrained_layout=True)
    if _n == 1:
        _axes = [_axes]

    for _ax, _ins in zip(_axes, insar):
        _x   = np.array(_ins.x)
        _y   = np.array(_ins.y)
        _los = np.array([_pt.d[0][0] for _pt in _ins.points])
        _vm  = np.nanpercentile(np.abs(_los), 98)
        _sc  = _ax.scatter(_x, _y, c=_los, cmap='RdBu_r',
                           vmin=-_vm, vmax=_vm, s=1, rasterized=True)
        plt.colorbar(_sc, ax=_ax, label='LOS (mm)', shrink=0.7)

        # ── limits from InSAR data (before overlays so GMT can't expand them) ─
        _pad = max(_x.max() - _x.min(), _y.max() - _y.min()) * 0.03
        _xlim_d = (_x.min() - _pad, _x.max() + _pad)
        _ylim_d = (_y.min() - _pad, _y.max() + _pad)

        _polys = []
        for _seg in flt.segments:
            for _p in _seg.patches:
                _corners = _patch_corners(_p.x1, _p.x2, _p.x3,
                                          _p.length, _p.width,
                                          _p.strike, _p.dip)
                _polys.append(_mpatches.Polygon(_corners, closed=True))
        _ax.add_collection(_PC(_polys, facecolor='none',
                               edgecolor='k', linewidth=0.7, zorder=5))
        for _gf in (gmtfiles or []):
            _fx, _fy = _gf.load()
            for _ig in range(len(_fx)):
                _ax.plot(_fx[_ig], _fy[_ig], color=_gf.color, lw=_gf.width, zorder=4)

        # force extent to InSAR data after all overlays
        _ax.set_xlim(_xlim_d)
        _ax.set_ylim(_ylim_d)
        _ax.set_title(_ins.network.replace('.xylos', ''), fontsize=9)
        _ax.set_xlabel('Easting (km)')
        _ax.set_ylabel('Northing (km)')
        _ax.set_aspect('equal')
        _ax.grid(True, alpha=0.3)

    _fig.suptitle('InSAR data + fault model (pre-inversion)', fontsize=11)
    os.makedirs(outputdir, exist_ok=True)
    _out_preview = os.path.join(outputdir, 'gti_data_preview.pdf')
    _fig.savefig(_out_preview, dpi=150, bbox_inches='tight')
    print(f'Saved preview: {_out_preview}')
    plt.show()
    plt.close('all')  # isoler les figures de résultats

print()
print('-' * 75)
print('Inversion')
print('-' * 75)
print()

inv = inversion(
    flt=flt, kernels=kernels, timeseries=timeseries, insar=insar,
    Lambda1=lambda1, outputdir=outputdir, normodel=normodel,
    orb=orb, base=base, mogi=mogi, mogisource=mogi_source,
    xini=xini, bndmin=bndmin, bndmax=bndmax,
    xlim=xlim, ylim=ylim, zlim=zlim, gmtfiles=gmtfiles,
)

if smoothing == 'yes':
    inv.buildsmooth()

if constraint == 'yes':
    inv.buildcons()

inv.solve()
inv.variance()
inv.info()

# Plots — save all open figures to a single PDF in outputdir
if coplot    == 'yes': inv.plotco(x0, y0)
if displot   == 'yes': inv.plotdisp(x0, y0)
if timeplot  == 'yes': inv.plotxy()
if residual  == 'yes': inv.graphr()
if insarplot == 'yes': inv.plotInSAR(x0, y0, name0)
if faultplot == 'yes':
    inv.plotRes()
    inv.plot3D()
if plotlos   == 'yes': inv.plotlos()

plt.show()
