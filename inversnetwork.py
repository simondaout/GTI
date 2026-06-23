#!/usr/bin/env python3

"""
Network classes for GPS time series and InSAR data.

@author: simon daout
"""

import numpy as np
import math
import sys
from os import path
from flatten import flatten


# =============================================================================
# Base point classes
# =============================================================================

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class gpspoint(point):
    def __init__(self, x, y, name, proj):
        super().__init__(x, y)
        self.name = name
        self.t = []
        self.d = []        # [East, North, Down]
        self.sigmad = []   # [East, North, Down]
        self.Nt = 0
        self.tmin = 0
        self.tmax = 0
        self.proj = proj

    def info(self):
        print('GPS station:', self.name)


# =============================================================================
# GPS time series (one file per station, multiple epochs)
# =============================================================================

class gpstimeseries:
    """GPS time series: one displacement file per station."""

    def __init__(self, network, reduction, dim, wdir, basis,
                 scale=1., weight=1., proj=[1., 1., 1.], extension='.dat'):
        self.network = network
        self.reduction = reduction
        self.dim = dim
        self.wdir = wdir
        self.basis = flatten(basis)
        self.scale = scale
        self.sigmad = np.atleast_1d(1. / weight)
        self.proj = proj
        self.extension = extension

        self.Mbasis = len(self.basis)
        self.points = []
        self.Npoint = 0
        self.Ndata = 0
        self.x, self.y = [], []

    def load(self, xlim, ylim):
        fname = self.wdir + self.network
        print('Input file:', fname)
        with open(fname, 'r') as f:
            name, x, y = np.loadtxt(f, comments='#', unpack=True,
                                     dtype='U4,f,f')
        index = np.nonzero(
            (x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])
        )
        name = np.delete(name, index)
        x    = np.delete(x, index)
        y    = np.delete(y, index)
        name, self.x, self.y = np.atleast_1d(name, x, y)
        self.Npoint = len(name)

        print('Loading time series...')
        for i in range(self.Npoint):
            station = self.wdir + self.reduction + '/' + name[i] + self.extension
            print(station)
            if not path.isfile(station):
                raise ValueError('Invalid file name: ' + station)

            self.points.append(gpspoint(self.x[i], self.y[i], name[i], self.proj))

            if self.dim == 3:
                dated, east, north, down, esigma, nsigma, dsigma = np.loadtxt(
                    station, comments='#', usecols=(0, 1, 2, 3, 4, 5, 6),
                    unpack=True, dtype='f,f,f,f,f,f,f'
                )
                dated, east, north, down, esigma, nsigma, dsigma = np.atleast_1d(
                    dated, east, north, down, esigma, nsigma, dsigma
                )
                self.points[i].d = [east * self.scale,
                                    north * self.scale,
                                    down * self.scale]
                self.points[i].sigmad = [esigma * self.sigmad * self.scale,
                                         nsigma * self.sigmad * self.scale,
                                         dsigma * self.sigmad * self.scale]
                self.plot = ['east', 'north', 'down']
                self.points[i].t = np.atleast_1d(dated)
                self.points[i].veast  = (east[-1]  - east[0])  / (dated[-1] - dated[0])
                self.points[i].vnorth = (north[-1] - north[0]) / (dated[-1] - dated[0])
                self.points[i].vdown  = (down[-1]  - down[0])  / (dated[-1] - dated[0])

            elif self.dim == 2:
                dated, east, north, esigma, nsigma = np.loadtxt(
                    station, comments='#', usecols=(0, 1, 2, 3, 4),
                    unpack=True, dtype='f,f,f,f,f'
                )
                dated, east, north, esigma, nsigma = np.atleast_1d(
                    dated, east, north, esigma, nsigma
                )
                self.points[i].d = [east * self.scale, north * self.scale]
                self.points[i].sigmad = [esigma * self.sigmad * self.scale,
                                         nsigma * self.sigmad * self.scale]
                self.plot = ['east', 'north']
                self.points[i].t = np.atleast_1d(dated)
                self.points[i].veast  = (east[-1]  - east[0])  / (dated[-1] - dated[0])
                self.points[i].vnorth = (north[-1] - north[0]) / (dated[-1] - dated[0])

            self.points[i].tmin, self.points[i].tmax = min(dated), max(dated)
            if len(dated) == 1:
                self.points[i].tmin = self.points[i].tmax - 1.
            self.points[i].Nt = len(self.points[i].t)
            self.Ndata += self.points[i].Nt * self.dim


# =============================================================================
# GPS stacked (velocity) data
# =============================================================================

class gpsstack:
    """GPS stacked displacements: one time-averaged value per station."""

    def __init__(self, network, reduction, dim, wdir, basis,
                 scale=1., weight=1., proj=[1., 1., 1.],
                 tmin=0., tmax=1., extension='.dat'):
        self.network = network
        self.reduction = reduction
        self.dim = dim
        self.wdir = wdir
        self.basis = flatten(basis)
        self.scale = scale
        self.sigmad = np.atleast_1d(1. / weight)
        self.proj = proj
        self.tmin = np.atleast_1d(tmin)
        self.tmax = np.atleast_1d(tmax)
        self.extension = extension

        self.Mbasis = len(self.basis)
        self.points = []
        self.Npoint = 0
        self.Ndata = 0
        self.x, self.y = [], []

    def load(self, xlim, ylim):
        fname = self.wdir + self.network
        print('Input file:', fname)
        with open(fname, 'r') as f:
            name, x, y = np.loadtxt(f, comments='#', unpack=True,
                                     dtype='U4,f,f')
        index = np.nonzero(
            (x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])
        )
        name = np.delete(name, index)
        x    = np.delete(x, index)
        y    = np.delete(y, index)
        name, self.x, self.y = np.atleast_1d(name, x, y)
        self.Npoint = len(name)

        print('Loading time series...')
        for i in range(self.Npoint):
            station = self.wdir + self.reduction + '/' + name[i] + self.extension
            print(station)
            if not path.isfile(station):
                raise ValueError('Invalid file name: ' + station)

            self.points.append(gpspoint(self.x[i], self.y[i], name[i], self.proj))
            self.points[i].tmin = self.tmin
            self.points[i].tmax = self.tmax
            dt = float(np.squeeze(self.tmax) - np.squeeze(self.tmin))

            if self.dim == 3:
                date, dated, east, north, down, esigma, nsigma, dsigma = np.loadtxt(
                    station, comments='#', usecols=(0, 1, 2, 3, 4, 5, 6, 7),
                    unpack=True, dtype='f,f,f,f,f,f,f,f'
                )
                dated, east, north, down, esigma, nsigma, dsigma = np.atleast_1d(
                    dated, east, north, down, esigma, nsigma, dsigma
                )
                self.points[i].d = [east * self.scale * dt,
                                    north * self.scale * dt,
                                    down * self.scale * dt]
                self.points[i].sigmad = [self.sigmad * self.scale * dt,
                                         self.sigmad * self.scale * dt,
                                         self.sigmad * self.scale * dt]
                self.plot = ['east', 'north', 'down']
                self.points[i].t = np.atleast_1d(self.tmax)

            elif self.dim == 2:
                dated, east, north, esigma, nsigma = np.loadtxt(
                    station, comments='#', usecols=(0, 1, 2, 3, 4),
                    unpack=True, dtype='f,f,f,f,f'
                )
                dated, east, north, esigma, nsigma = np.atleast_1d(
                    dated, east, north, esigma, nsigma
                )
                self.points[i].d = [east * self.scale * dt,
                                    north * self.scale * dt]
                self.points[i].sigmad = [self.sigmad * self.scale * dt,
                                         self.sigmad * self.scale * dt]
                self.plot = ['east', 'north']
                self.points[i].t = np.atleast_1d(self.tmax)

            elif self.dim == 1:
                dated, disp, edisp = np.loadtxt(
                    station, comments='#', usecols=(0, 1, 2),
                    unpack=True, dtype='f,f,f'
                )
                dated, disp, edisp = np.atleast_1d(dated, disp, edisp)
                self.points[i].d = [disp * self.scale * dt]
                self.points[i].sigmad = [self.sigmad * self.scale * dt]
                self.plot = ['fault parallel']
                self.points[i].t = np.atleast_1d(self.tmax)

            self.points[i].Nt = len(self.points[i].t)
            self.Ndata += self.points[i].Nt * self.dim


# =============================================================================
# InSAR point and stack classes
# =============================================================================

class insarpoint(point):
    """A single InSAR LOS observation."""

    def __init__(self, x, y, los, proj, tmin, tmax, sigmad):
        super().__init__(x, y)
        self.name = ''
        self.proj = proj   # [East, North, Down] LOS unit vector
        self.d = [[los]]
        self.sigmad = [[sigmad]]
        self.tmin = tmin
        self.tmax = tmax
        self.t = np.atleast_1d(tmax)
        self.Nt = 1


class insarstack:
    """InSAR stacked displacement (one scene).

    LOS projection can be specified in three ways (mutually exclusive):

    1. Scalar average angles (most common):
         av_los=23., av_heading=-76.
       los  = incidence angle from vertical (degrees): 0 = vertical, 90 = horizontal
       head = satellite flight-direction azimuth from North (degrees)
       → proj vector is constant across all pixels.

    2. Per-pixel angles read from the data file:
         los=True, head=True   → cols 4 & 5: los_angle  head_angle
         los=True, head=False  → col  4    : los_angle  (av_heading used for heading)
         los=False, head=True  → col  4    : head_angle (av_los used for incidence)

    Convention:
        projz =  cos(inc_rad)
        projx =  sin(inc_rad) * sin(head_rad)
        projy = -sin(inc_rad) * cos(head_rad)
    """

    def __init__(self, network, wdir,
                 av_los=None, av_heading=None,
                 los=False, head=False,
                 tmin=0., tmax=1., weight=1., scale=1., samp=1,
                 errorfile=None):
        self.network    = network
        self.dim        = 1
        self.wdir       = wdir
        self.tmin       = np.atleast_1d(tmin)
        self.tmax       = np.atleast_1d(tmax)
        self.sigmad     = 1. / weight
        self.scale      = scale
        self.av_los     = av_los
        self.av_heading = av_heading if av_heading is not None else 0.
        self.los        = bool(los)
        self.head       = bool(head)
        self.samp       = int(samp)
        self.errorfile  = errorfile   # optional: file with per-pixel uncertainties (x y sigma)

        if not self.los and not self.head and av_los is None:
            raise ValueError(
                f'{network}: provide av_los + av_heading  OR  los=True / head=True')

        self.points = []
        self.plot   = 'los'
        self.Ndata  = 0
        self.Npoint = 0

    @staticmethod
    def _proj_from_angles(look_deg, head_deg):
        """LOS unit vector [E, N, Up] from look angle and heading.

        look_deg : incidence angle from vertical (degrees)
                   0° = vertical (nadir),  90° = horizontal
        head_deg : satellite flight-direction azimuth from North (degrees)

        Convention (same as NSBas / profile.py):
            phi   = -90 - heading   (azimuth from East, math convention)
            theta =  90 - look      (elevation above horizontal)
            projE = cos(theta) * cos(phi)
            projN = cos(theta) * sin(phi)
            projU = sin(theta)
        """
        phi   = math.radians(-90. - head_deg)
        theta = math.radians(90.  - look_deg)
        return (math.cos(theta) * math.cos(phi),   # E
                math.cos(theta) * math.sin(phi),   # N
                math.sin(theta))                   # Up

    def load(self, xlim, ylim):
        fname = self.wdir + self.network
        print('Input file:', fname)
        if not path.isfile(fname):
            raise ValueError('Invalid file name: ' + fname)

        print('Loading interferograms...')
        with open(fname, 'r') as f:
            if self.los and self.head:
                # cols: x  y  los  look_angle  head_angle
                x, y, los, look_col, head_col = np.loadtxt(
                    f, comments='#', unpack=True, dtype='f,f,f,f,f')
                phi   = np.deg2rad(-90. - head_col)
                theta = np.deg2rad(90.  - look_col)
                self.projx = np.cos(theta) * np.cos(phi)
                self.projy = np.cos(theta) * np.sin(phi)
                self.projz = np.sin(theta)
            elif self.los:
                # col 4: look_angle; heading from av_heading
                x, y, los, look_col = np.loadtxt(
                    f, comments='#', unpack=True, dtype='f,f,f,f')
                phi   = math.radians(-90. - self.av_heading)
                theta = np.deg2rad(90. - look_col)
                self.projx = np.cos(theta) * math.cos(phi)
                self.projy = np.cos(theta) * math.sin(phi)
                self.projz = np.sin(theta)
            elif self.head:
                # col 4: head_angle; look from av_los
                x, y, los, head_col = np.loadtxt(
                    f, comments='#', unpack=True, dtype='f,f,f,f')
                phi   = np.deg2rad(-90. - head_col)
                theta = math.radians(90. - self.av_los)
                self.projx = np.cos(theta) * np.cos(phi)
                self.projy = np.cos(theta) * np.sin(phi)
                self.projz = math.sin(theta) * np.ones(len(x))
            else:
                # scalar average angles → constant proj
                x, y, los = np.loadtxt(f, comments='#', unpack=True,
                                        dtype='f,f,f')
                px, py, pz = self._proj_from_angles(self.av_los, self.av_heading)
                self.projx = np.full(len(x), px)
                self.projy = np.full(len(x), py)
                self.projz = np.full(len(x), pz)

        index = np.nonzero(
            (x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])
        )
        los, x, y = np.delete(los, index), np.delete(x, index), np.delete(y, index)
        self.projx = np.delete(self.projx, index)
        self.projy = np.delete(self.projy, index)
        self.projz = np.delete(self.projz, index)

        # ── subsampling ──────────────────────────────────────────────────────
        if self.samp > 1:
            x, y, los = x[::self.samp], y[::self.samp], los[::self.samp]
            self.projx = self.projx[::self.samp]
            self.projy = self.projy[::self.samp]
            self.projz = self.projz[::self.samp]

        losm = los - np.median(los)
        self.x, self.y, losm = np.atleast_1d(x, y, losm)
        self.Ndata = self.Npoint = len(losm)

        # ── per-pixel uncertainties ───────────────────────────────────────────
        if self.errorfile is not None:
            efname = self.errorfile if path.isabs(self.errorfile) else self.wdir + self.errorfile
            print('Error file:', efname)
            if not path.isfile(efname):
                raise ValueError('Invalid error file name: ' + efname)
            with open(efname, 'r') as ef:
                _ex, _ey, _esig = np.loadtxt(ef, comments='#', unpack=True,
                                              dtype='f,f,f')
            # apply same spatial filter then subsampling
            _emask = np.nonzero(
                (_ex < xlim[0]) | (_ex > xlim[1]) | (_ey < ylim[0]) | (_ey > ylim[1])
            )
            _esig = np.delete(_esig, _emask)
            if self.samp > 1:
                _esig = _esig[::self.samp]
            sigmad_arr = _esig * self.scale
        else:
            sigmad_arr = np.full(self.Npoint, self.sigmad)

        dt = float(np.squeeze(self.tmax) - np.squeeze(self.tmin))
        for i in range(self.Npoint):
            self.points.append(insarpoint(
                x[i], y[i],
                self.scale * losm[i] * dt,
                [self.projx[i], self.projy[i], self.projz[i]],
                self.tmin, self.tmax, sigmad_arr[i]
            ))


class insardisp:
    """InSAR displacement scene with full LOS unit vectors per pixel."""

    def __init__(self, network, wdir, tmin, tmax, sigmad):
        self.network = network
        self.dim = 1
        self.wdir = wdir
        self.tmin = np.atleast_1d(tmin)
        self.tmax = np.atleast_1d(tmax)
        self.sigmad = sigmad
        self.points = []
        self.plot = 'los'
        self.Ndata = 0
        self.Npoint = 0

    def load(self):
        fname = self.wdir + self.network
        print('Input file:', fname)
        if not path.isfile(fname):
            raise ValueError('Invalid file name: ' + fname)

        print('Loading interferograms...')
        with open(fname, 'r') as f:
            x, y, los, losx, losy, losz = np.loadtxt(
                f, comments='#', unpack=True, dtype='f,f,f,f,f,f'
            )
        x, y, los, losx, losy, losz = np.atleast_1d(x, y, los, losx, losy, losz)
        self.Ndata = self.Npoint = len(los)

        for i in range(self.Npoint):
            self.points.append(insarpoint(
                x[i], y[i], los[i],
                [losx[i], losy[i], losz[i]],
                self.tmin, self.tmax, self.sigmad
            ))
