#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot InSAR data and fault model patches as a quick-look map.

Usage
-----
    python3 GTI/plot_gti_data.py work/gti/input_3dinv.py

Displays:
  - InSAR LOS displacement fields (one panel per dataset)
  - Fault patches projected onto the map
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from os import path

# Make GTI modules importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inversfault import fault
from inversnetwork import insarstack

try:
    from inverskernel import interseismic
    from readgmt import gmt
except ImportError:
    pass


def patch_corners(x1, x2, x3, length, width, strike_deg, dip_deg):
    """Return (4,2) array of (x, y) km corners for one fault patch."""
    d2r = np.pi / 180.
    s, d = strike_deg * d2r, dip_deg * d2r

    # Unit vectors along-strike (Sv) and down-dip (Dv) — horizontal projection
    sv = np.array([ np.cos(s),  np.sin(s)])
    dv = np.array([-np.cos(d) * np.sin(s),
                    np.cos(d) * np.cos(s)])

    # x2 = Easting, x1 = Northing of top-left corner
    orig = np.array([x2, x1])

    c0 = orig
    c1 = orig + length * sv
    c2 = orig + length * sv + width * dv
    c3 = orig               + width * dv
    return np.array([c0, c1, c2, c3])   # shape (4, 2)


def load_input(fname):
    """exec the input file and return its globals."""
    g = {}
    sys.path.append(path.dirname(path.abspath(fname)))
    modname = path.splitext(path.basename(fname))[0]
    exec(f'from {modname} import *', g)
    return g


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    fname = sys.argv[1]
    print('Input file:', fname)
    g = load_input(fname)

    insar_list = g['insar']
    flt        = g['flt']
    maindir    = g.get('maindir', '')

    # ── load fault patches ───────────────────────────────────────────────────
    flt.getseg()

    # ── load InSAR data (no spatial filter) ─────────────────────────────────
    for ins in insar_list:
        ins.load([-9999., 9999.], [-9999., 9999.])

    # ── figure ───────────────────────────────────────────────────────────────
    n_insar = len(insar_list)
    fig, axes = plt.subplots(1, n_insar, figsize=(7 * n_insar, 6),
                             constrained_layout=True)
    if n_insar == 1:
        axes = [axes]

    for ax, ins in zip(axes, insar_list):
        x   = np.array(ins.x)
        y   = np.array(ins.y)
        los = np.array([pt.d[0][0] for pt in ins.points])

        vmax = np.nanpercentile(np.abs(los), 98)
        sc = ax.scatter(x, y, c=los, cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax,
                        s=1, rasterized=True)
        plt.colorbar(sc, ax=ax, label='LOS (mm)', shrink=0.7)

        # ── overlay fault patches ────────────────────────────────────────────
        all_polys = []
        for seg in flt.segments:
            for p in seg.patches:
                corners = patch_corners(p.x1, p.x2, p.x3,
                                        p.length, p.width,
                                        p.strike, p.dip)
                poly = mpatches.Polygon(corners, closed=True)
                all_polys.append(poly)

        pc = PatchCollection(all_polys, facecolor='none',
                             edgecolor='black', linewidth=0.6, zorder=5)
        ax.add_collection(pc)

        # label the projection direction
        label = ins.network.replace('.xylos', '')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle('InSAR data + fault model', fontsize=12)
    out = os.path.join(os.path.dirname(fname), 'gti_data_preview.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('Saved:', out)
    plt.show()


if __name__ == '__main__':
    main()
