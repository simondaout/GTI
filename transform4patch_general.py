#!/usr/bin/env python3

import numpy as np
import math


def transform4patch_general(x1, y1, z1, dl, dw, dip, strike):
    """
    Compute the four corners of each fault patch from its reference point and geometry.

    Parameters
    ----------
    x1, y1, z1 : array-like
        Top-left corner of each patch (Easting, Northing, depth).
    dl : array-like
        Along-strike length of each patch.
    dw : array-like
        Down-dip width of each patch.
    dip : array-like
        Dip angle of each patch (degrees).
    strike : array-like
        Strike angle of each patch (degrees).

    Returns
    -------
    xp, yp, zp : ndarray, shape (ntot, 4)
        Four corners of each patch (Easting, Northing, depth).
    """
    ntot = len(x1)
    d2r = math.pi / 180

    xp = np.zeros((ntot, 4))
    yp = np.zeros((ntot, 4))
    zp = np.zeros((ntot, 4))

    for k in range(ntot):
        s = [math.sin(strike[k] * d2r), math.cos(strike[k] * d2r), 0]
        d = [
            math.cos(dip[k] * d2r) * math.cos(strike[k] * d2r),
            -math.cos(dip[k] * d2r) * math.sin(strike[k] * d2r),
            math.sin(dip[k] * d2r),
        ]

        # Top-right corner
        x2 = x1[k] + dl[k] * s[0]
        y2 = y1[k] + dl[k] * s[1]
        z2 = z1[k]
        # Bottom-right corner
        x3 = x2 + dw[k] * d[0]
        y3 = y2 + dw[k] * d[1]
        z3 = z2 + dw[k] * d[2]
        # Bottom-left corner
        x4 = x1[k] + dw[k] * d[0]
        y4 = y1[k] + dw[k] * d[1]
        z4 = z1[k] + dw[k] * d[2]

        xp[k, :] = [x1[k], x2, x3, x4]
        yp[k, :] = [y1[k], y2, y3, y4]
        zp[k, :] = [z1[k], z2, z3, z4]

    return xp, yp, zp
