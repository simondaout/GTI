#!/usr/bin/env python3
"""
seg2flt.py — subsample a fault patch into smaller segments.

Segment length and width start from lo and wo respectively,
increasing geometrically with down-dip distance with factors
alphal and alphaw (alphal > 1 for increase).

Input segment file columns (13 values per line):
  i  x1(north)  x2(east)  x3(depth)  L  W  strike  dip  rake  lo  wo  alphal  alphaw

Output columns written to stdout:
  i  x1  x2  x3  length  width  strike  dip  rake

Usage
-----
  python3 seg2flt.py file.seg
  cat file.seg | python3 seg2flt.py
  python3 seg2flt.py --with-slip file.seg

@author: sbarbot (original) — translated to Python 3 by simon daout
"""

import sys
import getopt
import numpy as np


def usage():
    print('seg2flt.py converts a segment definition to a finely sampled fault file')
    print('')
    print('usage: seg2flt.py [--with-slip] file.seg')
    print('')
    print('or from standard input:')
    print('       cat file.seg | seg2flt.py')
    print('')
    print('writes the list of patches to standard output')
    print('')
    print('options:')
    print('  --with-slip  read and propagate a slip value (14-column input)')
    sys.exit()


_FMT_NOSLIP = "{:4d} {:8.4f} {:8.4f} {:8.4f} {:8.3f} {:8.3f} {:8.2f} {:5.2f} {:4.1f}"
_FMT_SLIP   = "{:4d} {:+10.3e} {:8.4f} {:8.4f} {:8.4f} {:8.3f} {:8.3f} {:8.2f} {:5.2f} {:4.1f}"


def seg2flt(index, x1o, x2o, x3o, L, W, strike, dip, rake,
            lo, wo, alphal, alphaw, slip=None):
    """Subsample one fault segment into patches with geometric size growth.

    Parameters
    ----------
    index : int
        Running patch counter (incremented in-place).
    x1o, x2o, x3o : float
        Origin of the segment (north, east, depth).
    L, W : float
        Total along-strike length and down-dip width.
    strike, dip, rake : float
        Angles in degrees.
    lo, wo : float
        Approximate initial patch length and width.
    alphal, alphaw : float
        Geometric growth factors (1 = uniform, >1 = growing with depth).
    slip : float or None
        Slip value to propagate (only with --with-slip).

    Returns
    -------
    index : int
        Updated patch counter.
    """
    d2r = np.pi / 180.

    # ── build down-dip width array ──────────────────────────────────────────
    Wc = W
    k  = 0
    w  = np.array([0.])
    while Wc > 0:
        Wt = wo * alphaw ** k
        if Wt > Wc / 2:
            Wt = Wc
        wn = min(Wt, Wc)
        w  = np.append(w, wn)
        k += 1
        Wc -= wn
    Nw = k

    # ── unit vectors along strike and down-dip ──────────────────────────────
    Sv = [ np.cos(strike * d2r),
           np.sin(strike * d2r),
           0.]
    Dv = [-np.cos(dip * d2r) * np.sin(strike * d2r),
           np.cos(dip * d2r) * np.cos(strike * d2r),
           np.sin(dip * d2r)]

    # ── loop over down-dip rows, then along-strike columns ──────────────────
    for j in range(Nw):
        lt = lo * alphal ** j
        Nl = int(np.ceil(L / lt))
        lt = L / Nl          # adjust so patches tile L exactly
        dw = w[:j + 1].sum()

        for i in range(Nl):
            x1 = x1o + i * lt * Sv[0] + dw * Dv[0]
            x2 = x2o + i * lt * Sv[1] + dw * Dv[1]
            x3 = x3o + i * lt * Sv[2] + dw * Dv[2]
            index += 1
            if slip is None:
                print(_FMT_NOSLIP.format(
                    index, x1, x2, x3, lt, w[j + 1], strike, dip, rake))
            else:
                print(_FMT_SLIP.format(
                    index, slip, x1, x2, x3, lt, w[j + 1], strike, dip, rake))

    return index


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "with-slip"])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    is_with_slip = False
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
        elif o == "--with-slip":
            is_with_slip = True
        else:
            assert False, f"unhandled option: {o}"

    fid = open(args[0], 'r') if args else sys.stdin

    if is_with_slip:
        print('# nb       slip       x1       x2       x3   length    width   strike   dip  rake')
    else:
        print('# nb       x1       x2       x3   length    width   strike   dip  rake')

    k = 0
    for line in fid:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        numbers = list(map(float, line.split()))
        s = None
        if len(numbers) == 13:
            i, x1, x2, x3, length, width, strike, dip, rake, Lo, Wo, al, aw = numbers
        elif len(numbers) == 14:
            if is_with_slip:
                i, s, x1, x2, x3, length, width, strike, dip, rake, Lo, Wo, al, aw = numbers
            else:
                # 14th column = rake derivative (ignored)
                i, x1, x2, x3, length, width, strike, dip, rake, _, Lo, Wo, al, aw = numbers
        else:
            raise ValueError(
                f"invalid number of columns ({len(numbers)}) — expected 13 or 14")

        k = seg2flt(k, x1, x2, x3, length, width, strike, dip, rake, Lo, Wo, al, aw, slip=s)

    if fid is not sys.stdin:
        fid.close()


if __name__ == "__main__":
    main()
