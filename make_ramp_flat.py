#!/usr/bin/env python3
"""
make_ramp_flat.py — Generate a ramp-flat fault segment file (.seg) and
optionally run seg2flt to produce the patch file (.flt).

Usage
-----
    python3 GTI/make_ramp_flat.py [options]

Options
-------
    --n0        Starting Northing (km)          default: -22.0
    --e0        Starting Easting  (km)          default:  57.0
    --strike    Strike angle (degrees)          default: -78.0
    --rake      Rake angle (degrees)            default:  90.0
    --length    Along-strike length L (km)      default:  20.0
    --dips      Dip angles, comma-separated     default: 78,14,4
    --widths    Down-dip widths (km), comma-sep default: 2,13,30
    --lo        Initial patch length (km)       default: 1.0
    --wos       Initial patch widths per seg    default: 1.0,1.5,2.0
    --alphaws   Down-dip growth factors         default: 1.0,1.3,1.2
    --out       Output .seg file path           default: work/data/faults/ramp-flat_seg.flt
    --flt       Output .flt file path (optional, runs seg2flt if given)

Example
-------
    python3 GTI/make_ramp_flat.py --n0 -22 --e0 57 \\
        --dips 78,14,4 --widths 2,13,30 \\
        --flt work/data/faults/ramp-flat.flt
"""

import math
import argparse
import subprocess
import sys
import os


def bas_plan(n0, e0, d0, dip_deg, width, north_shift, east_shift):
    """Return (north, east, depth) of the bottom edge of a segment."""
    horiz = width * math.cos(math.radians(dip_deg))
    vert  = width * math.sin(math.radians(dip_deg))
    return (n0 + north_shift * horiz,
            e0 + east_shift  * horiz,
            d0 + vert)


def make_seg(n0, e0, strike, rake, length, dips, widths, lo,
             wos, alphaws, out_seg, out_flt=None):

    assert len(dips) == len(widths) == len(wos) == len(alphaws), \
        "dips, widths, wos, alphaws must all have the same length"

    east_shift  =  math.cos(math.radians(strike))
    north_shift = -math.sin(math.radians(strike))

    # ── compute segment origins ───────────────────────────────────────────────
    origins = [(n0, e0, 0.0)]
    for dip, w in zip(dips[:-1], widths[:-1]):
        n, e, d = bas_plan(*origins[-1], dip, w, north_shift, east_shift)
        origins.append((n, e, d))

    # ── write .seg file ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_seg)), exist_ok=True)
    with open(out_seg, 'w') as f:
        f.write('# Ramp-flat fault geometry\n')
        f.write(f'# strike={strike}°  rake={rake}°  L={length} km\n')
        f.write('#\n')
        f.write('# Columns: i  x1(north)  x2(east)  x3(depth)  L  W  strike  dip  rake  lo  wo  alphal  alphaw\n')
        f.write('#\n')

        # segment summary in comments
        n_prev, e_prev, d_prev = n0, e0, 0.0
        for k, (dip, w) in enumerate(zip(dips, widths)):
            n_bot, e_bot, d_bot = bas_plan(n_prev, e_prev, d_prev,
                                            dip, w, north_shift, east_shift)
            f.write(f'# Seg{k+1}: top=({n_prev:.3f}, {e_prev:.3f}, {d_prev:.3f})  '
                    f'dip={dip}°  W={w} km  → base d={d_bot:.3f} km\n')
            n_prev, e_prev, d_prev = n_bot, e_bot, d_bot
        f.write('#\n')

        for k, (n, e, d, dip, w, wo, aw) in enumerate(
                zip(*zip(*origins), dips, widths, wos, alphaws)):
            f.write(f'  {k+1}  {n:8.4f}  {e:8.4f}  {d:8.4f}  '
                    f'{length:.1f}  {w:.3f}  {strike:.1f}  {dip:.2f}  '
                    f'{rake:.1f}  {lo:.1f}  {wo:.1f}  1.0  {aw:.1f}\n')

    print(f'Written: {out_seg}')

    # ── optionally run seg2flt ────────────────────────────────────────────────
    if out_flt:
        script = os.path.join(os.path.dirname(__file__), 'seg2flt.py')
        os.makedirs(os.path.dirname(os.path.abspath(out_flt)), exist_ok=True)
        with open(out_flt, 'w') as flt:
            subprocess.run([sys.executable, script, out_seg],
                           stdout=flt, check=True)
        n_patches = sum(1 for line in open(out_flt)
                        if line.strip() and not line.startswith('#'))
        print(f'Written: {out_flt}  ({n_patches} patches)')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n0',      type=float, default=-22.0)
    p.add_argument('--e0',      type=float, default=57.0)
    p.add_argument('--strike',  type=float, default=-78.0)
    p.add_argument('--rake',    type=float, default=90.0)
    p.add_argument('--length',  type=float, default=20.0)
    p.add_argument('--dips',    type=str,   default='78,14,4')
    p.add_argument('--widths',  type=str,   default='2,13,30')
    p.add_argument('--lo',      type=float, default=1.0)
    p.add_argument('--wos',     type=str,   default='1.0,1.5,2.0')
    p.add_argument('--alphaws', type=str,   default='1.0,1.3,1.2')
    p.add_argument('--out',     type=str,   default='work/data/faults/ramp-flat_seg.flt')
    p.add_argument('--flt',     type=str,   default=None)
    args = p.parse_args()

    dips    = [float(x) for x in args.dips.split(',')]
    widths  = [float(x) for x in args.widths.split(',')]
    wos     = [float(x) for x in args.wos.split(',')]
    alphaws = [float(x) for x in args.alphaws.split(',')]

    make_seg(
        n0=args.n0, e0=args.e0,
        strike=args.strike, rake=args.rake, length=args.length,
        dips=dips, widths=widths,
        lo=args.lo, wos=wos, alphaws=alphaws,
        out_seg=args.out, out_flt=args.flt,
    )


if __name__ == '__main__':
    main()
