# GTI — Geodetic Time-series Inversion

Python 3 toolkit for inverting GPS time series and InSAR stacked displacements into elastic dislocation slip on 3-D fault patch models (Okada 1992).

## Overview

GTI solves the linear problem:

```
G · m = d
```

where **d** is the observed surface displacement (InSAR LOS, GPS), **G** is the elastic Green's function matrix (Okada), and **m** is the slip model on fault patches. The inversion includes optional Laplacian smoothing, slip positivity constraints, and orbital ramp estimation.

## Features

- **Multi-dataset inversion**: combine InSAR (stacked or time-series) and GPS networks
- **Flexible LOS projection**: scalar average angles (`av_los`, `av_heading`) or per-pixel geometry read from data files
- **Ramp-flat fault geometry**: three-segment fault model (steep ramp → flat décollement) with automatic patch discretisation
- **Smoothing**: Laplacian regularisation with spatially variable weight based on resolution
- **Constrained inversion**: slip bounds (e.g. no back-slip) via `bndmin`/`bndmax`
- **Result figures**: InSAR data / model / residual maps, 3-D fault slip, resolution maps

## Dependencies

```
numpy scipy matplotlib
```

**Okada4py** (elastic dislocation, C++ extension):

```bash
pip install meson meson-python ninja
git clone https://github.com/jolivetr/okada4py.git
cd okada4py && pip install .
```

## Repository structure

```
GTI/
├── 3dinversion.py          # Main inversion script
├── inversfault.py          # Fault patch model + Okada Green's function builder
├── inversnetwork.py        # InSAR / GPS network classes
├── inverskernel.py         # Time-series basis functions (interseismic, coseismic…)
├── consInv.py              # Constrained least-squares solver
├── seg2flt.py              # Subsample a fault segment into patches
├── make_ramp_flat.py       # Generate a ramp-flat fault segment file
├── transform4patch_general.py  # Compute 4 corners of each fault patch
├── compute_laplacian.py    # Laplacian smoothing matrix
├── readgmt.py              # Read GMT multi-segment files
├── model2d.py              # 2-D analytical fault models
├── flatten.py              # Utility: flatten nested lists
└── okada4py/               # Local okada4py package (compiled C++ extension)
```

## Usage

### 1. Create a fault model

Define a fault segment file (e.g. `work/gti/ramp-flat_seg.flt`) and discretise it into patches:

```bash
# Using make_ramp_flat.py (ramp-flat geometry)
python3 GTI/make_ramp_flat.py \
    --n0 -22 --e0 58 --strike -70 \
    --dips 70,15,5 --widths 5,15,15 --length 25 \
    --flt work/data/faults/ramp-flat.flt

# Or directly with seg2flt.py for any geometry
python3 GTI/seg2flt.py work/gti/my_seg.flt > work/data/faults/my_fault.flt
```

**Segment file format** (13 columns):

```
# i  x1(N)  x2(E)  x3(depth)  L  W  strike  dip  rake  lo  wo  alphal  alphaw
  1  -22.0   58.0    0.0       25  5  -70    70   90    1.0  1.0  1.0   1.0
```

### 2. Write an input file

```python
# work/gti/input_3dinv.py

maindir   = 'work/data/'
outputdir = 'work/gti/output/'

kernels = [interseismic(name='interseismic', date=0., inversion='kernel', sigmam=1.)]

flt = fault(name='ramp-flat', wdir=maindir+'faults/', model=['ramp-flat.flt'])

insar = [
    # Decomposed vertical component (av_los=0 → purely vertical projection)
    insarstack(network='vertical.xylos', wdir=maindir+'insar/',
               tmin=0., tmax=1., av_los=0., av_heading=0., samp=5),

    # Real InSAR with per-pixel incidence and heading (cols 4 & 5)
    insarstack(network='asc_envisat.xylos', wdir=maindir+'insar/',
               tmin=2003., tmax=2010., los=True, head=True, samp=3),
]

lambda1   = 1.      # Laplacian smoothing weight (larger = smoother)
plot_data = 'yes'   # Preview data + fault model before inversion
```

### 3. Run the inversion

```bash
python3 GTI/3dinversion.py work/gti/input_3dinv.py
```

Results are saved in `outputdir`:

| File | Content |
|------|---------|
| `insar_results.pdf` | Data / model / residual maps per InSAR network |
| `fault_kernel{i}.pdf` | Slip map view + 3-D view per kernel |
| `resolution_kernel{i}.pdf` | Resolution map per kernel |
| `gti_data_preview.pdf` | Pre-inversion data + fault model preview |

## Input file options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda1` | `1.` | Laplacian smoothing weight |
| `smoothing` | `'yes'` | Enable Laplacian smoothing |
| `constraint` | `'yes'` | Enable slip positivity constraint |
| `orb` | `'yes'` | Estimate orbital ramp per InSAR network |
| `base` | `'yes'` | Estimate baseline offset |
| `bndmin` | `[]` | Lower bounds `[gps, slip, orb, anti, base, mogi, edge]` |
| `bndmax` | `[]` | Upper bounds (same order) |
| `normodel` | `[1,1,500,1,1,1,1]` | Prior model uncertainty per parameter type |
| `xlim/ylim/zlim` | auto | Spatial filter on data loading |
| `plot_data` | `'no'` | Preview before inversion |
| `insarplot` | `'yes'` | Save InSAR result maps |
| `faultplot` | `'yes'` | Save fault slip + resolution maps |

## LOS projection conventions

`insarstack` accepts two modes:

```python
# Scalar average angles (decomposed or approximate geometry)
insarstack(..., av_los=23., av_heading=-76.)
# av_los=0  → purely vertical;  av_los=90 → purely horizontal

# Per-pixel geometry from data file columns 4 [& 5]
insarstack(..., los=True, head=True)   # file: x  y  los  look  heading
insarstack(..., los=True, av_heading=-76.)  # file: x  y  los  look
```

Projection formula (standard geodetic convention):
```
phi   = -90 - heading  [degrees from East, math convention]
theta =  90 - look     [elevation above horizontal]
projE = cos(theta) * cos(phi)
projN = cos(theta) * sin(phi)
projU = sin(theta)
```

## Author

Simon Daout
