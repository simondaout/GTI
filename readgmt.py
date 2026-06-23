#!/usr/bin/env python3

import numpy as np


class gmt:
    """Reads multi-segment GMT files."""

    def __init__(self, name, wdir, filename, color='black', width=2.):
        self.name = name
        self.wdir = wdir
        self.filename = filename
        self.color = color
        self.width = width

    def load(self):
        """Load GMT segments. Returns (x, y) as lists of float lists (one per segment)."""
        x = [[]]
        y = [[]]
        i = 0
        with open(self.wdir + self.filename, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('>'):
                    i += 1
                    x.append([])
                    y.append([])
                else:
                    cols = line.split()
                    x[i].append(float(cols[0]))
                    y[i].append(float(cols[1]))
        return x, y
