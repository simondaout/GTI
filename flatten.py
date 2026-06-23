#!/usr/bin/env python3

import numpy as np
import math, sys


def flatten(sequence, recursive=True):
    """Returns a flat list containing all elements in the sequence in depth-first order.

    By default flattens recursively. Set recursive=False to flatten only one level.
    """
    result = []
    if recursive:
        stack = []
        i = iter(sequence)
        while True:
            try:
                e = next(i)
                if hasattr(e, "__iter__") and not isinstance(e, str):
                    stack.append(i)
                    i = iter(e)
                else:
                    result.append(e)
            except StopIteration:
                try:
                    i = stack.pop()
                except IndexError:
                    return result
    else:
        for e in sequence:
            if hasattr(e, "__iter__") and not isinstance(e, str):
                result.extend(e)
            else:
                result.append(e)
        return result
