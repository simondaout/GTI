#!/usr/bin/env python3

import numpy as np
import scipy.optimize as opt
import scipy.linalg as lst


def consInvert(A, b, sigmad=1, eq=[None, None], ineq=[None, None], bnd=None,
               cond=1.0e-10, iter=250, acc=1e-06, x0=None):
    """Solve the constrained inversion problem.

    Minimize:
        ||Ax - b||^2

    Subject to:
        Cx  = d      (equality constraints)
        Ex >= f      (inequality constraints)
        lb <= x <= ub  (bounds)

    Parameters
    ----------
    A : ndarray, shape (N, M)
    b : ndarray, shape (N,)
    sigmad : float or ndarray
        Data uncertainties (used as weights).
    eq : [C, d] or [None, None]
        Equality constraint matrix and vector.
    ineq : [E, f] or [None, None]
        Inequality constraint matrix and vector.
    bnd : list of (lb, ub) tuples, or None
        Bounds for each model parameter.
    """
    Ain = A
    bin = b

    if Ain.shape[0] != len(bin):
        raise ValueError('Incompatible dimensions for A and b')

    Cin, din = eq[0], eq[1]
    if Cin is not None:
        if Cin.shape[0] != len(din):
            raise ValueError('Incompatible dimensions for C and d')
        if Cin.shape[1] != Ain.shape[1]:
            raise ValueError('Incompatible dimensions for A and C')

    Ein, fin = ineq[0], ineq[1]
    if Ein is not None:
        if Ein.shape[0] != len(fin):
            raise ValueError('Incompatible shape for E and f')
        if Ein.shape[1] != Ain.shape[1]:
            raise ValueError('Incompatible shape for A and E')

    # Objective function and gradient
    _func   = lambda x: np.sum(((np.dot(Ain, x) - bin) / sigmad) ** 2)
    _fprime = lambda x: 2 * np.dot(Ain.T, (np.dot(Ain, x) - bin) / sigmad**2)

    # Equality constraints
    if Cin is not None:
        _f_cons      = lambda x: np.dot(Cin, x) - din
        _fprime_cons = lambda x: Cin

    # Inequality constraints
    if Ein is not None:
        _f_ieqcons      = lambda x: np.dot(Ein, x) - fin
        _fprime_ieqcons = lambda x: Ein

    # Initial guess
    if x0 is None:
        temp = lst.lstsq(Ain, bin, cond=cond)
        x0 = temp[0]

    # Solve
    if Cin is None:
        if Ein is None:
            if bnd is None:
                res = temp
            else:
                res = opt.fmin_slsqp(_func, x0, bounds=bnd, fprime=_fprime,
                                     iter=iter, full_output=True, acc=acc)
                if res[3] != 0:
                    print('Exit mode %d: %s' % (res[3], res[4]))
        else:
            if bnd is None:
                res = opt.fmin_slsqp(_func, x0, f_ieqcons=_f_ieqcons,
                                     fprime=_fprime, fprime_ieqcons=_fprime_ieqcons,
                                     iter=iter, full_output=True, acc=acc)
            else:
                res = opt.fmin_slsqp(_func, x0, f_ieqcons=_f_ieqcons, bounds=bnd,
                                     fprime=_fprime, fprime_ieqcons=_fprime_ieqcons,
                                     iter=iter, full_output=True, acc=acc)
            if res[3] != 0:
                print('Exit mode %d: %s' % (res[3], res[4]))
    else:
        if bnd is None:
            if Ein is None:
                res = opt.fmin_slsqp(_func, x0, f_eqcons=_f_cons, fprime=_fprime,
                                     fprime_eqcons=_fprime_cons,
                                     iter=iter, full_output=True, acc=acc)
            else:
                res = opt.fmin_slsqp(_func, x0, f_eqcons=_f_cons,
                                     f_ieqcons=_f_ieqcons, fprime=_fprime,
                                     fprime_eqcons=_fprime_cons,
                                     fprime_ieqcons=_fprime_ieqcons,
                                     iter=iter, full_output=True, acc=acc)
        else:
            if Ein is None:
                res = opt.fmin_slsqp(_func, x0, f_eqcons=_f_cons, bounds=bnd,
                                     fprime=_fprime, fprime_eqcons=_fprime_cons,
                                     iter=iter, full_output=True, acc=acc)
            else:
                res = opt.fmin_slsqp(_func, x0, f_eqcons=_f_cons, bounds=bnd,
                                     f_ieqcons=_f_ieqcons, fprime=_fprime,
                                     fprime_eqcons=_fprime_cons,
                                     fprime_ieqcons=_fprime_ieqcons,
                                     iter=iter, full_output=True, acc=acc)
        if res[3] != 0:
            print('Exit mode %d: %s' % (res[3], res[4]))

    return res[0]
