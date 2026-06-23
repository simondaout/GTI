#!/usr/bin/env python3

import numpy as np
import math
import sys

# COMPUTE_LAPLACIAN   Generate Laplacian with respect to a set of points
#   LAP = COMPUTE_LAPLACIAN(X,Y,Z,N) takes in the coordinates of a set of
#   points that are randomly scattered on an unknown surface and generates
#   a discrete approximation of the Laplacian using the nearest N points.
#   X, Y, and Z are column vectors of the same length, and N must be a
#   positive integer.
#
#   By Andrew Kositsky
#   Copyright 2009-2010 Tectonics Observatory


def compute_laplacian(x, y, z, N):

    N_nearest = N
    n_points = x.size

    x = x.reshape(n_points, 1)
    y = y.reshape(n_points, 1)
    z = z.reshape(n_points, 1)
    data = np.hstack([x[:], y[:], z[:]])

    distance_matrix = np.zeros((n_points, n_points), dtype=float)
    Lap = np.zeros((n_points, n_points), dtype=float)

    for ii in range(n_points - 1):
        for jj in range(ii + 1, n_points):
            distance_matrix[ii, jj] = dist_fcn(data[ii, :], data[jj, :])

    distance_matrix = distance_matrix + distance_matrix.conj().T

    I_dist = np.argsort(distance_matrix, axis=1, kind='mergesort')
    NearestNPoints = I_dist[:, 1:1 + N_nearest]

    for ii in range(n_points):
        translated_data = data[NearestNPoints[ii, :], :] - np.tile(data[ii, :], (N_nearest, 1))

        u, s, vh = np.linalg.svd(translated_data, full_matrices=False)
        v = vh.T

        theta, r = cart2pol(np.dot(u[:, 0], s[0]), np.dot(u[:, 1], s[1]))

        theta_index = np.argsort(theta)
        thetat = theta[theta_index]

        NearestNPoints[ii, :] = NearestNPoints[ii, theta_index]
        rt = r[theta_index]

        delta_theta = np.diff(np.hstack([thetat, thetat[0]]), n=1, axis=-1)
        theta_plus = delta_theta
        theta_minus = np.hstack([delta_theta[-1], delta_theta[:-1]])
        r_bar = np.mean(rt)

        Theta_tot = calc_Theta_tot(theta_plus, theta_minus)

        for jj in range(N_nearest):
            Lap[ii, NearestNPoints[ii, jj]] = (
                (4 / r_bar) * (1 / Theta_tot) * (1 / rt[jj])
                * calc_Theta(theta_plus[jj], theta_minus[jj])
            )

        Lap[ii, ii] = -sum(Lap[ii, NearestNPoints[ii, :]])

    return Lap


def dist_fcn(x1, x2):
    return np.linalg.norm(x1 - x2)


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return theta, rho


def calc_Theta_tot(theta_plus, theta_minus):
    Theta_tot = 0
    for i in range(len(theta_plus)):
        Theta_tot += calc_Theta(theta_plus[i], theta_minus[i])
    return Theta_tot


def calc_Theta(theta_plus, theta_minus):
    return calc_half_Theta(theta_plus) + calc_half_Theta(theta_minus)


def calc_half_Theta(theta_diff):
    tol = 10e-10
    if abs(theta_diff) < tol:
        return 0
    return (1 - math.cos(theta_diff)) / math.sin(theta_diff)
