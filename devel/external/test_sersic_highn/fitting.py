# Copyright 2012-2014 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#

import numpy as np

def _calculateSvalues(xarr, yarr, sigma2=1.):
    """Calculates the intermediate S values required for basic linear regression.

    For a pedagogical description of the reasoning (but *not* the code used here, which is original)
    see, e.g., Numerical Recipes (Press et al 1992) Section 15.2.
    """
    if len(xarr) != len(yarr):
        raise ValueError("Input xarr and yarr differ in length!")
    if len(xarr) <= 1:
        raise ValueError("Input arrays must have 2 or more values elements.")

    S = len(xarr) / sigma2
    Sx = np.sum(xarr / sigma2)
    Sy = np.sum(yarr / sigma2)
    Sxx = np.sum(xarr * xarr / sigma2)
    Sxy = np.sum(xarr * yarr / sigma2)
    return (S, Sx, Sy, Sxx, Sxy)

def fitline(xarr, yarr):
    """Fit a line y = a + b * x to input x and y arrays by least squares.

    Returns the tuple (a, b, Var(a), Cov(a, b), Var(b)), after performing an internal estimate of
    measurement errors from the best-fitting model residuals.

    For a pedagogical description of the reasoning (but *not* the code used here, which is original)
    see, e.g., Numerical Recipes (Press et al 1992) Section 15.2.
    """
    # Get the S values (use default sigma2, best fit a and b still valid for stationary data)
    S, Sx, Sy, Sxx, Sxy = _calculateSvalues(xarr, yarr)

    # Get the best fit a and b
    Del = S * Sxx - Sx * Sx
    a = (Sxx * Sy - Sx * Sxy) / Del
    b = (S * Sxy - Sx * Sy) / Del

    # Use these to estimate the sigma^2 by residuals from the best-fitting model
    ymodel = a + b * xarr
    sigma2 = np.mean((yarr - ymodel)**2)
    # And use this to get model parameter error estimates
    var_a  = sigma2 * Sxx / Del
    cov_ab = - sigma2 * Sx / Del
    var_b  = sigma2 * S / Del
    return (a, b, var_a, cov_ab, var_b)
