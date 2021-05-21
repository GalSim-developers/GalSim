# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import numpy as np
from functools import reduce
import galsim
from ..errors import GalSimRangeError

def trapz(func, min, max, points=10000):
    """Simple wrapper around 'numpy.trapz' to take function and limits as inputs.

    Example usage::

        >>> def func(x): return x**2
        >>> galsim.integ.trapz(func, 0, 1)
        0.33333333500033341
        >>> galsim.integ.trapz(func, 0, 1, 1e6)
        0.33333333333349996
        >>> galsim.integ.trapz(func, 0, 1, np.linspace(0, 1, 1e3))
        0.33333350033383402

    Parameters:
        func:       The function to be integrated.  y = func(x) should be valid.
        min:        The lower end of the integration bounds.
        max:        The upper end of the integration bounds.
        points:     If integer, the number of points to sample the integrand. If array-like, then
                    the points to sample the integrand at. [default: 1000].
    """
    from . import depr
    depr('galsim.integ.trapz', 2.3, 'galsim.integ.int1d')

    if not np.isscalar(points):
        if (np.max(points) > max) or (np.min(points) < min):
            raise GalSimRangeError("Points outside of specified range", points, min, max)
    elif int(points) != points:
        raise TypeError("npoints must be integer type or array")
    else:
        points = np.linspace(min, max, points)

    return np.trapz(func(points),points)

def midpt(fvals, x):
    """Midpoint rule for integration.

    Parameters:
        fvals:    Samples of the integrand
        x:        Locations at which the integrand was sampled.

    Returns:
        midpoint rule approximation of the integral.
    """
    from . import depr
    depr('galsim.integ.midpt', 2.3, 'np.trapz or galsim.trapz')

    dx = [x[1]-x[0]]
    dx.extend(0.5*(x[2:]-x[0:-2]))
    dx.append(x[-1]-x[-2])
    weighted_fvals = [w*f for w,f in zip(dx, fvals)]
    return reduce(lambda y,z:y+z, weighted_fvals)


galsim.integ.trapz = trapz
galsim.integ.midpt = midpt
