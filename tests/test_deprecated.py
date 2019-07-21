# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

from __future__ import print_function
import os
import sys
import numpy as np

import galsim
from galsim_test_helpers import *


def check_dep(f, *args, **kwargs):
    """Check that some function raises a GalSimDeprecationWarning as a warning, but not an error.
    """
    # Check that f() raises a warning, but not an error.
    with assert_warns(galsim.GalSimDeprecationWarning):
        res = f(*args, **kwargs)
    return res


@timer
def test_gsparams():
    check_dep(galsim.GSParams, allowed_flux_variation=0.90)
    check_dep(galsim.GSParams, range_division_for_extrema=50)
    check_dep(galsim.GSParams, small_fraction_of_flux=1.e-6)


@timer
def test_phase_psf():
    atm = galsim.Atmosphere(screen_size=10.0, altitude=0, r0_500=0.15, suppress_warning=True)
    psf = atm.makePSF(exptime=0.02, time_step=0.01, diam=1.1, lam=1000.0)
    check_dep(galsim.PhaseScreenPSF.__getattribute__, psf, "img")
    check_dep(galsim.PhaseScreenPSF.__getattribute__, psf, "finalized")

@timer
def test_interpolant():
    d = check_dep(galsim.Delta, tol=1.e-2)
    assert d.gsparams.kvalue_accuracy == 1.e-2
    assert d.tol == d.gsparams.kvalue_accuracy
    n = check_dep(galsim.Nearest, tol=1.e-2)
    assert n.gsparams.kvalue_accuracy == 1.e-2
    assert n.tol == n.gsparams.kvalue_accuracy
    s = check_dep(galsim.SincInterpolant, tol=1.e-2)
    assert s.gsparams.kvalue_accuracy == 1.e-2
    assert s.tol == s.gsparams.kvalue_accuracy
    l = check_dep(galsim.Linear, tol=1.e-2)
    assert l.gsparams.kvalue_accuracy == 1.e-2
    assert l.tol == l.gsparams.kvalue_accuracy
    c = check_dep(galsim.Cubic, tol=1.e-2)
    assert c.gsparams.kvalue_accuracy == 1.e-2
    assert c.tol == c.gsparams.kvalue_accuracy
    q = check_dep(galsim.Quintic, tol=1.e-2)
    assert q.gsparams.kvalue_accuracy == 1.e-2
    assert q.tol == q.gsparams.kvalue_accuracy
    l3 = check_dep(galsim.Lanczos, 3, tol=1.e-2)
    assert l3.gsparams.kvalue_accuracy == 1.e-2
    assert l3.tol == l3.gsparams.kvalue_accuracy
    ldc = check_dep(galsim.Lanczos, 3, False, tol=1.e-2)
    assert ldc.gsparams.kvalue_accuracy == 1.e-2
    assert ldc.tol == ldc.gsparams.kvalue_accuracy
    l8 = check_dep(galsim.Lanczos, 8, tol=1.e-2)
    assert l8.gsparams.kvalue_accuracy == 1.e-2
    assert l8.tol == l8.gsparams.kvalue_accuracy
    l11 = check_dep(galsim.Interpolant.from_name, 'lanczos11', tol=1.e-2)
    assert l11.gsparams.kvalue_accuracy == 1.e-2
    assert l11.tol == l11.gsparams.kvalue_accuracy

if __name__ == "__main__":
    test_gsparams()
    test_phase_psf()
    test_interpolant()
