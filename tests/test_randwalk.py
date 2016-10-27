# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
import numpy
import os
import sys
import galsim

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


@timer
def test_randwalk_defaults():
    """
    Create two objects, each with three exposures. Save them to a MEDS file.
    Load the MEDS file. Compare the created objects with the one read by MEDS.
    """

    # try constructing with mostly defaults
    npoints=100
    hlr = 8.0
    rw=galsim.RandomWalk(npoints, hlr)

    assert rw.npoints==npoints,"expected npoints==%d" % (npoints, rw.npoints)
    assert rw.input_half_light_radius==hlr,\
        "expected hlr==%g, got %g" % (hlr, rw.input_half_light_radius)

    g=rw.gaussians
    ngauss=len(g)
    assert ngauss == npoints==npoints,"expected %d gaussians, got %d" % (npoints, ngauss)

@timer
def test_randwalk_valid_inputs():
    """
    Create two objects, each with three exposures. Save them to a MEDS file.
    Load the MEDS file. Compare the created objects with the one read by MEDS.
    """

    # try constructing with mostly defaults
    npoints=100
    hlr = 8.0
    nstep = 40
    flux = 3.5

    seed=35
    rng=galsim.UniformDeviate(seed)

    rw=galsim.RandomWalk(npoints, hlr, nstep=nstep, flux=flux, rng=rng)

    assert rw.npoints==npoints,"expected npoints==%d" % (npoints, rw.npoints)
    assert rw.input_half_light_radius==hlr,\
        "expected hlr==%g, got %g" % (hlr, rw.input_half_light_radius)
    assert rw.flux==flux,\
        "expected flux==%g, got %g" % (flux, rw.flux)

    assert rw.nstep==nstep,"expected nstep==%d" % (nstep, rw.nstep)
    assert rw.nstep==nstep,"expected nstep==%d" % (nstep, rw.nstep)


    g=rw.gaussians
    ngauss=len(g)
    assert ngauss == npoints==npoints,"expected %d gaussians, got %d" % (npoints, ngauss)

@timer
def test_randwalk_invalid_inputs():
    """
    Create two objects, each with three exposures. Save them to a MEDS file.
    Load the MEDS file. Compare the created objects with the one read by MEDS.
    """

    # try with rng wrong type

    npoints=100
    hlr = 8.0
    rng=37

    args=(npoints, hlr)
    kwargs={'rng':rng}
    np.testing.assert_raises(TypeError, galsim.RandomWalk, *args, **kwargs)


    # try sending wrong type for npoints
    npoints=[35]
    hlr = 8.0
    args=(npoints, hlr)
    np.testing.assert_raises(TypeError, galsim.RandomWalk, *args)

    # try sending wrong type for nstep
    npoints=100
    hlr=8.0
    nstep=[40]
    args=(npoints, hlr)
    kwargs={'nstep':nstep}
    np.testing.assert_raises(TypeError, galsim.RandomWalk, *args, **kwargs)

    # try sending wrong type for hlr
    npoints=100
    hlr=[1.5]
    args=(npoints, hlr)
    np.testing.assert_raises(TypeError, galsim.RandomWalk, *args)

    # try sending wrong type for flux
    npoints=100
    hlr=8.0
    flux=[3.5]
    args=(npoints, hlr)
    kwargs={'flux':flux}
    np.testing.assert_raises(TypeError, galsim.RandomWalk, *args, **kwargs)

    # send bad value for npoints

    npoints=-35
    hlr = 8.0
    args=(npoints, hlr)
    np.testing.assert_raises(ValueError, galsim.RandomWalk, *args)

    # try sending bad value for nstep
    npoints=100
    hlr=8.0
    nstep=-35
    args=(npoints, hlr)
    kwargs={'nstep':nstep}
    np.testing.assert_raises(ValueError, galsim.RandomWalk, *args, **kwargs)

    # try sending bad value for hlr
    npoints=100
    hlr=-1.5
    args=(npoints, hlr)
    np.testing.assert_raises(ValueError, galsim.RandomWalk, *args)

    # try sending wrong type for flux
    npoints=100
    hlr=8.0
    flux=-35.0
    args=(npoints, hlr)
    kwargs={'flux':flux}
    np.testing.assert_raises(ValueError, galsim.RandomWalk, *args, **kwargs)


if __name__ == "__main__":
    test_randwalk_defaults()
    test_randwalk_valid_inputs()
    test_randwalk_invalid_inputs()
