# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def test_hlr():
    """Test the calculateHLR method.
    """
    import time
    t1 = time.time()

    # Compare the calculation for a simple Gaussian.
    g1 = galsim.Gaussian(sigma=5, flux=1.7)

    print 'g1 native hlr = ',g1.half_light_radius
    print 'g1.calculateHLR = ',g1.calculateHLR()
    print 'nyquist scale = ',g1.nyquistScale()
    # These should be exactly equal.
    np.testing.assert_equal(g1.half_light_radius, g1.calculateHLR(),
                            err_msg="Gaussian.calculateHLR() returned wrong value.")

    # Check for a convolution of two Gaussians.  Should be equivalent, but now will need to 
    # do the calculation.
    g2 = galsim.Convolve(galsim.Gaussian(sigma=3, flux=1.3), galsim.Gaussian(sigma=4, flux=23))
    test_hlr = g2.calculateHLR()
    print 'g2.calculateHLR = ',test_hlr
    print 'ratio - 1 = ',test_hlr/g1.half_light_radius-1
    np.testing.assert_almost_equal(test_hlr/g1.half_light_radius, 1.0, decimal=1,
                                   err_msg="Gaussian.calculateHLR() is not accurate.")

    # The default scale is only accurate to around 1 dp.  Using scale = 0.1 is accurate to 3 dp.
    # Note: Nyquist scale is about 4.23 for this profile.
    test_hlr = g2.calculateHLR(scale=0.1)
    print 'g2.calculateHLR(scale=0.1) = ',test_hlr
    print 'ratio - 1 = ',test_hlr/g1.half_light_radius-1
    np.testing.assert_almost_equal(test_hlr/g1.half_light_radius, 1.0, decimal=3,
                                   err_msg="Gaussian.calculateHLR(scale=0.1) is not accurate.")

    # Finally, we don't expect this to be accurate, but make sure the code can handle having
    # more than half the flux in the central pixel.
    test_hlr = g2.calculateHLR(scale=15)
    print 'g2.calculateHLR(scale=15) = ',test_hlr
    print 'ratio - 1 = ',test_hlr/g1.half_light_radius-1
    np.testing.assert_almost_equal(test_hlr/g1.half_light_radius/10, 0.1, decimal=1,
                                   err_msg="Gaussian.calculateHLR(scale=15) is not accurate.")

    # Next, use an Exponential profile
    e1 = galsim.Exponential(scale_radius=5, flux=1.7)

    print 'e1 native hlr = ',e1.half_light_radius
    print 'e1.calculateHLR = ',e1.calculateHLR()
    print 'nyquist scale = ',e1.nyquistScale()
    # These should be exactly equal.
    np.testing.assert_equal(e1.half_light_radius, e1.calculateHLR(),
                            err_msg="Exponential.calculateHLR() returned wrong value.")

    # Check for a convolution with a delta function.  Should be equivalent, but now will need to 
    # do the calculation.
    e2 = galsim.Convolve(galsim.Exponential(scale_radius=5, flux=1.3), 
                         galsim.Gaussian(sigma=1.e-4, flux=23))
    test_hlr = e2.calculateHLR()
    print 'e2.calculateHLR = ',test_hlr
    print 'ratio - 1 = ',test_hlr/e1.half_light_radius-1
    np.testing.assert_almost_equal(test_hlr/e1.half_light_radius, 1.0, decimal=1,
                                   err_msg="Exponential.calculateHLR() is not accurate.")

    # The default scale is only accurate to around 1 dp.  Using scale = 0.1 is accurate to 3 dp.
    # Note: Nyquist scale is about 1.57 for this profile.
    # We can also decrease the size, which should still be accurate, but maybe a little faster.
    # Go a bit more that 2*hlr in units of the pixels.
    size = 2.1 * e1.half_light_radius / 0.1
    test_hlr = e2.calculateHLR(scale=0.1, size=size)
    print 'e2.calculateHLR(scale=0.1) = ',test_hlr
    print 'ratio - 1 = ',test_hlr/e1.half_light_radius-1
    np.testing.assert_almost_equal(test_hlr/e1.half_light_radius, 1.0, decimal=3,
                                   err_msg="Exponential.calculateHLR(scale=0.1) is not accurate.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_hlr()
