# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

path, filename = os.path.split(__file__)
imgdir = os.path.join(path, "SBProfile_comparison_images") # Directory containing the reference

def test_spergelet():
    """ Generate some spergelets in python to check the c++ layer.
    """
    for nu in [-0.5, 0.5]:
        for (j,q) in [(0,0), (5,-5), (5,0), (5,5)]:
            filename = "spergelet_nu{:.2f}_j{}_q{}.fits".format(nu, j, q)
            savedImg = galsim.fits.read(os.path.join(imgdir, filename))
            dx = 0.2
            spergelet = galsim.Spergelet(nu=nu, scale_radius=1.0, j=j, q=q)
            myImg = spergelet.drawKImage(nx=63, ny=63, scale=0.2)[0]

            np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 5,
                err_msg="GSObject Spergelet disagrees with expected result")

def test_series_draw():
    """ Test that we can draw Series objects
    """
    a = galsim.SpergelSeries(nu=0.0, scale_radius=1.0, jmax=4)
    im = a.drawImage(nx=15, ny=15, scale=0.2)
    re, im = a.drawKImage(nx=15, ny=15, scale=0.2)
    
    b = galsim.SeriesConvolution(a, a)
    im = b.drawImage(nx=15, ny=15, scale=0.2)
    re, im = b.drawKImage(nx=15, ny=15, scale=0.2)
    
def test_series_gsobject_convolution():
    """ Test that we can convolve a Series and a GSObject.
    """
    a = galsim.SpergelSeries(nu=0.0, scale_radius=1.0, jmax=4)
    b = galsim.SeriesConvolution(a, galsim.Gaussian(fwhm=1))
    im = b.drawImage(nx=15, ny=15, scale=0.2)
    re, im = b.drawKImage(nx=15, ny=15, scale=0.2)
    
def test_spergelseries_decomposeA():
    """ Test that the SpergelSeries decomposition of the A matrix works.
    """
    pass

# def test_spergelseries_coeff():
#     """Check a_jq coefficient by generating a_jmn coefficients and forming the appropriate sum.
#     """
#     a = (galsim.SpergelSeries(nu=0.3, scale_radius=1.0, jmax=10)
#          .dilate(1.05)
#          .shear(e1=0.2, e2=-0.3)
#          .rotate(33*galsim.degrees))
#     ellip, phi0, Delta = a._decomposeA()
#     print ellip, phi0, Delta

#     def a_jmn(j,m,n):
#         ret = np.cos(2*(2*n-m)*phi0) + 1j*np.sin(2*(2*n-m)*phi0)
#         ret *= Delta**(j-m)*(1-Delta)**m*(-ellip)**m
#         return ret

#     indices = a.indices
#     coeffs = a.getCoeffs()
#     for idx, coeff in zip(indices, coeffs):
#         j, q = idx[2:4]
#         # loop through m in 0..j and n in 0..m, add up entries where 2n-m == q.
#         c = 0j
#         for m in range(j):
#             for n in range(m):
#                 if 2*n-m == q:
#                     c += a_jmn(j,m,n)
#         print j, q, coeff, c

def test_spergelseries():
    """Test that SpergelSeries converges to Spergel.
    """
    pass

def test_spergelseries_dilate():
    """ Check that SpergelSeries(nu=nu, scale_radius=1.0).dilate(1.1) gives the same results as
    SpergelSeries(nu=nu, scale_radius=1.1)
    """
    pass

if __name__ == "__main__":
    test_spergelet()
    test_series_draw()
    test_series_gsobject_convolution()
    test_spergelseries_decomposeA()
    test_spergelseries()
    test_spergelseries_dilate()
