# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
import os
import platform

import galsim
from galsim_test_helpers import *

path, filename = os.path.split(__file__) # Get the path to this file for use below...

# For reference tests:
TESTDIR=os.path.join(path, "interpolant_comparison_files")

# Some arbitrary kx, ky k space values to test
KXVALS = np.array((1.30, 0.71, -4.30)) * np.pi / 2.
KYVALS = np.array((0.80, -0.02, -0.31,)) * np.pi / 2.


@pytest.fixture
def ref():
    # This reference image will be used in a number of tests below, so make it at the start.
    g1 = galsim.Gaussian(sigma = 3.1, flux=2.4).shear(g1=0.2,g2=0.1)
    g2 = galsim.Gaussian(sigma = 1.9, flux=3.1).shear(g1=-0.4,g2=0.3).shift(-0.3,0.5)
    g3 = galsim.Gaussian(sigma = 4.1, flux=1.6).shear(g1=0.1,g2=-0.1).shift(0.7,-0.2)
    final = g1 + g2 + g3
    ref_image = galsim.ImageD(128,128)
    scale = 0.4
    # The reference image was drawn with the old convention, which is now use_true_center=False
    final.drawImage(image=ref_image, scale=scale, method='sb', use_true_center=False)
    return final, ref_image


@timer
def test_roundtrip(ref):
    """Test round trip from Image to InterpolatedImage back to Image.
    """
    final, ref_image = ref
    # for each type, try to make an InterpolatedImage, and check that when we draw an image from
    # that InterpolatedImage that it is the same as the original
    ftypes = [np.float32, np.float64]
    ref_array = np.array([
        [0.01, 0.08, 0.07, 0.02],
        [0.13, 0.38, 0.52, 0.06],
        [0.09, 0.41, 0.44, 0.09],
        [0.04, 0.11, 0.10, 0.01] ])
    test_scale = 2.0

    for array_type in ftypes:
        image_in = galsim.Image(ref_array.astype(array_type))
        np.testing.assert_array_equal(
                ref_array.astype(array_type),image_in.array,
                err_msg="Array from input Image differs from reference array for type %s"%
                        array_type)
        test_array = np.zeros(ref_array.shape, dtype=array_type)

        for wcs in [ galsim.PixelScale(2.0),
                     galsim.JacobianWCS(2.1, 0.3, -0.4, 2.3),
                     galsim.AffineTransform(-0.3, 2.1, 1.8, 0.1, galsim.PositionD(0.3, -0.4)) ]:
            interp = galsim.InterpolatedImage(image_in, wcs=wcs)
            image_out = galsim.Image(test_array, wcs=wcs)
            interp.drawImage(image_out, method='no_pixel')
            np.testing.assert_almost_equal(
                    ref_array.astype(array_type),image_out.array,
                    err_msg="Output Image differs from reference for type %s, wcs %s"%
                            (array_type,wcs))

        # And using scale, which is equivalent to the first pass above (but hits a different
        # code path).
        interp = galsim.InterpolatedImage(image_in, scale=test_scale)
        image_out = galsim.Image(test_array, scale=test_scale)
        interp.drawImage(image_out, method='no_pixel')
        np.testing.assert_array_equal(
                ref_array.astype(array_type),image_out.array,
                err_msg="Output Image differs from reference for type %s, scale %s"%
                        (array_type,test_scale))

        gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
        interp2 = galsim.InterpolatedImage(image_in, scale=test_scale, gsparams=gsp)
        assert interp2 != interp
        assert interp2 == interp.withGSParams(gsp)
        assert interp2 == interp.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
        assert interp2.x_interpolant.gsparams == gsp
        assert interp2.k_interpolant.gsparams == gsp
        assert interp.x_interpolant.gsparams != gsp
        assert interp.k_interpolant.gsparams != gsp

        # Lanczos doesn't quite get the flux right.  Wrong at the 5th decimal place.
        # Gary says that's expected -- Lanczos isn't technically flux conserving.
        # He applied the 1st order correction to the flux, but expect to be wrong at around
        # the 10^-5 level.
        # Anyway, Quintic seems to be accurate enough.
        # And this is now the default, so no need to do anything special here.

        check_basic(interp, "InterpolatedImage", approx_maxsb=True)

        do_shoot(interp,image_out,"InterpolatedImage")

        # Test kvalues
        test_im = galsim.Image(16,16,scale=0.2)
        do_kvalue(interp,test_im,"InterpolatedImage")

        # Check picklability
        check_pickle(interp, lambda x: x.drawImage(method='no_pixel'))
        check_pickle(interp)

    # Test using a non-c-contiguous image  (.T transposes the image, making it Fortran order)
    image_T = galsim.Image(ref_array.astype(array_type).T)
    interp = galsim.InterpolatedImage(image_T, scale=test_scale)
    test_array = np.zeros(ref_array.T.shape, dtype=array_type)
    image_out = galsim.Image(test_array, scale=test_scale)
    interp.drawImage(image_out, method='no_pixel')
    np.testing.assert_array_equal(
            ref_array.T.astype(array_type),image_out.array,
            err_msg="Transposed array failed InterpolatedImage roundtrip.")
    check_basic(interp, "InterpolatedImage (Fortran ordering)", approx_maxsb=True)

    # Check that folding_threshold and maxk_threshold update stepk and maxk
    # Do this with the larger ref_image, since this one is too small to have any difference
    # from different folding_threshold values.
    scale = 0.3
    ii1 = galsim.InterpolatedImage(ref_image, scale=scale)
    gsp = galsim.GSParams(folding_threshold=1.e-4, maxk_threshold=1.e-4)
    ii2 = galsim.InterpolatedImage(ref_image, scale=scale, gsparams=gsp)
    assert ii2 == ii1.withGSParams(gsp)
    assert ii2.stepk != ii1.stepk
    assert ii2.maxk != ii1.maxk
    assert ii2.stepk == ii1.withGSParams(folding_threshold=1.e-4).stepk
    assert ii2.maxk == ii1.withGSParams(maxk_threshold=1.e-4).maxk


@timer
def test_interpolant():
    from scipy.special import sici
    # Test aspects of using the interpolants directly.

    # Test function for pickle tests
    im = galsim.Gaussian(sigma=4).drawImage()
    test_func = lambda x : (
        galsim.InterpolatedImage(im, x_interpolant=x).drawImage(method='no_pixel'))

    x = np.linspace(-10, 10, 141)

    # Delta
    d = galsim.Delta()
    print(repr(d.gsparams))
    print(repr(galsim.GSParams()))
    assert d.gsparams == galsim.GSParams()
    assert d.xrange == 0
    assert d.ixrange == 0
    assert np.isclose(d.krange, 2.*np.pi / d.gsparams.kvalue_accuracy)
    assert np.isclose(d.krange, 2.*np.pi * d._i.urange())
    assert d.positive_flux == 1
    assert d.negative_flux == 0
    print(repr(d))
    check_pickle(d, test_func)
    check_pickle(galsim.Delta())
    check_pickle(galsim.Interpolant.from_name('delta'))

    true_xval = np.zeros_like(x)
    true_xval[np.abs(x) < d.gsparams.kvalue_accuracy/2] = 1./d.gsparams.kvalue_accuracy
    np.testing.assert_allclose(d.xval(x), true_xval)
    np.testing.assert_allclose(d.kval(x), 1.)
    assert np.isclose(d.xval(x[12]), true_xval[12])
    assert np.isclose(d.kval(x[12]), 1.)

    # Nearest
    n = galsim.Nearest()
    assert n.gsparams == galsim.GSParams()
    assert n.xrange == 0.5
    assert n.ixrange == 1
    assert np.isclose(n.krange, 2. / n.gsparams.kvalue_accuracy)
    assert n.positive_flux == 1
    assert n.negative_flux == 0
    check_pickle(n, test_func)
    check_pickle(galsim.Nearest())
    check_pickle(galsim.Interpolant.from_name('nearest'))

    true_xval = np.zeros_like(x)
    true_xval[np.abs(x) < 0.5] = 1
    np.testing.assert_allclose(n.xval(x), true_xval)
    true_kval = np.sinc(x/2/np.pi)
    np.testing.assert_allclose(n.kval(x), true_kval)
    assert np.isclose(n.xval(x[12]), true_xval[12])
    assert np.isclose(n.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    # Most interpolants (not Delta above) conserve a constant (DC) flux.
    # This means input points separated by 1 pixel with any subpixel phase
    # will sum to 1.  The input x array has 7 phases, so the total sum is 7.
    print('Nearest sum = ',np.sum(n.xval(x)))
    assert np.isclose(np.sum(n.xval(x)), 7.0)

    # SincInterpolant
    s = galsim.SincInterpolant()
    assert s.gsparams == galsim.GSParams()
    assert np.isclose(s.xrange, 1./(np.pi * s.gsparams.kvalue_accuracy))
    assert s.ixrange == 2*np.ceil(s.xrange)
    assert np.isclose(s.krange, np.pi)
    assert np.isclose(s.krange, 2.*np.pi * s._i.urange())
    assert np.isclose(s.positive_flux, 3.18726437) # Empirical -- this is a regression test
    assert np.isclose(s.negative_flux, s.positive_flux-1., rtol=1.e-4)
    check_pickle(galsim.SincInterpolant())
    check_pickle(galsim.Interpolant.from_name('sinc'))

    true_xval = np.sinc(x)
    np.testing.assert_allclose(s.xval(x), true_xval)
    true_kval = np.zeros_like(x)
    true_kval[np.abs(x) < np.pi] = 1.
    np.testing.assert_allclose(s.kval(x), true_kval)
    assert np.isclose(s.xval(x[12]), true_xval[12])
    assert np.isclose(s.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    # This one would conserve dc flux, but we don't go out far enough.
    # At +- 10 pixels, it's only about 6.86
    print('Sinc sum = ',np.sum(s.xval(x)))
    assert np.isclose(np.sum(s.xval(x)), 7.0, rtol=0.02)

    # Linear
    l = galsim.Linear()
    assert l.gsparams == galsim.GSParams()
    assert l.xrange == 1.
    assert l.ixrange == 2
    assert np.isclose(l.krange, 2./l.gsparams.kvalue_accuracy**0.5)
    assert np.isclose(l.krange, 2.*np.pi * l._i.urange())
    assert l.positive_flux == 1
    assert l.negative_flux == 0
    check_pickle(l, test_func)
    check_pickle(galsim.Linear())
    check_pickle(galsim.Interpolant.from_name('linear'))

    true_xval = np.zeros_like(x)
    true_xval[np.abs(x) < 1] = 1. - np.abs(x[np.abs(x) < 1])
    np.testing.assert_allclose(l.xval(x), true_xval)
    true_kval = np.sinc(x/2/np.pi)**2
    np.testing.assert_allclose(l.kval(x), true_kval)
    assert np.isclose(l.xval(x[12]), true_xval[12])
    assert np.isclose(l.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    print('Linear sum = ',np.sum(l.xval(x)))
    assert np.isclose(np.sum(l.xval(x)), 7.0)

    # Cubic
    c = galsim.Cubic()
    assert c.gsparams == galsim.GSParams()
    assert c.xrange == 2.
    assert c.ixrange == 4
    assert np.isclose(c.krange, 2. * (3**1.5 / 8 / c.gsparams.kvalue_accuracy)**(1./3.))
    assert np.isclose(c.krange, 2.*np.pi * c._i.urange())
    assert np.isclose(c.positive_flux, 13./12.)
    assert np.isclose(c.negative_flux, 1./12.)
    check_pickle(c, test_func)
    check_pickle(galsim.Cubic())
    check_pickle(galsim.Interpolant.from_name('cubic'))

    true_xval = np.zeros_like(x)
    ax = np.abs(x)
    m = ax < 1
    true_xval[m] = 1. + ax[m]**2 * (1.5*ax[m]-2.5)
    m = (1 <= ax) & (ax < 2)
    true_xval[m] = -0.5 * (ax[m]-1) * (2.-ax[m])**2
    np.testing.assert_allclose(c.xval(x), true_xval)
    sx = np.sinc(x/2/np.pi)
    cx = np.cos(x/2)
    true_kval = sx**3 * (3*sx - 2*cx)
    np.testing.assert_allclose(c.kval(x), true_kval)
    assert np.isclose(c.xval(x[12]), true_xval[12])
    assert np.isclose(c.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    print('Cubic sum = ',np.sum(c.xval(x)))
    assert np.isclose(np.sum(c.xval(x)), 7.0)

    # Quintic
    q = galsim.Quintic()
    assert q.gsparams == galsim.GSParams()
    assert q.xrange == 3.
    assert q.ixrange == 6
    assert np.isclose(q.krange, 2. * (5**2.5 / 108 / q.gsparams.kvalue_accuracy)**(1./3.))
    assert np.isclose(q.krange, 2.*np.pi * q._i.urange())
    assert np.isclose(q.positive_flux, (13018561. / 11595672.) + (17267. / 14494590.) * 31**0.5)
    assert np.isclose(q.negative_flux, q.positive_flux-1.)
    check_pickle(q, test_func)
    check_pickle(galsim.Quintic())
    check_pickle(galsim.Interpolant.from_name('quintic'))

    true_xval = np.zeros_like(x)
    ax = np.abs(x)
    m = ax < 1.
    true_xval[m] = 1. + ax[m]**3 * (-95./12. + 23./2.*ax[m] - 55./12.*ax[m]**2)
    m = (1 <= ax) & (ax < 2)
    true_xval[m] = (ax[m]-1) * (2.-ax[m]) * (23./4. - 29./2.*ax[m] + 83./8.*ax[m]**2
                                             - 55./24.*ax[m]**3)
    m = (2 <= ax) & (ax < 3)
    true_xval[m] = (ax[m]-2) * (3.-ax[m])**2 * (-9./4. + 25./12.*ax[m] - 11./24.*ax[m]**2)
    np.testing.assert_allclose(q.xval(x), true_xval)
    sx = np.sinc(x/2/np.pi)
    cx = np.cos(x/2)
    true_kval = sx**5 * (sx*(55.-19./4. * x**2) + cx*(x**2/2. - 54.))
    np.testing.assert_allclose(q.kval(x), true_kval)
    assert np.isclose(q.xval(x[12]), true_xval[12])
    assert np.isclose(q.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    print('Quintic sum = ',np.sum(q.xval(x)))
    assert np.isclose(np.sum(q.xval(x)), 7.0)

    # Lanczos
    l3 = galsim.Lanczos(3)
    assert l3.gsparams == galsim.GSParams()
    assert l3.conserve_dc == True
    assert l3.n == 3
    assert l3.xrange == l3.n
    assert l3.ixrange == 2*l3.n
    assert np.isclose(l3.krange, 2.*np.pi * l3._i.urange())  # No analytic version for this one.
    print(l3.positive_flux, l3.negative_flux)
    assert np.isclose(l3.positive_flux, 1.1793639)  # Empirical -- this is a regression test
    assert np.isclose(l3.negative_flux, l3.positive_flux-1., rtol=1.e-4)
    check_pickle(l3, test_func)
    check_pickle(galsim.Lanczos(n=7, conserve_dc=False))
    check_pickle(galsim.Lanczos(3))
    check_pickle(galsim.Interpolant.from_name('lanczos7'))
    check_pickle(galsim.Interpolant.from_name('lanczos9F'))
    check_pickle(galsim.Interpolant.from_name('lanczos8T'))
    assert_raises(ValueError, galsim.Interpolant.from_name, 'lanczos3A')
    assert_raises(ValueError, galsim.Interpolant.from_name, 'lanczosF')
    assert_raises(ValueError, galsim.Interpolant.from_name, 'lanzos')

    # Note: 1-7 all have special case code, so check them. 8 uses the generic code.
    for n in [1, 2, 3, 4, 5, 6, 7, 8]:
        ln = galsim.Lanczos(n, conserve_dc=False)
        assert ln.conserve_dc == False
        assert ln.n == n
        true_xval = np.zeros_like(x)
        true_xval[np.abs(x) < n] = np.sinc(x[np.abs(x)<n]) * np.sinc(x[np.abs(x)<n]/n)
        np.testing.assert_allclose(ln.xval(x), true_xval, rtol=1.e-5, atol=1.e-10)
        assert np.isclose(ln.xval(x[12]), true_xval[12])

        # Lanczos notably does not conserve dc flux
        print('Lanczos(%s,conserve_dc=False) sum = '%n,np.sum(ln.xval(x)))

        # With conserve_dc=True, it does a bit better, but still only to 1.e-4 accuracy.
        lndc = galsim.Lanczos(n, conserve_dc=True)
        np.testing.assert_allclose(lndc.xval(x), true_xval, rtol=0.3, atol=1.e-10)
        print('Lanczos(%s,conserve_dc=True) sum = '%n,np.sum(lndc.xval(x)))
        assert np.isclose(np.sum(lndc.xval(x)), 7.0, rtol=1.e-4)

        # The math for kval (at least when conserve_dc=False) is complicated, but tractable.
        # It ends up using the Si function, which is in scipy as scipy.special.sici
        vp = n * (x/np.pi + 1)
        vm = n * (x/np.pi - 1)
        true_kval = ( (vm-1) * sici(np.pi*(vm-1))[0]
                     -(vm+1) * sici(np.pi*(vm+1))[0]
                     -(vp-1) * sici(np.pi*(vp-1))[0]
                     +(vp+1) * sici(np.pi*(vp+1))[0] ) / (2*np.pi)
        np.testing.assert_allclose(ln.kval(x), true_kval, rtol=1.e-4, atol=1.e-8)
        assert np.isclose(ln.kval(x[12]), true_kval[12])

    # Base class is invalid.
    assert_raises(NotImplementedError, galsim.Interpolant)

    # 2d arrays are invalid.
    x2d = np.ones((5,5))
    with assert_raises(galsim.GalSimValueError):
        q.xval(x2d)
    with assert_raises(galsim.GalSimValueError):
        q.kval(x2d)


@timer
def test_unit_integrals():
    # Test Interpolant.unit_integrals

    interps = [galsim.Delta(),
               galsim.Nearest(),
               galsim.SincInterpolant(),
               galsim.Linear(),
               galsim.Cubic(),
               galsim.Quintic(),
               galsim.Lanczos(3),
               galsim.Lanczos(3, conserve_dc=False),
               galsim.Lanczos(17),
              ]
    for interp in interps:
        print(str(interp))
        # Compute directly with int1d
        n = interp.ixrange//2 + 1
        direct_integrals = np.zeros(n)
        if isinstance(interp, galsim.Delta):
            # int1d doesn't handle this well.
            direct_integrals[0] = 1
        else:
            for k in range(n):
                direct_integrals[k] = galsim.integ.int1d(interp.xval, k-0.5, k+0.5)
        print('direct: ',direct_integrals)

        # Get from unit_integrals method (sometimes using analytic formulas)
        integrals = interp.unit_integrals()
        print('integrals: ',len(integrals),integrals)

        assert len(integrals) == n
        np.testing.assert_allclose(integrals, direct_integrals, atol=1.e-12)

        if n > 10:
            print('n>10 for ',repr(interp))
            integrals2 = interp.unit_integrals(max_len=10)
            assert len(integrals2) == 10
            np.testing.assert_equal(integrals2, integrals[:10])

    # Test making shorter versions before longer ones
    interp = galsim.Lanczos(11)
    short = interp.unit_integrals(max_len=5)
    long = interp.unit_integrals(max_len=10)
    med = interp.unit_integrals(max_len=8)
    full = interp.unit_integrals()

    assert len(full) > 10
    np.testing.assert_equal(short, full[:5])
    np.testing.assert_equal(med, full[:8])
    np.testing.assert_equal(long, full[:10])


@timer
def test_fluxnorm():
    """Test that InterpolatedImage class responds properly to instructions about flux normalization.
    """
    # define values
    # Note that im_lin_scale should be even, since the auto-sized drawImage() command always
    # produces an even-sized image.  If the even/odd-ness doesn't match then the interpolant will
    # come into play, and the exact checks will fail.
    im_lin_scale = 6 # make an image with this linear scale
    im_fill_value = 3. # fill it with this number
    im_scale = 1.3
    test_flux = 0.7

    # First, make some Image with some total flux value (sum of pixel values) and scale
    im = galsim.ImageF(im_lin_scale, im_lin_scale, scale=im_scale, init_value=im_fill_value)
    total_flux = im_fill_value*(im_lin_scale**2)
    np.testing.assert_equal(total_flux, im.array.sum(),
                            err_msg='Created array with wrong total flux')

    # Check that if we make an InterpolatedImage with flux normalization, it keeps that flux
    interp = galsim.InterpolatedImage(im) # note, flux normalization is the default
    np.testing.assert_almost_equal(total_flux, interp.flux, decimal=9,
                                   err_msg='Did not keep flux normalization')
    # Check that this is preserved when drawing
    im2 = interp.drawImage(scale = im_scale, method='no_pixel')
    np.testing.assert_almost_equal(total_flux, im2.array.sum(), decimal=9,
                                   err_msg='Drawn image does not have expected flux normalization')
    check_pickle(interp, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(interp)

    # Now make an InterpolatedImage but tell it sb normalization
    interp_sb = galsim.InterpolatedImage(im, normalization = 'sb')
    # Check that when drawing, the sum is equal to what we expect if the original image had been
    # surface brightness
    im3 = interp_sb.drawImage(scale = im_scale, method='no_pixel')
    np.testing.assert_almost_equal(total_flux*(im_scale**2)/im3.array.sum(), 1.0, decimal=6,
                                   err_msg='Did not use surface brightness normalization')
    # Check that when drawing with sb normalization, the sum is the same as the original
    im4 = interp_sb.drawImage(scale = im_scale, method='sb')
    np.testing.assert_almost_equal(total_flux/im4.array.sum(), 1.0, decimal=6,
                                   err_msg='Failed roundtrip for sb normalization')
    np.testing.assert_almost_equal(
            im4.array.max(), interp_sb.max_sb, 5,
            err_msg="InterpolatedImage max_sb did not match maximum pixel value")

    check_pickle(interp_sb, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(interp_sb)

    # Finally make an InterpolatedImage but give it some other flux value
    interp_flux = galsim.InterpolatedImage(im, flux=test_flux)
    # Check that it has that flux
    np.testing.assert_equal(test_flux, interp_flux.flux,
                            err_msg = 'InterpolatedImage did not use flux keyword')
    # Check that this is preserved when drawing
    im5 = interp_flux.drawImage(scale = im_scale, method='no_pixel')
    np.testing.assert_almost_equal(test_flux/im5.array.sum(), 1.0, decimal=6,
                                   err_msg = 'Drawn image does not reflect flux keyword')
    check_pickle(interp_flux, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(interp_flux)


@timer
def test_exceptions():
    """Test failure modes for InterpolatedImage class.
    """
    # Check that provided image has valid bounds
    with assert_raises(galsim.GalSimUndefinedBoundsError):
        galsim.InterpolatedImage(image=galsim.ImageF(scale=1.))

    # Scale must be set
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        galsim.InterpolatedImage(image=galsim.ImageF(5, 5))

    # Image must be real type (F or D)
    with assert_raises(galsim.GalSimValueError):
        galsim.InterpolatedImage(image=galsim.ImageI(5, 5, scale=1))

    # Image must have non-zero flux
    with assert_raises(galsim.GalSimValueError):
        galsim.InterpolatedImage(image=galsim.ImageF(5, 5, scale=1, init_value=0.))

    # Can't shoot II with SincInterpolant
    ii = galsim.InterpolatedImage(image=galsim.ImageF(5, 5, scale=1, init_value=1.),
                                  x_interpolant='sinc',
                                  # Use larger than normal kvalue_accuracy to avoid image being
                                  # really huge for SincInterpolant before exception is raised.
                                  gsparams=galsim.GSParams(kvalue_accuracy=1.e-3))
    with assert_raises(galsim.GalSimError):
        ii.drawImage(method='phot')
    with assert_raises(galsim.GalSimError):
        ii.shoot(n_photons=3)

    # Check types of inputs
    im = galsim.ImageF(5, 5, scale=1., init_value=10.)
    assert_raises(TypeError, galsim.InterpolatedImage, image=im.array)
    assert_raises(TypeError, galsim.InterpolatedImage, im, wcs=galsim.PixelScale(1.), scale=1.)
    assert_raises(TypeError, galsim.InterpolatedImage, im, wcs=1.)
    assert_raises(TypeError, galsim.InterpolatedImage, im, pad_image=im.array)
    assert_raises(TypeError, galsim.InterpolatedImage, im, noise_pad_size=33)
    assert_raises(TypeError, galsim.InterpolatedImage, im, noise_pad=33)

    # Other invalid values:
    assert_raises(ValueError, galsim.InterpolatedImage, im, normalization='invalid')
    assert_raises(ValueError, galsim.InterpolatedImage, im, x_interpolant='invalid')
    assert_raises(ValueError, galsim.InterpolatedImage, im, k_interpolant='invalid')
    assert_raises(ValueError, galsim.InterpolatedImage, im, pad_factor=0.)
    assert_raises(ValueError, galsim.InterpolatedImage, im, pad_factor=-1.)
    assert_raises(ValueError, galsim.InterpolatedImage, im, noise_pad_size=33, noise_pad=im.wcs)
    assert_raises(ValueError, galsim.InterpolatedImage, im, noise_pad_size=33, noise_pad=-1.)
    assert_raises(ValueError, galsim.InterpolatedImage, im, noise_pad_size=-33, noise_pad=1.)


@timer
def test_operations_simple(run_slow):
    """Simple test of operations on InterpolatedImage: shear, magnification, rotation, shifting."""
    # Make some nontrivial image that can be described in terms of sums and convolutions of
    # GSObjects.  We want this to be somewhat hard to describe, but should be at least
    # critically-sampled, so put in an Airy PSF.
    gal_flux = 1000.
    pix_scale = 0.03 # arcsec
    bulge_frac = 0.3
    bulge_hlr = 0.3 # arcsec
    bulge_e = 0.15
    bulge_pos_angle = 30.*galsim.degrees
    disk_hlr = 0.6 # arcsec
    disk_e = 0.5
    disk_pos_angle = 60.*galsim.degrees
    lam = 800              # nm    NB: don't use lambda - that's a reserved word.
    tel_diam = 2.4         # meters
    lam_over_diam = lam * 1.e-9 / tel_diam # radians
    lam_over_diam *= 206265  # arcsec
    im_size = 512

    # define subregion for comparison
    comp_region=30 # compare the central region of this linear size
    comp_bounds = galsim.BoundsI(1,comp_region,1,comp_region)
    comp_bounds = comp_bounds.shift(galsim.PositionI((im_size-comp_region)/2,
                                                     (im_size-comp_region)/2))

    bulge = galsim.Sersic(4, half_light_radius=bulge_hlr)
    bulge = bulge.shear(e=bulge_e, beta=bulge_pos_angle)
    disk = galsim.Exponential(half_light_radius = disk_hlr)
    disk = disk.shear(e=disk_e, beta=disk_pos_angle)
    gal = bulge_frac*bulge + (1.-bulge_frac)*disk
    gal = gal.withFlux(gal_flux)
    psf = galsim.Airy(lam_over_diam)
    obj = galsim.Convolve([gal, psf])
    im = obj.drawImage(scale=pix_scale)

    # Turn it into an InterpolatedImage with default param settings
    int_im = galsim.InterpolatedImage(im)

    # Shear it, and compare with expectations from GSObjects directly
    test_g1=-0.07
    test_g2=0.1
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    test_int_im = int_im.shear(g1=test_g1, g2=test_g2)
    ref_obj = obj.shear(g1=test_g1, g2=test_g2)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.drawImage(image=im, scale=pix_scale, method='no_pixel')
    ref_obj.drawImage(image=ref_im, scale=pix_scale)
    # define subregion for comparison
    im_sub = im.subImage(comp_bounds)
    ref_im_sub = ref_im.subImage(comp_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Sheared InterpolatedImage disagrees with reference')

    # Also test drawing into a larger image to test some of the indexing adjustments
    # in fillXImage.
    big_im = galsim.Image(2*im_size,2*im_size, scale=pix_scale)
    test_int_im.drawImage(image=big_im, method='no_pixel')
    big_comp_bounds = galsim.BoundsI(1,comp_region,1,comp_region)
    big_comp_bounds = big_comp_bounds.shift(galsim.PositionI((2*im_size-comp_region)/2,
                                                             (2*im_size-comp_region)/2))
    big_im_sub = big_im.subImage(big_comp_bounds)
    print('comp_bounds = ',comp_bounds)
    print('big_comp_bounds = ',big_comp_bounds)
    print('center = ',big_im[big_im.center])
    print('sub center = ',big_im_sub[big_im_sub.center])
    print('ref center = ',ref_im[ref_im.center])
    np.testing.assert_allclose(big_im_sub.array, ref_im_sub.array, rtol=0.01)

    # The check_pickle tests should all pass below, but the a == eval(repr(a)) check can take a
    # really long time, so we only do that if run_slow is True.
    irreprable = not run_slow
    check_pickle(test_int_im, lambda x: x.drawImage(nx=5, ny=5, scale=0.1, method='no_pixel'),
              irreprable=irreprable)
    check_pickle(test_int_im, irreprable=irreprable)

    # Magnify it, and compare with expectations from GSObjects directly
    test_mag = 1.08
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.magnify(test_mag)
    ref_obj = obj.magnify(test_mag)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.drawImage(image=im, scale=pix_scale, method='no_pixel')
    ref_obj.drawImage(image=ref_im, scale=pix_scale)
    # define subregion for comparison
    im_sub = im.subImage(comp_bounds)
    ref_im_sub = ref_im.subImage(comp_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Magnified InterpolatedImage disagrees with reference')
    check_pickle(test_int_im, lambda x: x.drawImage(nx=5, ny=5, scale=0.1, method='no_pixel'),
              irreprable=irreprable)
    check_pickle(test_int_im, irreprable=irreprable)

    # Lens it (shear and magnify), and compare with expectations from GSObjects directly
    test_g1 = -0.03
    test_g2 = -0.04
    test_mag = 0.74
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.lens(test_g1, test_g2, test_mag)
    ref_obj = obj.lens(test_g1, test_g2, test_mag)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.drawImage(image=im, scale=pix_scale, method='no_pixel')
    ref_obj.drawImage(image=ref_im, scale=pix_scale)
    # define subregion for comparison
    im_sub = im.subImage(comp_bounds)
    ref_im_sub = ref_im.subImage(comp_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Lensed InterpolatedImage disagrees with reference')
    check_pickle(test_int_im, lambda x: x.drawImage(nx=5, ny=5, scale=0.1, method='no_pixel'),
              irreprable=irreprable)
    check_pickle(test_int_im, irreprable=irreprable)

    # Rotate it, and compare with expectations from GSObjects directly
    test_rot_angle = 32.*galsim.degrees
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.rotate(test_rot_angle)
    ref_obj = obj.rotate(test_rot_angle)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.drawImage(image=im, scale=pix_scale, method='no_pixel')
    ref_obj.drawImage(image=ref_im, scale=pix_scale)
    # define subregion for comparison
    im_sub = im.subImage(comp_bounds)
    ref_im_sub = ref_im.subImage(comp_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Rotated InterpolatedImage disagrees with reference')
    check_pickle(test_int_im, lambda x: x.drawImage(nx=5, ny=5, scale=0.1, method='no_pixel'),
              irreprable=irreprable)
    check_pickle(test_int_im, irreprable=irreprable)

    # Shift it, and compare with expectations from GSObjects directly
    x_shift = -0.31
    y_shift = 0.87
    test_decimal=2 # in % difference, i.e. 2 means 1% agreement
    comp_region=30 # compare the central region of this linear size
    test_int_im = int_im.shift(x_shift, y_shift)
    ref_obj = obj.shift(x_shift, y_shift)
    # make large images
    im = galsim.ImageD(im_size, im_size)
    ref_im = galsim.ImageD(im_size, im_size)
    test_int_im.drawImage(image=im, scale=pix_scale, method='no_pixel')
    ref_obj.drawImage(image=ref_im, scale=pix_scale)
    # define subregion for comparison
    im_sub = im.subImage(comp_bounds)
    ref_im_sub = ref_im.subImage(comp_bounds)
    diff_im=im_sub-ref_im_sub
    rel = diff_im/im_sub
    zeros_arr = np.zeros((comp_region, comp_region))
    # require relative difference to be smaller than some amount
    np.testing.assert_array_almost_equal(rel.array, zeros_arr,
        test_decimal,
        err_msg='Shifted InterpolatedImage disagrees with reference')
    check_pickle(test_int_im, lambda x: x.drawImage(nx=5, ny=5, scale=0.1, method='no_pixel'),
              irreprable=irreprable)
    check_pickle(test_int_im, irreprable=irreprable)


@timer
def test_operations():
    """Test of operations on complicated InterpolatedImage: shear, magnification, rotation,
    shifting.
    """
    test_decimal = 3

    # Make some nontrivial image
    im = galsim.fits.read('./real_comparison_images/test_images.fits') # read in first real galaxy
                                                                       # in test catalog
    int_im = galsim.InterpolatedImage(im)
    orig_mom = im.FindAdaptiveMom()

    # Magnify by some amount and make sure change is as expected
    mu = 0.92
    new_int_im = int_im.magnify(mu)
    test_im = galsim.ImageF(im.bounds)
    new_int_im.drawImage(image = test_im, scale = im.scale, method='no_pixel')
    new_mom = test_im.FindAdaptiveMom()
    np.testing.assert_almost_equal(new_mom.moments_sigma/np.sqrt(mu),
        orig_mom.moments_sigma, test_decimal,
        err_msg = 'Size of magnified InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.observed_shape.e1, orig_mom.observed_shape.e1,
        test_decimal,
        err_msg = 'e1 of magnified InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.observed_shape.e2, orig_mom.observed_shape.e2,
        test_decimal,
        err_msg = 'e2 of magnified InterpolatedImage from HST disagrees with expectations')
    check_pickle(new_int_im, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(new_int_im)

    # Shift, make sure change in moments is as expected
    x_shift = 0.92
    y_shift = -0.16
    new_int_im = int_im.shift(x_shift, y_shift)
    test_im = galsim.ImageF(im.bounds)
    new_int_im.drawImage(image = test_im, scale = im.scale, method='no_pixel')
    new_mom = test_im.FindAdaptiveMom()
    np.testing.assert_almost_equal(new_mom.moments_sigma, orig_mom.moments_sigma,
        test_decimal,
        err_msg = 'Size of shifted InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.moments_centroid.x-x_shift, orig_mom.moments_centroid.x,
        test_decimal,
        err_msg = 'x centroid of shifted InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.moments_centroid.y-y_shift, orig_mom.moments_centroid.y,
        test_decimal,
        err_msg = 'y centroid of shifted InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.observed_shape.e1, orig_mom.observed_shape.e1,
        test_decimal,
        err_msg = 'e1 of shifted InterpolatedImage from HST disagrees with expectations')
    np.testing.assert_almost_equal(new_mom.observed_shape.e2, orig_mom.observed_shape.e2,
        test_decimal,
        err_msg = 'e2 of shifted InterpolatedImage from HST disagrees with expectations')
    check_pickle(new_int_im, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(new_int_im)


@timer
def test_uncorr_padding(run_slow):
    """Test for uncorrelated noise padding of InterpolatedImage."""
    # Set up some defaults: use weird image sizes / shapes and noise variances.
    decimal_precise=5
    decimal_coarse=2
    orig_nx = 147
    orig_ny = 174
    noise_var = 1.73
    big_nx = 519
    big_ny = 482
    orig_seed = 151241

    # first, make a noise image
    orig_img = galsim.ImageF(orig_nx, orig_ny, scale=1.)
    gd = galsim.GaussianDeviate(orig_seed, mean=0., sigma=np.sqrt(noise_var))
    orig_img.addNoise(galsim.DeviateNoise(gd))

    # make it into an InterpolatedImage with some zero-padding
    # (note that default is zero-padding, by factors of several)
    int_im = galsim.InterpolatedImage(orig_img)
    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny)
    int_im.drawImage(big_img, scale=1., method='no_pixel')
    # check that variance is diluted by expected amount - should be exact, so check precisely!
    # Note that this only works if the big image has the same even/odd-ness in the two sizes.
    # Otherwise the center of the original image will fall between pixels in the big image.
    # Then the variance will be smoothed somewhat by the interpolant.
    big_var_expected = np.var(orig_img.array)*float(orig_nx*orig_ny)/(big_nx*big_ny)
    np.testing.assert_almost_equal(
        np.var(big_img.array), big_var_expected, decimal=decimal_precise,
        err_msg='Variance not diluted by expected amount when zero-padding')
    if run_slow:
        check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im)

    # make it into an InterpolatedImage with noise-padding
    int_im = galsim.InterpolatedImage(orig_img, noise_pad=noise_var,
                                      noise_pad_size=max(big_nx,big_ny),
                                      rng = galsim.GaussianDeviate(orig_seed))
    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny)
    int_im.drawImage(big_img, scale=1., method='no_pixel')
    # check that variance is same as original - here, we cannot be too precise because the padded
    # region is not huge and the comparison will be, well, noisy.
    print('measured var = ',np.var(big_img.array))
    np.testing.assert_almost_equal(
        np.var(big_img.array), noise_var, decimal=decimal_coarse,
        err_msg='Variance not correct after padding image with noise')
    if run_slow:
        check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im)
    else:
        # On pytest runs, use a smaller noise_pad_size for the pickle tests so it doesn't take
        # so long to serialize.
        int_im = galsim.InterpolatedImage(orig_img, noise_pad=noise_var,
                                          pad_factor=1,
                                          noise_pad_size=max(orig_nx+10,orig_ny+10),
                                          rng = galsim.GaussianDeviate(orig_seed))
        check_pickle(int_im)

    # check that if we pass in a RNG, it is actually used to pad with the same noise field
    # basically, redo all of the above steps and draw into a new image, make sure it's the same as
    # previous.
    int_im = galsim.InterpolatedImage(orig_img, noise_pad=noise_var,
                                      noise_pad_size=max(big_nx,big_ny),
                                      rng = galsim.GaussianDeviate(orig_seed))
    big_img_2 = galsim.ImageF(big_nx, big_ny)
    int_im.drawImage(big_img_2, scale=1., method='no_pixel')
    np.testing.assert_array_almost_equal(
        big_img_2.array, big_img.array, decimal=decimal_precise,
        err_msg='Cannot reproduce noise-padded image with same choice of seed')
    if run_slow:
        check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im)

    # Finally check inputs: what if we give it an input variance that is neg?  A list?
    with assert_raises(ValueError):
        galsim.InterpolatedImage(orig_img, noise_pad=-1., noise_pad_size=20)


@timer
def test_pad_image(run_slow):
    """Test padding an InterpolatedImage with a pad_image."""
    decimal=2  # all are coarse, since there are slight changes from odd/even centering issues.
    noise_sigma = 1.73
    noise_var = noise_sigma**2
    orig_seed = 12345
    rng = galsim.BaseDeviate(orig_seed)
    noise = galsim.GaussianNoise(rng, sigma=noise_sigma)

    # make the original image
    orig_nx = 64
    orig_ny = 64
    orig_img = galsim.ImageF(orig_nx, orig_ny, scale=1.)
    galsim.Exponential(scale_radius=1.7,flux=1000).drawImage(orig_img, method='no_pixel')
    orig_img.addNoise(noise)
    orig_img.setCenter(0,0)

    # We'll draw into a larger image for the tests
    pad_factor = 4
    big_nx = pad_factor*orig_nx
    big_ny = pad_factor*orig_ny
    big_img = galsim.ImageF(big_nx, big_ny, scale=1.)
    big_img.setCenter(0,0)

    # Use a few different kinds of shapes for that padding.
    for (pad_nx, pad_ny) in [ (160,160), (179,191), (256,256), (305, 307) ]:

        # make the pad_image
        pad_img = galsim.ImageF(pad_nx, pad_ny, scale=1.)
        pad_img.addNoise(noise)
        pad_img.setCenter(0,0)

        # make an interpolated image padded with the pad_image, and outside of that
        int_im = galsim.InterpolatedImage(orig_img, pad_image=pad_img, use_true_center=False)

        # draw into the larger image
        int_im.drawImage(big_img, use_true_center=False, method='no_pixel')

        # check that variance is diluted by expected amount
        # Note -- we don't use np.var, since that computes the variance relative to the
        # actual mean value.  We just want sum(I^2)/Npix relative to the nominal I=0 value.
        var1 = np.sum(orig_img.array**2)
        if pad_nx > big_nx and pad_ny > big_ny:
            var2 = np.sum(pad_img[big_img.bounds].array**2)
        else:
            var2 = np.sum(pad_img.array**2)
        var2 -= np.sum(pad_img[orig_img.bounds].array**2)
        var_expected = (var1 + var2) / (big_nx*big_ny)
        big_img.setCenter(0,0)
        np.testing.assert_almost_equal(
            np.mean(big_img.array**2), var_expected, decimal=decimal,
            err_msg='Variance not correct when padding with image')
        if run_slow:
            check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
            check_pickle(int_im)

        if pad_nx < big_nx and pad_ny < big_ny:
            # now also pad with noise_pad outside of the pad_image
            int_im = galsim.InterpolatedImage(orig_img, pad_image=pad_img, noise_pad=noise_var/2,
                                              noise_pad_size=max(big_nx,big_ny),
                                              rng=rng, use_true_center=False)
            int_im.drawImage(big_img, use_true_center=False, method='no_pixel')

            var3 = (noise_var/2) * float(big_nx*big_ny - pad_nx*pad_ny)
            var_expected = (var1 + var2 + var3) / (big_nx*big_ny)
            np.testing.assert_almost_equal(
                np.mean(big_img.array**2), var_expected, decimal=decimal,
                err_msg='Variance not correct after padding with image and extra noise')
            if run_slow:
                check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
                check_pickle(int_im)


@timer
def test_corr_padding(run_slow):
    """Test for correlated noise padding of InterpolatedImage."""
    # Set up some defaults for tests.
    decimal_precise=4
    decimal_coarse=2
    imgfile = 'fits_files/blankimg.fits'
    orig_nx = 187
    orig_ny = 164
    big_nx = 319
    big_ny = 322
    orig_seed = 151241

    # Read in some small image of a noise field from HST.
    im = galsim.fits.read(imgfile)
    # Make a CorrrlatedNoise out of it.
    cn = galsim.CorrelatedNoise(im, galsim.BaseDeviate(orig_seed))

    # first, make a noise image
    orig_img = galsim.ImageF(orig_nx, orig_ny, scale=1.)
    orig_img.addNoise(cn)

    # make it into an InterpolatedImage with some zero-padding
    # (note that default is zero-padding, by factors of several)
    int_im = galsim.InterpolatedImage(orig_img)
    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny)
    int_im.drawImage(big_img, scale=1., method='no_pixel')
    # check that variance is diluted by expected amount - should be exact, so check precisely!
    big_var_expected = np.var(orig_img.array)*float(orig_nx*orig_ny)/(big_nx*big_ny)
    np.testing.assert_almost_equal(np.var(big_img.array), big_var_expected, decimal=decimal_precise,
        err_msg='Variance not diluted by expected amount when zero-padding')
    if run_slow:
        check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im)

    # make it into an InterpolatedImage with noise-padding
    int_im = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                      noise_pad=im, noise_pad_size=max(big_nx,big_ny))

    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny, scale=1.)
    int_im.drawImage(big_img, method='no_pixel')
    # check that variance is same as original - here, we cannot be too precise because the padded
    # region is not huge and the comparison will be, well, noisy.
    np.testing.assert_almost_equal(np.var(big_img.array), np.var(orig_img.array),
        decimal=decimal_coarse,
        err_msg='Variance not correct after padding image with correlated noise')
    if run_slow:
        check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im)

    # Check the option to read the image from a file and also cache the resulting noise object
    int_im2 = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                       noise_pad=imgfile, noise_pad_size=max(big_nx,big_ny))
    big_img2 = galsim.ImageF(big_nx, big_ny)
    big_img2 = int_im2.drawImage(big_img.copy(), method='no_pixel')
    np.testing.assert_array_equal(big_img2.array, big_img.array)
    # Repeating the same file should use the cached value.
    int_im3 = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                       noise_pad=imgfile, noise_pad_size=max(big_nx,big_ny))
    big_img3 = int_im3.drawImage(big_img.copy(), method='no_pixel')
    np.testing.assert_array_equal(big_img3.array, big_img.array)
    # Unless we tell it not to.  (Functionality is the same, but less efficient.)
    int_im4 = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                       noise_pad=imgfile, noise_pad_size=max(big_nx,big_ny),
                                       use_cache=False)
    big_img4 = int_im4.drawImage(big_img.copy(), method='no_pixel')
    np.testing.assert_array_equal(big_img4.array, big_img.array)
    # If we don't provide the rng for a cached noise object, it keeps using the one it had.
    int_im5 = galsim.InterpolatedImage(orig_img,
                                       noise_pad=imgfile, noise_pad_size=max(big_nx,big_ny))
    big_img5 = int_im5.drawImage(big_img.copy(), method='no_pixel')
    assert not np.all(big_img5.array == big_img.array)
    np.testing.assert_almost_equal(np.var(big_img5.array), np.var(orig_img.array),
        decimal=decimal_coarse,
        err_msg='Variance not correct using cached noise, without resetting rng')

    # draw into a larger image
    big_img = galsim.ImageF(big_nx, big_ny)
    int_im.drawImage(big_img, scale=1., method='no_pixel')
    # check that variance is same as original - here, we cannot be too precise because the padded
    # region is not huge and the comparison will be, well, noisy.
    np.testing.assert_almost_equal(np.var(big_img.array), np.var(orig_img.array),
        decimal=decimal_coarse,
        err_msg='Variance not correct after padding image with correlated noise')
    if run_slow:
        check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im)

    # check that if we pass in a RNG, it is actually used to pad with the same noise field
    # basically, redo all of the above steps and draw into a new image, make sure it's the same as
    # previous.
    int_im = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                      noise_pad=cn, noise_pad_size=max(big_nx,big_ny))
    big_img_2 = galsim.ImageF(big_nx, big_ny)
    int_im.drawImage(big_img_2, scale=1., method='no_pixel')
    np.testing.assert_array_almost_equal(big_img_2.array, big_img.array, decimal=decimal_precise,
        err_msg='Cannot reproduce correlated noise-padded image with same choice of seed')
    if run_slow:
        check_pickle(int_im, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im)

    # Finally, check inputs:
    # what if we give it a screwy way of defining the image padding?
    with assert_raises(ValueError):
        galsim.InterpolatedImage(orig_img, noise_pad=-1., noise_pad_size=20)

    # also, check that whether we give it a string, image, or cn, it gives the same noise field
    # (given the same random seed)
    infile = 'fits_files/blankimg.fits'
    inimg = galsim.fits.read(infile)
    incf = galsim.CorrelatedNoise(inimg, galsim.GaussianDeviate()) # input RNG will be ignored below
    int_im2 = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                       noise_pad=inimg, noise_pad_size=max(big_nx,big_ny))
    int_im3 = galsim.InterpolatedImage(orig_img, rng=galsim.GaussianDeviate(orig_seed),
                                       noise_pad=incf, noise_pad_size=max(big_nx,big_ny))
    big_img2 = galsim.ImageF(big_nx, big_ny)
    big_img3 = galsim.ImageF(big_nx, big_ny)
    int_im2.drawImage(big_img2, scale=1., method='no_pixel')
    int_im3.drawImage(big_img3, scale=1., method='no_pixel')
    np.testing.assert_equal(big_img2.array, big_img3.array,
                            err_msg='Diff ways of specifying correlated noise give diff answers')
    if run_slow:
        check_pickle(int_im2, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im3, lambda x: x.drawImage(nx=200, ny=200, scale=1, method='no_pixel'))
        check_pickle(int_im2)
        check_pickle(int_im3)


@timer
def test_realspace_conv(run_slow):
    """Test that real-space convolution of an InterpolatedImage matches the FFT result
    """
    # Note: It is not usually a good idea to use real-space convolution with an InterpolatedImage.
    # It will almost always be much slower than the FFT convolution.  So it's probably only
    # a good idea if the image is very small and/or you absolutely need to avoid the ringing
    # that can show up in FFT convolutions.
    # That said, we still need to make sure the code is correct.  Especially since it
    # didn't used to be, as reported in issue #432.  So, here goes.

    # set up image scale and size
    raw_scale = 0.23
    raw_size = 15

    # We draw onto a smaller image so the unit test doesn't take forever!
    target_scale = 0.7
    target_size = 3

    gal = galsim.Exponential(flux=1.7, half_light_radius=1.2)
    gal_im = gal.drawImage(scale=raw_scale, nx=raw_size, ny=raw_size, method='no_pixel')

    psf1 = galsim.Gaussian(flux=1, half_light_radius=0.77)
    psf_im = psf1.drawImage(scale=raw_scale, nx=raw_size, ny=raw_size, method='no_pixel')

    if run_slow:
        interp_list = ['linear', 'cubic', 'quintic', 'lanczos3', 'lanczos5', 'lanczos7']
    else:
        interp_list = ['linear', 'cubic', 'quintic']
    for interp in interp_list:
        # Note 1: The Lanczos interpolants pass these tests just fine.  They just take a long
        # time to run, even with the small images we are working with.  So skip them for regular
        # unit testing.  Developers working on this should re-enable those while testing.

        # Note 2: I couldn't get 'nearest' to pass the tests.  Specifically the im3 == im4 test.
        # I don't know whether there is a bug in the Nearest class functions (seems unlikely since
        # they are so simple) or in the real-space convolver or if the nature of the Nearest
        # interpolation (with its very large extent in k-space and hard edges in real space) is
        # such that we don't actually expect the test to pass.  Anyway, it also takes a very long
        # time to run (before failing), so it's probably not a good idea to use it for
        # real-space convolution anyway.

        print('interp = ',interp)

        gal = galsim.InterpolatedImage(gal_im, x_interpolant=interp)

        # First convolve with a Gaussian:
        psf = psf1
        c1 = galsim.Convolve([gal,psf], real_space=True)
        c2 = galsim.Convolve([gal,psf], real_space=False)

        im1 = c1.drawImage(scale=target_scale, nx=target_size, ny=target_size, method='no_pixel')
        im2 = c2.drawImage(scale=target_scale, nx=target_size, ny=target_size, method='no_pixel')
        np.testing.assert_array_almost_equal(im1.array, im2.array, 5)

        # Now make the psf also an InterpolatedImage:
        psf=galsim.InterpolatedImage(psf_im, x_interpolant=interp, flux=1)
        c3 = galsim.Convolve([gal,psf], real_space=True)
        c4 = galsim.Convolve([gal,psf], real_space=False)

        im3 = c3.drawImage(scale=target_scale, nx=target_size, ny=target_size, method='no_pixel')
        im4 = c4.drawImage(scale=target_scale, nx=target_size, ny=target_size, method='no_pixel')
        np.testing.assert_array_almost_equal(im1.array, im3.array, 2)
        # Note: only 2 d.p. since the interpolated image version of the psf is really a different
        # profile from the original.  Especially for the lower order interpolants.  So we don't
        # expect these images to be equal to many decimal places.
        np.testing.assert_array_almost_equal(im3.array, im4.array, 5)

        check_pickle(c1, lambda x: x.xValue(1.123,-0.179))
        check_pickle(c3, lambda x: x.xValue(0.439,4.234))
        check_pickle(c1)
        check_pickle(c3)


@timer
def test_Cubic_ref(ref):
    """Test use of Cubic interpolant against some reference values
    """
    final, ref_image = ref
    interp = galsim.Cubic()
    scale = 0.4
    testobj = galsim.InterpolatedImage(ref_image, x_interpolant=interp, scale=scale,
                                       normalization='sb')
    testKvals = np.zeros(len(KXVALS))
    # Make test kValues
    for i in range(len(KXVALS)):
        posk = galsim.PositionD(KXVALS[i], KYVALS[i])
        testKvals[i] = np.abs(testobj.kValue(posk))
    # Compare with saved array
    refKvals = np.loadtxt(os.path.join(TESTDIR, "absfKCubic_test.txt"))
    print('ref = ',refKvals)
    print('test = ',testKvals)
    print('kValue(0) = ',testobj.kValue(galsim.PositionD(0.,0.)))
    np.testing.assert_array_almost_equal(
            refKvals/testKvals, 1., 5,
            err_msg="kValues do not match reference values for Cubic interpolant.")

    check_pickle(testobj, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(testobj)


@timer
def test_Quintic_ref(ref):
    """Test use of Quintic interpolant against some reference values
    """
    final, ref_image = ref
    interp = galsim.Quintic()
    scale = 0.4
    testobj = galsim.InterpolatedImage(ref_image, x_interpolant=interp, scale=scale,
                                       normalization='sb')
    testKvals = np.zeros(len(KXVALS))
    # Make test kValues
    for i in range(len(KXVALS)):
        posk = galsim.PositionD(KXVALS[i], KYVALS[i])
        testKvals[i] = np.abs(testobj.kValue(posk))
    # Compare with saved array
    refKvals = np.loadtxt(os.path.join(TESTDIR, "absfKQuintic_test.txt"))
    print('ref = ',refKvals)
    print('test = ',testKvals)
    np.testing.assert_array_almost_equal(
            refKvals/testKvals, 1., 5,
            err_msg="kValues do not match reference values for Quintic interpolant.")

    check_pickle(testobj, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(testobj)


@timer
def test_Lanczos5_ref(ref):
    """Test use of Lanczos5 interpolant against some reference values
    """
    final, ref_image = ref
    interp = galsim.Lanczos(5, conserve_dc=False)
    scale = 0.4
    testobj = galsim.InterpolatedImage(ref_image, x_interpolant=interp, scale=scale,
                                       normalization='sb')
    testKvals = np.zeros(len(KXVALS))
    # Make test kValues
    for i in range(len(KXVALS)):
        posk = galsim.PositionD(KXVALS[i], KYVALS[i])
        testKvals[i] = np.abs(testobj.kValue(posk))
    # Compare with saved array
    refKvals = np.loadtxt(os.path.join(TESTDIR, "absfKLanczos5_test.txt"))
    print('ref = ',refKvals)
    print('test = ',testKvals)
    np.testing.assert_array_almost_equal(
            refKvals/testKvals, 1., 5,
            err_msg="kValues do not match reference values for Lanczos-5 interpolant.")

    check_pickle(testobj, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(testobj)


@timer
def test_Lanczos7_ref(ref):
    """Test use of Lanczos7 interpolant against some reference values
    """
    final, ref_image = ref
    interp = galsim.Lanczos(7, conserve_dc=False)
    scale = 0.4
    testobj = galsim.InterpolatedImage(ref_image, x_interpolant=interp, scale=scale,
                                       normalization='sb')
    testKvals = np.zeros(len(KXVALS))
    # Make test kValues
    for i in range(len(KXVALS)):
        posk = galsim.PositionD(KXVALS[i], KYVALS[i])
        testKvals[i] = np.abs(testobj.kValue(posk))
    # Compare with saved array
    refKvals = np.loadtxt(os.path.join(TESTDIR, "absfKLanczos7_test.txt"))
    print('ref = ',refKvals)
    print('test = ',testKvals)
    np.testing.assert_array_almost_equal(
            refKvals/testKvals, 1., 5,
            err_msg="kValues do not match reference values for Lanczos-7 interpolant.")

    check_pickle(testobj, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(testobj)


@timer
def test_conserve_dc():
    """Test that the conserve_dc option for Lanczos does so.
    Note: the idea of conserving flux is a bit of a misnomer.  No interpolant does so
    precisely in general.  What we are really testing is that a flat background input
    image has a relatively flat output image.
    """
    im1_size = 40
    scale1 = 0.23
    init_val = 1.

    im2_size = 100
    scale2 = 0.011

    im1 = galsim.ImageF(im1_size, im1_size, scale=scale1, init_value=init_val)

    # im2 has a much smaller scale, but the same size, so interpolating an "infinite"
    # constant field.
    im2 = galsim.ImageF(im2_size, im2_size, scale=scale2)

    for interp in ['linear', 'cubic', 'quintic']:
        print('Testing interpolant ',interp)
        obj = galsim.InterpolatedImage(im1, x_interpolant=interp, normalization='sb')
        obj.drawImage(im2, method='sb')
        print('The maximum error is ',np.max(abs(im2.array-init_val)))
        np.testing.assert_array_almost_equal(
                im2.array,init_val,5,
                '%s did not preserve a flat input flux using xvals.'%interp)

        # Convolve with a delta function to force FFT drawing.
        delta = galsim.Gaussian(sigma=1.e-8)
        obj2 = galsim.Convolve([obj,delta])
        obj2.drawImage(im2, method='sb')
        print('The maximum error is ',np.max(abs(im2.array-init_val)))
        np.testing.assert_array_almost_equal(
                im2.array,init_val,5,
                '%s did not preserve a flat input flux using uvals.'%interp)

        check_pickle(obj, lambda x: x.drawImage(method='no_pixel'))
        check_pickle(obj2, lambda x: x.drawImage(method='no_pixel'))
        check_pickle(obj)
        check_pickle(obj2)


    for n in [3,4,5,6,7,8]:  # n=8 tests the generic formulae, since not specialized.
        print('Testing Lanczos interpolant with n = ',n)
        lan = galsim.Lanczos(n, conserve_dc=True)
        obj = galsim.InterpolatedImage(im1, x_interpolant=lan, normalization='sb')
        obj.drawImage(im2, method='sb')
        print('The maximum error is ',np.max(abs(im2.array-init_val)))
        np.testing.assert_array_almost_equal(
                im2.array,init_val,5,
                'Lanczos %d did not preserve a flat input flux using xvals.'%n)

        # Convolve with a delta function to force FFT drawing.
        delta = galsim.Gaussian(sigma=1.e-8)
        obj2 = galsim.Convolve([obj,delta])
        obj2.drawImage(im2, method='sb')
        print('The maximum error is ',np.max(abs(im2.array-init_val)))
        np.testing.assert_array_almost_equal(
                im2.array,init_val,5,
                'Lanczos %d did not preserve a flat input flux using uvals.'%n)

        check_pickle(obj, lambda x: x.drawImage(method='no_pixel'))
        check_pickle(obj2, lambda x: x.drawImage(method='no_pixel'))
        check_pickle(obj)
        check_pickle(obj2)


@timer
def test_stepk_maxk():
    """Test options to specify (or not) stepk and maxk.
    """
    scale = 0.18
    n = 101 # use an odd number so profile doesn't get recentered at all, modifying stepk

    obj = galsim.Exponential(half_light_radius=2.*scale)
    im = galsim.Image(n, n)
    im.setCenter(0,0)
    im = obj.drawImage(image=im, scale=scale)
    int_im = galsim.InterpolatedImage(im)

    # These values get calculated automatically with calculateStepK() and calculateMaxK()
    stepk_val = int_im.stepk
    maxk_val = int_im.maxk
    print('From calculate:')
    print('stepk = ',stepk_val)
    print('maxk = ',maxk_val)

    # Check the default values of these (without calculate or force)
    raw_int_im = galsim._InterpolatedImage(im)
    print('Raw values:')
    print('stepk = ',raw_int_im.stepk)
    print('maxk = ',raw_int_im.maxk)
    print('2pi/image_size = ',2.*np.pi/(n*scale))
    print('krange/pixel_scale = ',galsim.Quintic().krange/scale)
    np.testing.assert_allclose(raw_int_im.stepk, 2*np.pi/(n*scale), rtol=0.01,
                               err_msg="Raw stepk value not as expected")
    np.testing.assert_allclose(raw_int_im.maxk, galsim.Quintic().krange/scale, rtol=0.01,
                               err_msg="Raw stepk value not as expected")

    # Now check that we can force the value to be something else
    mult_val = 0.9
    new_int_im = galsim.InterpolatedImage(im, _force_stepk=mult_val*stepk_val,
                                          _force_maxk=mult_val*maxk_val)
    np.testing.assert_almost_equal(
        new_int_im.stepk, mult_val*stepk_val, decimal=7,
        err_msg='InterpolatedImage did not adopt forced value for stepk')
    np.testing.assert_almost_equal(
        new_int_im.maxk, mult_val*maxk_val, decimal=7,
        err_msg='InterpolatedImage did not adopt forced value for maxk')

    alt_int_im = galsim._InterpolatedImage(im, force_stepk=mult_val*stepk_val,
                                           force_maxk=mult_val*maxk_val)
    np.testing.assert_almost_equal(
        alt_int_im.stepk, mult_val*stepk_val, decimal=7,
        err_msg='_InterpolatedImage did not adopt forced value for stepk')
    np.testing.assert_almost_equal(
        alt_int_im.maxk, mult_val*maxk_val, decimal=7,
        err_msg='_InterpolatedImage did not adopt forced value for maxk')

    # Finally if _InterpolatedImage gets an already good fft size, then it doesn't expand,
    # so check that case too.
    alt_int_im = galsim._InterpolatedImage(int_im._xim, force_stepk=mult_val*stepk_val,
                                           force_maxk=mult_val*maxk_val)
    np.testing.assert_almost_equal(
        alt_int_im.stepk, mult_val*stepk_val, decimal=7,
        err_msg='_InterpolatedImage did not adopt forced value for stepk')
    np.testing.assert_almost_equal(
        alt_int_im.maxk, mult_val*maxk_val, decimal=7,
        err_msg='_InterpolatedImage did not adopt forced value for maxk')

    check_pickle(int_im, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(new_int_im, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(int_im)
    check_pickle(new_int_im)
    check_pickle(raw_int_im, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(raw_int_im)
    check_pickle(alt_int_im, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(alt_int_im)


@timer
def test_kroundtrip(ref):
    """ Test that GSObjects `a` and `b` are the same when b = InterpolatedKImage(a.drawKImage)
    """
    final, ref_image = ref
    a = final
    kim_a = a.drawKImage()
    b = galsim.InterpolatedKImage(kim_a)

    # Check picklability
    check_pickle(b)
    check_pickle(b, lambda x: x.drawImage())

    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    b2 = galsim.InterpolatedKImage(kim_a, gsparams=gsp)
    assert b2 != b
    assert b2 == b.withGSParams(gsp)
    assert b2 == b.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    assert b2.k_interpolant.gsparams == gsp
    assert b.k_interpolant.gsparams != gsp

    check_basic(b, "InterpolatedKImage", approx_maxsb=True)

    for kx, ky in zip(KXVALS, KYVALS):
        np.testing.assert_almost_equal(a.kValue(kx, ky), b.kValue(kx, ky), 3,
            err_msg=("InterpolatedKImage evaluated incorrectly at ({0:},{1:})"
                     .format(kx, ky)))

    np.testing.assert_almost_equal(a.flux, b.flux, 6) #Fails at 7th decimal

    kim_b = b.drawKImage(kim_a.copy())
    # Fails at 4th decimal
    np.testing.assert_array_almost_equal(kim_b.array, kim_a.array, 3,
                                         "InterpolatedKImage kimage drawn incorrectly.")

    img_a = a.drawImage()
    img_b = b.drawImage(img_a.copy())
    # This is the one that matters though; fails at 6th decimal
    np.testing.assert_array_almost_equal(img_a.array, img_b.array, 5,
                                         "InterpolatedKImage image drawn incorrectly.")

    # Check that we can construct an interpolatedKImage without a wcs.
    kim_c = a.drawKImage(scale=1)
    c = galsim.InterpolatedKImage(kim_c)
    d = galsim.InterpolatedKImage(galsim.ImageCD(kim_c.array))
    assert c == d, "Failed to construct InterpolatedKImage without wcs."
    check_pickle(d)
    check_pickle(d, lambda x: x.drawImage())

    # Try some (slightly larger maxk) non-even kimages:
    for dx, dy in zip((2,3,3), (3,2,3)):
        shape = kim_a.array.shape
        kim_a = a.drawKImage(nx=shape[1]+dx, ny=shape[0]+dy, scale=kim_a.scale)
        b = galsim.InterpolatedKImage(kim_a)

        np.testing.assert_almost_equal(a.flux, b.flux, 6) #Fails at 7th decimal
        img_b = b.drawImage(img_a.copy())
        # One of these fails at 6th decimal
        np.testing.assert_array_almost_equal(img_a.array, img_b.array, 5)

    # Try some additional transformations:
    a = a.shear(g1=0.2, g2=-0.2).shift(1.1, -0.2).dilate(0.7)
    b = b.shear(g1=0.2, g2=-0.2).shift(1.1, -0.2).dilate(0.7)
    img_a = a.drawImage()
    img_b = b.drawImage(img_a.copy())
    # Fails at 6th decimal
    np.testing.assert_array_almost_equal(img_a.array, img_b.array, 5,
                                         "Transformed InterpolatedKImage image drawn incorrectly.")

    # Does the stepk parameter do anything?
    a = final
    kim_a = a.drawKImage()
    b = galsim.InterpolatedKImage(kim_a)
    c = galsim.InterpolatedKImage(kim_a, stepk=2*b.stepk)
    np.testing.assert_almost_equal(b.stepk, kim_a.scale)
    np.testing.assert_almost_equal(2*b.stepk, c.stepk)
    np.testing.assert_almost_equal(b.maxk, c.maxk)

    # Smaller stepk is overridden.
    with assert_warns(galsim.GalSimWarning):
        d = galsim.InterpolatedKImage(kim_a, stepk=0.5*b.stepk)
    np.testing.assert_almost_equal(b.stepk, d.stepk)
    np.testing.assert_almost_equal(b.maxk, d.maxk)

    # Test centroid
    for dx, dy in zip(KXVALS, KYVALS):
        a = final.shift(dx, dy)
        b = galsim.InterpolatedKImage(a.drawKImage())
        np.testing.assert_almost_equal(a.centroid.x, b.centroid.x, 4) #Fails at 5th decimal
        np.testing.assert_almost_equal(a.centroid.y, b.centroid.y, 4)

    # Test convolution with another object.
    a = final
    b = galsim.InterpolatedKImage(a.drawKImage())
    c = galsim.Kolmogorov(fwhm=0.8).shear(e1=0.01, e2=0.02).shift(0.01, 0.02)
    a_conv_c = galsim.Convolve(a, c)
    b_conv_c = galsim.Convolve(b, c)
    a_conv_c_img = a_conv_c.drawImage()
    b_conv_c_img = b_conv_c.drawImage(image=a_conv_c_img.copy())
    # Fails at 6th decimal.
    np.testing.assert_array_almost_equal(a_conv_c_img.array, b_conv_c_img.array, 5,
                                         "Convolution of InterpolatedKImage drawn incorrectly.")


@timer
def test_kexceptions():
    """Test failure modes for InterpolatedKImage class.
    """
    # Check that provided image has valid bounds
    with assert_raises(galsim.GalSimUndefinedBoundsError):
        galsim.InterpolatedKImage(kimage=galsim.ImageCD(scale=1.))

    # Image must be complex type (CF or CD)
    with assert_raises(galsim.GalSimValueError):
        galsim.InterpolatedKImage(kimage=galsim.ImageD(5, 5, scale=1))

    # Check types of inputs
    im = galsim.ImageCD(5, 5, scale=1., init_value=10.)
    assert_raises(TypeError, galsim.InterpolatedKImage)
    assert_raises(TypeError, galsim.InterpolatedKImage, kimage=im.array)
    assert_raises(TypeError, galsim.InterpolatedKImage, real_kimage=im.real, imag_kimage=4)
    assert_raises(TypeError, galsim.InterpolatedKImage, real_kimage=3, imag_kimage=im.imag)
    assert_raises(TypeError, galsim.InterpolatedKImage, kimage=im,
                  real_kimage=im.real, imag_kimage=im.imag)

    # Other invalid values:
    assert_raises(ValueError, galsim.InterpolatedKImage, im, k_interpolant='invalid')
    assert_raises(ValueError, galsim.InterpolatedKImage, real_kimage=im.real)
    assert_raises(ValueError, galsim.InterpolatedKImage, imag_kimage=im.imag)
    assert_raises(ValueError, galsim.InterpolatedKImage, real_kimage=im, imag_kimage=im)
    assert_raises(ValueError, galsim.InterpolatedKImage, real_kimage=im.real,
                  imag_kimage=galsim.ImageD(4,4,scale=1.))
    assert_raises(ValueError, galsim.InterpolatedKImage, real_kimage=im.real,
                  imag_kimage=galsim.ImageD(5,5,scale=2.))
    assert_raises(ValueError, galsim.InterpolatedKImage,
                  kimage=galsim.ImageCD(5, 5, wcs=galsim.JacobianWCS(2.1, 0.3, -0.4, 2.3)))


@timer
def test_multihdu_readin():
    """Test the ability to read in from a file with multiple FITS extensions.
    """
    # Check that when we read in from the different HDUs using the keyword, we get the expected set
    # of shear values after drawing.  The file was created using
    # fits_files/make_interpim_hdu_test.py, so if that script gets changed, the test has to change
    # too.
    g2_vals = [0., 0.1, 0.7, 0.3]
    scale = 0.2
    infile = os.path.join(path, "fits_files", 'interpim_hdu_test.fits')
    for ind,g2 in enumerate(g2_vals):
        obj = galsim.InterpolatedImage(image=infile, hdu=ind)
        im = obj.drawImage(scale=scale, method='no_pixel')
        test_g2 = im.FindAdaptiveMom().observed_shape.g2
        np.testing.assert_almost_equal(
            test_g2, g2, decimal=3,
            err_msg='Did not get right shape image after reading from HDU')

    # Repeat for InterpolatedKImage, drawing in k space for the check.
    kfile = os.path.join(path, "fits_files", 'interpkim_hdu_test.fits')
    for ind,g2 in enumerate(g2_vals):
        obj2 = galsim.InterpolatedKImage(real_kimage=kfile, real_hdu=2*ind,
                                         imag_kimage=kfile, imag_hdu=2*ind+1)
        im = obj2.drawKImage(scale=scale)
        test_g2 = im.real.FindAdaptiveMom().observed_shape.g2
        np.testing.assert_almost_equal(
            test_g2, -g2, decimal=3,
            err_msg='Did not get right shape image after reading real_kimage from HDU')

    # Check for exception with invalid HDU.
    assert_raises(OSError, galsim.InterpolatedImage, infile, hdu=37)
    assert_raises(OSError, galsim.InterpolatedKImage,
                  real_kimage=infile, imag_kimage=infile, real_hdu=37, imag_hdu=1)
    assert_raises(OSError, galsim.InterpolatedKImage,
                  real_kimage=infile, imag_kimage=infile, real_hdu=1, imag_hdu=37)


@timer
def test_ii_shoot(run_slow):
    """Test InterpolatedImage with photon shooting.  Particularly the flux of the final image.
    """
    rng = galsim.BaseDeviate(1234)
    ref_array = np.array([
        [0.01, 0.08, 0.07, 0.02],
        [0.13, 0.38, 0.52, 0.06],
        [0.09, 0.41, 0.44, 0.09],
        [0.04, 0.11, 0.10, 0.01] ])
    image_in = galsim.Image(ref_array)
    interp_list = ['nearest', 'delta', 'linear', 'cubic', 'quintic',
                   'lanczos3', 'lanczos5', 'lanczos7']
    im = galsim.Image(100,100, scale=1)
    im.setCenter(0,0)
    if run_slow:
        flux = 1.e6
    else:
        flux = 1.e4
    for interp in interp_list:
        obj = galsim.InterpolatedImage(image_in, x_interpolant=interp, scale=3.3, flux=flux)
        added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
        print('interp = ',interp)
        print('obj.flux = ',obj.flux)
        print('added_flux = ',added_flux)
        print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
        print('sum = ',photons.flux.sum())
        print('image flux = ',im.array.sum())
        if obj.negative_flux > 0:
            # Where there are negative flux photons, there is an implicit poisson variance of the
            # flux based on how many end up positive vs how many end up negative.
            # This gets better as flux increases, but slower of course, so only use 1.e6
            # when name==main.
            rtol = (np.sqrt(obj.positive_flux) + np.sqrt(obj.negative_flux)) / obj.flux
            print('using rtol = ',rtol)
        else:
            rtol = 1.e-7
        assert np.isclose(added_flux, obj.flux, rtol=rtol)
        assert np.isclose(im.array.sum(), obj.flux, rtol=rtol)
        photons2 = obj.makePhot(poisson_flux=False, rng=rng.duplicate())
        assert photons2 == photons, "InterpolatedImage makePhot not equivalent to drawPhot"

        # Can treat as a convolution of a delta function and put it in a photon_ops list.
        delta = galsim.DeltaFunction(flux=flux)
        psf = galsim.InterpolatedImage(image_in, x_interpolant=interp, scale=3.3, flux=1)
        photons3 = delta.makePhot(poisson_flux=False, rng=rng.duplicate(), photon_ops=[psf],
                                  n_photons=len(photons))
        np.testing.assert_allclose(photons3.x, photons.x)
        np.testing.assert_allclose(photons3.y, photons.y)
        np.testing.assert_allclose(photons3.flux, photons.flux)


@timer
def test_ne(ref):
    """ Check that inequality works as expected for corner cases where the reprs of two
    unequal InterpolatedImages or InterpolatedKImages may be the same due to truncation.
    """
    final, ref_image = ref
    obj1 = galsim.InterpolatedImage(ref_image, flux=20, calculate_maxk=False, calculate_stepk=False)

    # Copy ref_image and perturb it slightly in the middle, away from where the InterpolatedImage
    # repr string will report.
    perturb_image = ref_image.copy()
    perturb_image.array[64, 64] *= 1000
    obj2 = galsim.InterpolatedImage(perturb_image, flux=20, calculate_maxk=False, calculate_stepk=False)

    with galsim.utilities.printoptions(threshold=128*128):
        assert repr(obj1) != repr(obj2), "Reprs unexpectedly agree: %r"%obj1

    with galsim.utilities.printoptions(threshold=1000):
        assert repr(obj1) == repr(obj2), "Reprs disagree: repr(obj1)=%r\nrepr(obj2)=%r"%(
                obj1, obj2)

    assert obj1 != obj2

    # Now repeat for InterpolatedKImage
    kim = obj1.drawKImage(nx=128, ny=128, scale=1)
    obj3 = galsim.InterpolatedKImage(kim)
    perturb = kim.copy()
    x = np.arange(128)
    x, y = np.meshgrid(x, x)
    w = ((perturb.real.array**2 - perturb.imag.array**2 > 1e-10) &
         (50 < x) & (x < (128-50)) &
         (50 < y) & (y < (128-50)))
    perturb.array[w] *= 2

    obj4 = galsim.InterpolatedKImage(perturb)

    with galsim.utilities.printoptions(threshold=128*128):
        assert repr(obj3) != repr(obj4)

    with galsim.utilities.printoptions(threshold=1000):
        assert repr(obj3) == repr(obj4)

    assert obj3 != obj4

    # Test that slightly different objects compare and hash appropriately.
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)
    gals = [galsim.InterpolatedImage(ref_image),
            galsim.InterpolatedImage(ref_image, calculate_maxk=False),
            galsim.InterpolatedImage(ref_image, calculate_stepk=False),
            galsim.InterpolatedImage(ref_image, flux=1.1),
            galsim.InterpolatedImage(ref_image, offset=(0.0, 1.1)),
            galsim.InterpolatedImage(ref_image, x_interpolant='Linear'),
            galsim.InterpolatedImage(ref_image, k_interpolant='Linear'),
            galsim.InterpolatedImage(ref_image, pad_factor=1.),
            galsim.InterpolatedImage(ref_image, normalization='sb'),
            galsim.InterpolatedImage(ref_image, noise_pad_size=100, noise_pad=0.1),
            galsim.InterpolatedImage(ref_image, noise_pad_size=100, noise_pad=0.2),
            galsim.InterpolatedImage(ref_image, noise_pad_size=100, noise_pad=0.2),
            galsim.InterpolatedImage(ref_image, _force_stepk=1.0),
            galsim.InterpolatedImage(ref_image, _force_maxk=1.0),
            galsim.InterpolatedImage(ref_image, scale=0.2),
            galsim.InterpolatedImage(ref_image, use_true_center=False),
            galsim.InterpolatedImage(ref_image, gsparams=gsp)]
    check_all_diff(gals)

    # And repeat for InterpolatedKImage
    gals = [galsim.InterpolatedKImage(kim),
            galsim.InterpolatedKImage(kim, k_interpolant='Linear'),
            galsim.InterpolatedKImage(kim, stepk=1.1),
            galsim.InterpolatedKImage(kim, gsparams=gsp)]
    check_all_diff(gals)


@timer
def test_quintic_glagn():
    """This is code that was giving a seg fault.  cf. Issue 1079.
    """

    fname = os.path.join('fits_files','GLAGN_host_427_0_disk.fits')
    for interpolant in 'linear cubic quintic'.split():
        print(interpolant)
        fits_image = galsim.InterpolatedImage(fname, scale=0.04, x_interpolant=interpolant)
        fits_image.withFlux(529.2666077544975)

        atm = galsim.Kolmogorov(lam_over_r0=0.7763974062349864)
        gaus = galsim.Gaussian(sigma=0.18368260871093112)
        gsobj = galsim.Convolve(fits_image, atm, gaus) * 4.7367431900462575

        image = galsim.Image(bounds=galsim.BoundsI(1391,1440,3416,3465), dtype=np.float32)

        gsobj.drawImage(method='phot', image=image, add_to_image=True)


@timer
def test_depixelize():
    # True, non-II profile.  Something not too symmetric or simple.
    true_prof = galsim.Convolve(
                    galsim.Kolmogorov(fwhm=1.1, flux=10).shear(g1=0.1, g2=0.2),
                    galsim.OpticalPSF(lam_over_diam=0.6, obscuration=0.4, nstruts=4,
                                      defocus=0.15, astig1=0.13, astig2=-0.14,
                                      coma1=-0.06, trefoil1=-0.08))

    # Make these unequal to test indexing
    nx = 32
    ny = 25
    scale = 0.3
    im1 = true_prof.drawImage(nx=nx, ny=ny, scale=scale, dtype=float)

    interp = galsim.Lanczos(11)

    import time
    t0 = time.time()
    ii_with_pixel = galsim.InterpolatedImage(im1, x_interpolant=interp)
    t1 = time.time()

    # The normal use of InterpolatedImage requires drawing with no_pixel to match original.
    im2 = ii_with_pixel.drawImage(nx=nx, ny=ny, scale=scale, method='no_pixel')
    print('with_pixel: max error = ',np.max(np.abs(im2.array-im1.array)))
    np.testing.assert_allclose(im2.array, im1.array, atol=1.e-9)
    t2 = time.time()

    nopix_image = im1.depixelize(x_interpolant=interp)
    t3 = time.time()
    ii_without_pixel = galsim.InterpolatedImage(nopix_image, x_interpolant=interp)
    t4 = time.time()

    # The depixelize function is basically exact for real-space convolution.
    im3 = ii_without_pixel.drawImage(nx=nx, ny=ny, scale=scale, method='real_space')
    print('real-space drawing: max error = ',np.max(np.abs(im3.array-im1.array)))
    np.testing.assert_allclose(im3.array, im1.array, atol=1.e-9)
    t5 = time.time()

    # With FFT convolution, it's not as close, but this is just due to the approximations that
    # we always have in FFT convolutions.
    im4 = ii_without_pixel.drawImage(nx=nx, ny=ny, scale=scale, method='fft')
    print('fft drawing: max error = ',np.max(np.abs(im4.array-im1.array)))
    np.testing.assert_allclose(im4.array, im1.array, atol=1.e-4)
    t6 = time.time()

    # We can make this a lot better by increasing maxk artificially.
    # However, it's not currently possible to make the InterpolatedImage have this high a
    # maxk using just maxk_threshold, since CalculateMaxK hits the edge of the FFT image at
    # around maxk=10, not matter what the threshold is.
    # So using this high a maxk is actually wrapping around the FFT image multiple times.
    # This feels like a clue that something could be improved in some of the choices we make
    # wrt the InterpolatedImage FFT rendering, but I'm going to leave this alone for now,
    # since I'm not actually sure what the better thing to do would be.
    alt = galsim.InterpolatedImage(nopix_image, x_interpolant=interp, _force_maxk=50)
    im5 = alt.drawImage(nx=nx, ny=ny, scale=scale, method='fft')
    print('high-maxk fft drawing: max error = ',np.max(np.abs(im5.array-im1.array)))
    np.testing.assert_allclose(im5.array, im1.array, atol=1.e-7)
    t7 = time.time()

    # Second time with the same size image is much faster, since uses a cache.
    nopix_image2 = im1.depixelize(x_interpolant=interp)
    t8 = time.time()
    if platform.python_implementation() != 'PyPy':
        # PyPy timings can be fairly arbitrary at times.
        assert t8-t7 < (t3-t2)/5

    # Even if the image is different.
    nopix_image3 = im4.depixelize(x_interpolant=interp)
    t9 = time.time()
    if platform.python_implementation() != 'PyPy':
        assert t9-t8 < (t3-t2)/5

    # But not if you clear the cache.
    galsim.Image.clear_depixelize_cache()
    nopix_image4 = im4.depixelize(x_interpolant=interp)
    t10 = time.time()
    if platform.python_implementation() != 'PyPy':
        assert t10-t9 > (t3-t2)/5

    print('times:')
    print('make ii_with_pixel: ',t1-t0)
    print('draw ii_with_pixel: ',t2-t1)
    print('depixelize: ',t3-t2)
    print('make ii_without_pixel: ',t4-t3)
    print('draw ii_without_pixel, real: ',t5-t4)
    print('draw ii_without_pixel, fft: ',t6-t5)
    print('draw ii_without_pixel, high maxk: ',t7-t6)
    print('depixelize #2: ',t8-t7)
    print('depixelize #3: ',t9-t8)
    print('depixelize #4: ',t10-t9)

    # Use the simpler API that we expect users to typically prefer.
    # Should be exactly equivalent to the above two-step process.
    # (But only if the original image is double-precision.  Otherwise, this gets to dtype=float
    # sooner than the other method.)
    ii_without_pixel2 = galsim.InterpolatedImage(im1, x_interpolant=interp, depixelize=True)
    assert ii_without_pixel2 == ii_without_pixel

    # Check with a non-trivial WCS
    wcs = galsim.AffineTransform(0.07, -0.31, 0.33, 0.03,
                                 galsim.PositionD(5.3,7.1), galsim.PositionD(293, 800))
    im6 = true_prof.drawImage(nx=nx, ny=ny, wcs=wcs)
    ii = galsim.InterpolatedImage(im6, x_interpolant=interp, _force_maxk=50, depixelize=True)
    im7 = ii.drawImage(nx=nx, ny=ny, wcs=wcs, method='auto')
    print('affine wcs max error = ',np.max(np.abs(im7.array-im6.array)),'  time = ',t2-t1)
    np.testing.assert_allclose(im7.array, im6.array, atol=1.e-7)

    # Check a variety of interpolants.
    interps = [galsim.Delta(),
               galsim.Nearest(),
               galsim.SincInterpolant(gsparams=galsim.GSParams(kvalue_accuracy=0.01)),
               galsim.Linear(),
               galsim.Cubic(),
               galsim.Quintic(),
               galsim.Lanczos(3),
               galsim.Lanczos(5, conserve_dc=False),
               galsim.Lanczos(17),
              ]
    for interp in interps:
        t1 = time.time()
        ii = galsim.InterpolatedImage(im1, x_interpolant=interp, _force_maxk=50, depixelize=True,
                                      gsparams=interp.gsparams)
        t2 = time.time()

        im6 = ii.drawImage(nx=nx, ny=ny, scale=scale, method='auto')
        # Most of these are better than 1.e-5, but Delta and Nearest are much worse.
        # So just use tol=1.e-2 here.
        np.testing.assert_allclose(im6.array, im1.array, atol=1.e-2)
        print(interp,' max error = ',np.max(np.abs(im6.array-im1.array)),'  time = ',t2-t1)


@timer
def test_drawreal_seg_fault():
    """Test to reproduce bug report in Issue #1164 that was causing seg faults
    """

    import pickle

    prof_file = 'input/test_interpolatedimage_seg_fault_prof.pkl'
    with open(prof_file, 'rb') as f:
        prof = pickle.load(f)
    print(repr(prof))

    image = galsim.Image(
        galsim.BoundsI(
            xmin=-12,
            xmax=12,
            ymin=-12,
            ymax=12
        ),
        dtype=float,
        scale=1
    )

    image.fill(3)
    prof.drawReal(image)

    # The problem was that the object is shifted fully off the target image and that was leading
    # to an attempt to create a stack of length -1, which caused the seg fault.
    # So mostly this test just confirms that this runs without seg faulting.
    # But we can check that the image is now correctly all zeros.
    np.testing.assert_array_equal(image.array, 0)


if __name__ == "__main__":
    runtests(__file__)
