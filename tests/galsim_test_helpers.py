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

import pytest
import numpy as np
import galsim

from galsim.utilities import check_pickle, check_all_diff, timer, CaptureLog, Profile

# We used to roll our own versions of these, but numpy.testing has good ones now.
from numpy.testing import assert_raises
from numpy.testing import assert_warns

__all__ = [
    "default_params",
    "gsobject_compare",
    "printval",
    "convertToShear",
    "check_basic_x",
    "check_basic_k",
    "assert_floatlike",
    "assert_intlike",
    "check_basic",
    "do_shoot",
    "do_kvalue",
    "radial_integrate",
    "drawNoise",
    "check_pickle",
    "check_all_diff",
    "timer",
    "CaptureLog",
    "assert_raises",
    "assert_warns",
    "Profile",
    "galsim_backend",
    "is_jax_galsim",
    "pytest",
]

# This file has some helper functions that are used by tests from multiple files to help
# avoid code duplication.

# These are the default GSParams used when unspecified.  We'll check that specifying
# these explicitly produces the same results.
default_params = galsim.GSParams(
        minimum_fft_size = 128,
        maximum_fft_size = 8192,
        folding_threshold = 5.e-3,
        maxk_threshold = 1.e-3,
        kvalue_accuracy = 1.e-5,
        xvalue_accuracy = 1.e-5,
        shoot_accuracy = 1.e-5,
        realspace_relerr = 1.e-4,
        realspace_abserr = 1.e-6,
        integration_relerr = 1.e-6,
        integration_abserr = 1.e-8)


def galsim_backend():
    if "jax_galsim/__init__.py" in galsim.__file__:
        return "jax_galsim"
    else:
        return "galsim"


def is_jax_galsim():
    return galsim_backend() == "jax_galsim"


def gsobject_compare(obj1, obj2, conv=None, decimal=10):
    """Helper function to check that two GSObjects are equivalent
    """
    if conv:
        obj1 = galsim.Convolve([obj1,conv.withGSParams(obj1.gsparams)])
        obj2 = galsim.Convolve([obj2,conv.withGSParams(obj2.gsparams)])

    im1 = galsim.ImageD(16,16)
    im2 = galsim.ImageD(16,16)
    if isinstance(obj1,galsim.BaseCorrelatedNoise):
        obj1.drawImage(scale=0.2, image=im1)
        obj2.drawImage(scale=0.2, image=im2)
    else:
        obj1.drawImage(scale=0.2, image=im1, method='no_pixel')
        obj2.drawImage(scale=0.2, image=im2, method='no_pixel')
    np.testing.assert_array_almost_equal(im1.array, im2.array, decimal=decimal)


def printval(image1, image2, show=False):
    print("New, saved array sizes: ", np.shape(image1.array), np.shape(image2.array))
    print("Sum of values: ", np.sum(image1.array, dtype=float), np.sum(image2.array, dtype=float))
    print("Minimum image value: ", np.min(image1.array), np.min(image2.array))
    print("Maximum image value: ", np.max(image1.array), np.max(image2.array))
    print("Peak location: ", image1.array.argmax(), image2.array.argmax())

    fmt = "      {0:<15.8g}  {1:<15.8g}  {2:<15.8g}  {3:<15.8g}  {4:<15.8g}"

    mom1 = galsim.utilities.unweighted_moments(image1)
    print("Moments Mx, My, Mxx, Myy, Mxy for new array: ")
    print(fmt.format(mom1['Mx'], mom1['My'], mom1['Mxx'], mom1['Myy'], mom1['Mxy']))

    mom2 = galsim.utilities.unweighted_moments(image2)
    print("Moments Mx, My, Mxx, Myy, Mxy for saved array: ")
    print(fmt.format(mom2['Mx'], mom2['My'], mom2['Mxx'], mom2['Myy'], mom2['Mxy']))

    if show:
        import matplotlib.pylab as plt
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        ax1.imshow(image1.array)
        ax2.imshow(image2.array)
        plt.show()

def convertToShear(e1,e2):
    # Convert a distortion (e1,e2) to a shear (g1,g2)
    import math
    e = math.sqrt(e1*e1 + e2*e2)
    g = math.tanh( 0.5 * math.atanh(e) )
    g1 = e1 * (g/e)
    g2 = e2 * (g/e)
    return (g1,g2)

def check_basic_x(prof, name, approx_maxsb=False, scale=None):
    """Test drawImage using sb method.
    """
    #print('  nyquist_scale, stepk, maxk = ', prof.nyquist_scale, prof.stepk, prof.maxk)
    image = prof.drawImage(method='sb', scale=scale, use_true_center=False)
    image.setCenter(0,0)
    dx = image.scale
    #print('  image scale,bounds = ',dx,image.bounds)
    if scale is None:
        assert image.scale == prof.nyquist_scale
    print('  flux: ',prof.flux, image.array.sum(dtype=float)*dx**2, image.added_flux)
    np.testing.assert_allclose(
            image.array.sum(dtype=float) * dx**2, image.added_flux, 1.e-5,
            err_msg="%s profile drawImage(method='sb') returned wrong added_flux"%name)
    np.testing.assert_allclose(
            image.added_flux, prof.flux, rtol=0.1,  # Not expected to be all that close, since sb.
            err_msg="%s profile flux not close to sum of pixel values"%name)

    print('  maxsb: ',prof.max_sb, image.array.max())
    #print('  image = ',image[galsim.BoundsI(-2,2,-2,2)].array)
    if approx_maxsb:
        np.testing.assert_array_less(
                np.abs(image.array).max(), np.abs(prof.max_sb) * 1.4,
                err_msg="%s profile max_sb smaller than maximum pixel value"%name)
    else:
        np.testing.assert_allclose(
                np.abs(image.array).max(), np.abs(prof.max_sb), rtol=1.e-5,
                err_msg="%s profile max_sb did not match maximum pixel value"%name)
    for i,j in ( (2,3), (-4,1), (0,-5), (-3,-3) ):
        x = i*dx
        y = j*dx
        print('  x: i,j = ',i,j,image(i,j),prof.xValue(x,y))
        np.testing.assert_allclose(
                image(i,j), prof.xValue(x,y), rtol=1.e-5,
                err_msg="%s profile sb image does not match xValue at %d,%d"%(name,i,j))
        np.testing.assert_allclose(
                image(i,j), prof._xValue(galsim.PositionD(x,y)), rtol=1.e-5,
                err_msg="%s profile sb image does not match _xValue at %d,%d"%(name,i,j))
    if is_jax_galsim():
        for line in galsim.GSObject.withFlux.__doc__.splitlines():
            if line.strip() and "LAX" not in line:
                assert line.strip() in prof.withFlux.__doc__, (
                    prof.withFlux.__doc__, galsim.GSObject.withFlux.__doc__,
                )
        for line in galsim.GSObject.withFlux.__doc__.splitlines():
            if line.strip() and "LAX" not in line:
                assert line.strip() in prof.__class__.withFlux.__doc__, (
                    prof.__class__.withFlux.__doc__, galsim.GSObject.withFlux.__doc__,
                )
    else:
        assert prof.withFlux.__doc__ == galsim.GSObject.withFlux.__doc__
        assert prof.__class__.withFlux.__doc__ == galsim.GSObject.withFlux.__doc__

    # Check negative flux:
    neg_image = prof.withFlux(-prof.flux).drawImage(method='sb', scale=scale, use_true_center=False)
    np.testing.assert_array_almost_equal(neg_image.array/prof.flux, -image.array/prof.flux, 7,
                                         '%s negative flux drawReal is not negative of +flux image'%name)

    # Direct call to drawReal should also work and be equivalent to the above with scale = 1.
    prof.drawImage(image, method='sb', scale=1., use_true_center=False)
    image2 = image.copy()
    prof.drawReal(image2)
    np.testing.assert_array_equal(image2.array, image.array,
                                  err_msg="%s drawReal not equivalent to drawImage"%name)

    # If supposed to be axisymmetric, make sure it is.
    if prof.is_axisymmetric:
        for r in [0.2, 1.3, 33.4]:
            ref_value = prof.xValue(0, r)  # Effectively theta = pi/2
            test_values = [prof.xValue(r*np.cos(t), r*np.sin(t)) for t in [0., 0.3, 0.9, 1.3, 2.9]]
            print(ref_value,test_values)
            np.testing.assert_allclose(test_values, ref_value, rtol=1.e-5,
                                       err_msg="%s profile not axisymmetric in xValues"%name)


def check_basic_k(prof, name):
    """Check drawKImage
    """
    print('  nyquist_scale, stepk, maxk = ', prof.nyquist_scale, prof.stepk, prof.maxk)
    if prof.maxk/prof.stepk > 2000.:
        # Don't try to draw huge images!
        kimage = prof.drawKImage(nx=2000,ny=2000)
    elif prof.maxk/prof.stepk < 12.:
        # or extremely small ones.
        kimage = prof.drawKImage(nx=12,ny=12)
    else:
        kimage = prof.drawKImage()
    kimage.setCenter(0,0)
    dk = kimage.scale
    print('  kimage scale,bounds = ',dk,kimage.bounds)
    assert kimage.scale == prof.stepk
    print('  k flux: ',prof.flux, prof.kValue(0,0), kimage(0,0))
    np.testing.assert_allclose(
            prof.kValue(0,0), prof.flux, rtol=1.e-10,
            err_msg="%s profile kValue(0,0) did not match flux"%name)
    np.testing.assert_allclose(
            kimage(0,0), prof.flux, rtol=1.e-10,
            err_msg="%s profile kimage(0,0) did not match flux"%name)
    for i,j in ( (2,3), (-4,1), (0,-5), (-3,-3) ):
        kx = i*dk
        ky = j*dk
        print('  k: i,j = ',i,j,kimage(i,j),prof.kValue(kx,ky))
        np.testing.assert_allclose(
                kimage(i,j), prof.kValue(kx,ky), rtol=1.e-5,
                err_msg="%s profile kimage does not match kValue at %d,%d"%(name,i,j))
        np.testing.assert_allclose(
                kimage(i,j), prof._kValue(galsim.PositionD(kx,ky)), rtol=1.e-5,
                err_msg="%s profile kimage does not match _kValue at %d,%d"%(name,i,j))

    # Check negative flux:
    neg_image = prof.withFlux(-prof.flux).drawKImage(kimage.copy())
    np.testing.assert_array_almost_equal(neg_image.array/prof.flux, -kimage.array/prof.flux, 7,
                                   '%s negative flux drawK is not negative of +flux image'%name)

    # If supposed to be axisymmetric, make sure it is in the kValues.
    if prof.is_axisymmetric:
        for r in [0.2, 1.3, 33.4]:
            ref_value = prof.kValue(0, r)  # Effectively theta = pi/2
            test_values = [prof.kValue(r*np.cos(t), r*np.sin(t)) for t in [0., 0.3, 0.9, 1.3, 2.9]]
            print(ref_value,test_values)
            np.testing.assert_allclose(test_values, ref_value, rtol=1.e-5,
                                       err_msg="%s profile not axisymmetric in kValues"%name)

def assert_floatlike(val):
    assert (
        isinstance(val, float)
        or (
            is_jax_galsim()
            and hasattr(val, "shape")
            and val.shape == ()
            and hasattr(val, "dtype")
            and val.dtype.name in ["float", "float32", "float64"]
        )
    ), "Value is not float-like: type(%r) = %r" % (val, type(val))

def assert_intlike(val):
    assert (
        isinstance(val, int)
        or (
            is_jax_galsim()
            and hasattr(val, "shape")
            and val.shape == ()
            and hasattr(val, "dtype")
            and val.dtype.name in ["int", "int32", "int64"]
        )
    ), "Value is not int-like: type(%r) = %r" % (val, type(val))

def check_basic(prof, name, approx_maxsb=False, scale=None, do_x=True, do_k=True):
    """Do some basic sanity checks that should work for all profiles.
    """
    print('Testing',name)
    if do_x and prof.is_analytic_x:
        check_basic_x(prof, name, approx_maxsb, scale)
    if do_k and prof.is_analytic_k:
        check_basic_k(prof, name)

    # A few things that should work regardless of what is analytic
    np.testing.assert_almost_equal(
            prof.positive_flux - prof.negative_flux, prof.flux,
            err_msg="%s profile flux not equal to posflux + negflux"%name)
    assert isinstance(prof.centroid, galsim.PositionD)
    assert_floatlike(prof.flux)
    assert_floatlike(prof.positive_flux)
    assert_floatlike(prof.negative_flux)
    assert_floatlike(prof.max_sb)
    assert_floatlike(prof.stepk)
    assert_floatlike(prof.maxk)
    assert isinstance(prof.has_hard_edges, bool)
    assert isinstance(prof.is_axisymmetric, bool)
    assert isinstance(prof.is_analytic_x, bool)
    assert isinstance(prof.is_analytic_k, bool)
    assert np.isclose(prof.positive_flux - prof.negative_flux, prof.flux)
    assert np.isclose(prof._flux_per_photon, prof.flux / (prof.positive_flux + prof.negative_flux))

    # When made with the same gsparams, it returns itself
    assert prof.withGSParams(prof.gsparams) is prof
    alt_gsp = galsim.GSParams(xvalue_accuracy=0.2, folding_threshold=0.03)
    prof_alt = prof.withGSParams(alt_gsp)
    prof_alt2 = prof.withGSParams(xvalue_accuracy=0.2, folding_threshold=0.03)
    assert isinstance(prof_alt, prof.__class__)
    assert prof_alt.gsparams == alt_gsp
    assert prof_alt2.gsparams == prof.gsparams.withParams(xvalue_accuracy=0.2,
                                                          folding_threshold=0.03)
    assert prof_alt != prof  # Assuming none of our tests use this exact gsparams choice.
    # Back to the original, ==, but not is
    assert prof_alt.withGSParams(prof.gsparams) is not prof
    assert prof_alt.withGSParams(prof.gsparams) == prof
    assert prof_alt2.withGSParams(xvalue_accuracy=prof.gsparams.xvalue_accuracy,
                                  folding_threshold=prof.gsparams.folding_threshold) == prof

    # Repeat for a rotated version of the profile.
    # The rotated version is mathematically the same for most profiles (all axisymmetric ones),
    # but it forces the draw codes to pass through different functions.  Specifically, it uses
    # the versions of fillXImage with dxy and dyx rather than icenter and jcenter, so this call
    # serves an important function for code coverage.
    prof = prof.rotate(17*galsim.degrees)
    name = "Rotated " + name
    print('Testing',name)
    if do_x and prof.is_analytic_x:
        check_basic_x(prof, name, approx_maxsb, scale)
    if do_k and prof.is_analytic_k:
        check_basic_k(prof, name)


def do_shoot(prof, img, name):
    # For photon shooting, we calculate the number of photons to use based on the target
    # accuracy we are shooting for.  (Pun intended.)
    # For each pixel,
    # uncertainty = sqrt(N_pix) * flux_photon = sqrt(N_tot * flux_pix / flux_tot) * flux_tot / N_tot
    #             = sqrt(flux_pix) * sqrt(flux_tot) / sqrt(N_tot)
    # This is largest for the brightest pixel.  So we use:
    # uncertainty = rtol * flux_max
    # => N_tot = flux_tot / flux_max / rtol**2
    # Then we multiply by 10 to get a 3 sigma buffer
    rtol = 2.e-2

    test_flux = 1.8

    print('Start do_shoot')
    # Verify that shoot with rng=None runs
    prof.shoot(100, rng=None)
    # And also verify 0, 1, or 2 photons.
    prof.shoot(0)
    prof.shoot(1)
    prof.shoot(2)

    # Test photon shooting for a particular profile (given as prof).
    prof.drawImage(img)
    flux_max = img.array.max()
    print('prof.flux = ',prof.flux)
    print('flux_max = ',flux_max)
    flux_tot = img.array.sum(dtype=float)
    print('flux_tot = ',flux_tot)
    atol = flux_max * rtol * 3

    nphot = flux_tot / flux_max / rtol**2
    print('nphot = ',nphot)
    img2 = img.copy()

    if is_jax_galsim():
        rtol *= 3

    # Use a deterministic random number generator so we don't fail tests because of rare flukes
    # in the random numbers.
    rng = galsim.UniformDeviate(12345)

    prof.drawImage(img2, n_photons=nphot, poisson_flux=False, rng=rng, method='phot')
    print('img2.sum => ',img2.array.sum(dtype=float))
    print('img2.max = ',img2.array.max())
    printval(img2,img)
    np.testing.assert_allclose(
            img2.array, img.array, rtol=rtol, atol=atol,
            err_msg="Photon shooting for %s disagrees with expected result"%name)

    # Test normalization
    dx = img.scale
    # Test with a large image to make sure we capture enough of the flux
    # even for slow convergers like Airy (which needs a _very_ large image) or Sersic.
    print('stepk, maxk = ',prof.stepk,prof.maxk)
    if 'Airy' in name:
        img = galsim.ImageD(2048,2048, scale=dx)
    elif 'Sersic' in name or 'DeVauc' in name or 'Spergel' in name or 'VonKarman' in name:
        img = galsim.ImageD(512,512, scale=dx)
    else:
        img = galsim.ImageD(128,128, scale=dx)
    prof = prof.withFlux(test_flux)
    prof.drawImage(img)
    print('img.sum = ',img.array.sum(dtype=float),'  cf. ',test_flux)
    np.testing.assert_allclose(img.array.sum(dtype=float), test_flux, rtol=1.e-4,
            err_msg="Flux normalization for %s disagrees with expected result"%name)
    # max_sb is not always very accurate, but it should be an overestimate if wrong.
    assert img.array.max() <= prof.max_sb*dx**2 * 1.4, "max_sb for %s is too small."%name

    scale = test_flux / flux_tot # from above
    nphot *= scale * scale
    print('nphot -> ',nphot)
    if 'InterpolatedImage' in name or 'PhaseScreen' in name:
        nphot *= 10
        print('nphot -> ',nphot)
    prof.drawImage(img, n_photons=nphot, poisson_flux=False, rng=rng, method='phot')
    print('img.sum = ',img.array.sum(dtype=float),'  cf. ',test_flux)
    np.testing.assert_allclose(
            img.array.sum(dtype=float), test_flux, rtol=rtol, atol=atol,
            err_msg="Photon shooting normalization for %s disagrees with expected result"%name)
    print('img.max = ',img.array.max(),'  cf. ',prof.max_sb*dx**2)
    print('ratio = ',img.array.max() / (prof.max_sb*dx**2))
    assert img.array.max() <= prof.max_sb*dx**2 * 1.4, \
            "Photon shooting for %s produced too high max pixel."%name

    # Test negative flux
    prof = prof.withFlux(-test_flux)
    prof.drawImage(img, n_photons=nphot, poisson_flux=False, rng=rng, method='phot')
    print('img.sum = ',img.array.sum(dtype=float),'  cf. ',-test_flux)
    np.testing.assert_allclose(
            img.array.sum(dtype=float), -test_flux, rtol=rtol, atol=atol,
            err_msg="Photon shooting normalization for %s disagrees when flux < 0"%name)


def do_kvalue(prof, im1, name):
    """Test that the k-space values are consistent with the real-space values by drawing the
    profile directly (without any convolution, so using fillXValues) and convolved by a tiny
    Gaussian (effectively a delta function).
    """

    prof.drawImage(im1, method='no_pixel')

    delta = galsim.Gaussian(sigma = 1.e-8)
    conv = galsim.Convolve([prof,delta])
    im2 = conv.drawImage(im1.copy(), method='no_pixel')
    printval(im1,im2)
    np.testing.assert_array_almost_equal(
            im2.array, im1.array, 3,
            err_msg = name +
            " convolved with a delta function is inconsistent with real-space image.")

def radial_integrate(prof, minr, maxr):
    """A simple helper that calculates int 2pi r f(r) dr, from rmin to rmax
       for an axially symmetric profile.
    """
    assert prof.is_axisymmetric
    # In this tight loop, it is worth optimizing away the parse_pos_args step.
    # It makes a rather significant difference in the running time of this function.
    # (I.e., use prof._xValue() instead of prof.xValue() )
    f = lambda r: 2 * np.pi * r * prof._xValue(galsim.PositionD(r,0))
    return galsim.integ.int1d(f, minr, maxr)

# A short helper function to test pickling of noise objects
def drawNoise(noise):
    im = galsim.ImageD(10,10)
    im.addNoise(noise)
    return im.array.astype(np.float32).tolist()


# Define a fixture to check if the tests were run via python script
@pytest.fixture
def run_slow(pytestconfig):
    return pytestconfig.getoption("--run_slow", default=False)


def runtests(filename, parser=None):
    if parser is None:
        from argparse import ArgumentParser
        parser = ArgumentParser()
    parser.add_argument('--profile', action='store_true', help='Profile tests')
    parser.add_argument('--prof_out', default=None, help="Profiler output file")
    args, unknown_args = parser.parse_known_args()
    pytest_args = [filename] + unknown_args + ["--run_slow", "--tb=short", "-s"]

    if args.profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
    pytest.main(pytest_args)
    if args.profile:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
        if args.prof_out:
            ps.dump_stats(args.prof_out)
