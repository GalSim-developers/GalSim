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

import galsim
from galsim_test_helpers import *


# for flux normalization tests
test_flux = 1.8

# A helper function used by both test_draw and test_drawk to check that the drawn image
# is a radially symmetric exponential with the right scale.
def CalculateScale(im):
    # We just determine the scale radius of the drawn exponential by calculating
    # the second moments of the image.
    # int r^2 exp(-r/s) 2pir dr = 12 s^4 pi
    # int exp(-r/s) 2pir dr = 2 s^2 pi
    x, y = np.meshgrid(np.arange(np.shape(im.array)[0]), np.arange(np.shape(im.array)[1]))
    if np.iscomplexobj(im.array):
        T = complex
    else:
        T = float
    flux = im.array.astype(T).sum()
    mx = (x * im.array.astype(T)).sum() / flux
    my = (y * im.array.astype(T)).sum() / flux
    mxx = (((x-mx)**2) * im.array.astype(T)).sum() / flux
    myy = (((y-my)**2) * im.array.astype(T)).sum() / flux
    mxy = ((x-mx) * (y-my) * im.array.astype(T)).sum() / flux
    s2 = mxx+myy
    print(flux,mx,my,mxx,myy,mxy)
    np.testing.assert_almost_equal((mxx-myy)/s2, 0, 5, "Found e1 != 0 for Exponential draw")
    np.testing.assert_almost_equal(2*mxy/s2, 0, 5, "Found e2 != 0 for Exponential draw")
    return np.sqrt(s2/6) * im.scale


@timer
def test_drawImage():
    """Test the various optional parameters to the drawImage function.
       In particular test the parameters image and dx in various combinations.
    """
    # We use a simple Exponential for our object:
    obj = galsim.Exponential(flux=test_flux, scale_radius=2)

    # First test drawImage() with method='no_pixel'.  It should:
    #   - create a new image
    #   - return the new image
    #   - set the scale to obj.nyquist_scale
    #   - set the size large enough to contain 99.5% of the flux
    im1 = obj.drawImage(method='no_pixel')
    nyq_scale = obj.nyquist_scale
    np.testing.assert_almost_equal(im1.scale, nyq_scale, 9,
                                   "obj.drawImage() produced image with wrong scale")
    np.testing.assert_equal(im1.bounds, galsim.BoundsI(1,56,1,56),
                            "obj.drawImage() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im1), 2, 1,
                                   "Measured wrong scale after obj.drawImage()")

    # The flux is only really expected to come out right if the object has been
    # convoled with a pixel:
    obj2 = galsim.Convolve([ obj, galsim.Pixel(im1.scale) ])
    im2 = obj2.drawImage(method='no_pixel')
    nyq_scale = obj2.nyquist_scale
    np.testing.assert_almost_equal(im2.scale, nyq_scale, 9,
                                   "obj2.drawImage() produced image with wrong scale")
    np.testing.assert_almost_equal(im2.array.astype(float).sum(), test_flux, 2,
                                   "obj2.drawImage() produced image with wrong flux")
    np.testing.assert_equal(im2.bounds, galsim.BoundsI(1,56,1,56),
                            "obj2.drawImage() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im2), 2, 1,
                                   "Measured wrong scale after obj2.drawImage()")
    # This should be the same as obj with method='auto'
    im2 = obj.drawImage()
    np.testing.assert_almost_equal(im2.scale, nyq_scale, 9,
                                   "obj2.drawImage() produced image with wrong scale")
    np.testing.assert_almost_equal(im2.array.astype(float).sum(), test_flux, 2,
                                   "obj2.drawImage() produced image with wrong flux")
    np.testing.assert_equal(im2.bounds, galsim.BoundsI(1,56,1,56),
                            "obj2.drawImage() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im2), 2, 1,
                                   "Measured wrong scale after obj2.drawImage()")

    # Test if we provide an image argument.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj2.nyquist_scale
    #   - zero out any existing data
    im3 = galsim.ImageD(56,56)
    im4 = obj.drawImage(im3)
    np.testing.assert_almost_equal(im3.scale, nyq_scale, 9,
                                   "obj.drawImage(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3.array.sum(), test_flux, 2,
                                   "obj.drawImage(im3) produced image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), im2.array.astype(float).sum(), 6,
                                   "obj.drawImage(im3) produced image with different flux than im2")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj.drawImage(im3)")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "im4 = obj.drawImage(im3) produced im4 != im3")
    im3.fill(9.8)
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "im4 = obj.drawImage(im3) produced im4 is not im3")
    im4 = obj.drawImage(im3)
    np.testing.assert_almost_equal(im3.array.sum(), im2.array.astype(float).sum(), 6,
                                   "obj.drawImage(im3) doesn't zero out existing data")

    # Test if we provide an image with undefined bounds.  It should:
    #   - resize the provided image
    #   - also return that image
    #   - set the scale to obj2.nyquist_scale
    im5 = galsim.ImageD()
    obj.drawImage(im5)
    np.testing.assert_almost_equal(im5.scale, nyq_scale, 9,
                                   "obj.drawImage(im5) produced image with wrong scale")
    np.testing.assert_almost_equal(im5.array.sum(), test_flux, 2,
                                   "obj.drawImage(im5) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im5), 2, 1,
                                   "Measured wrong scale after obj.drawImage(im5)")
    np.testing.assert_almost_equal(
        im5.array.sum(), im2.array.astype(float).sum(), 6,
        "obj.drawImage(im5) produced image with different flux than im2")
    np.testing.assert_equal(im5.bounds, galsim.BoundsI(1,56,1,56),
                            "obj.drawImage(im5) produced image with wrong bounds")

    # Test if we provide a dx to use.  It should:
    #   - create a new image using that dx for the scale
    #   - return the new image
    #   - set the size large enough to contain 99.5% of the flux
    scale = 0.51   # Just something different from 1 or dx_nyq
    im7 = obj.drawImage(scale=scale,method='no_pixel')
    np.testing.assert_almost_equal(im7.scale, scale, 9,
                                   "obj.drawImage(dx) produced image with wrong scale")
    np.testing.assert_almost_equal(im7.array.astype(float).sum(), test_flux, 2,
                                   "obj.drawImage(dx) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im7), 2, 1,
                                   "Measured wrong scale after obj.drawImage(dx)")
    np.testing.assert_equal(im7.bounds, galsim.BoundsI(1,68,1,68),
                            "obj.drawImage(dx) produced image with wrong bounds")

    # If also providing center, then same size, but centered near that center.
    for center in [(3,3), (210.2, 511.9), (10.55, -23.8), (0.5,0.5)]:
        im8 = obj.drawImage(scale=scale, center=center)
        np.testing.assert_almost_equal(im8.scale, scale, 9)
        # Note: it doesn't have to come out 68,68. If the offset is zero from the integer center,
        #       it drops down to (66, 66)
        if center == (3,3):
            np.testing.assert_equal(im8.array.shape, (66, 66))
        else:
            np.testing.assert_equal(im8.array.shape, (68, 68))
        np.testing.assert_almost_equal(im8.array.astype(float).sum(), test_flux, 2)
        print('center, true = ',center,im8.true_center)
        assert abs(center[0] - im8.true_center.x) <= 0.5
        assert abs(center[1] - im8.true_center.y) <= 0.5

    # Test if we provide an image with a defined scale.  It should:
    #   - write to the existing image
    #   - use the image's scale
    nx = 200  # Some randome size
    im9 = galsim.ImageD(nx,nx, scale=scale)
    obj.drawImage(im9)
    np.testing.assert_almost_equal(im9.scale, scale, 9,
                                   "obj.drawImage(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj.drawImage(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj.drawImage(im9)")

    # Test if we provide an image with a defined scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj2.nyquist_scale
    im9.scale = -scale
    im9.setZero()
    obj.drawImage(im9)
    np.testing.assert_almost_equal(im9.scale, nyq_scale, 9,
                                   "obj.drawImage(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj.drawImage(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj.drawImage(im9)")
    im9.scale = 0
    im9.setZero()
    obj.drawImage(im9)
    np.testing.assert_almost_equal(im9.scale, nyq_scale, 9,
                                   "obj.drawImage(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj.drawImage(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj.drawImage(im9)")


    # Test if we provide an image and dx.  It should:
    #   - write to the existing image
    #   - use the provided dx
    #   - write the new dx value to the image's scale
    im9.scale = 0.73
    im9.setZero()
    obj.drawImage(im9, scale=scale)
    np.testing.assert_almost_equal(im9.scale, scale, 9,
                                   "obj.drawImage(im9,dx) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj.drawImage(im9,dx) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj.drawImage(im9,dx)")

    # Test if we provide an image and dx <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj2.nyquist_scale
    im9.scale = 0.73
    im9.setZero()
    obj.drawImage(im9, scale=-scale)
    np.testing.assert_almost_equal(im9.scale, nyq_scale, 9,
                                   "obj.drawImage(im9,dx<0) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj.drawImage(im9,dx<0) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj.drawImage(im9,dx<0)")
    im9.scale = 0.73
    im9.setZero()
    obj.drawImage(im9, scale=0)
    np.testing.assert_almost_equal(im9.scale, nyq_scale, 9,
                                   "obj.drawImage(im9,scale=0) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj.drawImage(im9,scale=0) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj.drawImage(im9,scale=0)")

    # Test if we provide nx, ny, and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    ny = 100  # Make it non-square
    im10 = obj.drawImage(nx=nx, ny=ny, scale=scale)
    np.testing.assert_equal(im10.array.shape, (ny, nx),
                                   "obj.drawImage(nx,ny,scale) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, scale, 9,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong flux")
    mom = galsim.utilities.unweighted_moments(im10)
    np.testing.assert_almost_equal(
        mom['Mx'], (nx+1.)/2., 4, "obj.drawImage(nx,ny,scale) (even) did not center in x correctly")
    np.testing.assert_almost_equal(
        mom['My'], (ny+1.)/2., 4, "obj.drawImage(nx,ny,scale) (even) did not center in y correctly")

    # Repeat with odd nx,ny
    im10 = obj.drawImage(nx=nx+1, ny=ny+1, scale=scale)
    np.testing.assert_equal(im10.array.shape, (ny+1, nx+1),
                                   "obj.drawImage(nx,ny,scale) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, scale, 9,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong flux")
    mom = galsim.utilities.unweighted_moments(im10)
    np.testing.assert_almost_equal(
        mom['Mx'], (nx+1.+1.)/2., 4,
        "obj.drawImage(nx,ny,scale) (odd) did not center in x correctly")
    np.testing.assert_almost_equal(
        mom['My'], (ny+1.+1.)/2., 4,
        "obj.drawImage(nx,ny,scale) (odd) did not center in y correctly")

    # Test if we provide nx, ny, and no scale.  It should:
    #   - create a new image with the right size
    #   - set the scale to obj2.nyquist_scale
    im10 = obj.drawImage(nx=nx, ny=ny)
    np.testing.assert_equal(im10.array.shape, (ny, nx),
                                   "obj.drawImage(nx,ny) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, nyq_scale, 9,
                                   "obj.drawImage(nx,ny) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(nx,ny) produced image with wrong flux")
    mom = galsim.utilities.unweighted_moments(im10)
    np.testing.assert_almost_equal(
        mom['Mx'], (nx+1.)/2., 4, "obj.drawImage(nx,ny) (even) did not center in x correctly")
    np.testing.assert_almost_equal(
        mom['My'], (ny+1.)/2., 4, "obj.drawImage(nx,ny) (even) did not center in y correctly")

    # Repeat with odd nx,ny
    im10 = obj.drawImage(nx=nx+1, ny=ny+1)
    np.testing.assert_equal(im10.array.shape, (ny+1, nx+1),
                                   "obj.drawImage(nx,ny) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, nyq_scale, 9,
                                   "obj.drawImage(nx,ny) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(nx,ny) produced image with wrong flux")
    mom = galsim.utilities.unweighted_moments(im10)
    np.testing.assert_almost_equal(
        mom['Mx'], (nx+1.+1.)/2., 4, "obj.drawImage(nx,ny) (odd) did not center in x correctly")
    np.testing.assert_almost_equal(
        mom['My'], (ny+1.+1.)/2., 4, "obj.drawImage(nx,ny) (odd) did not center in y correctly")

    # Test if we provide bounds and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    bounds = galsim.BoundsI(1,nx,1,ny+1)
    im10 = obj.drawImage(bounds=bounds, scale=scale)
    np.testing.assert_equal(im10.array.shape, (ny+1, nx),
                                   "obj.drawImage(bounds,scale) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, scale, 9,
                                   "obj.drawImage(bounds,scale) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(bounds,scale) produced image with wrong flux")
    mom = galsim.utilities.unweighted_moments(im10)
    np.testing.assert_almost_equal(mom['Mx'], (nx+1.)/2., 4,
                                   "obj.drawImage(bounds,scale) did not center in x correctly")
    np.testing.assert_almost_equal(mom['My'], (ny+1.+1.)/2., 4,
                                   "obj.drawImage(bounds,scale) did not center in y correctly")

    # Test if we provide bounds and no scale.  It should:
    #   - create a new image with the right size
    #   - set the scale to obj2.nyquist_scale
    bounds = galsim.BoundsI(1,nx,1,ny+1)
    im10 = obj.drawImage(bounds=bounds)
    np.testing.assert_equal(im10.array.shape, (ny+1, nx),
                                   "obj.drawImage(bounds) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, nyq_scale, 9,
                                   "obj.drawImage(bounds) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(bounds) produced image with wrong flux")
    mom = galsim.utilities.unweighted_moments(im10)
    np.testing.assert_almost_equal(mom['Mx'], (nx+1.)/2., 4,
                                   "obj.drawImage(bounds) did not center in x correctly")
    np.testing.assert_almost_equal(mom['My'], (ny+1.+1.)/2., 4,
                                   "obj.drawImage(bounds) did not center in y correctly")

    # Test if we provide nx, ny, scale, and center.  It should:
    #   - create a new image with the right size
    #   - set the scale
    #   - set the center to be as close as possible to center
    for center in [(3,3), (10.2, 11.9), (10.55, -23.8)]:
        im11 = obj.drawImage(nx=nx, ny=ny, scale=scale, center=center)
        np.testing.assert_equal(im11.array.shape, (ny, nx))
        np.testing.assert_almost_equal(im11.scale, scale, 9)
        np.testing.assert_almost_equal(im11.array.sum(), test_flux, 4)
        print('center, true = ',center,im8.true_center)
        assert abs(center[0] - im11.true_center.x) <= 0.5
        assert abs(center[1] - im11.true_center.y) <= 0.5

        # Repeat with odd nx,ny
        im11 = obj.drawImage(nx=nx+1, ny=ny+1, scale=scale, center=center)
        np.testing.assert_equal(im11.array.shape, (ny+1, nx+1))
        np.testing.assert_almost_equal(im11.scale, scale, 9)
        np.testing.assert_almost_equal(im11.array.sum(), test_flux, 4)
        assert abs(center[0] - im11.true_center.x) <= 0.5
        assert abs(center[1] - im11.true_center.y) <= 0.5

    # Combinations that raise errors:
    assert_raises(TypeError, obj.drawImage, image=im10, bounds=bounds)
    assert_raises(TypeError, obj.drawImage, image=im10, dtype=int)
    assert_raises(TypeError, obj.drawImage, nx=3, ny=4, image=im10, scale=scale)
    assert_raises(TypeError, obj.drawImage, nx=3, ny=4, image=im10)
    assert_raises(TypeError, obj.drawImage, nx=3, ny=4, bounds=bounds)
    assert_raises(TypeError, obj.drawImage, nx=3, ny=4, add_to_image=True)
    assert_raises(TypeError, obj.drawImage, nx=3, ny=4, center=True)
    assert_raises(TypeError, obj.drawImage, nx=3, ny=4, center=23)
    assert_raises(TypeError, obj.drawImage, bounds=bounds, add_to_image=True)
    assert_raises(TypeError, obj.drawImage, image=galsim.Image(), add_to_image=True)
    assert_raises(TypeError, obj.drawImage, nx=3)
    assert_raises(TypeError, obj.drawImage, ny=3)
    assert_raises(TypeError, obj.drawImage, nx=3, ny=3, invalid=True)
    assert_raises(TypeError, obj.drawImage, bounds=bounds, scale=scale, wcs=galsim.PixelScale(3))
    assert_raises(TypeError, obj.drawImage, bounds=bounds, wcs=scale)
    assert_raises(TypeError, obj.drawImage, image=im10.array)
    assert_raises(TypeError, obj.drawImage, wcs=galsim.FitsWCS('fits_files/tpv.fits'))

    assert_raises(ValueError, obj.drawImage, bounds=galsim.BoundsI())
    assert_raises(ValueError, obj.drawImage, image=im10, gain=0.)
    assert_raises(ValueError, obj.drawImage, image=im10, gain=-1.)
    assert_raises(ValueError, obj.drawImage, image=im10, area=0.)
    assert_raises(ValueError, obj.drawImage, image=im10, area=-1.)
    assert_raises(ValueError, obj.drawImage, image=im10, exptime=0.)
    assert_raises(ValueError, obj.drawImage, image=im10, exptime=-1.)
    assert_raises(ValueError, obj.drawImage, image=im10, method='invalid')

    # These options are invalid unless metho=phot
    assert_raises(TypeError, obj.drawImage, image=im10, n_photons=3)
    assert_raises(TypeError, obj.drawImage, rng=galsim.BaseDeviate(234))
    assert_raises(TypeError, obj.drawImage, max_extra_noise=23)
    assert_raises(TypeError, obj.drawImage, poisson_flux=True)
    assert_raises(TypeError, obj.drawImage, maxN=10000)
    assert_raises(TypeError, obj.drawImage, save_photons=True)


@timer
def test_draw_methods():
    """Test the the different method options do the right thing.
    """
    # We use a simple Exponential for our object:
    obj = galsim.Exponential(flux=test_flux, scale_radius=1.09)
    test_scale = 0.28
    pix = galsim.Pixel(scale=test_scale)
    obj_pix = galsim.Convolve(obj, pix)

    N = 64
    im1 = galsim.ImageD(N, N, scale=test_scale)

    # auto and fft should be equivalent to drawing obj_pix with no_pixel
    im1 = obj.drawImage(image=im1)
    im2 = obj_pix.drawImage(image=im1.copy(), method='no_pixel')
    print('im1 flux diff = ',abs(im1.array.sum() - test_flux))
    np.testing.assert_almost_equal(
            im1.array.sum(), test_flux, 2,
            "obj.drawImage() produced image with wrong flux")
    print('im2 flux diff = ',abs(im2.array.sum() - test_flux))
    np.testing.assert_almost_equal(
            im2.array.sum(), test_flux, 2,
            "obj_pix.drawImage(no_pixel) produced image with wrong flux")
    print('im1, im2 max diff = ',abs(im1.array - im2.array).max())
    np.testing.assert_array_almost_equal(
            im1.array, im2.array, 6,
            "obj.drawImage() differs from obj_pix.drawImage(no_pixel)")
    im3 = obj.drawImage(image=im1.copy(), method='fft')
    print('im1, im3 max diff = ',abs(im1.array - im3.array).max())
    np.testing.assert_array_almost_equal(
            im1.array, im3.array, 6,
            "obj.drawImage(fft) differs from obj.drawImage")

    # real_space should be similar, but not precisely equal.
    im4 = obj.drawImage(image=im1.copy(), method='real_space')
    print('im1, im4 max diff = ',abs(im1.array - im4.array).max())
    np.testing.assert_array_almost_equal(
            im1.array, im4.array, 4,
            "obj.drawImage(real_space) differs from obj.drawImage")

    # sb should match xValue for pixel centers.  And be scale**2 factor different from no_pixel.
    im5 = obj.drawImage(image=im1.copy(), method='sb', use_true_center=False)
    im5.setCenter(0,0)
    print('im5(0,0) = ',im5(0,0))
    print('obj.xValue(0,0) = ',obj.xValue(0.,0.))
    np.testing.assert_almost_equal(
            im5(0,0), obj.xValue(0.,0.), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    np.testing.assert_almost_equal(
            im5(3,2), obj.xValue(3*test_scale, 2*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    im5 = obj.drawImage(image=im5, method='sb')
    print('im5(0,0) = ',im5(0,0))
    print('obj.xValue(dx/2,dx/2) = ',obj.xValue(test_scale/2., test_scale/2.))
    np.testing.assert_almost_equal(
            im5(0,0), obj.xValue(0.5*test_scale, 0.5*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    np.testing.assert_almost_equal(
            im5(3,2), obj.xValue(3.5*test_scale, 2.5*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    im6 = obj.drawImage(image=im1.copy(), method='no_pixel')
    print('im6, im5*scale**2 max diff = ',abs(im6.array - im5.array*test_scale**2).max())
    np.testing.assert_array_almost_equal(
            im5.array * test_scale**2, im6.array, 6,
            "obj.drawImage(sb) * scale**2 differs from obj.drawImage(no_pixel)")

    # Drawing a truncated object, auto should be identical to real_space
    obj = galsim.Sersic(flux=test_flux, n=3.7, half_light_radius=2, trunc=4)
    obj_pix = galsim.Convolve(obj, pix)

    # auto and real_space should be equivalent to drawing obj_pix with no_pixel
    im1 = obj.drawImage(image=im1)
    im2 = obj_pix.drawImage(image=im1.copy(), method='no_pixel')
    print('im1 flux diff = ',abs(im1.array.sum() - test_flux))
    np.testing.assert_almost_equal(
            im1.array.sum(), test_flux, 2,
            "obj.drawImage() produced image with wrong flux")
    print('im2 flux diff = ',abs(im2.array.sum() - test_flux))
    np.testing.assert_almost_equal(
            im2.array.sum(), test_flux, 2,
            "obj_pix.drawImage(no_pixel) produced image with wrong flux")
    print('im1, im2 max diff = ',abs(im1.array - im2.array).max())
    np.testing.assert_array_almost_equal(
            im1.array, im2.array, 6,
            "obj.drawImage() differs from obj_pix.drawImage(no_pixel)")
    im4 = obj.drawImage(image=im1.copy(), method='real_space')
    print('im1, im4 max diff = ',abs(im1.array - im4.array).max())
    np.testing.assert_array_almost_equal(
            im1.array, im4.array, 6,
            "obj.drawImage(real_space) differs from obj.drawImage")

    # fft should be similar, but not precisely equal.
    with assert_warns(galsim.GalSimWarning):
        # This emits a warning about convolving two things with hard edges.
        im3 = obj.drawImage(image=im1.copy(), method='fft')
    print('im1, im3 max diff = ',abs(im1.array - im3.array).max())
    np.testing.assert_array_almost_equal(
            im1.array, im3.array, 3, # Should be close, but not exact.
            "obj.drawImage(fft) differs from obj.drawImage")

    # sb should match xValue for pixel centers.  And be scale**2 factor different from no_pixel.
    im5 = obj.drawImage(image=im1.copy(), method='sb')
    im5.setCenter(0,0)
    print('im5(0,0) = ',im5(0,0))
    print('obj.xValue(dx/2,dx/2) = ',obj.xValue(test_scale/2., test_scale/2.))
    np.testing.assert_almost_equal(
            im5(0,0), obj.xValue(0.5*test_scale, 0.5*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    np.testing.assert_almost_equal(
            im5(3,2), obj.xValue(3.5*test_scale, 2.5*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    im6 = obj.drawImage(image=im1.copy(), method='no_pixel')
    print('im6, im5*scale**2 max diff = ',abs(im6.array - im5.array*test_scale**2).max())
    np.testing.assert_array_almost_equal(
            im5.array * test_scale**2, im6.array, 6,
            "obj.drawImage(sb) * scale**2 differs from obj.drawImage(no_pixel)")


@timer
def test_drawKImage():
    """Test the various optional parameters to the drawKImage function.
       In particular test the parameters image, and scale in various combinations.
    """
    # We use a Moffat profile with beta = 1.5, since its real-space profile is
    #    flux / (2 pi rD^2) * (1 + (r/rD)^2)^3/2
    # and the 2-d Fourier transform of that is
    #    flux * exp(-rD k)
    # So this should draw in Fourier space the same image as the Exponential drawn in
    # test_drawImage().
    obj = galsim.Moffat(flux=test_flux, beta=1.5, scale_radius=0.5)
    obj = obj.withGSParams(maxk_threshold=1.e-4)

    # First test drawKImage() with no kwargs.  It should:
    #   - create new images
    #   - return the new images
    #   - set the scale to 2pi/(N*obj.nyquist_scale)
    im1 = obj.drawKImage()
    N = 1174
    np.testing.assert_equal(im1.bounds, galsim.BoundsI(-N/2,N/2,-N/2,N/2),
                            "obj.drawKImage() produced image with wrong bounds")
    stepk = obj.stepk
    np.testing.assert_almost_equal(im1.scale, stepk, 9,
                                   "obj.drawKImage() produced image with wrong scale")
    np.testing.assert_almost_equal(CalculateScale(im1), 2, 1,
                                   "Measured wrong scale after obj.drawKImage()")

    # The flux in Fourier space is just the value at k=0
    np.testing.assert_equal(im1.bounds.center, galsim.PositionI(0,0))
    np.testing.assert_almost_equal(im1(0,0), test_flux, 2,
                                   "obj.drawKImage() produced image with wrong flux")
    # Imaginary component should all be 0.
    np.testing.assert_almost_equal(im1.imag.array.sum(), 0., 3,
                                   "obj.drawKImage() produced non-zero imaginary image")

    # Test if we provide an image argument.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj.stepk
    #   - zero out any existing data
    im3 = galsim.ImageCD(1149,1149)
    im4 = obj.drawKImage(im3)
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3(0,0), test_flux, 2,
                                   "obj.drawKImage(im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.imag.array.sum(), 0., 3,
                                   "obj.drawKImage(im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(im3)")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "im4 = obj.drawKImage(im3) produced im4 != im3")
    im3.fill(9.8)
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "im4 = obj.drawKImage(im3) produced im4 is not im3")

    # Test if we provide an image with undefined bounds.  It should:
    #   - resize the provided image
    #   - also return that image
    #   - set the scale to obj.stepk
    im5 = galsim.ImageCD()
    obj.drawKImage(im5)
    np.testing.assert_almost_equal(im5.scale, stepk, 9,
                                   "obj.drawKImage(im5) produced image with wrong scale")
    np.testing.assert_almost_equal(im5(0,0), test_flux, 2,
                                   "obj.drawKImage(im5) produced image with wrong flux")
    np.testing.assert_almost_equal(im5.imag.array.sum(), 0., 3,
                                   "obj.drawKImage(im5) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im5), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(im5)")
    np.testing.assert_equal(im5.bounds, galsim.BoundsI(-N/2,N/2,-N/2,N/2),
                            "obj.drawKImage(im5) produced image with wrong bounds")

    # Test if we provide a scale to use.  It should:
    #   - create a new image using that scale for the scale
    #   - return the new image
    #   - set the size large enough to contain 99.5% of the flux
    scale = 0.51   # Just something different from 1 or stepk
    im7 = obj.drawKImage(scale=scale)
    np.testing.assert_almost_equal(im7.scale, scale, 9,
                                   "obj.drawKImage(dx) produced image with wrong scale")
    np.testing.assert_almost_equal(im7(0,0), test_flux, 2,
                                   "obj.drawKImage(dx) produced image with wrong flux")
    np.testing.assert_almost_equal(im7.imag.array.astype(float).sum(), 0., 2,
                                   "obj.drawKImage(dx) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im7), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(dx)")
    # This image is smaller because not using nyquist scale for stepk
    np.testing.assert_equal(im7.bounds, galsim.BoundsI(-37,37,-37,37),
                            "obj.drawKImage(dx) produced image with wrong bounds")

    # Test if we provide an image with a defined scale.  It should:
    #   - write to the existing image
    #   - use the image's scale
    nx = 401
    im9 = galsim.ImageCD(nx,nx, scale=scale)
    obj.drawKImage(im9)
    np.testing.assert_almost_equal(im9.scale, scale, 9,
                                   "obj.drawKImage(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9(0,0), test_flux, 4,
                                   "obj.drawKImage(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(im9.imag.array.sum(), 0., 5,
                                   "obj.drawKImage(im9) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(im9)")

    # Test if we provide an image with a defined scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj.stepk
    im3.scale = -scale
    im3.setZero()
    obj.drawKImage(im3)
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3(0,0), test_flux, 4,
                                   "obj.drawKImage(im3) produced image with wrong flux")
    np.testing.assert_almost_equal(im3.imag.array.sum(), 0., 5,
                                   "obj.drawKImage(im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(im3)")
    im3.scale = 0
    im3.setZero()
    obj.drawKImage(im3)
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3(0,0), test_flux, 4,
                                   "obj.drawKImage(im3) produced image with wrong flux")
    np.testing.assert_almost_equal(im3.imag.array.sum(), 0., 5,
                                   "obj.drawKImage(im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(im3)")

    # Test if we provide an image and dx.  It should:
    #   - write to the existing image
    #   - use the provided dx
    #   - write the new dx value to the image's scale
    im9.scale = scale + 0.3  # Just something other than scale
    im9.setZero()
    obj.drawKImage(im9, scale=scale)
    np.testing.assert_almost_equal(
            im9.scale, scale, 9,
            "obj.drawKImage(im9,scale) produced image with wrong scale")
    np.testing.assert_almost_equal(
            im9(0,0), test_flux, 4,
            "obj.drawKImage(im9,scale) produced image with wrong flux")
    np.testing.assert_almost_equal(
            im9.imag.array.sum(), 0., 5,
            "obj.drawKImage(im9,scale) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(im9,scale)")

    # Test if we provide an image and scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj.stepk
    im3.scale = scale + 0.3
    im3.setZero()
    obj.drawKImage(im3, scale=-scale)
    np.testing.assert_almost_equal(
            im3.scale, stepk, 9,
            "obj.drawKImage(im3,scale<0) produced image with wrong scale")
    np.testing.assert_almost_equal(
            im3(0,0), test_flux, 4,
            "obj.drawKImage(im3,scale<0) produced image with wrong flux")
    np.testing.assert_almost_equal(
            im3.imag.array.sum(), 0., 5,
            "obj.drawKImage(im3,scale<0) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(im3,scale<0)")
    im3.scale = scale + 0.3
    im3.setZero()
    obj.drawKImage(im3, scale=0)
    np.testing.assert_almost_equal(
            im3.scale, stepk, 9,
            "obj.drawKImage(im3,scale=0) produced image with wrong scale")
    np.testing.assert_almost_equal(
            im3(0,0), test_flux, 4,
            "obj.drawKImage(im3,scale=0) produced image with wrong flux")
    np.testing.assert_almost_equal(
            im3.imag.array.sum(), 0., 5,
            "obj.drawKImage(im3,scale=0) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(im3,scale=0)")

    # Test if we provide nx, ny, and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    nx = 200  # Some randome non-square size
    ny = 100
    im4 = obj.drawKImage(nx=nx, ny=ny, scale=scale)
    np.testing.assert_almost_equal(
            im4.scale, scale, 9,
            "obj.drawKImage(nx,ny,scale) produced image with wrong scale")
    np.testing.assert_equal(
            im4.array.shape, (ny, nx),
            "obj.drawKImage(nx,ny,scale) produced image with wrong shape")

    # Test if we provide nx, ny, and no scale.  It should:
    #   - create a new image with the right size
    #   - set the scale to obj.stepk
    im4 = obj.drawKImage(nx=nx, ny=ny)
    np.testing.assert_almost_equal(
            im4.scale, stepk, 9,
            "obj.drawKImage(nx,ny) produced image with wrong scale")
    np.testing.assert_equal(
            im4.array.shape, (ny, nx),
            "obj.drawKImage(nx,ny) produced image with wrong shape")

    # Test if we provide bounds and no scale.  It should:
    #   - create a new image with the right size
    #   - set the scale to obj.stepk
    bounds = galsim.BoundsI(1,nx,1,ny)
    im4 = obj.drawKImage(bounds=bounds)
    np.testing.assert_almost_equal(
            im4.scale, stepk, 9,
            "obj.drawKImage(bounds) produced image with wrong scale")
    np.testing.assert_equal(
            im4.array.shape, (ny, nx),
            "obj.drawKImage(bounds) produced image with wrong shape")

    # Test if we provide bounds and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    bounds = galsim.BoundsI(1,nx,1,ny)
    im4 = obj.drawKImage(bounds=bounds, scale=scale)
    np.testing.assert_almost_equal(
            im4.scale, scale, 9,
            "obj.drawKImage(bounds,scale) produced image with wrong scale")
    np.testing.assert_equal(
            im4.array.shape, (ny, nx),
            "obj.drawKImage(bounds,scale) produced image with wrong shape")

    # Test recenter = False option
    bounds6 = galsim.BoundsI(0, nx//3, 0, ny//4)
    im6 = obj.drawKImage(bounds=bounds6, scale=scale, recenter=False)
    np.testing.assert_equal(
            im6.bounds, bounds6,
            "obj.drawKImage(bounds,scale,recenter=False) produced image with wrong bounds")
    np.testing.assert_almost_equal(
            im6.scale, scale, 9,
            "obj.drawKImage(bounds,scale,recenter=False) produced image with wrong scale")
    np.testing.assert_equal(
            im6.array.shape, (ny//4+1, nx//3+1),
            "obj.drawKImage(bounds,scale,recenter=False) produced image with wrong shape")
    np.testing.assert_almost_equal(
            im6.array, im4[bounds6].array, 9,
            "obj.drawKImage(recenter=False) produced different values than recenter=True")

    # Test recenter = False option
    im6.setZero()
    obj.drawKImage(im6, recenter=False)
    np.testing.assert_almost_equal(
            im6.scale, scale, 9,
            "obj.drawKImage(image,recenter=False) produced image with wrong scale")
    np.testing.assert_almost_equal(
            im6.array, im4[bounds6].array, 9,
            "obj.drawKImage(image,recenter=False) produced different values than recenter=True")

    # Can add to image if recenter is False
    im6.setZero()
    obj.drawKImage(im6, recenter=False, add_to_image=True)
    np.testing.assert_almost_equal(
            im6.scale, scale, 9,
            "obj.drawKImage(image,add_to_image=True) produced image with wrong scale")
    np.testing.assert_almost_equal(
            im6.array, im4[bounds6].array, 9,
            "obj.drawKImage(image,add_to_image=True) produced different values than recenter=True")

    # .. or if image is centered.
    im7 = im4.copy()
    im7.setZero()
    im7.setCenter(0,0)
    obj.drawKImage(im7, add_to_image=True)
    np.testing.assert_almost_equal(
            im7.scale, scale, 9,
            "obj.drawKImage(image,add_to_image=True) produced image with wrong scale")
    np.testing.assert_almost_equal(
            im7.array, im4.array, 9,
            "obj.drawKImage(image,add_to_image=True) produced different values than recenter=True")

    # .. but otherwise not.
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        obj.drawKImage(image=im6, add_to_image=True)

    # Other error combinations:
    assert_raises(TypeError, obj.drawKImage, image=im6, bounds=bounds)
    assert_raises(TypeError, obj.drawKImage, image=im6, dtype=int)
    assert_raises(TypeError, obj.drawKImage, nx=3, ny=4, image=im6, scale=scale)
    assert_raises(TypeError, obj.drawKImage, nx=3, ny=4, image=im6)
    assert_raises(TypeError, obj.drawKImage, nx=3, ny=4, add_to_image=True)
    assert_raises(TypeError, obj.drawKImage, nx=3, ny=4, bounds=bounds)
    assert_raises(TypeError, obj.drawKImage, bounds=bounds, add_to_image=True)
    assert_raises(TypeError, obj.drawKImage, image=galsim.Image(dtype=complex), add_to_image=True)
    assert_raises(TypeError, obj.drawKImage, nx=3)
    assert_raises(TypeError, obj.drawKImage, ny=3)
    assert_raises(TypeError, obj.drawKImage, nx=3, ny=3, invalid=True)
    assert_raises(TypeError, obj.drawKImage, bounds=bounds, wcs=galsim.PixelScale(3))
    assert_raises(TypeError, obj.drawKImage, image=im6.array)
    assert_raises(ValueError, obj.drawKImage, image=galsim.ImageF(3,4))
    assert_raises(ValueError, obj.drawKImage, bounds=galsim.BoundsI())


@timer
def test_drawKImage_Gaussian():
    """Test the drawKImage function using known symmetries of the Gaussian Hankel transform.

    See http://en.wikipedia.org/wiki/Hankel_transform.
    """
    test_flux = 2.3     # Choose a non-unity flux
    test_sigma = 17.    # ...likewise for sigma
    test_imsize = 45    # Dimensions of comparison image, doesn't need to be large

    # Define a Gaussian GSObject
    gal = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    # Then define a related object which is in fact the opposite number in the Hankel transform pair
    # For the Gaussian this is straightforward in our definition of the Fourier transform notation,
    # and has sigma -> 1/sigma and flux -> flux * 2 pi / sigma**2
    gal_hankel = galsim.Gaussian(sigma=1./test_sigma, flux=test_flux*2.*np.pi/test_sigma**2)

    # Do a basic flux test: the total flux of the gal should equal gal_Hankel(k=(0, 0))
    np.testing.assert_almost_equal(
        gal.flux, gal_hankel.xValue(galsim.PositionD(0., 0.)), decimal=12,
        err_msg="Test object flux does not equal k=(0, 0) mode of its Hankel transform conjugate.")

    image_test = galsim.ImageD(test_imsize, test_imsize)
    kimage_test = galsim.ImageCD(test_imsize, test_imsize)

    # Then compare these two objects at a couple of different scale (reasonably matched for size)
    for scale_test in (0.03 / test_sigma, 0.4 / test_sigma):
        gal.drawKImage(image=kimage_test, scale=scale_test)
        gal_hankel.drawImage(image_test, scale=scale_test, use_true_center=False, method='sb')
        np.testing.assert_array_almost_equal(
            kimage_test.real.array, image_test.array, decimal=12,
            err_msg="Test object drawKImage() and drawImage() from Hankel conjugate do not match "
            "for grid spacing scale = "+str(scale_test))
        np.testing.assert_array_almost_equal(
            kimage_test.imag.array, 0., decimal=12,
            err_msg="Non-zero imaginary part for drawKImage from test object that is purely "
            "centred on the origin.")


@timer
def test_drawKImage_Exponential_Moffat():
    """Test the drawKImage function using known symmetries of the Exponential Hankel transform
    (which is a Moffat with beta=1.5).

    See http://mathworld.wolfram.com/HankelTransform.html.
    """
    test_flux = 4.1         # Choose a non-unity flux
    test_scale_radius = 13. # ...likewise for scale_radius
    test_imsize = 45        # Dimensions of comparison image, doesn't need to be large

    # Define an Exponential GSObject
    gal = galsim.Exponential(scale_radius=test_scale_radius, flux=test_flux)
    # Then define a related object which is in fact the opposite number in the Hankel transform pair
    # For the Exponential we need a Moffat, with scale_radius=1/scale_radius.  The total flux under
    # this Moffat with unit amplitude at r=0 is is pi * scale_radius**(-2) / (beta - 1)
    #  = 2. * pi * scale_radius**(-2) in this case, so it works analagously to the Gaussian above.
    gal_hankel = galsim.Moffat(beta=1.5, scale_radius=1. / test_scale_radius,
                               flux=test_flux * 2. * np.pi / test_scale_radius**2)

    # Do a basic flux test: the total flux of the gal should equal gal_Hankel(k=(0, 0))
    np.testing.assert_almost_equal(
        gal.flux, gal_hankel.xValue(galsim.PositionD(0., 0.)), decimal=12,
        err_msg="Test object flux does not equal k=(0, 0) mode of its Hankel transform conjugate.")

    image_test = galsim.ImageD(test_imsize, test_imsize)
    kimage_test = galsim.ImageCD(test_imsize, test_imsize)

    # Then compare these two objects at a couple of different scale (reasonably matched for size)
    for scale_test in (0.15 / test_scale_radius, 0.6 / test_scale_radius):
        gal.drawKImage(image=kimage_test, scale=scale_test)
        gal_hankel.drawImage(image_test, scale=scale_test, use_true_center=False, method='sb')
        np.testing.assert_array_almost_equal(
            kimage_test.real.array, image_test.array, decimal=12,
            err_msg="Test object drawKImageImage() and drawImage() from Hankel conjugate do not "+
            "match for grid spacing scale = "+str(scale_test))
        np.testing.assert_array_almost_equal(
            kimage_test.imag.array, 0., decimal=12,
            err_msg="Non-zero imaginary part for drawKImage from test object that is purely "+
            "centred on the origin.")


@timer
def test_offset():
    """Test the offset parameter to the drawImage function.
    """
    scale = 0.23

    # Use some more exact GSParams.  We'll be comparing FFT images to real-space convolved values,
    # so we don't want to suffer from our overall accuracy being only about 10^-3.
    # Update: It turns out the only one I needed to reduce to obtain the accuracy I wanted
    # below is maxk_threshold.  Perhaps this is a sign that we ought to lower it in general?
    params = galsim.GSParams(maxk_threshold=1.e-4)

    # We use a simple Exponential for our object:
    gal = galsim.Exponential(flux=test_flux, scale_radius=0.5, gsparams=params)
    pix = galsim.Pixel(scale, gsparams=params)
    obj = galsim.Convolve([gal,pix], gsparams=params)

    # The shapes of the images we will build
    # Make sure all combinations of odd/even are represented.
    shape_list = [ (256,256), (256,243), (249,260), (255,241), (270,260) ]

    # Some reasonable (x,y) values at which to test the xValues (near the center)
    xy_list = [ (128,128), (123,131), (126,124) ]

    # The offsets to test
    offset_list = [ (1,-3), (0.3,-0.1), (-2.3,-1.2) ]

    # Make the images somewhat large so the moments are measured accurately.
    for nx,ny in shape_list:

        # First check that the image agrees with our calculation of the center
        cenx = (nx+1.)/2.
        ceny = (ny+1.)/2.
        im = galsim.ImageD(nx,ny, scale=scale)
        true_center = im.bounds.true_center
        np.testing.assert_almost_equal(
                cenx, true_center.x, 6,
                "im.bounds.true_center.x is wrong for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                ceny, true_center.y, 6,
                "im.bounds.true_center.y is wrong for (nx,ny) = %d,%d"%(nx,ny))

        # Check that the default draw command puts the centroid in the center of the image.
        obj.drawImage(im, method='sb')
        mom = galsim.utilities.unweighted_moments(im)
        np.testing.assert_almost_equal(
                mom['Mx'], cenx, 5,
                "obj.drawImage(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                mom['My'], ceny, 5,
                "obj.drawImage(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))

        # Can also use center to explicitly say we want to use the true_center.
        im3 = obj.drawImage(im.copy(), method='sb', center=im.true_center)
        np.testing.assert_almost_equal(im3.array, im.array)

        # Test that a few pixel values match xValue.
        # Note: we don't expect the FFT drawn image to match the xValues precisely, since the
        # latter use real-space convolution, so they should just match to our overall accuracy
        # requirement, which is something like 1.e-3 or so.  But an image of just the galaxy
        # should use real-space drawing, so should be pretty much exact.
        im2 = galsim.ImageD(nx,ny, scale=scale)
        gal.drawImage(im2, method='sb')
        for x,y in xy_list:
            u = (x-cenx) * scale
            v = (y-ceny) * scale
            np.testing.assert_almost_equal(
                    im(x,y), obj.xValue(galsim.PositionD(u,v)), 2,
                    "im(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))
            np.testing.assert_almost_equal(
                    im2(x,y), gal.xValue(galsim.PositionD(u,v)), 6,
                    "im2(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))

        # Check that offset moves the centroid by the right amount.
        for offx, offy in offset_list:
            # For integer offsets, we expect the centroids to come out pretty much exact.
            # (Only edge effects of the image should produce any error, and those are very small.)
            # However, for non-integer effects, we don't actually expect the centroids to be
            # right, even with perfect image rendering.  To see why, imagine using a delta function
            # for the galaxy.  The centroid changes discretely, not continuously as the offset
            # varies.  The effect isn't as severe of course for our Exponential, but the effect
            # is still there in part.  Hence, only use 2 decimal places for non-integer offsets.
            if offx == int(offx) and offy == int(offy):
                decimal = 4
            else:
                decimal = 2

            offset = galsim.PositionD(offx,offy)
            obj.drawImage(im, method='sb', offset=offset)
            mom = galsim.utilities.unweighted_moments(im)
            np.testing.assert_almost_equal(
                    mom['Mx'], cenx+offx, decimal,
                    "obj.drawImage(im,offset) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
            np.testing.assert_almost_equal(
                    mom['My'], ceny+offy, decimal,
                    "obj.drawImage(im,offset) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
            # Test that a few pixel values match xValue
            gal.drawImage(im2, method='sb', offset=offset)
            for x,y in xy_list:
                u = (x-cenx-offx) * scale
                v = (y-ceny-offy) * scale
                np.testing.assert_almost_equal(
                        im(x,y), obj.xValue(galsim.PositionD(u,v)), 2,
                        "im(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))
                np.testing.assert_almost_equal(
                        im2(x,y), gal.xValue(galsim.PositionD(u,v)), 6,
                        "im2(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))

            # Check that shift also moves the centroid by the right amount.
            shifted_obj = obj.shift(offset * scale)
            shifted_obj.drawImage(im, method='sb')
            mom = galsim.utilities.unweighted_moments(im)
            np.testing.assert_almost_equal(
                    mom['Mx'], cenx+offx, decimal,
                    "shifted_obj.drawImage(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
            np.testing.assert_almost_equal(
                    mom['My'], ceny+offy, decimal,
                    "shifted_obj.drawImage(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
            # Test that a few pixel values match xValue
            shifted_gal = gal.shift(offset * scale)
            shifted_gal.drawImage(im2, method='sb')
            for x,y in xy_list:
                u = (x-cenx) * scale
                v = (y-ceny) * scale
                np.testing.assert_almost_equal(
                        im(x,y), shifted_obj.xValue(galsim.PositionD(u,v)), 2,
                        "im(%d,%d) does not match shifted xValue(%f,%f)"%(x,y,x-cenx,y-ceny))
                np.testing.assert_almost_equal(
                        im2(x,y), shifted_gal.xValue(galsim.PositionD(u,v)), 6,
                        "im2(%d,%d) does not match shifted xValue(%f,%f)"%(x,y,x-cenx,y-ceny))
                u = (x-cenx-offx) * scale
                v = (y-ceny-offy) * scale
                np.testing.assert_almost_equal(
                        im(x,y), obj.xValue(galsim.PositionD(u,v)), 2,
                        "im(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))
                np.testing.assert_almost_equal(
                        im2(x,y), gal.xValue(galsim.PositionD(u,v)), 6,
                        "im2(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))

            # Test that the center parameter can be used to do the same thing.
            center = galsim.PositionD(cenx + offx, ceny + offy)
            im3 = obj.drawImage(im.copy(), method='sb', center=center)
            np.testing.assert_almost_equal(im3.array, im.array)
            assert im3.bounds == im.bounds
            assert im3.wcs == im.wcs

            # Can also use both offset and center
            im3 = obj.drawImage(im.copy(), method='sb',
                                center=(cenx-1, ceny+1), offset=(offx+1, offy-1))
            np.testing.assert_almost_equal(im3.array, im.array)
            assert im3.bounds == im.bounds
            assert im3.wcs == im.wcs

        # Check the image's definition of the nominal center
        nom_cenx = (nx+2)//2
        nom_ceny = (ny+2)//2
        nominal_center = im.bounds.center
        np.testing.assert_almost_equal(
                nom_cenx, nominal_center.x, 6,
                "im.bounds.center.x is wrong for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                nom_ceny, nominal_center.y, 6,
                "im.bounds.center.y is wrong for (nx,ny) = %d,%d"%(nx,ny))

        # Check that use_true_center = false is consistent with an offset by 0 or 0.5 pixels.
        obj.drawImage(im, method='sb', use_true_center=False)
        mom = galsim.utilities.unweighted_moments(im)
        np.testing.assert_almost_equal(
                mom['Mx'], nom_cenx, 4,
                "obj.drawImage(im, use_true_center=False) not centered correctly for (nx,ny) = "+
                "%d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                mom['My'], nom_ceny, 4,
                "obj.drawImage(im, use_true_center=False) not centered correctly for (nx,ny) = "+
                "%d,%d"%(nx,ny))
        cen_offset = galsim.PositionD(nom_cenx - cenx, nom_ceny - ceny)
        obj.drawImage(im2, method='sb', offset=cen_offset)
        np.testing.assert_array_almost_equal(
                im.array, im2.array, 6,
                "obj.drawImage(im, offset=%f,%f) different from use_true_center=False")

        # Can also use center to explicitly say to use the integer center
        im3 = obj.drawImage(im.copy(), method='sb', center=im.center)
        np.testing.assert_almost_equal(im3.array, im.array)

@timer
def test_shoot():
    """Test drawImage(..., method='phot')

    Most tests of the photon shooting method are done using the `do_shoot` function calls
    in various places.  Here we test other aspects of photon shooting that are not fully
    covered by these other tests.
    """
    # This test comes from a bug report by Jim Chiang on issue #866.  There was a rounding
    # problem when the number of photons to shoot came out to 100,000 + 1.  It did the first
    # 100,000 and then was left with 1, but rounding errors (since it is a double, not an int)
    # was 1 - epsilon, and it ended up in a place where it shouldn't have been able to get to
    # in exact arithmetic.  We had an assert there which blew up in a not very nice way.
    obj = galsim.Gaussian(sigma=0.2398318) + 0.1*galsim.Gaussian(sigma=0.47966352)
    obj = obj.withFlux(100001)
    image1 = galsim.ImageF(32,32, init_value=100)
    rng = galsim.BaseDeviate(1234)
    obj.drawImage(image1, method='phot', poisson_flux=False, add_to_image=True, rng=rng,
                  maxN=100000)

    # The test here is really just that it doesn't crash.
    # But let's do something to check correctness.
    image2 = galsim.ImageF(32,32)
    rng = galsim.BaseDeviate(1234)
    obj.drawImage(image2, method='phot', poisson_flux=False, add_to_image=False, rng=rng,
                  maxN=100000)
    image2 += 100
    np.testing.assert_almost_equal(image2.array, image1.array, decimal=12)

    # Also check that you get the same answer with a smaller maxN.
    image3 = galsim.ImageF(32,32, init_value=100)
    rng = galsim.BaseDeviate(1234)
    obj.drawImage(image3, method='phot', poisson_flux=False, add_to_image=True, rng=rng, maxN=1000)
    # It's not exactly the same, since the rngs are realized in a different order.
    np.testing.assert_allclose(image3.array, image1.array, rtol=0.25)

    # Test that shooting with 0.0 flux makes a zero-photons image.
    image4 = (obj*0).drawImage(method='phot')
    np.testing.assert_equal(image4.array, 0)

    # Warns if flux is 1 and n_photons not given.
    psf = galsim.Gaussian(sigma=3)
    with assert_warns(galsim.GalSimWarning):
        psf.drawImage(method='phot')
    with assert_warns(galsim.GalSimWarning):
        psf.drawPhot(image4)
    with assert_warns(galsim.GalSimWarning):
        psf.makePhot()
    # With n_photons=1, it's fine.
    psf.drawImage(method='phot', n_photons=1)
    psf.drawPhot(image4, n_photons=1)
    psf.makePhot(n_photons=1)

    # Check negative flux shooting with poisson_flux=True
    # The do_shoot test in galsim_test_helpers checks negative flux with a fixed number of photons.
    # But we also want to check that the automatic number of photons is reaonable when the flux
    # is negative.
    obj = obj.withFlux(-1.e5)
    image3 = galsim.ImageF(64,64)
    obj.drawImage(image3, method='phot', poisson_flux=True, rng=rng)
    print('image3.sum = ',image3.array.sum())
    # Only accurate to about sqrt(1.e5) from Poisson realization
    np.testing.assert_allclose(image3.array.sum(), obj.flux, rtol=0.01)


@timer
def test_drawImage_area_exptime():
    """Test that area and exptime kwargs to drawImage() appropriately scale image."""
    exptime = 2
    area = 1.4

    # We will be photon shooting, so use largish flux.
    obj = galsim.Exponential(flux=1776., scale_radius=2)

    im1 = obj.drawImage(nx=24, ny=24, scale=0.3)
    im2 = obj.drawImage(image=im1.copy(), exptime=exptime, area=area)
    np.testing.assert_array_almost_equal(im1.array, im2.array/exptime/area, 5,
            "obj.drawImage() did not respect area and exptime kwargs.")

    # Now check with drawShoot().  Scaling the gain should just scale the image proportionally.
    # Scaling the area or exptime should actually produce a non-proportional image, though, since a
    # different number of photons will be shot.

    rng = galsim.BaseDeviate(1234)
    im1 = obj.drawImage(nx=24, ny=24, scale=0.3, method='phot', rng=rng.duplicate())
    im2 = obj.drawImage(image=im1.copy(), method='phot', rng=rng.duplicate())
    np.testing.assert_array_almost_equal(im1.array, im2.array, 5,
            "obj.drawImage(method='phot', rng=rng.duplicate()) did not produce image "
            "deterministically.")
    im3 = obj.drawImage(image=im1.copy(), method='phot', rng=rng.duplicate(), gain=2)
    np.testing.assert_array_almost_equal(im1.array, im3.array*2, 5,
            "obj.drawImage(method='phot', rng=rng.duplicate(), gain=2) did not produce image "
            "deterministically.")

    im4 = obj.drawImage(image=im1.copy(), method='phot', rng=rng.duplicate(),
                        area=area, exptime=exptime)
    msg = ("obj.drawImage(method='phot') unexpectedly produced proportional images with different "
           "area and exptime keywords.")
    assert not np.allclose(im1.array, im4.array/area/exptime), msg

    im5 = obj.drawImage(image=im1.copy(), method='phot', area=area, exptime=exptime)
    msg = "obj.drawImage(method='phot') unexpectedly produced equal images with different rng"
    assert not np.allclose(im5.array, im4.array), msg

    # Shooting with flux=1 raises a warning.
    obj1 = obj.withFlux(1)
    with assert_warns(galsim.GalSimWarning):
        obj1.drawImage(method='phot')
    # But not if we explicitly tell it to shoot 1 photon
    with assert_raises(AssertionError):
        assert_warns(galsim.GalSimWarning, obj1.drawImage, method='phot', n_photons=1)
    # Likewise for makePhot
    with assert_warns(galsim.GalSimWarning):
        obj1.makePhot()
    with assert_raises(AssertionError):
        assert_warns(galsim.GalSimWarning, obj1.makePhot, n_photons=1)
    # And drawPhot
    with assert_warns(galsim.GalSimWarning):
        obj1.drawPhot(im1)
    with assert_raises(AssertionError):
        assert_warns(galsim.GalSimWarning, obj1.drawPhot, im1, n_photons=1)


@timer
def test_fft():
    """Test the routines for calculating the fft of an image.
    """

    # Start with a really simple test of the round trip fft and then inverse_fft.
    # And run it for all input types to make sure they all work.
    types = [np.int16, np.int32, np.float32, np.float64, np.complex128, int, float, complex]
    for dt in types:
        xim = galsim.Image([ [0,2,4,2],
                             [2,4,6,4],
                             [4,6,8,4],
                             [2,4,6,6] ],
                           xmin=-2, ymin=-2, dtype=dt, scale=0.1)
        kim = xim.calculate_fft()
        xim2 = kim.calculate_inverse_fft()
        np.testing.assert_almost_equal(xim.array, xim2.array)

        # Now the other way, starting with a (real) k-space image.
        kim = galsim.Image([ [4,2,0],
                             [6,4,2],
                             [8,6,4],
                             [6,4,2] ],
                           xmin=0, ymin=-2, dtype=dt, scale=0.1)
        xim = kim.calculate_inverse_fft()
        kim2 = xim.calculate_fft()
        np.testing.assert_almost_equal(kim.array, kim2.array)

        # Test starting with a larger image that gets wrapped.
        kim3 = galsim.Image([ [0,1,2,1,0],
                              [1,4,6,4,1],
                              [2,6,8,6,2],
                              [1,4,6,4,1],
                              [0,1,2,1,0] ],
                            xmin=-2, ymin=-2, dtype=dt, scale=0.1)
        xim = kim3.calculate_inverse_fft()
        kim2 = xim.calculate_fft()
        np.testing.assert_almost_equal(kim.array, kim2.array)

        # Test padding X Image with zeros
        xim = galsim.Image([ [0,0,0,0],
                             [2,4,6,0],
                             [4,6,8,0],
                             [0,0,0,0] ],
                           xmin=-2, ymin=-2, dtype=dt, scale=0.1)
        xim2 = galsim.Image([ [2,4,6],
                              [4,6,8] ],
                            xmin=-2, ymin=-1, dtype=dt, scale=0.1)
        kim = xim.calculate_fft()
        kim2 = xim2.calculate_fft()
        np.testing.assert_almost_equal(kim.array, kim2.array)

        # Test padding K Image with zeros
        kim = galsim.Image([ [4,2,0],
                             [6,4,0],
                             [8,6,0],
                             [6,4,0] ],
                           xmin=0, ymin=-2, dtype=dt, scale=0.1)
        kim2 = galsim.Image([ [6,4],
                              [8,6],
                              [6,4],
                              [4,2] ],
                           xmin=0, ymin=-1, dtype=dt, scale=0.1)
        xim = kim.calculate_inverse_fft()
        xim2 = kim2.calculate_inverse_fft()
        np.testing.assert_almost_equal(xim.array, xim2.array)

    # Now use drawKImage (as above in test_drawKImage) to get a more realistic k-space image
    obj = galsim.Moffat(flux=test_flux, beta=1.5, scale_radius=0.5)
    obj = obj.withGSParams(maxk_threshold=1.e-4)
    im1 = obj.drawKImage()
    N = 1174  # NB. It is useful to have this come out not a multiple of 4, since some of the
              #     calculation needs to be different when N/2 is odd.
    np.testing.assert_equal(im1.bounds, galsim.BoundsI(-N/2,N/2,-N/2,N/2),
                            "obj.drawKImage() produced image with wrong bounds")
    nyq_scale = obj.nyquist_scale

    # If we inverse_fft the above automatic image, it should match the automatic real image
    # for method = 'sb' and use_true_center=False.
    im1_real = im1.calculate_inverse_fft()
    # Convolve by a delta function to force FFT drawing.
    obj2 = galsim.Convolve(obj, galsim.Gaussian(sigma=1.e-10))
    im1_alt_real = obj2.drawImage(method='sb', use_true_center=False)
    im1_alt_real.setCenter(0,0)  # This isn't done automatically.
    np.testing.assert_equal(
            im1_real.bounds, im1_alt_real.bounds,
            "inverse_fft did not produce the same bounds as obj2.drawImage(method='sb')")
    # The scale and array are only approximately equal, because drawImage rounds the size up to
    # an even number and uses Nyquist scale for dx.
    np.testing.assert_almost_equal(
            im1_real.scale, im1_alt_real.scale, 3,
            "inverse_fft produce a different scale than obj2.drawImage(method='sb')")
    np.testing.assert_almost_equal(
            im1_real.array, im1_alt_real.array, 3,
            "inverse_fft produce a different array than obj2.drawImage(method='sb')")

    # If we give both a good size to use and match up the scales, then they should produce the
    # same thing.
    N = galsim.Image.good_fft_size(N)
    assert N == 1536 == 3 * 2**9
    kscale = 2.*np.pi / (N * nyq_scale)
    im2 = obj.drawKImage(nx=N+1, ny=N+1, scale=kscale)
    im2_real = im2.calculate_inverse_fft()
    im2_alt_real = obj2.drawImage(nx=N, ny=N, method='sb', use_true_center=False, dtype=float)
    im2_alt_real.setCenter(0,0)
    np.testing.assert_equal(
            im2_real.bounds, im2_alt_real.bounds,
            "inverse_fft did not produce the same bounds as obj2.drawImage(nx,ny,method='sb')")
    np.testing.assert_almost_equal(
            im2_real.scale, im2_alt_real.scale, 9,
            "inverse_fft produce a different scale than obj2.drawImage(nx,ny,method='sb')")
    np.testing.assert_almost_equal(
            im2_real.array, im2_alt_real.array, 9,
            "inverse_fft produce a different array than obj2.drawImage(nx,ny,method='sb')")

    # wcs must be a PixelScale
    xim.wcs = galsim.JacobianWCS(1.1,0.1,0.1,1)
    with assert_raises(galsim.GalSimError):
        xim.calculate_fft()
    with assert_raises(galsim.GalSimError):
        xim.calculate_inverse_fft()
    xim.wcs = None
    with assert_raises(galsim.GalSimError):
        xim.calculate_fft()
    with assert_raises(galsim.GalSimError):
        xim.calculate_inverse_fft()

    # inverse needs image with 0,0
    xim.scale=1
    xim.setOrigin(1,1)
    with assert_raises(galsim.GalSimBoundsError):
        xim.calculate_inverse_fft()


@timer
def test_np_fft():
    """Test the equivalence between np.fft functions and the galsim versions
    """
    input_list = []
    input_list.append( np.array([ [0,1,2,1],
                                  [1,2,3,2],
                                  [2,3,4,3],
                                  [1,2,3,2] ], dtype=int ))
    input_list.append( np.array([ [0,1],
                                  [1,2],
                                  [2,3],
                                  [1,2] ], dtype=int ))
    noise = galsim.GaussianNoise(sigma=5, rng=galsim.BaseDeviate(1234))
    for N in [2,4,8,10]:
        xim = galsim.ImageD(N,N)
        xim.addNoise(noise)
        input_list.append(xim.array)

    for Nx,Ny in [ (2,4), (4,2), (10,6), (6,10) ]:
        xim = galsim.ImageD(Nx,Ny)
        xim.addNoise(noise)
        input_list.append(xim.array)

    for N in [2,4,8,10]:
        xim = galsim.ImageCD(N,N)
        xim.real.addNoise(noise)
        xim.imag.addNoise(noise)
        input_list.append(xim.array)

    for Nx,Ny in [ (2,4), (4,2), (10,6), (6,10) ]:
        xim = galsim.ImageCD(Nx,Ny)
        xim.real.addNoise(noise)
        xim.imag.addNoise(noise)
        input_list.append(xim.array)

    for xar in input_list:
        Ny,Nx = xar.shape
        print('Nx,Ny = ',Nx,Ny)
        if Nx + Ny < 10:
            print('xar = ',xar)
        kar1 = np.fft.fft2(xar)
        #print('numpy kar = ',kar1)
        kar2 = galsim.fft.fft2(xar)
        if Nx + Ny < 10:
            print('kar = ',kar2)
        np.testing.assert_almost_equal(kar1, kar2, 9, "fft2 not equivalent to np.fft.fft2")

        # Check that kar is Hermitian in the way that we describe in the doc for ifft2
        if not np.iscomplexobj(xar):
            for kx in range(Nx//2,Nx):
                np.testing.assert_almost_equal(kar2[0,kx], kar2[0,Nx-kx].conjugate())
                for ky in range(1,Ny):
                    np.testing.assert_almost_equal(kar2[ky,kx], kar2[Ny-ky,Nx-kx].conjugate())

        # Check shift_in
        kar3 = np.fft.fft2(np.fft.fftshift(xar))
        kar4 = galsim.fft.fft2(xar, shift_in=True)
        np.testing.assert_almost_equal(kar3, kar4, 9, "fft2(shift_in) failed")

        # Check shift_out
        kar5 = np.fft.fftshift(np.fft.fft2(xar))
        kar6 = galsim.fft.fft2(xar, shift_out=True)
        np.testing.assert_almost_equal(kar5, kar6, 9, "fft2(shift_out) failed")

        # Check both
        kar7 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(xar)))
        kar8 = galsim.fft.fft2(xar, shift_in=True, shift_out=True)
        np.testing.assert_almost_equal(kar7, kar8, 9, "fft2(shift_in,shift_out) failed")

        # ifft2
        #print('ifft2')
        xar1 = np.fft.ifft2(kar2)
        xar2 = galsim.fft.ifft2(kar2)
        if Nx + Ny < 10:
            print('xar2 = ',xar2)
        np.testing.assert_almost_equal(xar1, xar2, 9, "ifft2 not equivalent to np.fft.ifft2")
        np.testing.assert_almost_equal(xar2, xar, 9, "ifft2(fft2(a)) != a")

        xar3 = np.fft.ifft2(np.fft.fftshift(kar6))
        xar4 = galsim.fft.ifft2(kar6, shift_in=True)
        np.testing.assert_almost_equal(xar3, xar4, 9, "ifft2(shift_in) failed")
        np.testing.assert_almost_equal(xar4, xar, 9, "ifft2(fft2(a)) != a with shift_in/out")

        xar5 = np.fft.fftshift(np.fft.ifft2(kar4))
        xar6 = galsim.fft.ifft2(kar4, shift_out=True)
        np.testing.assert_almost_equal(xar5, xar6, 9, "ifft2(shift_out) failed")
        np.testing.assert_almost_equal(xar6, xar, 9, "ifft2(fft2(a)) != a with shift_out/in")

        xar7 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kar8)))
        xar8 = galsim.fft.ifft2(kar8, shift_in=True, shift_out=True)
        np.testing.assert_almost_equal(xar7, xar8, 9, "ifft2(shift_in,shift_out) failed")
        np.testing.assert_almost_equal(xar8, xar, 9, "ifft2(fft2(a)) != a with all shifts")

        if np.iscomplexobj(xar): continue

        # rfft2
        #print('rfft2')
        rkar1 = np.fft.rfft2(xar)
        rkar2 = galsim.fft.rfft2(xar)
        np.testing.assert_almost_equal(rkar1, rkar2, 9, "rfft2 not equivalent to np.fft.rfft2")

        rkar3 = np.fft.rfft2(np.fft.fftshift(xar))
        rkar4 = galsim.fft.rfft2(xar, shift_in=True)
        np.testing.assert_almost_equal(rkar3, rkar4, 9, "rfft2(shift_in) failed")

        rkar5 = np.fft.fftshift(np.fft.rfft2(xar),axes=(0,))
        rkar6 = galsim.fft.rfft2(xar, shift_out=True)
        np.testing.assert_almost_equal(rkar5, rkar6, 9, "rfft2(shift_out) failed")

        rkar7 = np.fft.fftshift(np.fft.rfft2(np.fft.fftshift(xar)),axes=(0,))
        rkar8 = galsim.fft.rfft2(xar, shift_in=True, shift_out=True)
        np.testing.assert_almost_equal(rkar7, rkar8, 9, "rfft2(shift_in,shift_out) failed")

        # irfft2
        #print('irfft2')
        xar1 = np.fft.irfft2(rkar1)
        xar2 = galsim.fft.irfft2(rkar1)
        np.testing.assert_almost_equal(xar1, xar2, 9, "irfft2 not equivalent to np.fft.irfft2")
        np.testing.assert_almost_equal(xar2, xar, 9, "irfft2(rfft2(a)) != a")

        xar3 = np.fft.irfft2(np.fft.fftshift(rkar6,axes=(0,)))
        xar4 = galsim.fft.irfft2(rkar6, shift_in=True)
        np.testing.assert_almost_equal(xar3, xar4, 9, "irfft2(shift_in) failed")
        np.testing.assert_almost_equal(xar4, xar, 9, "irfft2(rfft2(a)) != a with shift_in/out")

        xar5 = np.fft.fftshift(np.fft.irfft2(rkar4))
        xar6 = galsim.fft.irfft2(rkar4, shift_out=True)
        np.testing.assert_almost_equal(xar5, xar6, 9, "irfft2(shift_out) failed")
        np.testing.assert_almost_equal(xar6, xar, 9, "irfft2(rfft2(a)) != a with shift_out/in")

        xar7 = np.fft.fftshift(np.fft.irfft2(np.fft.fftshift(rkar8,axes=(0,))))
        xar8 = galsim.fft.irfft2(rkar8, shift_in=True, shift_out=True)
        np.testing.assert_almost_equal(xar7, xar8, 9, "irfft2(shift_in,shift_out) failed")
        np.testing.assert_almost_equal(xar8, xar, 9, "irfft2(rfft2(a)) != a with all shifts")

        # ifft can also accept real arrays
        xar9 = galsim.fft.fft2(galsim.fft.ifft2(xar))
        np.testing.assert_almost_equal(xar9, xar, 9, "fft2(ifft2(a)) != a")
        xar10 = galsim.fft.fft2(galsim.fft.ifft2(xar,shift_in=True),shift_out=True)
        np.testing.assert_almost_equal(xar10, xar, 9, "fft2(ifft2(a)) != a with shift_in/out")
        xar11 = galsim.fft.fft2(galsim.fft.ifft2(xar,shift_out=True),shift_in=True)
        np.testing.assert_almost_equal(xar11, xar, 9, "fft2(ifft2(a)) != a with shift_out/in")
        xar12 = galsim.fft.fft2(galsim.fft.ifft2(xar,shift_in=True,shift_out=True),
                                shift_in=True,shift_out=True)
        np.testing.assert_almost_equal(xar12, xar, 9, "fft2(ifft2(a)) != a with all shifts")

    # Check for invalid inputs
    # Must be 2-d arrays
    xar_1d = input_list[0].ravel()
    xar_3d = input_list[0].reshape(2,2,4)
    xar_4d = input_list[0].reshape(2,2,2,2)
    assert_raises(ValueError, galsim.fft.fft2, xar_1d)
    assert_raises(ValueError, galsim.fft.fft2, xar_3d)
    assert_raises(ValueError, galsim.fft.fft2, xar_4d)
    assert_raises(ValueError, galsim.fft.ifft2, xar_1d)
    assert_raises(ValueError, galsim.fft.ifft2, xar_3d)
    assert_raises(ValueError, galsim.fft.ifft2, xar_4d)
    assert_raises(ValueError, galsim.fft.rfft2, xar_1d)
    assert_raises(ValueError, galsim.fft.rfft2, xar_3d)
    assert_raises(ValueError, galsim.fft.rfft2, xar_4d)
    assert_raises(ValueError, galsim.fft.irfft2, xar_1d)
    assert_raises(ValueError, galsim.fft.irfft2, xar_3d)
    assert_raises(ValueError, galsim.fft.irfft2, xar_4d)

    # Must have even sizes
    xar_oo = input_list[0][:3,:3]
    xar_oe = input_list[0][:3,:]
    xar_eo = input_list[0][:,:3]
    assert_raises(ValueError, galsim.fft.fft2, xar_oo)
    assert_raises(ValueError, galsim.fft.fft2, xar_oe)
    assert_raises(ValueError, galsim.fft.fft2, xar_eo)
    assert_raises(ValueError, galsim.fft.ifft2, xar_oo)
    assert_raises(ValueError, galsim.fft.ifft2, xar_oe)
    assert_raises(ValueError, galsim.fft.ifft2, xar_eo)
    assert_raises(ValueError, galsim.fft.rfft2, xar_oo)
    assert_raises(ValueError, galsim.fft.rfft2, xar_oe)
    assert_raises(ValueError, galsim.fft.rfft2, xar_eo)
    assert_raises(ValueError, galsim.fft.irfft2, xar_oo)
    assert_raises(ValueError, galsim.fft.irfft2, xar_oe)
    # eo is ok, since the second dimension is actually N/2+1

def round_cast(array, dt):
    # array.astype(dt) doesn't round to the nearest for integer types.
    # This rounds first if dt is integer and then casts.
    if dt(0.5) != 0.5:
        array = np.around(array)
    return array.astype(dt)

@timer
def test_types():
    """Test drawing onto image types other than float32, float64.
    """

    # Methods test drawReal, drawFFT, drawPhot respectively
    for method in ['no_pixel', 'fft', 'phot']:
        if method == 'phot':
            rng = galsim.BaseDeviate(1234)
        else:
            rng = None
        obj = galsim.Exponential(flux=177, scale_radius=2)
        ref_im = obj.drawImage(method=method, dtype=float, rng=rng)

        for dt in [ np.float32, np.float64, np.int16, np.int32, np.uint16, np.uint32,
                    np.complex128, np.complex64 ]:
            if method == 'phot': rng.reset(1234)
            print('Checking',method,'with dt =', dt)
            im = obj.drawImage(method=method, dtype=dt, rng=rng)
            np.testing.assert_equal(im.scale, ref_im.scale,
                                    "wrong scale when drawing onto dt=%s"%dt)
            np.testing.assert_equal(im.bounds, ref_im.bounds,
                                    "wrong bounds when drawing onto dt=%s"%dt)
            np.testing.assert_almost_equal(im.array, round_cast(ref_im.array, dt), 6,
                                           "wrong array when drawing onto dt=%s"%dt)

            if method == 'phot':
                rng.reset(1234)
            obj.drawImage(im, method=method, add_to_image=True, rng=rng)
            np.testing.assert_almost_equal(im.array, round_cast(ref_im.array, dt) * 2, 6,
                                           "wrong array when adding to image with dt=%s"%dt)

@timer
def test_direct_scale():
    """Test the explicit functions with scale != 1

    The default behavior is to change the profile to image coordinates, and draw that onto an
    image with scale=1.  But the three direct functions allow the image to have a non-unit
    pixel scale.  (Not more complicated wcs though.)

    This test checks that the results are equivalent between the two calls.
    """

    scale = 0.35
    rng = galsim.BaseDeviate(1234)
    obj = galsim.Exponential(flux=177, scale_radius=2)
    obj_with_pixel = galsim.Convolve(obj, galsim.Pixel(scale))
    obj_sb = obj / scale**2

    # Make these odd, so we don't have to deal with the centering offset stuff.
    im1 = galsim.ImageD(65, 65, scale=scale)
    im2 = galsim.ImageD(65, 65, scale=scale)
    im2.setCenter(0,0)

    # One possibe use of the specific functions is to not automatically recenter on 0,0.
    # So make sure they work properly if 0,0 is not the center
    im3 = galsim.ImageD(32, 32, scale=scale)  # origin is (1,1)
    im4 = galsim.ImageD(32, 32, scale=scale)
    im5 = galsim.ImageD(32, 32, scale=scale)

    obj.drawImage(im1, method='no_pixel')
    obj.drawReal(im2)
    obj.drawReal(im3)
    # Note that cases 4 and 5 have objects that are logically identical (because obj is circularly
    # symmetric), but the code follows different paths in the SBProfile.draw function due to the
    # different jacobians in each case.
    obj.dilate(1.0).drawReal(im4)
    obj.rotate(0.3*galsim.radians).drawReal(im5)
    print('no_pixel: max diff = ',np.max(np.abs(im1.array - im2.array)))
    np.testing.assert_almost_equal(im1.array, im2.array, 15,
                                   "drawReal made different image than method='no_pixel'")
    np.testing.assert_almost_equal(im3.array, im2[im3.bounds].array, 15,
                                   "drawReal made different image when off-center")
    np.testing.assert_almost_equal(im4.array, im2[im3.bounds].array, 15,
                                   "drawReal made different image when jac is not None")
    np.testing.assert_almost_equal(im5.array, im2[im3.bounds].array, 15,
                                   "drawReal made different image when jac is not diagonal")

    obj.drawImage(im1, method='sb')
    obj_sb.drawReal(im2)
    obj_sb.drawReal(im3)
    obj_sb.dilate(1.0).drawReal(im4)
    obj_sb.rotate(0.3*galsim.radians).drawReal(im5)
    print('sb: max diff = ',np.max(np.abs(im1.array - im2.array)))
    np.testing.assert_almost_equal(im1.array, im2.array, 15,
                                   "drawReal made different image than method='sb'")
    np.testing.assert_almost_equal(im3.array, im2[im3.bounds].array, 15,
                                   "drawReal made different image when off-center")
    np.testing.assert_almost_equal(im4.array, im2[im3.bounds].array, 15,
                                   "drawReal made different image when jac is not None")
    np.testing.assert_almost_equal(im5.array, im2[im3.bounds].array, 14,
                                   "drawReal made different image when jac is not diagonal")

    obj.drawImage(im1, method='fft')
    obj_with_pixel.drawFFT(im2)
    obj_with_pixel.drawFFT(im3)
    obj_with_pixel.dilate(1.0).drawFFT(im4)
    obj_with_pixel.rotate(90 * galsim.degrees).drawFFT(im5)
    print('fft: max diff = ',np.max(np.abs(im1.array - im2.array)))
    np.testing.assert_almost_equal(im1.array, im2.array, 15,
                                   "drawFFT made different image than method='fft'")
    np.testing.assert_almost_equal(im3.array, im2[im3.bounds].array, 15,
                                   "drawFFT made different image when off-center")
    np.testing.assert_almost_equal(im4.array, im2[im3.bounds].array, 15,
                                   "drawFFT made different image when jac is not None")
    np.testing.assert_almost_equal(im5.array, im2[im3.bounds].array, 14,
                                   "drawFFT made different image when jac is not diagonal")

    obj.drawImage(im1, method='real_space')
    obj_with_pixel.drawReal(im2)
    obj_with_pixel.drawReal(im3)
    obj_with_pixel.dilate(1.0).drawReal(im4)
    obj_with_pixel.rotate(90 * galsim.degrees).drawReal(im5)
    print('real_space: max diff = ',np.max(np.abs(im1.array - im2.array)))
    # I'm not sure why this one comes out a bit less precisely equal.  But 12 digits is still
    # plenty accurate enough.
    np.testing.assert_almost_equal(im1.array, im2.array, 12,
                                   "drawReal made different image than method='real_space'")
    np.testing.assert_almost_equal(im3.array, im2[im3.bounds].array, 14,
                                   "drawReal made different image when off-center")
    np.testing.assert_almost_equal(im4.array, im2[im3.bounds].array, 14,
                                   "drawReal made different image when jac is not None")
    np.testing.assert_almost_equal(im5.array, im2[im3.bounds].array, 14,
                                   "drawReal made different image when jac is not diagonal")

    obj.drawImage(im1, method='phot', rng=rng.duplicate())
    _, phot1 = obj.drawPhot(im2, rng=rng.duplicate())
    _, phot2 = obj.drawPhot(im3, rng=rng.duplicate())
    phot3 = obj.makePhot(rng=rng.duplicate())
    phot3.scaleXY(1./scale)
    phot4 = im3.wcs.toImage(obj).makePhot(rng=rng.duplicate())
    print('phot: max diff = ',np.max(np.abs(im1.array - im2.array)))
    np.testing.assert_almost_equal(im1.array, im2.array, 15,
                                   "drawPhot made different image than method='phot'")
    np.testing.assert_almost_equal(im3.array, im2[im3.bounds].array, 15,
                                   "drawPhot made different image when off-center")
    assert phot2 == phot1, "drawPhot made different photons than method='phot'"
    assert phot3 == phot1, "makePhot made different photons than method='phot'"
    # phot4 has a different order of operations for the math, so it doesn't come out exact.
    np.testing.assert_almost_equal(phot4.x, phot3.x, 15,
                                   "two ways to have makePhot apply scale have different x")
    np.testing.assert_almost_equal(phot4.y, phot3.y, 15,
                                   "two ways to have makePhot apply scale have different y")
    np.testing.assert_almost_equal(phot4.flux, phot3.flux, 15,
                                   "two ways to have makePhot apply scale have different flux")

    # Check images with invalid wcs raise ValueError
    im4 = galsim.ImageD(65, 65)
    im5 = galsim.ImageD(65, 65, wcs=galsim.JacobianWCS(0.4,0.1,-0.1,0.5))
    assert_raises(ValueError, obj.drawReal, im4)
    assert_raises(ValueError, obj.drawReal, im5)
    assert_raises(ValueError, obj.drawFFT, im4)
    assert_raises(ValueError, obj.drawFFT, im5)
    assert_raises(ValueError, obj.drawPhot, im4)
    assert_raises(ValueError, obj.drawPhot, im5)
    # Also some other errors from drawPhot
    assert_raises(ValueError, obj.drawPhot, im2, n_photons=-20)
    assert_raises(TypeError, obj.drawPhot, im2, sensor=5)
    assert_raises(ValueError, obj.makePhot, n_photons=-20)

def test_center():
    # This test is in response to issue #1322, where it was found that giving nx,ny with center
    # would not always center the object at the right location.
    # This test is essentially the test proposed in that issue, which showed the bug.

    def compute_centroid(img):
        x, y = img.get_pixel_centers()
        flux = img.array
        xcen = np.sum(x * flux) / np.sum(flux)
        ycen = np.sum(y * flux) / np.sum(flux)
        return galsim.PositionD(xcen, ycen)

    wcs = galsim.PixelScale(0.2)
    gal = galsim.Gaussian(fwhm=0.9)
    sizes = [50,51]
    offsets = [-1.0, -0.63, -0.5, -0.12, 0., 0.33, 0.5, 0.78, 1.0]
    for xsize in sizes:
        for ysize in sizes:
            for x_center_offset in offsets:
                for y_center_offset in offsets:
                    print(xsize,ysize,x_center_offset,y_center_offset)
                    center = galsim.PositionD(
                        xsize//2 + x_center_offset,
                        ysize//2 + y_center_offset,
                    )
                    im = gal.drawImage(nx=xsize, ny=ysize, wcs=wcs, center=center)
                    centroid = compute_centroid(im)
                    print('  center = ', center)
                    print('  centroid = ', centroid)
                    np.testing.assert_allclose(centroid.x, center.x, atol=1e-4, rtol=0)
                    np.testing.assert_allclose(centroid.y, center.y, atol=1e-4, rtol=0)


if __name__ == "__main__":
    runtests(__file__)
