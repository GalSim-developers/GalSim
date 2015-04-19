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
import os
import sys

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

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
    flux = im.array.astype(float).sum()
    mx = (x * im.array.astype(float)).sum() / flux
    my = (y * im.array.astype(float)).sum() / flux
    mxx = (((x-mx)**2) * im.array.astype(float)).sum() / flux
    myy = (((y-my)**2) * im.array.astype(float)).sum() / flux
    mxy = ((x-mx) * (y-my) * im.array.astype(float)).sum() / flux
    s2 = mxx+myy
    print flux,mx,my,mxx,myy,mxy
    np.testing.assert_almost_equal((mxx-myy)/s2, 0, 5, "Found e1 != 0 for Exponential draw")
    np.testing.assert_almost_equal(2*mxy/s2, 0, 5, "Found e2 != 0 for Exponential draw")
    return np.sqrt(s2/6) * im.scale

def test_drawImage():
    """Test the various optional parameters to the draw function.
       In particular test the parameters image, dx, and wmult in various combinations.
    """
    import time
    t1 = time.time()

    # We use a simple Exponential for our object:
    obj = galsim.Exponential(flux=test_flux, scale_radius=2)

    # First test drawImage() with method='no_pixel'.  It should:
    #   - create a new image
    #   - return the new image
    #   - set the scale to obj.nyquistScale()
    #   - set the size large enough to contain 99.5% of the flux
    im1 = obj.drawImage(method='no_pixel')
    nyq_scale = obj.nyquistScale()
    np.testing.assert_almost_equal(im1.scale, nyq_scale, 9,
                                   "obj.drawImage() produced image with wrong scale")
    #print 'im1.bounds = ',im1.bounds
    assert im1.bounds == galsim.BoundsI(1,56,1,56),(
            "obj.drawImage() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im1), 2, 1,
                                   "Measured wrong scale after obj.drawImage()")

    # The flux is only really expected to come out right if the object has been
    # convoled with a pixel:
    obj2 = galsim.Convolve([ obj, galsim.Pixel(im1.scale) ])
    im2 = obj2.drawImage(method='no_pixel')
    nyq_scale = obj2.nyquistScale()
    np.testing.assert_almost_equal(im2.scale, nyq_scale, 9,
                                   "obj2.drawImage() produced image with wrong scale")
    np.testing.assert_almost_equal(im2.array.astype(float).sum(), test_flux, 2,
                                   "obj2.drawImage() produced image with wrong flux")
    assert im2.bounds == galsim.BoundsI(1,56,1,56),(
            "obj2.drawImage() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im2), 2, 1,
                                   "Measured wrong scale after obj2.drawImage()")
    # This should be the same as obj with method='auto'
    im2 = obj.drawImage()
    np.testing.assert_almost_equal(im2.scale, nyq_scale, 9,
                                   "obj2.drawImage() produced image with wrong scale")
    np.testing.assert_almost_equal(im2.array.astype(float).sum(), test_flux, 2,
                                   "obj2.drawImage() produced image with wrong flux")
    assert im2.bounds == galsim.BoundsI(1,56,1,56),(
            "obj2.drawImage() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im2), 2, 1,
                                   "Measured wrong scale after obj2.drawImage()")

    # Test if we provide an image argument.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj2.nyquistScale()
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
    #   - set the scale to obj2.nyquistScale()
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
    assert im5.bounds == galsim.BoundsI(1,56,1,56),(
            "obj.drawImage(im5) produced image with wrong bounds")

    # Test if we provide wmult.  It should:
    #   - create a new image that is wmult times larger in each direction.
    #   - return the new image
    #   - set the scale to obj2.nyquistScale()
    im6 = obj.drawImage(wmult=4.)
    np.testing.assert_almost_equal(im6.scale, nyq_scale, 9,
                                   "obj.drawImage(wmult) produced image with wrong scale")
    # Can assert accuracy to 4 decimal places now, since we're capturing much more
    # of the flux on the image.
    np.testing.assert_almost_equal(im6.array.astype(float).sum(), test_flux, 4,
                                   "obj.drawImage(wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im6), 2, 2,
                                   "Measured wrong scale after obj.drawImage(wmult)")
    #print 'im6.bounds = ',im6.bounds
    assert im6.bounds == galsim.BoundsI(1,220,1,220),(
            "obj.drawImage(wmult) produced image with wrong bounds")

    # Test if we provide an image argument and wmult.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj2.nyquistScale()
    #   - zero out any existing data
    #   - the calculation of the convolution should be slightly more accurate than for im3
    im3.setZero()
    im5.setZero()
    obj.drawImage(im3, wmult=4.)
    obj.drawImage(im5)
    np.testing.assert_almost_equal(im3.scale, nyq_scale, 9,
                                   "obj.drawImage(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3.array.sum(), test_flux, 2,
                                   "obj.drawImage(im3,wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj.drawImage(im3,wmult)")
    assert ((im3.array-im5.array)**2).sum() > 0, (
            "obj.drawImage(im3,wmult) produced the same image as without wmult")

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
    #print 'im7.bounds = ',im7.bounds
    assert im7.bounds == galsim.BoundsI(1,68,1,68),(
            "obj.drawImage(dx) produced image with wrong bounds")

    # Test with dx and wmult.  It should:
    #   - create a new image using that dx for the scale
    #   - set the size a factor of wmult times larger in each direction.
    #   - return the new image
    im8 = obj.drawImage(scale=scale, wmult=4.)
    np.testing.assert_almost_equal(im8.scale, scale, 9,
                                   "obj.drawImage(dx,wmult) produced image with wrong scale")
    np.testing.assert_almost_equal(im8.array.astype(float).sum(), test_flux, 4,
                                   "obj.drawImage(dx,wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im8), 2, 2,
                                   "Measured wrong scale after obj.drawImage(dx,wmult)")
    #print 'im8.bounds = ',im8.bounds
    assert im8.bounds == galsim.BoundsI(1,270,1,270),(
            "obj.drawImage(dx,wmult) produced image with wrong bounds")

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
    #   - set the scale to obj2.nyquistScale()
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
    #   - set the scale to obj2.nyquistScale()
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
    np.testing.assert_almost_equal(im10.array.shape, (ny, nx), 9,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, scale, 9,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong flux")
    mx, my, mxx, myy, mxy = getmoments(im10)
    np.testing.assert_almost_equal(
        mx, (nx+1.)/2., 4, "obj.drawImage(nx,ny,scale) (even) did not center in x correctly")
    np.testing.assert_almost_equal(
        my, (ny+1.)/2., 4, "obj.drawImage(nx,ny,scale) (even) did not center in y correctly")

    # Repeat with odd nx,ny
    im10 = obj.drawImage(nx=nx+1, ny=ny+1, scale=scale)
    np.testing.assert_almost_equal(im10.array.shape, (ny+1, nx+1), 9,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, scale, 9,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(nx,ny,scale) produced image with wrong flux")
    mx, my, mxx, myy, mxy = getmoments(im10)
    np.testing.assert_almost_equal(
        mx, (nx+1.+1.)/2., 4,
        "obj.drawImage(nx,ny,scale) (odd) did not center in x correctly")
    np.testing.assert_almost_equal(
        my, (ny+1.+1.)/2., 4,
        "obj.drawImage(nx,ny,scale) (odd) did not center in y correctly")

    try:
        # Test if we provide nx, ny, and no scale.  It should:
        #   - raise a ValueError
        im10 = galsim.ImageF()
        kwargs = {'nx':nx, 'ny':ny}
        np.testing.assert_raises(ValueError, obj.drawImage, kwargs)

        # Test if we provide nx, ny, scale, and an existing image.  It should:
        #   - raise a ValueError
        im10 = galsim.ImageF()
        kwargs = {'nx':nx, 'ny':ny, 'scale':scale, 'image':im10}
        np.testing.assert_raises(ValueError, obj2.drawImage, kwargs)
    except ImportError:
        print 'The assert_raises tests require nose'

    # Test if we provide bounds and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    bounds = galsim.BoundsI(1,nx,1,ny+1)
    im10 = obj.drawImage(bounds=bounds, scale=scale)
    np.testing.assert_almost_equal(im10.array.shape, (ny+1, nx), 9,
                                   "obj.drawImage(bounds,scale) produced image with wrong size")
    np.testing.assert_almost_equal(im10.scale, scale, 9,
                                   "obj.drawImage(bounds,scale) produced image with wrong scale")
    np.testing.assert_almost_equal(im10.array.sum(), test_flux, 4,
                                   "obj.drawImage(bounds,scale) produced image with wrong flux")
    mx, my, mxx, myy, mxy = getmoments(im10)
    np.testing.assert_almost_equal(mx, (nx+1.)/2., 4,
                                   "obj.drawImage(bounds,scale) did not center in x correctly")
    np.testing.assert_almost_equal(my, (ny+1.+1.)/2., 4,
                                   "obj.drawImage(bounds,scale) did not center in y correctly")

    try:
        # Test if we provide bounds and no scale.  It should:
        #   - raise a ValueError
        bounds = galsim.BoundsI(1,nx,1,ny)
        kwargs = {'bounds':bounds}
        np.testing.assert_raises(ValueError, obj.drawImage, kwargs)

        # Test if we provide bounds, scale, and an existing image.  It should:
        #   - raise a ValueError
        bounds = galsim.BoundsI(1,nx,1,ny)
        kwargs = {'bounds':bounds, 'scale':scale, 'image':im10}
        np.testing.assert_raises(ValueError, obj.drawImage, kwargs)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_draw_methods():
    """Test the the different method options do the right thing.
    """
    import time
    t1 = time.time()

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
    print 'im1 flux diff = ',abs(im1.array.sum() - test_flux)
    np.testing.assert_almost_equal(
            im1.array.sum(), test_flux, 2,
            "obj.drawImage() produced image with wrong flux")
    print 'im2 flux diff = ',abs(im2.array.sum() - test_flux)
    np.testing.assert_almost_equal(
            im2.array.sum(), test_flux, 2,
            "obj_pix.drawImage(no_pixel) produced image with wrong flux")
    print 'im1, im2 max diff = ',abs(im1.array - im2.array).max()
    np.testing.assert_array_almost_equal(
            im1.array, im2.array, 6,
            "obj.drawImage() differs from obj_pix.drawImage(no_pixel)")
    im3 = obj.drawImage(image=im1.copy(), method='fft')
    print 'im1, im3 max diff = ',abs(im1.array - im3.array).max()
    np.testing.assert_array_almost_equal(
            im1.array, im3.array, 6,
            "obj.drawImage(fft) differs from obj.drawImage")

    # real_space should be similar, but not precisely equal.
    im4 = obj.drawImage(image=im1.copy(), method='real_space')
    print 'im1, im4 max diff = ',abs(im1.array - im4.array).max()
    np.testing.assert_array_almost_equal(
            im1.array, im4.array, 4,
            "obj.drawImage(real_space) differs from obj.drawImage")

    # sb should match xValue for pixel centers.  And be scale**2 factor different from no_pixel.
    im5 = obj.drawImage(image=im1.copy(), method='sb', use_true_center=False)
    im5.setCenter(0,0)
    print 'im5(0,0) = ',im5(0,0)
    print 'obj.xValue(0,0) = ',obj.xValue(0.,0.)
    np.testing.assert_almost_equal(
            im5(0,0), obj.xValue(0.,0.), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    np.testing.assert_almost_equal(
            im5(3,2), obj.xValue(3*test_scale, 2*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    im5 = obj.drawImage(image=im5, method='sb')
    print 'im5(0,0) = ',im5(0,0)
    print 'obj.xValue(dx/2,dx/2) = ',obj.xValue(test_scale/2., test_scale/2.)
    np.testing.assert_almost_equal(
            im5(0,0), obj.xValue(0.5*test_scale, 0.5*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    np.testing.assert_almost_equal(
            im5(3,2), obj.xValue(3.5*test_scale, 2.5*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    im6 = obj.drawImage(image=im1.copy(), method='no_pixel')
    print 'im6, im5*scale**2 max diff = ',abs(im6.array - im5.array*test_scale**2).max()
    np.testing.assert_array_almost_equal(
            im5.array * test_scale**2, im6.array, 6,
            "obj.drawImage(sb) * scale**2 differs from obj.drawImage(no_pixel)")

    # Drawing a truncated object, auto should be identical to real_space
    obj = galsim.Sersic(flux=test_flux, n=3.7, half_light_radius=2, trunc=4)
    obj_pix = galsim.Convolve(obj, pix)

    # auto and real_space should be equivalent to drawing obj_pix with no_pixel
    im1 = obj.drawImage(image=im1)
    im2 = obj_pix.drawImage(image=im1.copy(), method='no_pixel')
    print 'im1 flux diff = ',abs(im1.array.sum() - test_flux)
    np.testing.assert_almost_equal(
            im1.array.sum(), test_flux, 2,
            "obj.drawImage() produced image with wrong flux")
    print 'im2 flux diff = ',abs(im2.array.sum() - test_flux)
    np.testing.assert_almost_equal(
            im2.array.sum(), test_flux, 2,
            "obj_pix.drawImage(no_pixel) produced image with wrong flux")
    print 'im1, im2 max diff = ',abs(im1.array - im2.array).max()
    np.testing.assert_array_almost_equal(
            im1.array, im2.array, 6,
            "obj.drawImage() differs from obj_pix.drawImage(no_pixel)")
    im4 = obj.drawImage(image=im1.copy(), method='real_space')
    print 'im1, im4 max diff = ',abs(im1.array - im4.array).max()
    np.testing.assert_array_almost_equal(
            im1.array, im4.array, 6,
            "obj.drawImage(real_space) differs from obj.drawImage")

    # fft should be similar, but not precisely equal.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # This emits a warning about convolving two things with hard edges.
        im3 = obj.drawImage(image=im1.copy(), method='fft')
    print 'im1, im3 max diff = ',abs(im1.array - im3.array).max()
    np.testing.assert_array_almost_equal(
            im1.array, im3.array, 3, # Should be close, but not exact.
            "obj.drawImage(fft) differs from obj.drawImage")

    # sb should match xValue for pixel centers.  And be scale**2 factor different from no_pixel.
    im5 = obj.drawImage(image=im1.copy(), method='sb')
    im5.setCenter(0,0)
    print 'im5(0,0) = ',im5(0,0)
    print 'obj.xValue(dx/2,dx/2) = ',obj.xValue(test_scale/2., test_scale/2.)
    np.testing.assert_almost_equal(
            im5(0,0), obj.xValue(0.5*test_scale, 0.5*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    np.testing.assert_almost_equal(
            im5(3,2), obj.xValue(3.5*test_scale, 2.5*test_scale), 6,
            "obj.drawImage(sb) values do not match surface brightness given by xValue")
    im6 = obj.drawImage(image=im1.copy(), method='no_pixel')
    print 'im6, im5*scale**2 max diff = ',abs(im6.array - im5.array*test_scale**2).max()
    np.testing.assert_array_almost_equal(
            im5.array * test_scale**2, im6.array, 6,
            "obj.drawImage(sb) * scale**2 differs from obj.drawImage(no_pixel)")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_drawKImage():
    """Test the various optional parameters to the drawKImage function.
       In particular test the parameters image, and scale in various combinations.
    """
    import time
    t1 = time.time()

    # We use a Moffat profile with beta = 1.5, since its real-space profile is
    #    flux / (2 pi rD^2) * (1 + (r/rD)^2)^3/2
    # and the 2-d Fourier transform of that is
    #    flux * exp(-rD k)
    # So this should draw in Fourier space the same image as the Exponential drawn in
    # test_drawImage().
    obj = galsim.Moffat(flux=test_flux, beta=1.5, scale_radius=0.5)

    # First test drawKImage() with no kwargs.  It should:
    #   - create new images
    #   - return the new images
    #   - set the scale to 2pi/(N*obj.nyquistScale())
    re1, im1 = obj.drawKImage()
    N = 1162
    assert re1.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawKImage() produced image with wrong bounds")
    assert im1.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawKImage() produced image with wrong bounds")
    nyq_scale = obj.nyquistScale()
    stepk = obj.stepK()
    #print 'nyq_scale = ',nyq_scale
    #print '2pi/(nyq_scale N) = ',2*np.pi/(nyq_scale*N)
    #print 'stepK = ',obj.stepK()
    #print 'maxK = ',obj.maxK()
    #print 'im1.scale = ',im1.scale
    #print 'im1.center = ',im1.bounds.center
    np.testing.assert_almost_equal(re1.scale, stepk, 9,
                                   "obj.drawKImage() produced real image with wrong scale")
    np.testing.assert_almost_equal(im1.scale, stepk, 9,
                                   "obj.drawKImage() produced imag image with wrong scale")
    np.testing.assert_almost_equal(CalculateScale(re1), 2, 1,
                                   "Measured wrong scale after obj.drawKImage()")

    # The flux in Fourier space is just the value at k=0
    np.testing.assert_almost_equal(re1(re1.bounds.center()), test_flux, 2,
                                   "obj.drawKImage() produced real image with wrong flux")
    # Imaginary component should all be 0.
    np.testing.assert_almost_equal(im1.array.sum(), 0., 3,
                                   "obj.drawKImage() produced non-zero imaginary image")

    # Test if we provide an image argument.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj.stepK()
    #   - zero out any existing data
    re3 = galsim.ImageD(1149,1149)
    im3 = galsim.ImageD(1149,1149)
    re4, im4 = obj.drawKImage(re3, im3)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 2,
                                   "obj.drawKImage(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 3,
                                   "obj.drawKImage(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3)")
    np.testing.assert_array_equal(re3.array, re4.array,
                                  "re4, im4 = obj.drawKImage(re3,im3) produced re4 != re3")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "re4, im4 = obj.drawKImage(re3,im3) produced im4 != im3")
    re3.fill(9.8)
    im3.fill(9.8)
    np.testing.assert_array_equal(re3.array, re4.array,
                                  "re4, im4 = obj.drawKImage(re3,im3) produced re4 is not re3")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "re4, im4 = obj.drawKImage(re3,im3) produced im4 is not im3")

    # Test if we provide an image with undefined bounds.  It should:
    #   - resize the provided image
    #   - also return that image
    #   - set the scale to obj.stepK()
    re5 = galsim.ImageD()
    im5 = galsim.ImageD()
    obj.drawKImage(re5, im5)
    np.testing.assert_almost_equal(re5.scale, stepk, 9,
                                   "obj.drawKImage(re5,im5) produced real image with wrong scale")
    np.testing.assert_almost_equal(im5.scale, stepk, 9,
                                   "obj.drawKImage(re5,im5) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re5(re5.bounds.center()), test_flux, 2,
                                   "obj.drawKImage(re5,im5) produced real image with wrong flux")
    np.testing.assert_almost_equal(im5.array.sum(), 0., 3,
                                   "obj.drawKImage(re5,im5) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re5), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re5,im5)")
    assert im5.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawKImage(re5,im5) produced image with wrong bounds")

    # Test if we provide a scale to use.  It should:
    #   - create a new image using that dx for the scale
    #   - return the new image
    #   - set the size large enough to contain 99.5% of the flux
    scale = 0.51   # Just something different from 1 or dx_nyq
    re7, im7 = obj.drawKImage(scale=scale)
    np.testing.assert_almost_equal(re7.scale, scale, 9,
                                   "obj.drawKImage(dx) produced real image with wrong scale")
    np.testing.assert_almost_equal(im7.scale, scale, 9,
                                   "obj.drawKImage(dx) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re7(re7.bounds.center()), test_flux, 2,
                                   "obj.drawKImage(dx) produced real image with wrong flux")
    np.testing.assert_almost_equal(im7.array.astype(float).sum(), 0., 2,
                                   "obj.drawKImage(dx) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re7), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(dx)")
    assert im7.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawKImage(dx) produced image with wrong bounds")

    # Test if we provide an image with a defined scale.  It should:
    #   - write to the existing image
    #   - use the image's scale
    nx = 401
    re9 = galsim.ImageD(nx,nx, scale=scale)
    im9 = galsim.ImageD(nx,nx, scale=scale)
    obj.drawKImage(re9, im9)
    np.testing.assert_almost_equal(re9.scale, scale, 9,
                                   "obj.drawKImage(re9,im9) produced real image with wrong scale")
    np.testing.assert_almost_equal(im9.scale, scale, 9,
                                   "obj.drawKImage(re9,im9) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re9(re9.bounds.center()), test_flux, 4,
                                   "obj.drawKImage(re9,im9) produced real image with wrong flux")
    np.testing.assert_almost_equal(im9.array.sum(), 0., 5,
                                   "obj.drawKImage(re9,im9) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re9), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re9,im9)")

    # Test if we provide an image with a defined scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj.stepK()
    re3.scale = -scale
    im3.scale = -scale
    re3.setZero()
    obj.drawKImage(re3, im3)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 4,
                                   "obj.drawKImage(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 5,
                                   "obj.drawKImage(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3)")
    re3.scale = 0
    im3.scale = 0
    re3.setZero()
    obj.drawKImage(re3, im3)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 4,
                                   "obj.drawKImage(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 5,
                                   "obj.drawKImage(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3)")

    # Test if we provide an image and dx.  It should:
    #   - write to the existing image
    #   - use the provided dx
    #   - write the new dx value to the image's scale
    re9.scale = scale + 0.3  # Just something other than scale
    im9.scale = scale + 0.3
    re9.setZero()
    obj.drawKImage(re9, im9, scale=scale)
    np.testing.assert_almost_equal(
            re9.scale, scale, 9,
            "obj.drawKImage(re9,im9,scale) produced real image with wrong scale")
    np.testing.assert_almost_equal(
            im9.scale, scale, 9,
            "obj.drawKImage(re9,im9,scale) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
            re9(re9.bounds.center()), test_flux, 4,
            "obj.drawKImage(re9,im9,scale) produced real image with wrong flux")
    np.testing.assert_almost_equal(
            im9.array.sum(), 0., 5,
            "obj.drawKImage(re9,im9,scale) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re9), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re9,im9,scale)")

    # Test if we provide an image and scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj.stepK()
    re3.scale = scale + 0.3
    im3.scale = scale + 0.3
    re3.setZero()
    obj.drawKImage(re3, im3, scale=-scale)
    np.testing.assert_almost_equal(
            re3.scale, stepk, 9,
            "obj.drawKImage(re3,im3,scale<0) produced real image with wrong scale")
    np.testing.assert_almost_equal(
            im3.scale, stepk, 9,
            "obj.drawKImage(re3,im3,scale<0) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
            re3(re3.bounds.center()), test_flux, 4,
            "obj.drawKImage(re3,im3,scale<0) produced real image with wrong flux")
    np.testing.assert_almost_equal(
            im3.array.sum(), 0., 5,
            "obj.drawKImage(re3,im3,scale<0) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3,scale<0)")
    re3.scale = scale + 0.3
    im3.scale = scale + 0.3
    re3.setZero()
    obj.drawKImage(re3, im3, scale=0)
    np.testing.assert_almost_equal(
        re3.scale, stepk, 9,
        "obj.drawKImage(re3,im3,scale=0) produced real image with wrong scale")
    np.testing.assert_almost_equal(
        im3.scale, stepk, 9,
        "obj.drawKImage(re3,im3,scale=0) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
        re3(re3.bounds.center()), test_flux, 4,
        "obj.drawKImage(re3,im3,scale=0) produced real image with wrong flux")
    np.testing.assert_almost_equal(
        im3.array.sum(), 0., 5,
        "obj.drawKImage(re3,im3,scale=0) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3,scale=0)")

    # Test if we provide nx, ny, and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    nx = 200  # Some randome non-square size
    ny = 100
    re4, im4 = obj.drawKImage(nx=nx, ny=ny, scale=scale)
    np.testing.assert_almost_equal(
        re4.scale, scale, 9,
        "obj.drawKImage(nx,ny,scale) produced real image with wrong scale")
    np.testing.assert_almost_equal(
        im4.scale, scale, 9,
        "obj.drawKImage(nx,ny,scale) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
        re4.array.shape, (ny, nx), 9,
        "obj.drawKImage(nx,ny,scale) produced real image with wrong shape")
    np.testing.assert_almost_equal(
        im4.array.shape, (ny, nx), 9,
        "obj.drawKImage(nx,ny,scale) produced imag image with wrong shape")

    # Test if we provide bounds and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    bounds = galsim.BoundsI(1,nx,1,ny)
    re4, im4 = obj.drawKImage(bounds=bounds, scale=stepk)
    np.testing.assert_almost_equal(
        re4.scale, stepk, 9,
        "obj.drawKImage(bounds,scale) produced real image with wrong scale")
    np.testing.assert_almost_equal(
        im4.scale, stepk, 9,
        "obj.drawKImage(bounds,scale) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
        re4.array.shape, (ny, nx), 9,
        "obj.drawKImage(bounds,scale) produced real image with wrong shape")
    np.testing.assert_almost_equal(
        im4.array.shape, (ny, nx), 9,
        "obj.drawKImage(bounds,scale) produced imag image with wrong shape")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_drawKImage_Gaussian():
    """Test the drawKImage function using known symmetries of the Gaussian Hankel transform.

    See http://en.wikipedia.org/wiki/Hankel_transform.
    """
    import time
    t1 = time.time()

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
        gal.getFlux(), gal_hankel.xValue(galsim.PositionD(0., 0.)), decimal=12,
        err_msg="Test object flux does not equal k=(0, 0) mode of its Hankel transform conjugate.")

    image_test = galsim.ImageD(test_imsize, test_imsize)
    rekimage_test = galsim.ImageD(test_imsize, test_imsize)
    imkimage_test = galsim.ImageD(test_imsize, test_imsize)

    # Then compare these two objects at a couple of different scale (reasonably matched for size)
    for scale_test in (0.03 / test_sigma, 0.4 / test_sigma):
        gal.drawKImage(re=rekimage_test, im=imkimage_test, scale=scale_test)
        gal_hankel.drawImage(image_test, scale=scale_test, use_true_center=False, method='sb')
        np.testing.assert_array_almost_equal(
            rekimage_test.array, image_test.array, decimal=12,
            err_msg="Test object drawKImage() and drawImage() from Hankel conjugate do not match "
            "for grid spacing scale = "+str(scale_test))
        np.testing.assert_array_almost_equal(
            imkimage_test.array, np.zeros_like(imkimage_test.array), decimal=12,
            err_msg="Non-zero imaginary part for drawKImage from test object that is purely "
            "centred on the origin.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_drawKImage_Exponential_Moffat():
    """Test the drawKImage function using known symmetries of the Exponential Hankel transform 
    (which is a Moffat with beta=1.5).

    See http://mathworld.wolfram.com/HankelTransform.html.
    """
    import time
    t1 = time.time()

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
        gal.getFlux(), gal_hankel.xValue(galsim.PositionD(0., 0.)), decimal=12,
        err_msg="Test object flux does not equal k=(0, 0) mode of its Hankel transform conjugate.")

    image_test = galsim.ImageD(test_imsize, test_imsize)
    rekimage_test = galsim.ImageD(test_imsize, test_imsize)
    imkimage_test = galsim.ImageD(test_imsize, test_imsize)

    # Then compare these two objects at a couple of different scale (reasonably matched for size)
    for scale_test in (0.15 / test_scale_radius, 0.6 / test_scale_radius):
        gal.drawKImage(re=rekimage_test, im=imkimage_test, scale=scale_test)
        gal_hankel.drawImage(image_test, scale=scale_test, use_true_center=False, method='sb')
        np.testing.assert_array_almost_equal(
            rekimage_test.array, image_test.array, decimal=12,
            err_msg="Test object drawKImageImage() and drawImage() from Hankel conjugate do not "+
            "match for grid spacing scale = "+str(scale_test))
        np.testing.assert_array_almost_equal(
            imkimage_test.array, np.zeros_like(imkimage_test.array), decimal=12,
            err_msg="Non-zero imaginary part for drawKImage from test object that is purely "+
            "centred on the origin.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_offset():
    """Test the offset parameter to the drawImage function.
    """
    import time
    t1 = time.time()

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
        #print '\n\n\nnx,ny = ',nx,ny

        # First check that the image agrees with our calculation of the center
        cenx = (nx+1.)/2.
        ceny = (ny+1.)/2.
        #print 'cen = ',cenx,ceny
        im = galsim.ImageD(nx,ny, scale=scale)
        true_center = im.bounds.trueCenter()
        np.testing.assert_almost_equal(
                cenx, true_center.x, 6,
                "im.bounds.trueCenter().x is wrong for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                ceny, true_center.y, 6,
                "im.bounds.trueCenter().y is wrong for (nx,ny) = %d,%d"%(nx,ny))

        # Check that the default draw command puts the centroid in the center of the image.
        obj.drawImage(im, method='sb')
        moments = getmoments(im)
        #print 'moments = ',moments
        np.testing.assert_almost_equal(
                moments[0], cenx, 5,
                "obj.drawImage(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                moments[1], ceny, 5,
                "obj.drawImage(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))

        # Test that a few pixel values match xValue.
        # Note: we don't expect the FFT drawn image to match the xValues precisely, since the
        # latter use real-space convolution, so they should just match to our overall accuracy
        # requirement, which is something like 1.e-3 or so.  But an image of just the galaxy
        # should use real-space drawing, so should be pretty much exact.
        im2 = galsim.ImageD(nx,ny, scale=scale)
        gal.drawImage(im2, method='sb')
        for x,y in xy_list:
            #print 'x,y = ',x,y
            #print 'im(x,y) = ',im(x,y)
            u = (x-cenx) * scale
            v = (y-ceny) * scale
            #print 'xval(x-cenx,y-ceny) = ',obj.xValue(galsim.PositionD(u,v))
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

            #print 'offx,offy = ',offx,offy
            offset = galsim.PositionD(offx,offy)
            obj.drawImage(im, method='sb', offset=offset)
            moments = getmoments(im)
            #print 'moments = ',moments
            np.testing.assert_almost_equal(
                    moments[0], cenx+offx, decimal,
                    "obj.drawImage(im,offset) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
            np.testing.assert_almost_equal(
                    moments[1], ceny+offy, decimal,
                    "obj.drawImage(im,offset) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
            # Test that a few pixel values match xValue
            gal.drawImage(im2, method='sb', offset=offset)
            for x,y in xy_list:
                #print 'x,y = ',x,y
                #print 'im(x,y) = ',im(x,y)
                u = (x-cenx-offx) * scale
                v = (y-ceny-offy) * scale
                #print 'xval(x-cenx-offx,y-ceny-offy) = ',obj.xValue(galsim.PositionD(u,v))
                np.testing.assert_almost_equal(
                        im(x,y), obj.xValue(galsim.PositionD(u,v)), 2,
                        "im(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))
                np.testing.assert_almost_equal(
                        im2(x,y), gal.xValue(galsim.PositionD(u,v)), 6,
                        "im2(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))

            # Check that shift also moves the centroid by the right amount.
            shifted_obj = obj.shift(offset * scale)
            shifted_obj.drawImage(im, method='sb')
            moments = getmoments(im)
            #print 'moments = ',moments
            np.testing.assert_almost_equal(
                    moments[0], cenx+offx, decimal,
                    "shifted_obj.drawImage(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
            np.testing.assert_almost_equal(
                    moments[1], ceny+offy, decimal,
                    "shifted_obj.drawImage(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
            # Test that a few pixel values match xValue
            shifted_gal = gal.shift(offset * scale)
            shifted_gal.drawImage(im2, method='sb')
            for x,y in xy_list:
                #print 'x,y = ',x,y
                #print 'im(x,y) = ',im(x,y)
                u = (x-cenx) * scale
                v = (y-ceny) * scale
                #print 'shifted xval(x-cenx,y-ceny) = ',shifted_obj.xValue(galsim.PositionD(u,v))
                np.testing.assert_almost_equal(
                        im(x,y), shifted_obj.xValue(galsim.PositionD(u,v)), 2,
                        "im(%d,%d) does not match shifted xValue(%f,%f)"%(x,y,x-cenx,y-ceny))
                np.testing.assert_almost_equal(
                        im2(x,y), shifted_gal.xValue(galsim.PositionD(u,v)), 6,
                        "im2(%d,%d) does not match shifted xValue(%f,%f)"%(x,y,x-cenx,y-ceny))
                u = (x-cenx-offx) * scale
                v = (y-ceny-offy) * scale
                #print 'xval(x-cenx-offx,y-ceny-offy) = ',obj.xValue(galsim.PositionD(u,v))
                np.testing.assert_almost_equal(
                        im(x,y), obj.xValue(galsim.PositionD(u,v)), 2,
                        "im(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))
                np.testing.assert_almost_equal(
                        im2(x,y), gal.xValue(galsim.PositionD(u,v)), 6,
                        "im2(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))

        # Chcek the image's definition of the nominal center
        nom_cenx = (nx+2)/2
        nom_ceny = (ny+2)/2
        nominal_center = im.bounds.center()
        np.testing.assert_almost_equal(
                nom_cenx, nominal_center.x, 6,
                "im.bounds.center().x is wrong for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                nom_ceny, nominal_center.y, 6,
                "im.bounds.center().y is wrong for (nx,ny) = %d,%d"%(nx,ny))

        # Check that use_true_center = false is consistent with an offset by 0 or 0.5 pixels.
        obj.drawImage(im, method='sb', use_true_center=False)
        moments = getmoments(im)
        #print 'moments = ',moments
        np.testing.assert_almost_equal(
                moments[0], nom_cenx, 4,
                "obj.drawImage(im, use_true_center=False) not centered correctly for (nx,ny) = "+
                "%d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                moments[1], nom_ceny, 4,
                "obj.drawImage(im, use_true_center=False) not centered correctly for (nx,ny) = "+
                "%d,%d"%(nx,ny))
        cen_offset = galsim.PositionD(nom_cenx - cenx, nom_ceny - ceny)
        #print 'cen_offset = ',cen_offset
        obj.drawImage(im2, method='sb', offset=cen_offset)
        np.testing.assert_array_almost_equal(
                im.array, im2.array, 6,
                "obj.drawImage(im, offset=%f,%f) different from use_true_center=False")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_drawImage()
    test_draw_methods()
    test_drawKImage()
    test_drawKImage_Gaussian()
    test_drawKImage_Exponential_Moffat()
    test_offset()
