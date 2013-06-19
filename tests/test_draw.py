# Copyright 2012, 2013 The GalSim developers:
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
 
def test_draw():
    """Test the various optional parameters to the draw function.
       In particular test the parameters image, dx, and wmult in various combinations.
    """
    import time
    t1 = time.time()

    # We use a simple Exponential for our object:
    obj = galsim.Exponential(flux=test_flux, scale_radius=2)

    # First test draw() with no kwargs.  It should:
    #   - create a new image
    #   - return the new image
    #   - set the scale to obj.nyquistDx()
    #   - set the size large enough to contain 99.5% of the flux
    im1 = obj.draw()
    dx_nyq = obj.nyquistDx()
    np.testing.assert_almost_equal(im1.scale, dx_nyq, 9,
                                   "obj.draw() produced image with wrong scale")
    print 'im1.bounds = ',im1.bounds
    assert im1.bounds == galsim.BoundsI(1,54,1,54),(
            "obj.draw() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im1), 2, 1,
                                   "Measured wrong scale after obj.draw()")

    # The flux is only really expected to come out right if the object has been
    # convoled with a pixel:
    obj2 = galsim.Convolve([ obj, galsim.Pixel(im1.scale) ])
    im2 = obj2.draw()
    dx_nyq = obj2.nyquistDx()
    np.testing.assert_almost_equal(im2.scale, dx_nyq, 9,
                                   "obj2.draw() produced image with wrong scale")
    np.testing.assert_almost_equal(im2.array.astype(float).sum(), test_flux, 2,
                                   "obj2.draw() produced image with wrong flux")
    assert im2.bounds == galsim.BoundsI(1,54,1,54),(
            "obj2.draw() produced image with wrong bounds")
    np.testing.assert_almost_equal(CalculateScale(im2), 2, 1,
                                   "Measured wrong scale after obj2.draw()")

    # Test if we provide an image argument.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj2.nyquistDx()
    #   - zero out any existing data
    im3 = galsim.ImageD(54,54)
    im4 = obj2.draw(im3)
    np.testing.assert_almost_equal(im3.scale, dx_nyq, 9,
                                   "obj2.draw(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3.array.sum(), test_flux, 2,
                                   "obj2.draw(im3) produced image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), im2.array.astype(float).sum(), 6,
                                   "obj2.draw(im3) produced image with different flux than im2")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj2.draw(im3)")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "im4 = obj2.draw(im3) produced im4 != im3")
    im3.fill(9.8)
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "im4 = obj2.draw(im3) produced im4 is not im3")
    im4 = obj2.draw(im3)
    np.testing.assert_almost_equal(im3.array.sum(), im2.array.astype(float).sum(), 6,
                                   "obj2.draw(im3) doesn't zero out existing data")
    
    # Test if we provide an image with undefined bounds.  It should:
    #   - resize the provided image
    #   - also return that image
    #   - set the scale to obj2.nyquistDx()
    im5 = galsim.ImageD()
    obj2.draw(im5)
    np.testing.assert_almost_equal(im5.scale, dx_nyq, 9,
                                   "obj2.draw(im5) produced image with wrong scale")
    np.testing.assert_almost_equal(im5.array.sum(), test_flux, 2,
                                   "obj2.draw(im5) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im5), 2, 1,
                                   "Measured wrong scale after obj2.draw(im5)")
    np.testing.assert_almost_equal(im5.array.sum(), im2.array.astype(float).sum(), 6,
                                   "obj2.draw(im5) produced image with different flux than im2")
    assert im5.bounds == galsim.BoundsI(1,54,1,54),(
            "obj2.draw(im5) produced image with wrong bounds")

    # Test if we provide wmult.  It should:
    #   - create a new image that is wmult times larger in each direction.
    #   - return the new image
    #   - set the scale to obj2.nyquistDx()
    im6 = obj2.draw(wmult=4.)
    np.testing.assert_almost_equal(im6.scale, dx_nyq, 9,
                                   "obj2.draw(wmult) produced image with wrong scale")
    # Can assert accuracy to 4 decimal places now, since we're capturing much more
    # of the flux on the image.
    np.testing.assert_almost_equal(im6.array.astype(float).sum(), test_flux, 4,
                                   "obj2.draw(wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im6), 2, 2,
                                   "Measured wrong scale after obj2.draw(wmult)")
    print 'im6.bounds = ',im6.bounds
    assert im6.bounds == galsim.BoundsI(1,214,1,214),(
            "obj2.draw(wmult) produced image with wrong bounds")

    # Test if we provide an image argument and wmult.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj2.nyquistDx()
    #   - zero out any existing data
    #   - the calculation of the convolution should be slightly more accurate than for im3
    im3.setZero()
    im5.setZero()
    obj2.draw(im3, wmult=4.)
    obj2.draw(im5)
    np.testing.assert_almost_equal(im3.scale, dx_nyq, 9,
                                   "obj2.draw(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3.array.sum(), test_flux, 2,
                                   "obj2.draw(im3,wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj2.draw(im3,wmult)")
    assert ((im3.array-im5.array)**2).sum() > 0, (
            "obj2.draw(im3,wmult) produced the same image as without wmult")
    
    # Test if we provide a dx to use.  It should:
    #   - create a new image using that dx for the scale
    #   - return the new image
    #   - set the size large enough to contain 99.5% of the flux
    im7 = obj2.draw(dx=0.51)
    np.testing.assert_almost_equal(im7.scale, 0.51, 9,
                                   "obj2.draw(dx) produced image with wrong scale")
    np.testing.assert_almost_equal(im7.array.astype(float).sum(), test_flux, 2,
                                   "obj2.draw(dx) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im7), 2, 1,
                                   "Measured wrong scale after obj2.draw(dx)")
    print 'im7.bounds = ',im7.bounds
    assert im7.bounds == galsim.BoundsI(1,66,1,66),(
            "obj2.draw(dx) produced image with wrong bounds")

    # Test with dx and wmult.  It should:
    #   - create a new image using that dx for the scale
    #   - set the size a factor of wmult times larger in each direction.
    #   - return the new image
    im8 = obj2.draw(dx=0.51, wmult=4.)
    np.testing.assert_almost_equal(im8.scale, 0.51, 9,
                                   "obj2.draw(dx,wmult) produced image with wrong scale")
    np.testing.assert_almost_equal(im8.array.astype(float).sum(), test_flux, 4,
                                   "obj2.draw(dx,wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im8), 2, 2,
                                   "Measured wrong scale after obj2.draw(dx,wmult)")
    print 'im8.bounds = ',im8.bounds
    assert im8.bounds == galsim.BoundsI(1,264,1,264),(
            "obj2.draw(dx,wmult) produced image with wrong bounds")

    # Test if we provide an image with a defined scale.  It should:
    #   - write to the existing image
    #   - use the image's scale 
    im9 = galsim.ImageD(200,200)
    im9.setScale(0.51)
    obj2.draw(im9)
    np.testing.assert_almost_equal(im9.scale, 0.51, 9,
                                   "obj2.draw(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj2.draw(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9)")

    # Test if we provide an image with a defined scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj2.nyquistDx()
    im9.setScale(-0.51)
    im9.setZero()
    obj2.draw(im9)
    np.testing.assert_almost_equal(im9.scale, dx_nyq, 9,
                                   "obj2.draw(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj2.draw(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9)")
    im9.setScale(0)
    im9.setZero()
    obj2.draw(im9)
    np.testing.assert_almost_equal(im9.scale, dx_nyq, 9,
                                   "obj2.draw(im9) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj2.draw(im9) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9)")
    

    # Test if we provide an image and dx.  It should:
    #   - write to the existing image
    #   - use the provided dx
    #   - write the new dx value to the image's scale
    im9.setScale(0.73)
    im9.setZero()
    obj2.draw(im9, dx=0.51)
    np.testing.assert_almost_equal(im9.scale, 0.51, 9,
                                   "obj2.draw(im9,dx) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj2.draw(im9,dx) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9,dx)")

    # Test if we provide an image and dx <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj2.nyquistDx()
    im9.setScale(0.73)
    im9.setZero()
    obj2.draw(im9, dx=-0.51)
    np.testing.assert_almost_equal(im9.scale, dx_nyq, 9,
                                   "obj2.draw(im9,dx<0) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj2.draw(im9,dx<0) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9,dx<0)")
    im9.setScale(0.73)
    im9.setZero()
    obj2.draw(im9, dx=0)
    np.testing.assert_almost_equal(im9.scale, dx_nyq, 9,
                                   "obj2.draw(im9,dx=0) produced image with wrong scale")
    np.testing.assert_almost_equal(im9.array.sum(), test_flux, 4,
                                   "obj2.draw(im9,dx=0) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im9), 2, 2,
                                   "Measured wrong scale after obj2.draw(im9,dx=0)")
    
    
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_drawK():
    """Test the various optional parameters to the drawK function.
       In particular test the parameters image, and dk in various combinations.
    """
    import time
    t1 = time.time()

    # We use a Moffat profile with beta = 1.5, since its real-space profile is
    #    flux / (2 pi rD^2) * (1 + (r/rD)^2)^3/2
    # and the 2-d Fourier transform of that is
    #    flux * exp(-rD k)
    # So this should draw in Fourier space the same image as the Exponential drawn in test_draw().
    obj = galsim.Moffat(flux=test_flux, beta=1.5, scale_radius=0.5)

    # First test drawK() with no kwargs.  It should:
    #   - create new images
    #   - return the new images
    #   - set the scale to 2pi/(N*obj.nyquistDx())
    re1, im1 = obj.drawK()
    N = 1162
    assert re1.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawK() produced image with wrong bounds")
    assert im1.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawK() produced image with wrong bounds")
    dx_nyq = obj.nyquistDx()
    stepk = obj.stepK()
    print 'dx_nyq = ',dx_nyq
    print '2pi/(dx_nyq N) = ',2*np.pi/(dx_nyq*N)
    print 'stepK = ',obj.stepK()
    print 'maxK = ',obj.maxK()
    print 'im1.scale = ',im1.scale
    print 'im1.center = ',im1.bounds.center
    np.testing.assert_almost_equal(re1.scale, stepk, 9,
                                   "obj.drawK() produced real image with wrong scale")
    np.testing.assert_almost_equal(im1.scale, stepk, 9,
                                   "obj.drawK() produced imag image with wrong scale")
    np.testing.assert_almost_equal(CalculateScale(re1), 2, 1,
                                   "Measured wrong scale after obj.drawK()")

    # The flux in Fourier space is just the value at k=0
    np.testing.assert_almost_equal(re1(re1.bounds.center()), test_flux, 2,
                                   "obj.drawK() produced real image with wrong flux")
    # Imaginary component should all be 0.
    np.testing.assert_almost_equal(im1.array.sum(), 0., 3,
                                   "obj.drawK() produced non-zero imaginary image")

    # Test if we provide an image argument.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj.stepK()
    #   - zero out any existing data
    re3 = galsim.ImageD(1149,1149)
    im3 = galsim.ImageD(1149,1149)
    re4, im4 = obj.drawK(re3, im3)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawK(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawK(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 2,
                                   "obj.drawK(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 3,
                                   "obj.drawK(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawK(re3,im3)")
    np.testing.assert_array_equal(re3.array, re4.array,
                                  "re4, im4 = obj.drawK(re3,im3) produced re4 != re3")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "re4, im4 = obj.drawK(re3,im3) produced im4 != im3")
    re3.fill(9.8)
    im3.fill(9.8)
    np.testing.assert_array_equal(re3.array, re4.array,
                                  "re4, im4 = obj.drawK(re3,im3) produced re4 is not re3")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "re4, im4 = obj.drawK(re3,im3) produced im4 is not im3")
    
    # Test if we provide an image with undefined bounds.  It should:
    #   - resize the provided image
    #   - also return that image
    #   - set the scale to obj.stepK()
    re5 = galsim.ImageD()
    im5 = galsim.ImageD()
    obj.drawK(re5, im5)
    np.testing.assert_almost_equal(re5.scale, stepk, 9,
                                   "obj.drawK(re5,im5) produced real image with wrong scale")
    np.testing.assert_almost_equal(im5.scale, stepk, 9,
                                   "obj.drawK(re5,im5) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re5(re5.bounds.center()), test_flux, 2,
                                   "obj.drawK(re5,im5) produced real image with wrong flux")
    np.testing.assert_almost_equal(im5.array.sum(), 0., 3,
                                   "obj.drawK(re5,im5) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re5), 2, 1,
                                   "Measured wrong scale after obj.drawK(re5,im5)")
    assert im5.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawK(re5,im5) produced image with wrong bounds")

    # Test if we provide a dk to use.  It should:
    #   - create a new image using that dx for the scale
    #   - return the new image
    #   - set the size large enough to contain 99.5% of the flux
    re7, im7 = obj.drawK(dk=0.51)
    np.testing.assert_almost_equal(re7.scale, 0.51, 9,
                                   "obj.drawK(dx) produced real image with wrong scale")
    np.testing.assert_almost_equal(im7.scale, 0.51, 9,
                                   "obj.drawK(dx) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re7(re7.bounds.center()), test_flux, 2,
                                   "obj.drawK(dx) produced real image with wrong flux")
    np.testing.assert_almost_equal(im7.array.astype(float).sum(), 0., 2,
                                   "obj.drawK(dx) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re7), 2, 1,
                                   "Measured wrong scale after obj.drawK(dx)")
    assert im7.bounds == galsim.BoundsI(1,394,1,394),(
            "obj.drawK(dx) produced image with wrong bounds")

    # Test if we provide an image with a defined scale.  It should:
    #   - write to the existing image
    #   - use the image's scale 
    re9 = galsim.ImageD(401,401)
    im9 = galsim.ImageD(401,401)
    re9.setScale(0.51)
    im9.setScale(0.51)
    obj.drawK(re9, im9)
    np.testing.assert_almost_equal(re9.scale, 0.51, 9,
                                   "obj.drawK(re9,im9) produced real image with wrong scale")
    np.testing.assert_almost_equal(im9.scale, 0.51, 9,
                                   "obj.drawK(re9,im9) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re9(re9.bounds.center()), test_flux, 4,
                                   "obj.drawK(re9,im9) produced real image with wrong flux")
    np.testing.assert_almost_equal(im9.array.sum(), 0., 5,
                                   "obj.drawK(re9,im9) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re9), 2, 1,
                                   "Measured wrong scale after obj.drawK(re9,im9)")

    # Test if we provide an image with a defined scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj.stepK()
    re3.setScale(-0.51)
    im3.setScale(-0.51)
    re3.setZero()
    obj.drawK(re3, im3)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawK(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawK(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 4,
                                   "obj.drawK(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 5,
                                   "obj.drawK(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawK(re3,im3)")
    re3.setScale(0)
    im3.setScale(0)
    re3.setZero()
    obj.drawK(re3, im3)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawK(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawK(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 4,
                                   "obj.drawK(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 5,
                                   "obj.drawK(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawK(re3,im3)")
    
    # Test if we provide an image and dx.  It should:
    #   - write to the existing image
    #   - use the provided dx
    #   - write the new dx value to the image's scale
    re9.setScale(0.73)
    im9.setScale(0.73)
    re9.setZero()
    obj.drawK(re9, im9, dk=0.51)
    np.testing.assert_almost_equal(re9.scale, 0.51, 9,
                                   "obj.drawK(re9,im9,dk) produced real image with wrong scale")
    np.testing.assert_almost_equal(im9.scale, 0.51, 9,
                                   "obj.drawK(re9,im9,dk) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re9(re9.bounds.center()), test_flux, 4,
                                   "obj.drawK(re9,im9,dk) produced real image with wrong flux")
    np.testing.assert_almost_equal(im9.array.sum(), 0., 5,
                                   "obj.drawK(re9,im9,dk) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re9), 2, 1,
                                   "Measured wrong scale after obj.drawK(re9,im9,dk)")

    # Test if we provide an image and dk <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj.stepK()
    re3.setScale(0.73)
    im3.setScale(0.73)
    re3.setZero()
    obj.drawK(re3, im3, dk=-0.51)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawK(re3,im3,dk<0) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawK(re3,im3,dk<0) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 4,
                                   "obj.drawK(re3,im3,dk<0) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 5,
                                   "obj.drawK(re3,im3,dk<0) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawK(re3,im3,dk<0)")
    re3.setScale(0.73)
    im3.setScale(0.73)
    re3.setZero()
    obj.drawK(re3, im3, dk=0)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawK(re3,im3,dk=0) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawK(re3,im3,dk=0) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 4,
                                   "obj.drawK(re3,im3,dk=0) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 5,
                                   "obj.drawK(re3,im3,dk=0) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawK(re3,im3,dk=0)")
    
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_drawK_Gaussian():
    """Test the drawK function using known symmetries of the Gaussian Hankel transform.

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

    # Then compare these two objects at a couple of different dk (reasonably matched for size)
    for dk_test in (0.03 / test_sigma, 0.4 / test_sigma):
        gal.drawK(re=rekimage_test, im=imkimage_test, dk=dk_test) 
        gal_hankel.draw(image_test, dx=dk_test, use_true_center=False, normalization="sb")
        np.testing.assert_array_almost_equal(
            rekimage_test.array, image_test.array, decimal=12,
            err_msg="Test object drawK() and draw() from Hankel conjugate do not match for grid "+
            "spacing dk = "+str(dk_test))
        np.testing.assert_array_almost_equal(
            imkimage_test.array, np.zeros_like(imkimage_test.array), decimal=12,
            err_msg="Non-zero imaginary part for drawK from test object that is purely centred on "+
            "the origin.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_drawK_Exponential_Moffat():
    """Test the drawK function using known symmetries of the Exponential Hankel transform (which is
    a Moffat with beta=1.5).

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

    # Then compare these two objects at a couple of different dk (reasonably matched for size)
    for dk_test in (0.15 / test_scale_radius, 0.6 / test_scale_radius):
        gal.drawK(re=rekimage_test, im=imkimage_test, dk=dk_test) 
        gal_hankel.draw(image_test, dx=dk_test, use_true_center=False, normalization="sb")
        np.testing.assert_array_almost_equal(
            rekimage_test.array, image_test.array, decimal=12,
            err_msg="Test object drawK() and draw() from Hankel conjugate do not match for grid "+
            "spacing dk = "+str(dk_test))
        np.testing.assert_array_almost_equal(
            imkimage_test.array, np.zeros_like(imkimage_test.array), decimal=12,
            err_msg="Non-zero imaginary part for drawK from test object that is purely centred on "+
            "the origin.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_offset():
    """Test the offset parameter to the draw and drawShoot function.
    """
    import time
    t1 = time.time()

    scale = 0.23

    # We use a simple Exponential for our object:
    obj = galsim.Exponential(flux=test_flux, scale_radius=0.5)

    # Make the images somewhat large so the moments are measured accurately.
    for nx,ny in [ (256,256), (256,243), (279,240), (255,241)]:
        print '\n\n\nnx,ny = ',nx,ny

        # First check that the image agrees with our calculation of the center
        cenx = (nx+1.)/2.
        ceny = (ny+1.)/2.
        print 'cen = ',cenx,ceny
        im = galsim.ImageD(nx,ny)
        im.scale = scale
        true_center = im.bounds.trueCenter()
        np.testing.assert_almost_equal(
                cenx, true_center.x, 5, 
                "im.bounds.trueCenter().x is wrong for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                ceny, true_center.y, 5, 
                "im.bounds.trueCenter().y is wrong for (nx,ny) = %d,%d"%(nx,ny))

        # Check that the default draw command puts the centroid in the center of the image.
        obj.draw(im, normalization='sb')
        moments = getmoments(im)
        print 'moments = ',moments
        im.write('junk.fits')
        np.testing.assert_almost_equal(
                moments[0], cenx, 5,
                "obj.draw(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                moments[1], ceny, 5,
                "obj.draw(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
        # Thest that a few pixel values match xValue
        for x,y in [ (128,128), (123,131), (126,124) ]:
            print 'x,y = ',x,y
            print 'im(x,y) = ',im(x,y)
            u = (x-cenx) * scale
            v = (y-ceny) * scale
            print 'xval(x-cenx,y-ceny) = ',obj.xValue(galsim.PositionD(u,v))
            np.testing.assert_almost_equal(
                    im(x,y), obj.xValue(galsim.PositionD(u,v)), 5,
                    "im(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))

        # Check that offset moves the centroid by the right amount.
        offx = 1
        offy = -3
        offset = galsim.PositionD(offx,offy)
        obj.draw(im, normalization='sb', offset=offset)
        moments = getmoments(im)
        print 'moments = ',moments
        np.testing.assert_almost_equal(
                moments[0], cenx+offx, 5,
                "obj.draw(im,offset) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                moments[1], ceny+offy, 5,
                "obj.draw(im,offset) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
        # Thest that a few pixel values match xValue
        for x,y in [ (32,32), (31,33), (29,27) ]:
            print 'x,y = ',x,y
            print 'im(x,y) = ',im(x,y)
            u = (x-cenx-offx) * scale
            v = (y-ceny-offy) * scale
            print 'xval(x-cenx-offx,y-ceny-offy) = ',obj.xValue(galsim.PositionD(u,v))
            np.testing.assert_almost_equal(
                    im(x,y), obj.xValue(galsim.PositionD(u,v)), 5,
                    "im(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))

        # Check that applyShift also moves the centroid by the right amount.
        shifted_obj = obj.createShifted(offset * scale)
        shifted_obj.draw(im, normalization='sb')
        moments = getmoments(im)
        print 'moments = ',moments
        np.testing.assert_almost_equal(
                moments[0], cenx+offx, 5,
                "shifted_obj.draw(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                moments[1], ceny+offy, 5,
                "shifted_obj.draw(im) not centered correctly for (nx,ny) = %d,%d"%(nx,ny))
        # Thest that a few pixel values match xValue
        for x,y in [ (32,32), (31,33), (29,27) ]:
            print 'x,y = ',x,y
            print 'im(x,y) = ',im(x,y)
            u = (x-cenx) * scale
            v = (y-ceny) * scale
            print 'shifted xval(x-cenx,y-ceny) = ',shifted_obj.xValue(galsim.PositionD(u,v))
            np.testing.assert_almost_equal(
                    im(x,y), shifted_obj.xValue(galsim.PositionD(u,v)), 5,
                    "im(%d,%d) does not match shifted xValue(%f,%f)"%(x,y,x-cenx,y-ceny))
            u = (x-cenx-offx) * scale
            v = (y-ceny-offy) * scale
            print 'xval(x-cenx-offx,y-ceny-offy) = ',obj.xValue(galsim.PositionD(u,v))
            np.testing.assert_almost_equal(
                    im(x,y), obj.xValue(galsim.PositionD(u,v)), 5,
                    "im(%d,%d) does not match xValue(%f,%f)"%(x,y,u,v))

        # Chcek the image's definition of the nominal center
        nom_cenx = (nx+2)/2
        nom_ceny = (ny+2)/2
        nominal_center = im.bounds.center()
        np.testing.assert_almost_equal(
                nom_cenx, nominal_center.x, 5, 
                "im.bounds.center().x is wrong for (nx,ny) = %d,%d"%(nx,ny))
        np.testing.assert_almost_equal(
                nom_ceny, nominal_center.y, 5, 
                "im.bounds.center().y is wrong for (nx,ny) = %d,%d"%(nx,ny))

        # Check that use_true_center = false is consistent with an offset by 0 or 0.5 pixels.
        obj.draw(im, normalization='sb', use_true_center=False)
        moments = getmoments(im)
        print 'moments = ',moments
        np.testing.assert_almost_equal(
                moments[0], nom_cenx, 5,
                "obj.draw(im, use_true_center=False) not centered correctly for (nx,ny) = %d,%d"%(
                        nx,ny))
        np.testing.assert_almost_equal(
                moments[1], nom_ceny, 5,
                "obj.draw(im, use_true_center=False) not centered correctly for (nx,ny) = %d,%d"%(
                        nx,ny))
        im2 = galsim.ImageD(nx,ny)
        im2.scale = scale
        cen_offset = galsim.PositionD(nom_cenx - cenx, nom_ceny - ceny)
        print 'cen_offset = ',cen_offset
        obj.draw(im2, normalization='sb', offset=cen_offset)
        np.testing.assert_array_almost_equal(
                im.array, im2.array, 5,
                "obj.draw(im, offset=%f,%f) different from use_true_center=False")

if __name__ == "__main__":
    test_draw()
    test_drawK()
    test_drawK_Gaussian()
    test_drawK_Exponential_Moffat()
    test_offset()
