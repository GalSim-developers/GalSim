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

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images. 

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def printval(image1, image2):
    print "New, saved array sizes: ", np.shape(image1.array), np.shape(image2.array)
    print "Sum of values: ", np.sum(image1.array), np.sum(image2.array)
    print "Minimum image value: ", np.min(image1.array), np.min(image2.array)
    print "Maximum image value: ", np.max(image1.array), np.max(image2.array)
    print "Peak location: ", image1.array.argmax(), image2.array.argmax()
    print "Moments Mx, My, Mxx, Myy, Mxy for new array: "
    getmoments(image1)
    print "Moments Mx, My, Mxx, Myy, Mxy for saved array: "
    getmoments(image2)
    xcen = image2.array.shape[0]/2
    ycen = image2.array.shape[1]/2
    print "new image.center = ",image1.array[xcen-3:xcen+4,ycen-3:ycen+4]
    print "saved image.center = ",image2.array[xcen-3:xcen+4,ycen-3:ycen+4]

def getmoments(image1):
    xgrid, ygrid = np.meshgrid(np.arange(np.shape(image1.array)[0]) + image1.getXMin(), 
                               np.arange(np.shape(image1.array)[1]) + image1.getYMin())
    mx = np.mean(xgrid * image1.array) / np.mean(image1.array)
    my = np.mean(ygrid * image1.array) / np.mean(image1.array)
    mxx = np.mean(((xgrid-mx)**2) * image1.array) / np.mean(image1.array)
    myy = np.mean(((ygrid-my)**2) * image1.array) / np.mean(image1.array)
    mxy = np.mean((xgrid-mx) * (ygrid-my) * image1.array) / np.mean(image1.array)
    print "    ", mx-image1.getXMin(), my-image1.getYMin(), mxx, myy, mxy

def funcname():
    import inspect
    return inspect.stack()[1][3]

# define a series of tests

def test_shapelet_gaussian():
    """Test that the simplest Shapelet profile is equivalent to a Gaussian
    """
    import time
    t1 = time.time()

    ftypes = [np.float32, np.float64]
    dx = 0.2
    test_flux = 23.

    # First, a Shapelet with only b_00 = 1 should be identically a Gaussian
    im1 = galsim.ImageF(64,64)
    im1.scale = dx
    im2 = galsim.ImageF(64,64)
    im2.scale = dx
    for sigma in [1., 0.6, 2.4]:
        gauss = galsim.Gaussian(flux=test_flux, sigma=sigma)
        gauss.draw(im1)
        for order in [0, 2, 8]:
            bvec = np.zeros(galsim.LVectorSize(order))
            bvec[0] = test_flux
            shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
            shapelet.draw(im2)
            printval(im2,im1)
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, 5,
                    err_msg="Shapelet with (only) b00=1 disagrees with Gaussian result"
                    "for flux=%f, sigma=%f, order=%d"%(test_flux,sigma,order))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_draw():
    """Test some measured properties of a drawn shapelet against the supposed true values
    """
    import time
    t1 = time.time()

    ftypes = [np.float32, np.float64]
    dx = 0.2
    test_flux = 23.

    pix = galsim.Pixel(dx)
    im = galsim.ImageF(129,129)
    im.scale = dx
    for sigma in [1., 0.3, 2.4]:
        for order in [0, 2, 8]:
            shapelet = galsim.Shapelet(sigma=sigma, order=order)
            shapelet.setNM(0,0,1.)
            for n in range(1,order+1):
                if n%2 == 0:  # even n
                    #shapelet.setNM(n,0,0.23/(n*n))
                    shapelet.setPQ(n/2,n/2,0.23/(n*n))  # same thing.  Just test setPQ syntax.
                    if n >= 2:
                        shapelet.setNM(n,2,0.14/n,-0.08/n)
                else:  # odd n
                    if n >= 1:
                        shapelet.setNM(n,1,-0.08/n**3.2,0.05/n**2.1)
                    if n >= 3:
                        shapelet.setNM(n,3,0.31/n**4.2,-0.18/n**3.9)
            #print 'shapelet vector = ',shapelet.getBVec()

            # Test normalization  (This is normally part of do_shoot.  When we eventually 
            # implement photon shooting, we should go back to the normal do_shoot call, 
            # and remove this section.)
            shapelet.setFlux(test_flux)
            # Need to convolve with a pixel if we want the flux to come out right.
            conv = galsim.Convolve([pix,shapelet])
            conv.draw(im, normalization="surface brightness")
            flux = im.array.sum()
            print 'img.sum = ',flux,'  cf. ',test_flux/(dx*dx)
            np.testing.assert_almost_equal(flux * dx*dx / test_flux, 1., 4,
                    err_msg="Surface brightness normalization for Shapelet "
                    "disagrees with expected result")
            conv.draw(im, normalization="flux")
            flux = im.array.sum()
            print 'im.sum = ',flux,'  cf. ',test_flux
            np.testing.assert_almost_equal(flux / test_flux, 1., 4,
                    err_msg="Flux normalization for Shapelet disagrees with expected result")

            # Test centroid
            # Note: this only works if the image has odd sizes.  If they are even, then
            # setCenter doesn't actually set the center to the true center of the image 
            # (since it falls between pixels).
            im.setCenter(0,0)
            x,y = np.meshgrid(np.arange(im.array.shape[0]).astype(float) + im.getXMin(), 
                              np.arange(im.array.shape[1]).astype(float) + im.getYMin())
            x *= dx
            y *= dx
            flux = im.array.sum()
            mx = (x*im.array).sum() / flux
            my = (y*im.array).sum() / flux
            print 'centroid = ',mx,my,' cf. ',conv.centroid()
            np.testing.assert_almost_equal(mx, shapelet.centroid().x, 3,
                    err_msg="Measured centroid (x) for Shapelet disagrees with expected result")
            np.testing.assert_almost_equal(my, shapelet.centroid().y, 3,
                    err_msg="Measured centroid (y) for Shapelet disagrees with expected result")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_properties():
    """Test some specific numbers for a particular Shapelet profile.
    """
    import time
    t1 = time.time()

    # A semi-random particular vector of coefficients.
    sigma = 1.8
    order = 4
    bvec = [1.3,                               # n = 0
            0.02, 0.03,                        # n = 1
            0.23, -0.19, 0.08,                 # n = 2
            0.01, 0.02, 0.04, -0.03,           # n = 3
            -0.09, 0.07, -0.11, -0.08, 0.11]   # n = 4

    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)

    # Check flux
    flux = bvec[0] + bvec[5] + bvec[14]
    np.testing.assert_almost_equal(shapelet.getFlux(), flux, 10)
    # Check centroid
    cen = galsim.PositionD(bvec[1],-bvec[2]) + np.sqrt(2.) * galsim.PositionD(bvec[8],-bvec[9])
    cen *= 2. * sigma / flux
    np.testing.assert_almost_equal(shapelet.centroid().x, cen.x, 10)
    np.testing.assert_almost_equal(shapelet.centroid().y, cen.y, 10)
    # Check Fourier properties
    np.testing.assert_almost_equal(shapelet.maxK(), 4.61738371186, 10)
    np.testing.assert_almost_equal(shapelet.stepK(), 0.195133742529, 10)
    # Check image values in real and Fourier space
    zero = galsim.PositionD(0., 0.)
    np.testing.assert_almost_equal(shapelet.kValue(zero), flux+0j, 10)
    np.testing.assert_almost_equal(shapelet.xValue(zero), 0.0653321217013, 10)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shapelet_fit():
    """Test fitting a Shapelet decomposition of an image
    """
    import time
    t1 = time.time()

    for norm in ['f', 'sb']:
        # We fit a shapelet approximation of a distorted Moffat profile:
        flux = 20
        psf = galsim.Moffat(beta=3.4, half_light_radius=1.2, flux=flux)
        psf.applyShear(g1=0.11,g2=0.07)
        psf.applyShift(0.03,0.04)
        dx = 0.2
        pixel = galsim.Pixel(dx)
        conv = galsim.Convolve([psf,pixel])
        im1 = conv.draw(dx=dx, normalization=norm)

        sigma = 1.2  # Match half-light-radius as a decent first approximation.
        shapelet = galsim.Shapelet(sigma=sigma, order=10)
        shapelet.fitImage(im1, normalization=norm)
        #print 'fitted shapelet coefficients = ',shapelet.getBVec()

        # Check flux
        print 'flux = ',shapelet.getFlux(),'  cf. ',flux
        np.testing.assert_almost_equal(shapelet.getFlux() / flux, 1., 1,
                err_msg="Fitted shapelet has the wrong flux")

        # Test centroid
        print 'centroid = ',shapelet.centroid(),'  cf. ',conv.centroid()
        np.testing.assert_almost_equal(shapelet.centroid().x, conv.centroid().x, 2,
                err_msg="Fitted shapelet has the wrong centroid (x)")
        np.testing.assert_almost_equal(shapelet.centroid().y, conv.centroid().y, 2,
                err_msg="Fitted shapelet has the wrong centroid (y)")

        # Test drawing image from shapelet
        im2 = im1.copy()
        shapelet.draw(im2, normalization=norm)
        im2.write('junk2.fits')
        # Check that images are close to the same:
        print 'norm(diff) = ',np.sum((im1.array-im2.array)**2)
        print 'norm(im) = ',np.sum(im1.array**2)
        assert np.sum((im1.array-im2.array)**2) < 1.e-3 * np.sum(im1.array**2)

        # Remeasure -- should now be very close to the same.
        shapelet2 = shapelet.copy()
        shapelet2.fitImage(im2, normalization=norm)
        np.testing.assert_equal(shapelet.getSigma(), shapelet2.getSigma(),
                err_msg="Second fitted shapelet has the wrong sigma")
        np.testing.assert_equal(shapelet.getOrder(), shapelet2.getOrder(),
                err_msg="Second fitted shapelet has the wrong order")
        np.testing.assert_almost_equal(shapelet.getBVec(), shapelet2.getBVec(), 6,
                err_msg="Second fitted shapelet coefficients do not match original")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shapelet_adjustments():
    """Test that adjusting the Shapelet profile in various ways does the right thing
    """
    import time
    t1 = time.time()

    ftypes = [np.float32, np.float64]

    nx = 128
    ny = 128
    dx = 0.2
    im = galsim.ImageF(nx,ny)
    im.scale = dx

    sigma = 1.8
    order = 6
    bvec = [1.3,                                            # n = 0
            0.02, 0.03,                                     # n = 1
            0.23, -0.19, 0.08,                              # n = 2
            0.01, 0.02, 0.04, -0.03,                        # n = 3
            -0.09, 0.07, -0.11, -0.08, 0.11,                # n = 4
            -0.03, -0.02, -0.08, 0.01, -0.06, -0.03,        # n = 5
            0.06, -0.02, 0.00, -0.05, -0.04, 0.01, 0.09 ]   # n = 6

    ref_shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
    ref_im = galsim.ImageF(nx,ny)
    ref_shapelet.draw(ref_im, dx=dx)

    # Test setSigma
    shapelet = galsim.Shapelet(sigma=1., order=order, bvec=bvec)
    shapelet.setSigma(sigma)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setSigma disagrees with reference Shapelet")

    # Test setBVec
    shapelet = galsim.Shapelet(sigma=sigma, order=order)
    shapelet.setBVec(bvec)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setBVec disagrees with reference Shapelet")

    # Test setOrder
    shapelet = galsim.Shapelet(sigma=sigma, order=2)
    shapelet.setOrder(order)
    shapelet.setBVec(bvec)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setOrder disagrees with reference Shapelet")

    # Test that changing the order preserves the values to the extent possible.
    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
    shapelet.setOrder(10)
    np.testing.assert_array_equal(
        shapelet.getBVec()[0:28], bvec, 
        err_msg="Shapelet setOrder to larger doesn't preserve existing values.")
    np.testing.assert_array_equal(
        shapelet.getBVec()[28:66], np.zeros(66-28),
        err_msg="Shapelet setOrder to larger doesn't fill with zeros.")
    shapelet.setOrder(6)
    np.testing.assert_array_equal(
        shapelet.getBVec(), bvec, 
        err_msg="Shapelet setOrder back to original from larger doesn't preserve existing values.")
    shapelet.setOrder(3)
    np.testing.assert_array_equal(
        shapelet.getBVec()[0:10], bvec[0:10], 
        err_msg="Shapelet setOrder to smaller doesn't preserve existing values.")
    shapelet.setOrder(6)
    np.testing.assert_array_equal(
        shapelet.getBVec()[0:10], bvec[0:10], 
        err_msg="Shapelet setOrder back to original from smaller doesn't preserve existing values.")
    shapelet.setOrder(6)
    np.testing.assert_array_equal(
        shapelet.getBVec()[10:28], np.zeros(28-10),
        err_msg="Shapelet setOrder back to original from smaller doesn't fill with zeros.")

    # Test that setting a Shapelet with setNM gives the right profile
    shapelet = galsim.Shapelet(sigma=sigma, order=order)
    i = 0
    for n in range(order+1):
        for m in range(n,-1,-2):
            if m == 0:
                shapelet.setNM(n,m,bvec[i])
                i = i+1
            else:
                shapelet.setNM(n,m,bvec[i],bvec[i+1])
                i = i+2
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setNM disagrees with reference Shapelet")

    # Test that setting a Shapelet with setPQ gives the right profile
    shapelet = galsim.Shapelet(sigma=sigma, order=order)
    i = 0
    for n in range(order+1):
        for m in range(n,-1,-2):
            p = (n+m)/2
            q = (n-m)/2
            if m == 0:
                shapelet.setPQ(p,q,bvec[i])
                i = i+1
            else:
                shapelet.setPQ(p,q,bvec[i],bvec[i+1])
                i = i+2
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setPQ disagrees with reference Shapelet")

    # Test that the Shapelet setFlux does the same thing as the GSObject setFlux
    gsref_shapelet = galsim.GSObject(ref_shapelet)  # Make it opaque to the Shapelet versions
    gsref_shapelet.setFlux(23.)
    gsref_shapelet.draw(ref_im)
    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
    shapelet.setFlux(23.)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet setFlux disagrees with GSObject setFlux")

    # Test that the Shapelet scaleFlux does the same thing as the GSObject scaleFlux
    gsref_shapelet.scaleFlux(0.23)
    gsref_shapelet.draw(ref_im)
    shapelet.scaleFlux(0.23)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet setFlux disagrees with SObject scaleFlux")

    # Test that the Shapelet applyRotation does the same thing as the GSObject applyRotation
    gsref_shapelet.applyRotation(23. * galsim.degrees)
    gsref_shapelet.draw(ref_im)
    shapelet.applyRotation(23. * galsim.degrees)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet applyRotation disagrees with GSObject applyRotation")

    # Test that the Shapelet applyDilation does the same thing as the GSObject applyDilation
    gsref_shapelet.applyDilation(1.3)
    gsref_shapelet.draw(ref_im)
    shapelet.applyDilation(1.3)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet applyDilation disagrees with GSObject applyDilation")

    # Test that the Shapelet applyMagnification does the same thing as the GSObject 
    # applyMagnification
    gsref_shapelet.applyMagnification(0.8)
    gsref_shapelet.draw(ref_im)
    shapelet.applyMagnification(0.8)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet applyMagnification disagrees with GSObject applyMagnification")

    # Test that applyLensing works on Shapelet
    gsref_shapelet.applyLensing(-0.05, 0.15, 1.1)
    gsref_shapelet.draw(ref_im)
    shapelet.applyLensing(-0.05, 0.15, 1.1)
    shapelet.draw(im)
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet applyLensing disagrees with GSObject applyLensing")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



if __name__ == "__main__":
    test_shapelet_gaussian()
    test_shapelet_draw()
    test_shapelet_properties()
    test_shapelet_fit()
    test_shapelet_adjustments()
