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

# These positions will be used a few times below, so define them here.
# One of the tests requires that the last pair are integers, so don't change that.
near_x_list = [ 0, 0.242, -1.342, -5 ]
near_y_list = [ 0, -0.173, 2.003, -7 ]

far_x_list = [ 10, -31.7, -183.6, -700 ]
far_y_list = [ 10, 12.5, 103.3, 500 ]

# Make a few different profiles to check.  Make sure to include ones that 
# aren't symmetrical so we don't get fooled by symmetries.
prof1 = galsim.Gaussian(sigma = 1.7, flux = 100)
prof2 = prof1.createSheared(g1=0.3, g2=-0.12)
prof3 = prof2 + galsim.Exponential(scale_radius = 1.3, flux = 20).createShifted(-0.1,-0.4)
profiles = [ prof1, prof2, prof3 ]

if __name__ != "__main__":
    # Some of the classes we test here are not terribly fast.  WcsToolsWCS in particular.
    # So reduce the number of tests.  Keep the hardest ones, since the easier ones are mostly
    # useful as diagnostics when there are problems.  So they will get run when doing
    # python test_wcs.py.  But not during a nosetests run.
    near_x_list = near_x_list[-2:]
    near_y_list = near_y_list[-2:]
    far_x_list = far_x_list[-2:]
    far_y_list = far_y_list[-2:]
    profiles = [ prof3 ]

all_x_list = near_x_list + far_x_list
all_y_list = near_y_list + far_y_list

# How many digits of accuracy should we demand?  
# We test everything in units of arcsec, so this corresponds to 1.e-3 arcsec.  
# 1 mas should be plenty accurate for our purposes.  (And note that most classes do much
# better than this.  Just a few things that require iterative solutions for the world->image
# transformation or things that output to a fits file at slightly less than full precision
# do worse than 6 digits.)
digits = 3

# The HPX, TAN, TSC, STG, ZEA, and ZPN files were downloaded from the web site:
# 
# http://www.atnf.csiro.au/people/mcalabre/WCS/example_data.html
# 
# From that page: "The example maps and spectra are offered mainly as test material for 
# software that deals with FITS WCS."
#
# I picked the 4 that GSFitsWCS can do plus a couple others that struck me as interstingly 
# different, but there are a bunch more on that web page as well.  In particular, I included ZPN,
# since that uses PV values, which the others don't, and HPX, since it is not implemented in 
# wcstools.
# 
# The SIP, TPV, ZPX, REGION, and TNX are either new or "non-standard" that are not implemented 
# by (at least older versions of) wcslib.  They were downloaded from the web site:
#
# http://fits.gsfc.nasa.gov/fits_registry.html
#
# For each file, I use ds9 to pick out two reference points.  I generally try to pick two 
# points on opposite sides of the image so any non-linearities in the WCS are maximized.
# For most of them, I then use wcstools to get the ra and dec to 6 digits of accuracy.
# (Unfortunately, ds9's ra and dec information is only accurate to 2 or 3 digits.)
# The exception is HPX, for which I used the PyAst library to compute accurate values,
# since wcstools can't understand it.

references = {
    # Note: the four 1904-66 files use the brightest pixels in the same two stars.
    # The ra, dec are thus essentially the same (modulo the large pixel size of 3 arcmin).
    # However, the image positions are quite different.
    'HPX' : ('1904-66_HPX.fits' , 
            [ ('193916.551671', '-634247.346862', 114, 180, 13.59960),
              ('181935.761589', '-634608.860203', 144, 30, 11.49591) ] ),
    'TAN' : ('1904-66_TAN.fits' ,
            [ ('193930.753119', '-634259.217527', 117, 178, 13.43628),
              ('181918.652839', '-634903.833411', 153, 35, 11.44438) ] ),
    'TSC' : ('1904-66_TSC.fits' , 
            [ ('193939.996553', '-634114.585586', 113, 161, 12.48409),
              ('181905.985494', '-634905.781036', 141, 48, 11.65945) ] ),
    'STG' : ('1904-66_STG.fits' ,
            [ ('193914.752140', '-634420.882465', 112, 172, 13.1618),
              ('181937.824461', '-634624.483497', 147, 38, 11.6091) ] ),
    'ZEA' : ('1904-66_ZEA.fits' ,
            [ ('193926.871566', '-634326.059526', 110, 170, 13.253),
              ('181934.480902', '-634640.038427', 144, 39, 11.62) ] ),
    'ARC' : ('1904-66_ARC.fits' ,
            [ ('193928.622018', '-634153.658982', 111, 171, 13.7654),
              ('181947.020701', '-634622.381334', 145, 39, 11.2099) ] ),
    'ZPN' : ('1904-66_ZPN.fits' ,
            [ ('193924.948254', '-634643.636138', 95, 151, 12.84769),
              ('181924.149409', '-634937.453404', 122, 48, 11.01434) ] ),
    'SIP' : ('sipsample.fits' ,
            [ ('133001.474154', '471251.794474', 242, 75, 12.24437),
              ('132943.747626', '470913.879660', 12, 106, 5.30282) ] ),
    'TPV' : ('tpv.fits',
            [ ('033009.340034', '-284350.811107', 418, 78, 2859.53882),
              ('033015.728999', '-284501.488629', 148, 393, 2957.98584) ] ),
    # Strangely, zpx.fits is the same image as tpv.fits, but the WCS-computed RA, Dec 
    # values are not anywhere close to TELRA, TELDEC in the header.  It's a bit 
    # unfortunate, since my understanding is that ZPX can encode the same function as
    # TPV, so they could have produced the equivalent function.  But instead they just
    # inserted some totally off-the-wall different WCS transformation.
    'ZPX' : ('zpx.fits',
            [ ('212412.094326', '371034.575917', 418, 78, 2859.53882),
              ('212405.350816', '371144.596579', 148, 393, 2957.98584) ] ),
    # Older versions of the new TPV standard just used the TAN wcs name and expected
    # the code to notice the PV values and use them correctly.  This did not become a
    # FITS standard (or even a registered non-standard), but some old FITS files use
    # this, so we want to support it.  I just edited the tpv.fits to change the 
    # CTYPE values from TPV to TAN.
    'TAN-PV' : ('tanpv.fits',
            [ ('033009.340034', '-284350.811107', 418, 78, 2859.53882),
              ('033015.728999', '-284501.488629', 148, 393, 2957.98584) ] ),
    'REGION' : ('region.fits',
            [ ('140211.202432', '543007.702200', 80, 80, 2241),
              ('140417.341523', '541628.554326', 45, 54, 1227) ] ),
    # Strangely, ds9 seems to get this one wrong.  It differs by about 6 arcsec in dec.
    # But PyAst and wcstools agree on these values, so I'm taking them to be accurate.
    'TNX' : ('tnx.fits',
            [ ('174653.214511', '-300847.895372', 32, 91, 7140),
              ('174658.100741', '-300750.121787', 246, 326, 15022) ] ),
}
all_tags = references.keys()


def do_wcs_pos(wcs, ufunc, vfunc, name, x0=0, y0=0):
    # I would call this do_wcs_pos_tests, but nosetests takes any function with test 
    # _anywhere_ in the name an tries to run it.  So make sure the name doesn't 
    # have 'test' in it.  There are a bunch of other do* functions that work similarly.

    #print 'start do_wcs_pos for ',name, wcs
    # Check that (x,y) -> (u,v) and converse work correctly
    if 'local' in name or 'jacobian' in name or 'affine' in name:
        # If the "local" is really a non-local WCS which has been localized, then we cannot
        # count on the far positions to be sufficiently accurate. Just use near positions.
        x_list = near_x_list
        y_list = near_y_list
        # And even then, it sometimes fails at our normal 3 digits because of the 2nd derivative
        # coming into play.
        digits2 = 1
    else:
        x_list = all_x_list
        y_list = all_y_list
        digits2 = digits
    u_list = [ ufunc(x+x0,y+y0) for x,y in zip(x_list, y_list) ]
    v_list = [ vfunc(x+x0,y+y0) for x,y in zip(x_list, y_list) ]

    for x,y,u,v in zip(x_list, y_list, u_list, v_list):
        image_pos = galsim.PositionD(x+x0,y+y0)
        world_pos = galsim.PositionD(u,v)
        #print 'image_pos = ',image_pos
        #print 'world_pos = ',world_pos
        world_pos2 = wcs.toWorld(image_pos)
        #print 'world_pos2 = ',world_pos2
        np.testing.assert_almost_equal(
                world_pos.x, world_pos2.x, digits2,
                'wcs.toWorld returned wrong world position for '+name)
        np.testing.assert_almost_equal(
                world_pos.y, world_pos2.y, digits2,
                'wcs.toWorld returned wrong world position for '+name)

        scale = wcs.maxLinearScale(image_pos)
        try:
            # The reverse transformation is not guaranteed to be implemented,
            # so guard against NotImplementedError being raised:
            image_pos2 = wcs.toImage(world_pos)
            #print 'image_pos2 = ',image_pos2
            np.testing.assert_almost_equal(
                    image_pos.x*scale, image_pos2.x*scale, digits2,
                    'wcs.toImage returned wrong image position for '+name)
            np.testing.assert_almost_equal(
                    image_pos.y*scale, image_pos2.y*scale, digits2,
                    'wcs.toImage returned wrong image position for '+name)
        except NotImplementedError:
            pass

    if x0 == 0 and y0 == 0:
        # The last item in list should also work as a PositionI
        image_pos = galsim.PositionI(x,y)
        np.testing.assert_almost_equal(
                world_pos.x, wcs.toWorld(image_pos).x, digits2,
                'wcs.toWorld gave different value with PositionI image_pos for '+name)
        np.testing.assert_almost_equal(
                world_pos.y, wcs.toWorld(image_pos).y, digits2,
                'wcs.toWorld gave different value with PositionI image_pos for '+name)


def check_world(pos1, pos2, digits, err_msg):
    try:
        x = pos1.x
        y = pos2.y
    except AttributeError:
        np.testing.assert_almost_equal(pos1.distanceTo(pos2) / galsim.arcsec, 0, digits, err_msg)
    else:
        np.testing.assert_almost_equal(pos1.x, pos2.x, digits, err_msg)
        np.testing.assert_almost_equal(pos1.y, pos2.y, digits, err_msg)

def do_wcs_image(wcs, name, approx=False):
    
    print 'Start image tests for WCS '+name
    #print 'wcs = ',wcs

    # Use the "blank" image as our test image.  It's not blank in the sense of having all
    # zeros.  Rather, there are basically random values that we can use to test that 
    # the shifted values are correct.  And it is a conveniently small-ish, non-square image.
    dir = 'fits_files'
    file_name = 'blankimg.fits'
    im = galsim.fits.read(file_name, dir=dir)
    np.testing.assert_equal(im.origin().x, 1, "initial origin is not 1,1 as expected")
    np.testing.assert_equal(im.origin().y, 1, "initial origin is not 1,1 as expected")
    im.wcs = wcs
    world1 = im.wcs.toWorld(im.origin())
    value1 = im(im.origin())
    world2 = im.wcs.toWorld(im.center())
    value2 = im(im.center())
    offset = galsim.PositionI(11,13)
    image_pos = im.origin() + offset
    world3 = im.wcs.toWorld(image_pos)
    value3 = im(image_pos)

    # Test that im.shift does the right thing to the wcs
    # Also test parsing a position as x,y args.
    dx = 3
    dy = 9
    im.shift(3,9)
    image_pos = im.origin() + offset
    np.testing.assert_equal(im.origin().x, 1+dx, "shift set origin to wrong value")
    np.testing.assert_equal(im.origin().y, 1+dy, "shift set origin to wrong value")
    check_world(im.wcs.toWorld(im.origin()), world1, digits,
                "World position of origin after shift is wrong.")
    np.testing.assert_almost_equal(im(im.origin()), value1, digits,
                                   "Image value at origin after shift is wrong.")
    check_world(im.wcs.toWorld(im.center()), world2, digits,
                "World position of center after shift is wrong.")
    np.testing.assert_almost_equal(im(im.center()), value2, digits,
                                   "Image value at center after shift is wrong.")
    check_world(im.wcs.toWorld(image_pos), world3, digits,
                "World position of image_pos after shift is wrong.")
    np.testing.assert_almost_equal(im(image_pos), value3, digits,
                                   "image value at center after shift is wrong.")

    # Test writing the image to a fits file and reading it back in.
    # The new image doesn't have to have the same wcs type.  But it does have to produce
    # consistent values of the world coordinates.
    test_name = 'test_' + name + '.fits'
    im.write(test_name, dir=dir)
    im2 = galsim.fits.read(test_name, dir=dir)
    if approx:
        # Sometimes, the round trip doesn't preserve accuracy completely.
        # In these cases, only test the positions after write/read to 1 digit.
        digits2 = 1
    else:
        digits2 = digits
    np.testing.assert_equal(im2.origin().x, im.origin().x, "origin changed after write/read")
    np.testing.assert_equal(im2.origin().y, im.origin().y, "origin changed after write/read")
    check_world(im2.wcs.toWorld(im.origin()), world1, digits2,
                "World position of origin is wrong after write/read.")
    np.testing.assert_almost_equal(im2(im.origin()), value1, digits,
                                   "Image value at origin is wrong after write/read.")
    check_world(im2.wcs.toWorld(im.center()), world2, digits2,
                "World position of center is wrong after write/read.")
    np.testing.assert_almost_equal(im2(im.center()), value2, digits,
                                   "Image value at center is wrong after write/read.")
    check_world(im2.wcs.toWorld(image_pos), world3, digits2,
                "World position of image_pos is wrong after write/read.")
    np.testing.assert_almost_equal(im2(image_pos), value3, digits,
                                   "Image value at center is wrong after write/read.")

    if wcs.isUniform():
        # Test that the regular CD, CRPIX, CRVAL items that are written to the header
        # describe an equivalent WCS as this one.
        hdu, hdu_list, fin = galsim.fits.readFile(test_name, dir=dir)
        affine = galsim.AffineTransform._readHeader(hdu.header)
        affine = affine.setOrigin(galsim.PositionD(dx,dy))
        galsim.fits.closeHDUList(hdu_list, fin)
        check_world(affine.toWorld(im.origin()), world1, digits2,
                    "World position of origin is wrong after write/read.")
        check_world(affine.toWorld(im.center()), world2, digits2,
                    "World position of center is wrong after write/read.")
        check_world(affine.toWorld(image_pos), world3, digits2,
                    "World position of image_pos is wrong after write/read.")


    # Test that im.setOrigin does the right thing to the wcs
    # Also test parsing a position as a tuple.
    new_origin = (-3432, 1907)
    im.setOrigin(new_origin)
    image_pos = im.origin() + offset
    np.testing.assert_equal(im.origin().x, new_origin[0], "setOrigin set origin to wrong value")
    np.testing.assert_equal(im.origin().y, new_origin[1], "setOrigin set origin to wrong value")
    check_world(im.wcs.toWorld(im.origin()), world1, digits,
                "World position of origin after setOrigin is wrong.")
    np.testing.assert_almost_equal(im(im.origin()), value1, digits,
                                   "Image value at origin after setOrigin is wrong.")
    check_world(im.wcs.toWorld(im.center()), world2, digits,
                "World position of center after setOrigin is wrong.")
    np.testing.assert_almost_equal(im(im.center()), value2, digits,
                                   "Image value at center after setOrigin is wrong.")
    check_world(im.wcs.toWorld(image_pos), world3, digits,
                "World position of image_pos after setOrigin is wrong.")
    np.testing.assert_almost_equal(im(image_pos), value3, digits,
                                   "Image value at center after setOrigin is wrong.")

    # Test that im.setCenter does the right thing to the wcs.
    # Also test parsing a position as a PositionI object.
    new_center = galsim.PositionI(0,0)
    im.setCenter(new_center)
    image_pos = im.origin() + offset
    np.testing.assert_equal(im.center().x, new_center.x, "setCenter set center to wrong value")
    np.testing.assert_equal(im.center().y, new_center.y, "setCenter set center to wrong value")
    check_world(im.wcs.toWorld(im.origin()), world1, digits,
                "World position of origin after setCenter is wrong.")
    np.testing.assert_almost_equal(im(im.origin()), value1, digits,
                                   "Image value at origin after setCenter is wrong.")
    check_world(im.wcs.toWorld(im.center()), world2, digits,
                "World position of center after setCenter is wrong.")
    np.testing.assert_almost_equal(im(im.center()), value2, digits,
                                   "Image value at center after setCenter is wrong.")
    check_world(im.wcs.toWorld(image_pos), world3, digits,
                "World position of image_pos after setCenter is wrong.")
    np.testing.assert_almost_equal(im(image_pos), value3, digits,
                                   "Image value at center after setCenter is wrong.")

    # Test makeSkyImage
    new_origin = (-134, 128)
    im.setOrigin(new_origin)
    sky_level = 177
    wcs.makeSkyImage(im, sky_level)
    for x,y in [ (im.bounds.xmin, im.bounds.ymin), 
                 (im.bounds.xmax, im.bounds.ymin),
                 (im.bounds.xmin, im.bounds.ymax),
                 (im.bounds.xmax, im.bounds.ymax),
                 (im.center().x, im.center().y) ]:
        val = im(x,y)
        area = wcs.pixelArea(galsim.PositionD(x,y))
        np.testing.assert_almost_equal(val/(area*sky_level), 1., digits,
                                       "SkyImage at %d,%d is wrong"%(x,y))


def do_local_wcs(wcs, ufunc, vfunc, name):

    print 'Start testing local WCS '+name
    #print 'wcs = ',wcs

    # Check that local and setOrigin work correctly:
    wcs2 = wcs.local()
    assert wcs == wcs2, name+' local() is not == the original'
    new_origin = galsim.PositionI(123,321)
    wcs3 = wcs.setOrigin(new_origin)
    assert wcs != wcs3, name+' is not != wcs.setOrigin(pos)'
    assert wcs3 != wcs, name+' is not != wcs.setOrigin(pos) (reverse)'
    wcs2 = wcs3.local()
    assert wcs == wcs2, name+' is not equal after wcs.setOrigin(pos).local()'
    world_pos1 = wcs.toWorld(galsim.PositionD(0,0))
    world_pos2 = wcs3.toWorld(new_origin)
    np.testing.assert_almost_equal(
            world_pos2.x, world_pos1.x, digits,
            'setOrigin(new_origin) returned wrong world position')
    np.testing.assert_almost_equal(
            world_pos2.y, world_pos1.y, digits,
            'setOrigin(new_origin) returned wrong world position')
    new_world_origin = galsim.PositionD(5352.7, 9234.3)
    wcs4 = wcs.setOrigin(new_origin, new_world_origin)
    world_pos3 = wcs4.toWorld(new_origin)
    np.testing.assert_almost_equal(
            world_pos3.x, new_world_origin.x, digits,
            'setOrigin(new_origin, new_world_origin) returned wrong position')
    np.testing.assert_almost_equal(
            world_pos3.y, new_world_origin.y, digits,
            'setOrigin(new_origin, new_world_origin) returned wrong position')

    # Check inverse:
    image_pos = wcs.inverse().toWorld(world_pos1)
    np.testing.assert_almost_equal(
            image_pos.x, 0, digits,
            'wcs.inverse().toWorld(world_pos) returned wrong image position')
    np.testing.assert_almost_equal(
            image_pos.y, 0, digits,
            'wcs.inverse().toWorld(world_pos) returned wrong image position')
    image_pos = wcs4.toImage(new_world_origin)
    np.testing.assert_almost_equal(
            image_pos.x, new_origin.x, digits,
            'wcs4.toImage(new_world_origin) returned wrong image position')
    np.testing.assert_almost_equal(
            image_pos.y, new_origin.y, digits,
            'wcs4.toImage(new_world_origin) returned wrong image position')
    image_pos = wcs4.inverse().toWorld(new_world_origin)
    np.testing.assert_almost_equal(
            image_pos.x, new_origin.x, digits,
            'wcs4.inverse().toWorld(new_world_origin) returned wrong image position')
    np.testing.assert_almost_equal(
            image_pos.y, new_origin.y, digits,
            'wcs4.inverse().toWorld(new_world_origin) returned wrong image position')

    # Check that (x,y) -> (u,v) and converse work correctly
    do_wcs_pos(wcs, ufunc, vfunc, name)

    # Test the transformation of a GSObject
    # These only work for local WCS projections!

    near_u_list = [ ufunc(x,y) for x,y in zip(near_x_list, near_y_list) ]
    near_v_list = [ vfunc(x,y) for x,y in zip(near_x_list, near_y_list) ]

    im1 = galsim.Image(64,64, wcs=wcs)
    im2 = galsim.Image(64,64, scale=1.)

    for world_profile in profiles:
        #print 'profile = ',world_profile
        # The profiles build above are in world coordinates (as usual)
    
        # Convert to image coordinates
        image_profile = wcs.toImage(world_profile)

        # Also check round trip (starting with either one)
        world_profile2 = wcs.toWorld(image_profile)
        image_profile2 = wcs.toImage(world_profile2)

        for x,y,u,v in zip(near_x_list, near_y_list, near_u_list, near_v_list):
            #print x,y,u,v
            image_pos = galsim.PositionD(x,y)
            world_pos = galsim.PositionD(u,v)
            pixel_area = wcs.pixelArea(image_pos=image_pos)

            np.testing.assert_almost_equal(
                    image_profile.xValue(image_pos) / pixel_area,
                    world_profile.xValue(world_pos), digits,
                    'xValue for image_profile and world_profile differ for '+name)
            np.testing.assert_almost_equal(
                    image_profile.xValue(image_pos),
                    image_profile2.xValue(image_pos), digits,
                    'image_profile not equivalent after round trip through world for '+name)
            np.testing.assert_almost_equal(
                    world_profile.xValue(world_pos),
                    world_profile2.xValue(world_pos), digits,
                    'world_profile not equivalent after round trip through image for '+name)

        # The last item in list should also work as a PositionI
        image_pos = galsim.PositionI(x,y)
        np.testing.assert_almost_equal(
                pixel_area, wcs.pixelArea(image_pos=image_pos), digits,
                'pixelArea gave different result for PositionI image_pos for '+name)
        np.testing.assert_almost_equal(
                image_profile.xValue(image_pos) / pixel_area,
                world_profile.xValue(world_pos), digits,
                'xValue for image_profile gave different result for PositionI for '+name)
        np.testing.assert_almost_equal(
                image_profile.xValue(image_pos),
                image_profile2.xValue(image_pos), digits,
                'round trip xValue gave different result for PositionI for '+name)

        # Test drawing the profile on an image with the given wcs
        world_profile.draw(im1)
        image_profile.draw(im2)
        np.testing.assert_array_almost_equal(
                im1.array, im2.array, digits,
                'world_profile and image_profile were different when drawn for '+name)


def do_jac_decomp(wcs, name):

    #print 'Check deomposition for ',name,wcs

    scale, shear, theta, flip = wcs.getDecomposition()
    #print 'decomposition = ',scale, shear, theta, flip

    # First see if we can recreate the right matrix from this:
    S = np.matrix( [ [ 1.+shear.g1, shear.g2 ],
                     [ shear.g2, 1.-shear.g1 ] ] ) / np.sqrt(1.-shear.g1**2-shear.g2**2)
    R = np.matrix( [ [ np.cos(theta.rad()), -np.sin(theta.rad()) ],
                     [ np.sin(theta.rad()), np.cos(theta.rad()) ] ] )
    if flip:
        F = np.matrix( [ [ 0, 1 ],
                         [ 1, 0 ] ] )
    else:
        F = np.matrix( [ [ 1, 0 ],
                         [ 0, 1 ] ] )

    M = scale * S * R * F
    J = wcs.getMatrix()
    np.testing.assert_almost_equal(
            M, J, 8, "Decomposition was inconsistent with jacobian for "+name)

    # The minLinearScale is scale * (1-g) / sqrt(1-g^2)
    import math
    g = shear.getG()
    min_scale = scale * (1.-g) / math.sqrt(1.-g**2)
    np.testing.assert_almost_equal(wcs.minLinearScale(), min_scale, 6, "minLinearScale")
    # The maxLinearScale is scale * (1+g) / sqrt(1-g^2)
    max_scale = scale * (1.+g) / math.sqrt(1.-g**2)
    np.testing.assert_almost_equal(wcs.maxLinearScale(), max_scale, 6, "minLinearScale")

    # There are some relations between the decomposition and the inverse decomposition that should 
    # be true:
    scale2, shear2, theta2, flip2 = wcs.inverse().getDecomposition()
    #print 'inverse decomposition = ',scale2, shear2, theta2, flip2
    np.testing.assert_equal(flip, flip2, "inverse flip")
    np.testing.assert_almost_equal(scale, 1./scale2, 6, "inverse scale")
    if flip:
        np.testing.assert_almost_equal(theta.rad(), theta2.rad(), 6, "inverse theta")
    else:
        np.testing.assert_almost_equal(theta.rad(), -theta2.rad(), 6, "inverse theta")
    np.testing.assert_almost_equal(shear.getG(), shear2.getG(), 6, "inverse shear")
    # There is no simple relation between the directions of the shear in the two cases.
    # The shear direction gets mixed up by the rotation if that is non-zero.

    # Also check that the profile is transformed equivalently as advertised in the docstring
    # for getDecomposition.
    base_obj = galsim.Gaussian(sigma=2)
    # Make sure it doesn't have any initial symmetry!
    base_obj.applyShear(g1=0.1, g2=0.23)
    base_obj.applyShift(0.17, -0.37)

    obj1 = base_obj.copy()
    obj1.applyTransformation(wcs.dudx, wcs.dudy, wcs.dvdx, wcs.dvdy)

    obj2 = base_obj.copy()
    if flip:
        obj2.applyTransformation(0,1,1,0)
    obj2.applyRotation(theta)
    obj2.applyShear(shear)
    obj2.applyExpansion(scale)

    gsobject_compare(obj1, obj2)


def do_nonlocal_wcs(wcs, ufunc, vfunc, name):

    print 'Start testing non-local WCS '+name
    #print 'wcs = ',wcs

    # Check that setOrigin and local work correctly:
    new_origin = galsim.PositionI(123,321)
    wcs3 = wcs.setOrigin(new_origin)
    #print 'wcs3 = ',wcs3
    assert wcs != wcs3, name+' is not != wcs.setOrigin(pos)'
    wcs4 = wcs.local(wcs.origin)
    assert wcs != wcs4, name+' is not != wcs.local()'
    assert wcs4 != wcs, name+' is not != wcs.local() (reverse)'
    world_origin = wcs.toWorld(wcs.origin)
    if wcs.isUniform():
        if wcs.world_origin == galsim.PositionD(0,0):
            wcs2 = wcs.local(wcs.origin).setOrigin(wcs.origin)
            assert wcs == wcs2, name+' is not equal after wcs.local().setOrigin(origin)'
        wcs2 = wcs.local(wcs.origin).setOrigin(wcs.origin, wcs.world_origin)
        assert wcs == wcs2, name+' not equal after wcs.local().setOrigin(origin,world_origin)'
    world_pos1 = wcs.toWorld(galsim.PositionD(0,0))
    wcs3 = wcs.setOrigin(new_origin)
    world_pos2 = wcs3.toWorld(new_origin)
    np.testing.assert_almost_equal(
            world_pos2.x, world_pos1.x, digits,
            'setOrigin(new_origin) returned wrong world position')
    np.testing.assert_almost_equal(
            world_pos2.y, world_pos1.y, digits,
            'setOrigin(new_origin) returned wrong world position')
    if not wcs.isCelestial():
        new_world_origin = galsim.PositionD(5352.7, 9234.3)
        wcs5 = wcs.setOrigin(new_origin, new_world_origin)
        world_pos3 = wcs5.toWorld(new_origin)
        np.testing.assert_almost_equal(
                world_pos3.x, new_world_origin.x, digits,
                'setOrigin(new_origin, new_world_origin) returned wrong position')
        np.testing.assert_almost_equal(
                world_pos3.y, new_world_origin.y, digits,
                'setOrigin(new_origin, new_world_origin) returned wrong position')


    # Check that (x,y) -> (u,v) and converse work correctly
    # These tests work regardless of whether the WCS is local or not.
    do_wcs_pos(wcs, ufunc, vfunc, name)

    # The GSObject transformation tests are only valid for a local WCS. 
    # But it should work for wcs.local()

    far_u_list = [ ufunc(x,y) for x,y in zip(far_x_list, far_y_list) ]
    far_v_list = [ vfunc(x,y) for x,y in zip(far_x_list, far_y_list) ]

    full_im1 = galsim.Image(galsim.BoundsI(-1023,1024,-1023,1024), wcs=wcs)
    full_im2 = galsim.Image(galsim.BoundsI(-1023,1024,-1023,1024), scale=1.)

    for x0,y0,u0,v0 in zip(far_x_list, far_y_list, far_u_list, far_v_list):
        #print 'x0,y0 = ',x0,y0
        local_ufunc = lambda x,y: ufunc(x+x0,y+y0) - u0
        local_vfunc = lambda x,y: vfunc(x+x0,y+y0) - v0
        image_pos = galsim.PositionD(x0,y0)
        world_pos = galsim.PositionD(u0,v0)
        do_wcs_pos(wcs.local(image_pos), local_ufunc, local_vfunc, name+'.local(image_pos)')
        do_wcs_pos(wcs.jacobian(image_pos), local_ufunc, local_vfunc, name+'.jacobian(image_pos)')
        do_wcs_pos(wcs.affine(image_pos), ufunc, vfunc, name+'.affine(image_pos)', x0, y0)

        try:
            # The local call is not guaranteed to be implemented for world_pos.
            # So guard against NotImplementedError.
            do_wcs_pos(wcs.local(world_pos=world_pos), local_ufunc, local_vfunc,
                       name + '.local(world_pos)')
            do_wcs_pos(wcs.jacobian(world_pos=world_pos), local_ufunc, local_vfunc,
                       name + '.jacobian(world_pos)')
            do_wcs_pos(wcs.affine(world_pos=world_pos), ufunc, vfunc, name+'.affine(world_pos)',
                       x0, y0)
        except NotImplementedError:
            pass

        # Test drawing the profile on an image with the given wcs
        ix0 = int(x0)
        iy0 = int(y0)
        dx = x0 - ix0
        dy = y0 - iy0
        b = galsim.BoundsI(ix0-31, ix0+31, iy0-31, iy0+31)
        im1 = full_im1[b]
        im2 = full_im2[b]

        for world_profile in profiles:
            #print 'profile = ',world_profile
            image_profile = wcs.toImage(world_profile, image_pos=image_pos)

            world_profile.draw(im1, offset=(dx,dy))
            image_profile.draw(im2, offset=(dx,dy))
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, digits,
                    'world_profile and image_profile differed when drawn for '+name)

            try:
                # The toImage call is not guaranteed to be implemented for world_pos.
                # So guard against NotImplementedError.
                image_profile = wcs.toImage(world_profile, world_pos=world_pos)

                world_profile.draw(im1, offset=(dx,dy))
                image_profile.draw(im2, offset=(dx,dy))
                np.testing.assert_array_almost_equal(
                        im1.array, im2.array, digits,
                        'world_profile and image_profile differed when drawn for '+name)
            except NotImplementedError:
                pass


def do_celestial_wcs(wcs, name):
    # It's a bit harder to test WCS functions that return a CelestialCoord, since 
    # (usually) we don't have an exact formula to compare with.  So the tests here
    # are a bit sparer.

    print 'Start testing celestial WCS '+name
    #print 'wcs = ',wcs

    # Check that setOrigin and local work correctly:
    new_origin = galsim.PositionI(123,321)
    wcs3 = wcs.setOrigin(new_origin)
    assert wcs != wcs3, name+' is not != wcs.setOrigin(pos)'
    wcs4 = wcs.local(wcs.origin)
    assert wcs != wcs4, name+' is not != wcs.local()'
    assert wcs4 != wcs, name+' is not != wcs.local() (reverse)'
    world_pos1 = wcs.toWorld(galsim.PositionD(0,0))
    wcs3 = wcs.setOrigin(new_origin)
    world_pos2 = wcs3.toWorld(new_origin)
    np.testing.assert_almost_equal(
            world_pos2.distanceTo(world_pos1) / galsim.arcsec, 0, digits,
            'setOrigin(new_origin) returned wrong world position')

    world_origin = wcs.toWorld(wcs.origin)

    full_im1 = galsim.Image(galsim.BoundsI(-1023,1024,-1023,1024), wcs=wcs)
    full_im2 = galsim.Image(galsim.BoundsI(-1023,1024,-1023,1024), scale=1.)

    # Some of the FITS images have really huge pixel scales.  Lower the accuracy requirement
    # for them.  2 digits in arcsec corresponds to 4 digits in pixels.
    max_scale = wcs.maxLinearScale(wcs.origin)
    #print 'max_scale = ',max_scale
    if max_scale > 100:  # arcsec
        digits2 = 2
    else:
        digits2 = digits

    for x0,y0 in zip(near_x_list, near_y_list):
        #print 'x0,y0 = ',x0,y0
        image_pos = galsim.PositionD(x0,y0)
        world_pos = wcs.toWorld(image_pos)

        # Check the calculation of the jacobian
        w1 = wcs.toWorld(galsim.PositionD(x0+0.5,y0))
        w2 = wcs.toWorld(galsim.PositionD(x0-0.5,y0))
        w3 = wcs.toWorld(galsim.PositionD(x0,y0+0.5))
        w4 = wcs.toWorld(galsim.PositionD(x0,y0-0.5))
        cosdec = np.cos(world_pos.dec.rad())
        jac = wcs.jacobian(image_pos)
        np.testing.assert_array_almost_equal(
                jac.dudx, (w2.ra - w1.ra)/galsim.arcsec * cosdec, digits2,
                'jacobian dudx incorrect for '+name)
        np.testing.assert_array_almost_equal(
                jac.dudy, (w4.ra - w3.ra)/galsim.arcsec * cosdec, digits2,
                'jacobian dudy incorrect for '+name)
        np.testing.assert_array_almost_equal(
                jac.dvdx, (w1.dec - w2.dec)/galsim.arcsec, digits2,
                'jacobian dvdx incorrect for '+name)
        np.testing.assert_array_almost_equal(
                jac.dvdy, (w3.dec - w4.dec)/galsim.arcsec, digits2,
                'jacobian dvdy incorrect for '+name)

        # Test drawing the profile on an image with the given wcs
        ix0 = int(x0)
        iy0 = int(y0)
        dx = x0 - ix0
        dy = y0 - iy0
        b = galsim.BoundsI(ix0-31, ix0+31, iy0-31, iy0+31)
        im1 = full_im1[b]
        im2 = full_im2[b]

        for world_profile in profiles:
            #print 'profile = ',world_profile
            image_profile = wcs.toImage(world_profile, image_pos=image_pos)

            world_profile.draw(im1, offset=(dx,dy))
            image_profile.draw(im2, offset=(dx,dy))
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, digits,
                    'world_profile and image_profile differed when drawn for '+name)

            try:
                # The toImage call is not guaranteed to be implemented for world_pos.
                # So guard against NotImplementedError.
                image_profile = wcs.toImage(world_profile, world_pos=world_pos)

                world_profile.draw(im1, offset=(dx,dy))
                image_profile.draw(im2, offset=(dx,dy))
                np.testing.assert_array_almost_equal(
                        im1.array, im2.array, digits,
                        'world_profile and image_profile differed when drawn for '+name)
            except NotImplementedError:
                pass

def test_pixelscale():
    """Test the PixelScale class
    """
    import time
    t1 = time.time()

    scale = 0.23
    wcs = galsim.PixelScale(scale)

    # Check basic copy and == , !=:
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'PixelScale copy is not == the original'
    wcs3 = galsim.PixelScale(scale + 0.1234)
    assert wcs != wcs3, 'PixelScale is not != a different one'
   
    ufunc = lambda x,y: x*scale
    vfunc = lambda x,y: y*scale

    # Do generic tests that apply to all WCS types
    do_local_wcs(wcs, ufunc, vfunc, 'PixelScale')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'PixelScale')

    # Check jacobian()
    jac = wcs.jacobian()
    np.testing.assert_almost_equal(jac.dudx, scale, digits,
                                   'PixelScale dudx does not match expected value.')
    np.testing.assert_almost_equal(jac.dudy, 0., digits,
                                   'PixelScale dudy does not match expected value.')
    np.testing.assert_almost_equal(jac.dvdx, 0., digits,
                                   'PixelScale dvdx does not match expected value.')
    np.testing.assert_almost_equal(jac.dvdy, scale, digits,
                                   'PixelScale dvdy does not match expected value.')

    # Check the decomposition:
    do_jac_decomp(jac, 'PixelScale')

    # Add an image origin offset
    x0 = 1
    y0 = 1
    origin = galsim.PositionD(x0,y0)
    wcs = galsim.OffsetWCS(scale, origin)

    # Check basic copy and == , != for OffsetWCS:
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'OffsetWCS copy is not == the original'
    wcs3a = galsim.OffsetWCS(scale+0.123, origin)
    wcs3b = galsim.OffsetWCS(scale, origin*2)
    wcs3c = galsim.OffsetWCS(scale, origin, origin)
    assert wcs != wcs3a, 'OffsetWCS is not != a different one (scale)'
    assert wcs != wcs3b, 'OffsetWCS is not != a different one (origin)'
    assert wcs != wcs3c, 'OffsetWCS is not != a different one (world_origin)'

    ufunc = lambda x,y: scale*(x-x0)
    vfunc = lambda x,y: scale*(y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'OffsetWCS 1')

    # Add a world origin offset
    u0 = 124.3
    v0 = -141.9
    world_origin = galsim.PositionD(u0,v0)
    wcs = galsim.OffsetWCS(scale, world_origin=world_origin)
    ufunc = lambda x,y: scale*x + u0
    vfunc = lambda x,y: scale*y + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'OffsetWCS 2')

    # Add both kinds of offsets
    x0 = -3
    y0 = 104
    u0 = 1423.9
    v0 = 8242.7
    origin = galsim.PositionD(x0,y0)
    world_origin = galsim.PositionD(u0,v0)
    wcs = galsim.OffsetWCS(scale, origin=origin, world_origin=world_origin)
    ufunc = lambda x,y: scale*(x-x0) + u0
    vfunc = lambda x,y: scale*(y-y0) + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'OffsetWCS 3')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'OffsetWCS')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shearwcs():
    """Test the ShearWCS class
    """
    import time
    t1 = time.time()

    scale = 0.23
    g1 = 0.14
    g2 = -0.37
    shear = galsim.Shear(g1=g1,g2=g2)
    wcs = galsim.ShearWCS(scale, shear)

    # Check basic copy and == , !=:
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'ShearWCS copy is not == the original'
    wcs3a = galsim.ShearWCS(scale + 0.1234, shear)
    wcs3b = galsim.ShearWCS(scale, -shear)
    assert wcs != wcs3a, 'ShearWCS is not != a different one (scale)'
    assert wcs != wcs3b, 'ShearWCS is not != a different one (shear)'
    
    factor = 1./np.sqrt(1.-g1*g1-g2*g2)
    ufunc = lambda x,y: (x - g1*x - g2*y) * scale * factor
    vfunc = lambda x,y: (y + g1*y - g2*x) * scale * factor

    # Do generic tests that apply to all WCS types
    do_local_wcs(wcs, ufunc, vfunc, 'ShearWCS')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'ShearWCS')

    # Check jacobian()
    jac = wcs.jacobian()
    np.testing.assert_almost_equal(jac.dudx, (1.-g1) * scale * factor,  digits,
                                   'ShearWCS dudx does not match expected value.')
    np.testing.assert_almost_equal(jac.dudy, -g2 * scale * factor,  digits,
                                   'ShearWCS dudy does not match expected value.')
    np.testing.assert_almost_equal(jac.dvdx, -g2 * scale * factor,  digits,
                                   'ShearWCS dvdx does not match expected value.')
    np.testing.assert_almost_equal(jac.dvdy, (1.+g1) * scale * factor,  digits,
                                   'ShearWCS dvdy does not match expected value.')

    # Check the decomposition:
    do_jac_decomp(jac, 'ShearWCS')

    # Add an image origin offset
    x0 = 1
    y0 = 1
    origin = galsim.PositionD(x0,y0)
    wcs = galsim.OffsetShearWCS(scale, shear, origin)

    # Check basic copy and == , != for OffsetShearWCS:
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'OffsetShearWCS copy is not == the original'
    wcs3a = galsim.OffsetShearWCS(scale+0.123, shear, origin)
    wcs3b = galsim.OffsetShearWCS(scale, -shear, origin)
    wcs3c = galsim.OffsetShearWCS(scale, shear, origin*2)
    wcs3d = galsim.OffsetShearWCS(scale, shear, origin, origin)
    assert wcs != wcs3a, 'OffsetShearWCS is not != a different one (scale)'
    assert wcs != wcs3b, 'OffsetShearWCS is not != a different one (shear)'
    assert wcs != wcs3c, 'OffsetShearWCS is not != a different one (origin)'
    assert wcs != wcs3d, 'OffsetShearWCS is not != a different one (world_origin)'

    ufunc = lambda x,y: ((1-g1)*(x-x0) - g2*(y-y0)) * scale * factor
    vfunc = lambda x,y: ((1+g1)*(y-y0) - g2*(x-x0)) * scale * factor
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'OffsetShearWCS 1')

    # Add a world origin offset
    u0 = 124.3
    v0 = -141.9
    world_origin = galsim.PositionD(u0,v0)
    wcs = galsim.OffsetShearWCS(scale, shear, world_origin=world_origin)
    ufunc = lambda x,y: ((1-g1)*x - g2*y) * scale * factor + u0
    vfunc = lambda x,y: ((1+g1)*y - g2*x) * scale * factor + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'OffsetShearWCS 2')

    # Add both kinds of offsets
    x0 = -3
    y0 = 104
    u0 = 1423.9
    v0 = 8242.7
    origin = galsim.PositionD(x0,y0)
    world_origin = galsim.PositionD(u0,v0)
    wcs = galsim.OffsetShearWCS(scale, shear, origin=origin, world_origin=world_origin)
    ufunc = lambda x,y: ((1-g1)*(x-x0) - g2*(y-y0)) * scale * factor + u0
    vfunc = lambda x,y: ((1+g1)*(y-y0) - g2*(x-x0)) * scale * factor + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'OffsetShearWCS 3')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'OffsetShearWCS')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_affinetransform():
    """Test the AffineTransform class
    """
    import time
    t1 = time.time()

    # First a slight tweak on a simple scale factor
    dudx = 0.2342
    dudy = 0.0023
    dvdx = 0.0019
    dvdy = 0.2391

    wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)

    # Check basic copy and == , !=:
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'JacobianWCS copy is not == the original'
    wcs3a = galsim.JacobianWCS(dudx+0.123, dudy, dvdx, dvdy)
    wcs3b = galsim.JacobianWCS(dudx, dudy+0.123, dvdx, dvdy)
    wcs3c = galsim.JacobianWCS(dudx, dudy, dvdx+0.123, dvdy)
    wcs3d = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy+0.123)
    assert wcs != wcs3a, 'JacobianWCS is not != a different one (dudx)'
    assert wcs != wcs3b, 'JacobianWCS is not != a different one (dudy)'
    assert wcs != wcs3c, 'JacobianWCS is not != a different one (dvdx)'
    assert wcs != wcs3d, 'JacobianWCS is not != a different one (dvdy)'

    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    do_local_wcs(wcs, ufunc, vfunc, 'JacobianWCS 1')

    # Check the decomposition:
    do_jac_decomp(wcs, 'JacobianWCS 1')

    # Add an image origin offset
    x0 = 1
    y0 = 1
    origin = galsim.PositionD(x0,y0)
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin)

    # Check basic copy and == , != for AffineTransform:
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'AffineTransform copy is not == the original'
    wcs3a = galsim.AffineTransform(dudx+0.123, dudy, dvdx, dvdy, origin)
    wcs3b = galsim.AffineTransform(dudx, dudy+0.123, dvdx, dvdy, origin)
    wcs3c = galsim.AffineTransform(dudx, dudy, dvdx+0.123, dvdy, origin)
    wcs3d = galsim.AffineTransform(dudx, dudy, dvdx, dvdy+0.123, origin)
    wcs3e = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin*2)
    wcs3f = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin, origin)
    assert wcs != wcs3a, 'AffineTransform is not != a different one (dudx)'
    assert wcs != wcs3b, 'AffineTransform is not != a different one (dudy)'
    assert wcs != wcs3c, 'AffineTransform is not != a different one (dvdx)'
    assert wcs != wcs3d, 'AffineTransform is not != a different one (dvdy)'
    assert wcs != wcs3e, 'AffineTransform is not != a different one (origin)'
    assert wcs != wcs3f, 'AffineTransform is not != a different one (world_origin)'

    ufunc = lambda x,y: dudx*(x-x0) + dudy*(y-y0)
    vfunc = lambda x,y: dvdx*(x-x0) + dvdy*(y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'AffineTransform 1')

    # Next one with a flip and significant rotation and a large (u,v) offset
    dudx = 0.1432
    dudy = 0.2342
    dvdx = 0.2391
    dvdy = 0.1409

    wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    do_local_wcs(wcs, ufunc, vfunc, 'JacobianWCS 2')

    # Check the decomposition:
    do_jac_decomp(wcs, 'JacobianWCS 2')

    # Add a world origin offset
    u0 = 124.3
    v0 = -141.9
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, world_origin=galsim.PositionD(u0,v0))
    ufunc = lambda x,y: dudx*x + dudy*y + u0
    vfunc = lambda x,y: dvdx*x + dvdy*y + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'AffineTransform 2')

    # Finally a really crazy one that isn't remotely regular
    dudx = 0.2342
    dudy = -0.1432
    dvdx = 0.0924
    dvdy = -0.3013

    wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    do_local_wcs(wcs, ufunc, vfunc, 'Jacobian 3')

    # Check the decomposition:
    do_jac_decomp(wcs, 'JacobianWCS 3')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'JacobianWCS')

    # Add both kinds of offsets
    x0 = -3
    y0 = 104
    u0 = 1423.9
    v0 = 8242.7
    origin = galsim.PositionD(x0,y0)
    world_origin = galsim.PositionD(u0,v0)
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=origin, world_origin=world_origin)
    ufunc = lambda x,y: dudx*(x-x0) + dudy*(y-y0) + u0
    vfunc = lambda x,y: dvdx*(x-x0) + dvdy*(y-y0) + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'AffineTransform 3')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'AffineTransform')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def radial_u(x, y):
    """A cubic radial function used for a u(x,y) function """
    # Note: This is designed to be smooth enough that the local approximates are accurate
    # to 5 decimal places when we do the local tests.
    #
    # We will use a functional form of rho/r0 = r/r0 + a (r/r0)^3 
    # To be accurate to < 1.e-6 arcsec at an offset of 7 pixels at r = 700, we need:
    #     | rho(r+dr) - rho(r) - drho/dr(r) * dr | < 1.e-6
    #     1/2 |d2(rho)/dr^2| * dr^2  < 1.e-6
    #     rho = r + a/r0^2 r^3 
    #     rho' = 1 + 3a/r0^2 r^2 
    #     rho'' = 6a/r0^2 r 
    #     1/2 6|a| / 2000^2 * 700 * 7^2 < 1.e-6 
    #     |a| < 3.8e-5

    r0 = 2000.  # scale factor for function
    a = 2.3e-5
    rho_over_r = 1 + a * (x*x+y*y)/(r0*r0)
    return x * rho_over_r

def radial_v(x, y):
    """A radial function used for a u(x,y) function """
    r0 = 2000.
    a = 2.3e-5
    rho_over_r = 1 + a * (x*x+y*y)/(r0*r0)
    return y * rho_over_r

class Cubic(object):
    """A class that can act as a function, implementing a cubic radial function.  """
    def __init__(self, a, r0, whichuv): 
        self._a = a
        self._r0 = r0
        self._uv = whichuv

    def __call__(self, x, y): 
        rho_over_r = 1 + self._a * (x*x+y*y)/(self._r0*self._r0)
        if self._uv == 'u':
            return x * rho_over_r
        else:
            return y * rho_over_r


def test_uvfunction():
    """Test the UVFunction class
    """
    import time
    t1 = time.time()

    # First make some that are identical to simpler WCS classes:
    # 1. Like PixelScale
    scale = 0.17
    ufunc = lambda x,y: x * scale
    vfunc = lambda x,y: y * scale
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like PixelScale')

    # Also check with inverse functions.
    xfunc = lambda u,v: u / scale
    yfunc = lambda u,v: v / scale
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like PixelScale with inverse')
 
    # 2. Like ShearWCS
    scale = 0.23
    g1 = 0.14
    g2 = -0.37
    factor = 1./np.sqrt(1.-g1*g1-g2*g2)
    ufunc = lambda x,y: (x - g1*x - g2*y) * scale * factor
    vfunc = lambda x,y: (y + g1*y - g2*x) * scale * factor
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like ShearWCS')
    
    # Also check with inverse functions.
    xfunc = lambda u,v: (u + g1*u + g2*v) / scale * factor
    yfunc = lambda u,v: (v - g1*v + g2*u) / scale * factor
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like ShearWCS with inverse')

    # 3. Like an AffineTransform
    dudx = 0.2342
    dudy = 0.1432
    dvdx = 0.1409
    dvdy = 0.2391

    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like AffineTransform')

    # Check that passing functions as strings works correctly.
    wcs = galsim.UVFunction(ufunc='%f*x + %f*y'%(dudx,dudy), vfunc='%f*x + %f*y'%(dvdx,dvdy))
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction with string funcs')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_string')

    # Also check with inverse functions.
    det = dudx*dvdy - dudy*dvdx
    wcs = galsim.UVFunction(
            ufunc='%f*x + %f*y'%(dudx,dudy),
            vfunc='%f*x + %f*y'%(dvdx,dvdy),
            xfunc='(%f*u + %f*v)/(%.8f)'%(dvdy,-dudy,det),
            yfunc='(%f*u + %f*v)/(%.8f)'%(-dvdx,dudx,det) )
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction with string inverse funcs')

    # 4. Next some UVFunctions with non-trivial offsets
    x0 = 1.3
    y0 = -0.9
    u0 = 124.3
    v0 = -141.9
    origin = galsim.PositionD(x0,y0)
    world_origin = galsim.PositionD(u0,v0)
    ufunc2 = lambda x,y: dudx*(x-x0) + dudy*(y-y0) + u0
    vfunc2 = lambda x,y: dvdx*(x-x0) + dvdy*(y-y0) + v0
    wcs = galsim.UVFunction(ufunc2, vfunc2)
    do_nonlocal_wcs(wcs, ufunc2, vfunc2, 'UVFunction with origins in funcs')
    wcs = galsim.UVFunction(ufunc, vfunc, origin=origin, world_origin=world_origin)
    do_nonlocal_wcs(wcs, ufunc2, vfunc2, 'UVFunction with origin arguments')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_lambda')

    # Check basic copy and == , != for UVFunction
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'UVFunction copy is not == the original'
    wcs3a = galsim.UVFunction(vfunc, vfunc, origin=origin, world_origin=world_origin)
    wcs3b = galsim.UVFunction(ufunc, ufunc, origin=origin, world_origin=world_origin)
    wcs3c = galsim.UVFunction(ufunc, vfunc, origin=origin*2, world_origin=world_origin)
    wcs3d = galsim.UVFunction(ufunc, vfunc, origin=origin, world_origin=world_origin*2)
    assert wcs != wcs3a, 'UVFunction is not != a different one (ufunc)'
    assert wcs != wcs3b, 'UVFunction is not != a different one (vfunc)'
    assert wcs != wcs3c, 'UVFunction is not != a different one (origin)'
    assert wcs != wcs3d, 'UVFunction is not != a different one (world_origin)'

    # 5. Now some non-trivial 3rd order radial function.
    origin = galsim.PositionD(x0,y0)
    wcs = galsim.UVFunction(radial_u, radial_v, origin=origin)

    # Check jacobian()
    for x,y in zip(far_x_list, far_y_list):
        image_pos = galsim.PositionD(x,y)
        jac = wcs.jacobian(image_pos)
        # u = x * rho_over_r
        # v = y * rho_over_r
        # For simplicity of notation, let rho_over_r = w(r) = 1 + a r^2/r0^2 
        # dudx = w + x dwdr drdx = w + x (2ar/r0^2) (x/r) = w + 2a x^2/r0^2
        # dudy = x dwdr drdy = x (2ar/r0^2) (y/r) = 2a xy/r0^2
        # dvdx = y dwdr drdx = y (2ar/r0^2) (x/r) = 2a xy/r0^2
        # dvdy = w + y dwdr drdy = w + y (2ar/r0^2) (y/r) = w + 2a y^2/r0^2
        r0 = 2000.
        a = 2.3e-5
        factor = a/(r0*r0)
        w = 1. + factor*(x*x+y*y)
        np.testing.assert_almost_equal(jac.dudx, w + 2*factor*x*x, digits,
                                       'UVFunction dudx does not match expected value.')
        np.testing.assert_almost_equal(jac.dudy, 2*factor*x*y, digits,
                                       'UVFunction dudy does not match expected value.')
        np.testing.assert_almost_equal(jac.dvdx, 2*factor*x*y, digits,
                                       'UVFunction dvdx does not match expected value.')
        np.testing.assert_almost_equal(jac.dvdy, w + 2*factor*y*y, digits,
                                       'UVFunction dvdy does not match expected value.')

    ufunc = lambda x,y: radial_u(x-x0, y-y0)
    vfunc = lambda x,y: radial_v(x-x0, y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Cubic radial UVFunction')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_func')

    # 6. Repeat with a function object rather than a regular function.
    # Use a different `a` parameter for u and v to make things more interesting.
    cubic_u = Cubic(2.9e-5, 2000., 'u')
    cubic_v = Cubic(-3.7e-5, 2000., 'v')
    wcs = galsim.UVFunction(cubic_u, cubic_v, origin=galsim.PositionD(x0,y0))
    ufunc = lambda x,y: cubic_u(x-x0, y-y0)
    vfunc = lambda x,y: cubic_v(x-x0, y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Cubic object UVFunction')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_object')

    # 7. Test the UVFunction that is used in demo9 to confirm that I got the 
    # inverse function correct!
    ufunc = lambda x,y : 0.05 * x * (1. + 2.e-6 * (x**2 + y**2))
    vfunc = lambda x,y : 0.05 * y * (1. + 2.e-6 * (x**2 + y**2))
    # w = 0.05 (r + 2.e-6 r^3)
    # 0 = r^3 + 5e5 r - 1e7 w
    # 
    # Cardano's formula gives
    # (http://en.wikipedia.org/wiki/Cubic_function#Cardano.27s_method)
    # r = ( sqrt( (5e6 w)^2 + (5e5)^3/27 ) + (5e6 w) )^1/3 -
    #     ( sqrt( (5e6 w)^2 + (5e5)^3/27 ) - (5e6 w) )^1/3
    #   = 100 ( ( 5 sqrt( w^2 + 5.e3/27 ) + 5 w )^1/3 -
    #           ( 5 sqrt( w^2 + 5.e3/27 ) - 5 w )^1/3 )
    import math
    xfunc = lambda u,v : (
        lambda w: ( 0. if w==0. else
            100.*u/w*(( 5*math.sqrt(w**2+5.e3/27.)+5*w )**(1./3.) -
                      ( 5*math.sqrt(w**2+5.e3/27.)-5*w )**(1./3.))) )(math.sqrt(u**2+v**2))
    yfunc = lambda u,v : (
        lambda w: ( 0. if w==0. else
            100.*v/w*(( 5*math.sqrt(w**2+5.e3/27.)+5*w )**(1./3.) -
                      ( 5*math.sqrt(w**2+5.e3/27.)-5*w )**(1./3.))) )(math.sqrt(u**2+v**2))
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction from demo9')

    # This version doesn't work with numpy arrays because of the math functions.
    # This provides a test of that branch of the makeSkyImage function.
    ufunc = lambda x,y : 0.17 * x * (1. + 1.e-5 * math.sqrt(x**2 + y**2))
    vfunc = lambda x,y : 0.17 * y * (1. + 1.e-5 * math.sqrt(x**2 + y**2))
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction with math funcs')
    if __name__ == "__main__":
        do_wcs_image(wcs, 'UVFunction_math')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_radecfunction():
    """Test the RaDecFunction class
    """
    import time
    t1 = time.time()

    # Do a sterographic projection of the above UV functions around a given reference point.
    funcs = []

    scale = 0.17
    ufunc = lambda x,y: x * scale
    vfunc = lambda x,y: y * scale
    funcs.append( (ufunc, vfunc, 'like PixelScale') )
 
    scale = 0.23
    g1 = 0.14
    g2 = -0.37
    factor = 1./np.sqrt(1.-g1*g1-g2*g2)
    ufunc = lambda x,y: (x - g1*x - g2*y) * scale * factor
    vfunc = lambda x,y: (y + g1*y - g2*x) * scale * factor
    funcs.append( (ufunc, vfunc, 'like ShearWCS') )

    dudx = 0.2342
    dudy = 0.1432
    dvdx = 0.1409
    dvdy = 0.2391
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    funcs.append( (ufunc, vfunc, 'like JacobianWCS') )

    x0 = 1.3
    y0 = -0.9
    u0 = 124.3
    v0 = -141.9
    origin = galsim.PositionD(x0,y0)
    world_origin = galsim.PositionD(u0,v0)
    ufunc = lambda x,y: dudx*(x-x0) + dudy*(y-y0) + u0
    vfunc = lambda x,y: dvdx*(x-x0) + dvdy*(y-y0) + v0
    funcs.append( (ufunc, vfunc, 'like AffineTransform') )

    funcs.append( (radial_u, radial_v, 'Cubic radial') )

    ufunc = lambda x,y: radial_u(x-x0, y-y0)
    vfunc = lambda x,y: radial_v(x-x0, y-y0)
    funcs.append( (ufunc, vfunc, 'offset Cubic radial') )

    cubic_u = Cubic(2.9e-5, 2000., 'u')
    cubic_v = Cubic(-3.7e-5, 2000., 'v')
    funcs.append( (cubic_u, cubic_v, 'Cubic object') )

    ufunc = lambda x,y: cubic_u(x-x0, y-y0)
    vfunc = lambda x,y: cubic_v(x-x0, y-y0)
    funcs.append( (cubic_u, cubic_v, 'offset Cubic object') )


    # ra, dec of projection centers (in degrees)
    # Note: This won't work directly at a pole, but it should work rather close to the pole.
    #       Using dec = 89.5 (in the last one) failed some of our tests, but 89.1 succeeds.
    centers = [ (0.,0.), (34., 12.), (-190.4, -79.8), (234.56, 89.1) ]

    # We need this below.
    north_pole = galsim.CelestialCoord(0 * galsim.degrees, 90 * galsim.degrees)

    for ufunc, vfunc, name in funcs:
        u0 = ufunc(0.,0.)
        v0 = vfunc(0.,0.)
        wcs1 = galsim.UVFunction(ufunc, vfunc)
        for cenra, cendec in centers:
            center = galsim.CelestialCoord(cenra * galsim.degrees, cendec * galsim.degrees)
            # Unit test the hms, dms parsers:
            np.testing.assert_almost_equal(galsim.HMS_Angle(center.ra.hms()).wrap() / galsim.arcsec,
                                           center.ra.wrap() / galsim.arcsec, digits,
                                           'HMS parser error')
            np.testing.assert_almost_equal(galsim.DMS_Angle(center.dec.dms()) / galsim.arcsec,
                                           center.dec / galsim.arcsec, digits, 
                                           'DMS parser error')

            radec_func = lambda x,y: center.deproject_rad(ufunc(x,y), vfunc(x,y))
            wcs2 = galsim.RaDecFunction(radec_func)

            # Also test with one that doesn't work with numpy arrays to test that the 
            # code does the right thing in that case too, since local and makeSkyImage
            # try the numpy option first and do something else if it fails.
            alt_radec_func = ( lambda x,y: 
                    [ (c.ra.rad(), c.dec.rad()) for c in 
                            [ center.deproject(galsim.PositionD(ufunc(x,y), vfunc(x,y))) ] ][0] )
            wcs3 = galsim.RaDecFunction(alt_radec_func)

            # Check that distance, jacobian for some x,y positions match the UV values.
            for x,y in zip(far_x_list, far_y_list):

                # First do some basic checks of project, deproject for the given (u,v)
                u = ufunc(x,y)
                v = vfunc(x,y)
                coord = center.deproject(galsim.PositionD(u,v))
                ra, dec = radec_func(x,y)
                np.testing.assert_almost_equal(ra, coord.ra.rad(), 8,
                                               'rafunc produced wrong value')
                np.testing.assert_almost_equal(dec, coord.dec.rad(), 8,
                                               'decfunc produced wrong value')
                pos = center.project(coord)
                np.testing.assert_almost_equal(pos.x, u, digits, 'project x was inconsistent')
                np.testing.assert_almost_equal(pos.y, v, digits, 'project y was inconsistent')
                d1 = np.sqrt(u*u+v*v)
                d2 = center.distanceTo(coord)
                # The distances aren't expected to match.  Instead, for a Lambert projection,
                # d1 should match the straight line distance through the sphere.
                import math
                d2 = 2.*math.sin(d2.rad()/2) * galsim.radians / galsim.arcsec
                np.testing.assert_almost_equal(
                        d2, d1, digits, 'deprojected dist does not match expected value.')

                # Now test the two RaDec wcs classes
                for wcs in [ wcs2, wcs3 ]:
                    image_pos = galsim.PositionD(x,y)
                    world_pos1 = wcs1.toWorld(image_pos)
                    world_pos2 = wcs2.toWorld(image_pos)
                    origin = wcs2.toWorld(galsim.PositionD(0.,0.))
                    d3 = np.sqrt( world_pos1.x**2 + world_pos1.y**2 )
                    d4 = center.distanceTo(world_pos2)
                    d4 = 2.*math.sin(d4.rad()/2) * galsim.radians / galsim.arcsec
                    np.testing.assert_almost_equal(
                            d3, d1, digits, 'UV '+name+' dist does not match expected value.')
                    np.testing.assert_almost_equal(
                            d4, d1, digits, 'RaDec '+name+' dist does not match expected value.')

                    # Calculate the Jacobians for each wcs
                    jac1 = wcs1.jacobian(image_pos)
                    jac2 = wcs2.jacobian(image_pos)

                    # The pixel area should match pretty much exactly.  The Lambert projection
                    # is an area preserving projection.
                    np.testing.assert_almost_equal(
                            jac2.pixelArea(), jac1.pixelArea(), digits,
                            'RaDecFunction '+name+' pixelArea() does not match expected value.')
                    np.testing.assert_almost_equal(
                            wcs2.pixelArea(image_pos), jac1.pixelArea(), digits,
                            'RaDecFunction '+name+' pixelArea(pos) does not match expected value.')

                    # The distortion should be pretty small, so the min/max linear scale should
                    # match pretty well.
                    np.testing.assert_almost_equal(
                            jac2.minLinearScale(), jac1.minLinearScale(), digits,
                            'RaDecFunction '+name+' minScale() does not match expected value.')
                    np.testing.assert_almost_equal(
                            wcs2.minLinearScale(image_pos), jac1.minLinearScale(), digits,
                            'RaDecFunction '+name+' minScale(pos) does not match expected value.')
                    np.testing.assert_almost_equal(
                            jac2.maxLinearScale(), jac1.maxLinearScale(), digits,
                            'RaDecFunction '+name+' maxScale() does not match expected value.')
                    np.testing.assert_almost_equal(
                            wcs2.maxLinearScale(image_pos), jac1.maxLinearScale(), digits,
                            'RaDecFunction '+name+' maxScale(pos) does not match expected value.')

                    # The main discrepancy between the jacobians is a rotation term. 
                    # The pixels in the projected coordinates do not necessarily point north,
                    # since the direction to north changes over the field.  However, we can 
                    # calculate this expected discrepancy and correct for it to get a comparison 
                    # of the full jacobian that should be accurate to 5 digits.
                    # If A = coord, B = center, and C = the north pole, then the rotation angle is
                    # 180 deg - A - B.
                    A = coord.angleBetween(north_pole, center)
                    B = center.angleBetween(coord, north_pole)
                    C = north_pole.angleBetween(center, coord)
                    # The angle C should equal coord.ra - cneter.ra, so use this as a unit test of
                    # the angleBetween function:
                    np.testing.assert_almost_equal(
                            C / galsim.degrees, (coord.ra - center.ra) / galsim.degrees, digits,
                            'CelestialCoord calculated the wrong angle between center and coord')
                    angle = 180 * galsim.degrees - A - B

                    # Now we can use this angle to correct the jacobian from wcs2.
                    c = math.cos(angle.rad())
                    s = math.sin(angle.rad())
                    rot_dudx = c*jac2.dudx + s*jac2.dvdx
                    rot_dudy = c*jac2.dudy + s*jac2.dvdy
                    rot_dvdx = -s*jac2.dudx + c*jac2.dvdx
                    rot_dvdy = -s*jac2.dudy + c*jac2.dvdy

                    np.testing.assert_almost_equal(
                            rot_dudx, jac1.dudx, digits,
                            'RaDecFunction '+name+' dudx (rotated) does not match expected value.')
                    np.testing.assert_almost_equal(
                            rot_dudy, jac1.dudy, digits,
                            'RaDecFunction '+name+' dudy (rotated) does not match expected value.')
                    np.testing.assert_almost_equal(
                            rot_dvdx, jac1.dvdx, digits,
                            'RaDecFunction '+name+' dvdx (rotated) does not match expected value.')
                    np.testing.assert_almost_equal(
                            rot_dvdy, jac1.dvdy, digits,
                            'RaDecFunction '+name+' dvdy (rotated) does not match expected value.')

            if abs(center.dec/galsim.degrees) < 45:
                # The projections far to the north or the south don't pass all the tests in 
                # do_celestial because of the high non-linearities in the projection, so just
                # skip them.
                do_celestial_wcs(wcs2, 'RaDecFunc 1 centered at '+str(center.ra/galsim.degrees)+
                                 ', '+str(center.dec/galsim.degrees))
                do_celestial_wcs(wcs3, 'RaDecFunc 2 centered at '+str(center.ra/galsim.degrees)+
                                 ', '+str(center.dec/galsim.degrees))

    # Check that using a wcs in the context of an image works correctly
    # (Uses the last wcs2, wcs3 set in the above loops.)
    do_wcs_image(wcs2, 'RaDecFunction')
    if __name__ == "__main__":
        # As advertised, this is slow.  So only run it when doing python test_wcs.py.
        do_wcs_image(wcs3, 'RaDecFunction')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def do_ref(wcs, ref_list, name, approx=False, image=None):
    # Test that the given wcs object correctly converts the reference positions

    print 'Start reference testing for '+name
    for ref in ref_list:
        ra = galsim.HMS_Angle(ref[0])
        dec = galsim.DMS_Angle(ref[1])
        x = ref[2]
        y = ref[3]
        val = ref[4]
        #print 'Start ref: ',ra,dec,x,y,val

        # Check image -> world
        ref_coord = galsim.CelestialCoord(ra,dec)
        coord = wcs.toWorld(galsim.PositionD(x,y))
        #print 'ref_coord = ',ra.hms(), dec.dms()
        #print 'coord = ',coord.ra.hms(), coord.dec.dms()
        dist = ref_coord.distanceTo(coord) / galsim.arcsec
        #print 'delta(ra) = ',(ref_coord.ra - coord.ra)/galsim.arcsec
        #print 'delta(dec) = ',(ref_coord.dec - coord.dec)/galsim.arcsec
        #print 'dist = ',dist
        np.testing.assert_almost_equal(dist, 0, digits, 'wcs.toWorld differed from expected value')

        # Normally, we check the agreement to 1.e-3 arcsec.
        # However, we allow the caller to indicate the that inverse transform is
        # only approximate.  In this case, we only check to 1 digit.
        if approx:
            digits2 = 1
        else:
            digits2 = digits

        # Check world -> image
        pixel_scale = wcs.minLinearScale(galsim.PositionD(x,y))
        pos = wcs.toImage(galsim.CelestialCoord(ra,dec))
        #print 'x,y = ',x,y
        #print 'pos = ',pos
        np.testing.assert_almost_equal((x-pos.x)*pixel_scale, 0, digits2,
                                       'wcs.toImage differed from expected value')
        np.testing.assert_almost_equal((y-pos.y)*pixel_scale, 0, digits2,
                                       'wcs.toImage differed from expected value')
        if image:
            np.testing.assert_almost_equal(image(x,y), val, digits,
                                           'image(x,y) differed from reference value')

def test_astropywcs():
    """Test the AstropyWCS class
    """
    import time
    t1 = time.time()

    try:
        import astropy.wcs
    except ImportError:
        print 'Unable to import astropy.wcs.  Skipping AstropyWCS tests.'
        return

    # These all work, but it is quite slow, so only test one of them for the regular unit tests.
    # Test all of them when running python test_wcs.py.
    if __name__ == "__main__":
        test_tags = [ 'HPX', 'TAN', 'TSC', 'STG', 'ZEA', 'ARC', 'ZPN', 'SIP', 'REGION' ]
    else:
        test_tags = [ 'SIP' ]

    dir = 'fits_files'
    for tag in test_tags:
        file_name, ref_list = references[tag]
        #print tag,' file_name = ',file_name
        wcs = galsim.AstropyWCS(file_name, dir=dir)

        do_ref(wcs, ref_list, 'AstropyWCS '+tag)

        do_celestial_wcs(wcs, 'Astropy file '+file_name)

        do_wcs_image(wcs, 'AstropyWCS_'+tag)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_pyastwcs():
    """Test the PyAstWCS class
    """
    import time
    t1 = time.time()

    try:
        import starlink.Ast
    except ImportError:
        print 'Unable to import starlink.Ast.  Skipping PyAstWCS tests.'
        return

    # These all work, but it is quite slow, so only test one of them for the regular unit tests.
    # Test all of them when running python test_wcs.py.
    if __name__ == "__main__":
        test_tags = [ 'HPX', 'TAN', 'TSC', 'STG', 'ZEA', 'ARC', 'ZPN', 'SIP', 'TPV', 'ZPX',
                      'TAN-PV', 'REGION', 'TNX' ]
    else:
        test_tags = [ 'ZPX' ]

    dir = 'fits_files'
    for tag in test_tags:
        file_name, ref_list = references[tag]
        #print tag,' file_name = ',file_name
        wcs = galsim.PyAstWCS(file_name, dir=dir)

        # The PyAst implementation of the SIP type only gets the inverse transformation
        # approximately correct.  So we need to be a bit looser in that check.
        approx = tag in [ 'SIP' ]
        do_ref(wcs, ref_list, 'PyAstWCS '+tag, approx)

        do_celestial_wcs(wcs, 'PyAst file '+file_name)

        approx = tag in [ 'ZPX' ]
        do_wcs_image(wcs, 'PyAstWCS_'+tag, approx)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_wcstools():
    """Test the WcsToolsWCS class
    """
    import time
    t1 = time.time()

    # These all work, but it is quite slow, so only test one of them for the regular unit tests.
    # Test all of them when running python test_wcs.py.
    if __name__ == "__main__":
        # Note: TPV seems to work, but on one machine, repeated calls to xy2sky with the same
        # x,y values vary between two distinct ra,dec outputs.  I have no idea what's going on,
        # since I thought the calculation ought to be deterministic, but it clearly something 
        # isn't working right.  So just skip that test.
        test_tags = [ 'TAN', 'TSC', 'STG', 'ZEA', 'ARC', 'ZPN', 'SIP', 'ZPX', 'REGION', 'TNX' ]
    else:
        test_tags = [ 'TNX' ]

    dir = 'fits_files'
    try:
        galsim.WcsToolsWCS(references['TAN'][0], dir=dir)
    except OSError:
        print 'Unable to execute xy2sky.  Skipping WcsToolsWCS tests.'
        return

    for tag in test_tags:
        file_name, ref_list = references[tag]
        #print tag,' file_name = ',file_name
        wcs = galsim.WcsToolsWCS(file_name, dir=dir)

        # The wcstools implementation of the SIP and TPV types only gets the inverse 
        # transformations approximately correct.  So we need to be a bit looser in those checks.
        approx = tag in [ 'SIP', 'TPV' ]
        do_ref(wcs, ref_list, 'WcsToolsWCS '+tag, approx)

        # Recenter (x,y) = (0,0) at the image center to avoid wcstools warnings about going
        # off the image.
        im = galsim.fits.read(file_name, dir=dir)
        wcs = wcs.setOrigin(origin = -im.bounds.center())

        do_celestial_wcs(wcs, 'WcsToolsWCS '+file_name)

        do_wcs_image(wcs, 'WcsToolsWCS_'+tag)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_gsfitswcs():
    """Test the GSFitsWCS class
    """
    import time
    t1 = time.time()

    # These are all relatively fast (total time for all 7 and the TanWCS stuff below is about 
    # 1.6 seconds), but longer than my arbitrary 1 second goal for any unit test, so only do the 
    # two most important ones as part of the regular test suite runs.
    if __name__ == "__main__":
        test_tags = [ 'TAN', 'STG', 'ZEA', 'ARC', 'TPV', 'TAN-PV', 'TNX' ]
    else:
        test_tags = [ 'TAN', 'TPV' ]

    dir = 'fits_files'

    for tag in test_tags:
        file_name, ref_list = references[tag]
        #print tag,' file_name = ',file_name
        wcs = galsim.GSFitsWCS(file_name, dir=dir)

        do_ref(wcs, ref_list, 'GSFitsWCS '+tag)

        do_celestial_wcs(wcs, 'GSFitsWCS '+file_name)

        do_wcs_image(wcs, 'GSFitsWCS_'+tag)

    # Use TanWCS function to create TAN GSFitsWCS objects from scratch.
    # First a slight tweak on a simple scale factor
    dudx = 0.2342
    dudy = 0.0023
    dvdx = 0.0019
    dvdy = 0.2391
    x0 = 1
    y0 = 1
    origin = galsim.PositionD(x0,y0)
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin)
    center = galsim.CelestialCoord(0.*galsim.radians, 0.*galsim.radians)
    wcs = galsim.TanWCS(affine, center)
    do_celestial_wcs(wcs, 'TanWCS 1')

    # Next one with a flip and significant rotation and a large (u,v) offset
    dudx = 0.1432
    dudy = 0.2342
    dvdx = 0.2391
    dvdy = 0.1409
    u0 = 124.3
    v0 = -141.9
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, world_origin=galsim.PositionD(u0,v0))
    center = galsim.CelestialCoord(3.4 * galsim.hours, -17.9 * galsim.degrees)
    wcs = galsim.TanWCS(affine, center)
    do_celestial_wcs(wcs, 'TanWCS 2')

    # Finally a really crazy one that isn't remotely regular
    dudx = 0.2342
    dudy = -0.1432
    dvdx = 0.0924
    dvdy = -0.3013
    x0 = -3
    y0 = 104
    u0 = 1423.9
    v0 = 8242.7
    origin = galsim.PositionD(x0,y0)
    world_origin = galsim.PositionD(u0,v0)
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=origin, world_origin=world_origin)
    center = galsim.CelestialCoord(-241.4 * galsim.hours, 87.9 * galsim.degrees)
    wcs = galsim.TanWCS(affine, center)
    do_celestial_wcs(wcs, 'TanWCS 3')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_fitswcs():
    """Test the FitsWCS factory function
    """
    import time
    t1 = time.time()

    if __name__ == "__main__":
        # For more thorough unit tests (when running python test_wcs.py explicitly), this 
        # will test everything.  If you don't have everything installed (especially 
        test_tags = all_tags
    else:
        # These should always work, since GSFitsWCS will work on them.  So this 
        # mostly just tests the basic interface of the FitsWCS function.
        test_tags = [ 'TAN', 'TPV' ]

    dir = 'fits_files'

    for tag in test_tags:
        file_name, ref_list = references[tag]
        #print tag,' file_name = ',file_name
        wcs = galsim.FitsWCS(file_name, dir=dir)
        print 'FitsWCS is really ',type(wcs)

        if isinstance(wcs, galsim.AffineTransform):
            import warnings
            warnings.warn("None of the existing WCS classes were able to read "+file_name)
        else:
            approx1 = ( (tag == 'SIP' and isinstance(wcs, galsim.PyAstWCS)) or
                        (tag in ['SIP', 'TPV'] and isinstance(wcs, galsim.WcsToolsWCS)) )
            approx2 = tag == 'ZPX' and isinstance(wcs, galsim.PyAstWCS)
            do_ref(wcs, ref_list, 'FitsWCS '+tag, approx1)
            do_celestial_wcs(wcs, 'FitsWCS '+file_name)
            do_wcs_image(wcs, 'FitsWCS_'+tag, approx2)

            # Should also be able to build the file just from a fits.read() call, which 
            # uses FitsWCS behind the scenes.
            im = galsim.fits.read(file_name, dir=dir)
            do_ref(im.wcs, ref_list, 'WCS from fits.read '+tag, approx1, im)

        # Finally, also check that AffineTransform can read the file.
        # We don't really have any accuracy checks here.  This really just checks that the
        # read function doesn't raise an exception.
        hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir)
        affine = galsim.AffineTransform._readHeader(hdu.header)
        galsim.fits.closeHDUList(hdu_list, fin)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_pixelscale()
    test_shearwcs()
    test_affinetransform()
    test_uvfunction()
    test_radecfunction()
    test_astropywcs()
    test_pyastwcs()
    test_wcstools()
    test_gsfitswcs()
    test_fitswcs()
