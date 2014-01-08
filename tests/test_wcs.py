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

far_x_list = [ 10, 24.2, -134.2, -500 ]
far_y_list = [ 10, -17.3, 200.3, -700 ]

all_x_list = near_x_list + far_x_list
all_y_list = near_y_list + far_y_list

# Make a few different profiles to check.  Make sure to include ones that 
# aren't symmetrical so we don't get fooled by symmetries.
profiles = []
prof = galsim.Gaussian(sigma = 1.7, flux = 100)
profiles.append(prof)
prof = prof.createSheared(g1=0.3, g2=-0.12)
profiles.append(prof)
prof = prof + galsim.Exponential(scale_radius = 1.3, flux = 20).createShifted(-0.1,-0.4)
profiles.append(prof)

# How many digits of accuracy should we demand?  
# This mostly comes into play for functions with a non-trivial local approximation.
digits = 6

# The HPX, TAN, TSC, and ZPN files were downloaded from the web site:
# 
# http://www.atnf.csiro.au/people/mcalabre/WCS/example_data.html
# 
# From that page: "The example maps and spectra are offered mainly as test material for 
# software that deals with FITS WCS."
#
# I picked 4 that looked rather different in ds9, but there are a bunch more on that web 
# page as well.  In particular, I included ZPN, since that uses PV values, so it is a bit 
# different from the others.  Also, HPX is not implemented in wcstools, so that's also 
# worth including.
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
# The exception is HPX, for which I used the PyAst library to compute accurate values.

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
all_tags = [ 'HPX', 'TAN', 'TSC', 'ZPN', 'SIP', 'TPV', 'ZPX', 'TAN-PV', 'REGION', 'TNX' ]


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
    else:
        x_list = all_x_list
        y_list = all_y_list
    u_list = [ ufunc(x+x0,y+y0) for x,y in zip(x_list, y_list) ]
    v_list = [ vfunc(x+x0,y+y0) for x,y in zip(x_list, y_list) ]

    for x,y,u,v in zip(x_list, y_list, u_list, v_list):
        image_pos = galsim.PositionD(x+x0,y+y0)
        world_pos = galsim.PositionD(u,v)
        world_pos2 = wcs.toWorld(image_pos)
        np.testing.assert_almost_equal(
                world_pos.x, world_pos2.x, digits,
                'wcs.toWorld returned wrong world position for '+name)
        np.testing.assert_almost_equal(
                world_pos.y, world_pos2.y, digits,
                'wcs.toWorld returned wrong world position for '+name)
            
        try:
            # The reverse transformation is not guaranteed to be implemented,
            # so guard against NotImplementedError being raised:
            image_pos2 = wcs.toImage(world_pos)
            np.testing.assert_almost_equal(
                    image_pos.x, image_pos2.x, digits,
                    'wcs.toImage returned wrong image position for '+name)
            np.testing.assert_almost_equal(
                    image_pos.y, image_pos2.y, digits,
                    'wcs.toImage returned wrong image position for '+name)
        except NotImplementedError:
            pass

    if x0 == 0 and y0 == 0:
        # The last item in list should also work as a PositionI
        image_pos = galsim.PositionI(x,y)
        np.testing.assert_almost_equal(
                world_pos.x, wcs.toWorld(image_pos).x, digits,
                'wcs.toWorld gave different value with PositionI image_pos for '+name)
        np.testing.assert_almost_equal(
                world_pos.y, wcs.toWorld(image_pos).y, digits,
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
    #print 'im.wcs = ',im.wcs
    #print 'im2.wcs = ',im2.wcs
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
    wcs5 = wcs.setOrigin(new_origin, new_world_origin)
    world_pos3 = wcs5.toWorld(new_origin)
    np.testing.assert_almost_equal(
            world_pos3.x, new_world_origin.x, digits,
            'setOrigin(new_origin, new_world_origin) returned wrong position')
    np.testing.assert_almost_equal(
            world_pos3.y, new_world_origin.y, digits,
            'setOrigin(new_origin, new_world_origin) returned wrong position')

    # Check that (x,y) -> (u,v) and converse work correctly
    do_wcs_pos(wcs, ufunc, vfunc, name)

    # Test the transformation of a GSObject
    # These only work for local WCS projections!

    near_u_list = [ ufunc(x,y) for x,y in zip(near_x_list, near_y_list) ]
    near_v_list = [ vfunc(x,y) for x,y in zip(near_x_list, near_y_list) ]

    im1 = galsim.Image(64,64, wcs=wcs)
    im2 = galsim.Image(64,64, scale=1.)

    # This isn't normally necessary, but it is when we use UVFunction as a local WCS
    # The other classes just ignore the image_pos parameter.
    origin = galsim.PositionD(0,0)

    for world_profile in profiles:
        # The profiles build above are in world coordinates (as usual)
    
        # Convert to image coordinates
        image_profile = wcs.toImage(world_profile, image_pos=origin)

        # Also check round trip (starting with either one)
        world_profile2 = wcs.toWorld(image_profile, image_pos=origin)
        image_profile2 = wcs.toImage(world_profile2, image_pos=origin)

        for x,y,u,v in zip(near_x_list, near_y_list, near_u_list, near_v_list):
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


def do_nonlocal_wcs(wcs, ufunc, vfunc, name):

    print 'Start testing non-local WCS '+name
    #print 'wcs = ',wcs

    # Check that setOrigin and local work correctly:
    new_origin = galsim.PositionI(123,321)
    wcs3 = wcs.setOrigin(new_origin)
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
    if wcs.isUniform():
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
    global digits

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
    # for them.
    max_scale = wcs.maxLinearScale(wcs.origin)
    #print 'max_scale = ',max_scale
    orig_digits = digits
    if max_scale > 100:  # arcsec
        digits = 2

    for x0,y0 in zip(near_x_list, near_y_list):
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
                jac.dudx, (w2.ra - w1.ra)/galsim.arcsec * cosdec, digits,
                'jacobian dudx incorrect for '+name)
        np.testing.assert_array_almost_equal(
                jac.dudy, (w4.ra - w3.ra)/galsim.arcsec * cosdec, digits,
                'jacobian dudy incorrect for '+name)
        np.testing.assert_array_almost_equal(
                jac.dvdx, (w1.dec - w2.dec)/galsim.arcsec, digits,
                'jacobian dvdx incorrect for '+name)
        np.testing.assert_array_almost_equal(
                jac.dvdy, (w3.dec - w4.dec)/galsim.arcsec, digits,
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

    digits = orig_digits


def test_pixelscale():
    """Test the PixelScale class
    """
    import time
    t1 = time.time()

    global digits
    digits = 6
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

    global digits
    digits = 6
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

    global digits
    digits = 6
    # First a simple tweak on a simple scale factor
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
    dudy = 0.1432
    dudx = 0.2342
    dvdy = 0.2391
    dvdx = 0.1409

    wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    do_local_wcs(wcs, ufunc, vfunc, 'JacobianWCS 2')

    # Add a world origin offset
    u0 = 124.3
    v0 = -141.9
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, world_origin=galsim.PositionD(u0,v0))
    ufunc = lambda x,y: dudx*x + dudy*y + u0
    vfunc = lambda x,y: dvdx*x + dvdy*y + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'AffineTransform 2')

    # Finally a really crazy one that isn't remotely regular
    dudy = -0.1432
    dudx = 0.2342
    dvdy = -0.3013
    dvdx = 0.0924

    wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    do_local_wcs(wcs, ufunc, vfunc, 'Jacobian 3')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'JacobianWCS')

    # Add both kinds of offsets
    x0 = -3
    y0 = 104
    u0 = 1423.9
    v0 = 8242.7
    origin = galsim.PositionD(x0,y0)
    world_origin = galsim.PositionD(u0,v0)
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=origin,
                                 world_origin=world_origin)
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

    global digits
    digits = 6

    # First make some that are identical to simpler WCS classes:
    # 1. Like PixelScale
    scale = 0.17
    ufunc = lambda x,y: x * scale
    vfunc = lambda x,y: y * scale
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like PixelScale')
 
    # 2. Like ShearWCS
    scale = 0.23
    g1 = 0.14
    g2 = -0.37
    factor = 1./np.sqrt(1.-g1*g1-g2*g2)
    ufunc = lambda x,y: (x - g1*x - g2*y) * scale * factor
    vfunc = lambda x,y: (y + g1*y - g2*x) * scale * factor
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like ShearWCS')

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
    wcs = galsim.UVFunction(ufunc, vfunc, origin, world_origin)
    do_nonlocal_wcs(wcs, ufunc2, vfunc2, 'UVFunction with origin arguments')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_lambda')

    # Check basic copy and == , != for UVFunction
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'UVFunction copy is not == the original'
    wcs3a = galsim.UVFunction(radial_v, radial_v, origin, world_origin)
    wcs3b = galsim.UVFunction(radial_u, radial_u, origin, world_origin)
    wcs3c = galsim.UVFunction(radial_u, radial_v, origin*2, world_origin)
    wcs3d = galsim.UVFunction(radial_u, radial_v, origin, world_origin*2)
    assert wcs != wcs3a, 'UVFunction is not != a different one (ufunc)'
    assert wcs != wcs3b, 'UVFunction is not != a different one (vfunc)'
    assert wcs != wcs3c, 'UVFunction is not != a different one (origin)'
    assert wcs != wcs3d, 'UVFunction is not != a different one (world_origin)'

    # 5. Now some non-trivial 3rd order radial function.
    # This is only designed to be accurate to 5 digits, rather than usual 6.
    digits = 5
    origin = galsim.PositionD(x0,y0)
    wcs = galsim.UVFunction(radial_u, radial_v, origin)

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

    # Repeat with a function object rather than a regular function.
    # Use a different `a` parameter for u and v to make things more interesting.
    cubic_u = Cubic(2.9e-5, 2000., 'u')
    cubic_v = Cubic(-3.7e-5, 2000., 'v')
    wcs = galsim.UVFunction(cubic_u, cubic_v, origin=galsim.PositionD(x0,y0))
    ufunc = lambda x,y: cubic_u(x-x0, y-y0)
    vfunc = lambda x,y: cubic_v(x-x0, y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Cubic object UVFunction')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_object')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_radecfunction():
    """Test the RaDecFunction class
    """
    import time
    t1 = time.time()

    global digits
    digits = 6

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
            np.testing.assert_almost_equal(galsim.HMS_Angle(center.ra.hms()) / galsim.arcsec,
                                           center.ra / galsim.arcsec, 6, 'HMS parser error')
            np.testing.assert_almost_equal(galsim.DMS_Angle(center.dec.dms()) / galsim.arcsec,
                                           center.dec / galsim.arcsec, 6, 'DMS parser error')

            rafunc = lambda x,y: center.deproject(galsim.PositionD(ufunc(x,y), vfunc(x,y))).ra
            decfunc = lambda x,y: center.deproject(galsim.PositionD(ufunc(x,y), vfunc(x,y))).dec
            wcs2 = galsim.RaDecFunction(rafunc, decfunc)

            # Check that distance, jacobian for some x,y positions match the UV values.
            for x,y in zip(far_x_list, far_y_list):

                # First do some basic checks of project, deproject for the given (u,v)
                u = ufunc(x,y)
                v = vfunc(x,y)
                coord = center.deproject(galsim.PositionD(u,v))
                ra = rafunc(x,y)
                dec = decfunc(x,y)
                np.testing.assert_almost_equal(ra/galsim.degrees, coord.ra/galsim.degrees, 6,
                                               'rafunc produced wrong value')
                np.testing.assert_almost_equal(dec/galsim.degrees, coord.dec/galsim.degrees, 6,
                                               'decfunc produced wrong value')
                pos = center.project(coord)
                np.testing.assert_almost_equal(pos.x, u, 6, 'project x was inconsistent')
                np.testing.assert_almost_equal(pos.y, v, 6, 'project y was inconsistent')
                d1 = np.sqrt(u*u+v*v)
                d2 = center.distanceTo(coord)
                # The distances aren't expected to match.  Instead, for a Lambert projection,
                # d1 should match the straight line distance through the sphere.
                import math
                d2 = 2.*math.sin(d2.rad()/2) * galsim.radians / galsim.arcsec
                np.testing.assert_almost_equal(
                        d2, d1, 5, 'deprojected dist does not match expected value.')

                # Now test the wcs that uses deproject
                image_pos = galsim.PositionD(x,y)
                world_pos1 = wcs1.toWorld(image_pos)
                world_pos2 = wcs2.toWorld(image_pos)
                origin = wcs2.toWorld(galsim.PositionD(0.,0.))
                d3 = np.sqrt( world_pos1.x**2 + world_pos1.y**2 )
                d4 = center.distanceTo(world_pos2)
                d4 = 2.*math.sin(d4.rad()/2) * galsim.radians / galsim.arcsec
                np.testing.assert_almost_equal(
                        d3, d1, 5, 'UVFunction '+name+' dist does not match expected value.')
                np.testing.assert_almost_equal(
                        d4, d1, 5, 'RaDecFunction '+name+' dist does not match expected value.')

                # Calculate the Jacobians for each wcs
                jac1 = wcs1.jacobian(image_pos)
                jac2 = wcs2.jacobian(image_pos)

                # The pixel area should match pretty much exactly.  The Lambert projection
                # is an area preserving projection.
                np.testing.assert_almost_equal(
                        jac2.pixelArea(), jac1.pixelArea(), 7,
                        'RaDecFunction '+name+' pixelArea() does not match expected value.')
                np.testing.assert_almost_equal(
                        wcs2.pixelArea(image_pos), jac1.pixelArea(), 7,
                        'RaDecFunction '+name+' pixelArea(pos) does not match expected value.')

                # The distortion should be pretty small, so the min/max linear scale should
                # match pretty well.
                np.testing.assert_almost_equal(
                        jac2.minLinearScale(), jac1.minLinearScale(), 5,
                        'RaDecFunction '+name+' minLinearScale() does not match expected value.')
                np.testing.assert_almost_equal(
                        wcs2.minLinearScale(image_pos), jac1.minLinearScale(), 5,
                        'RaDecFunction '+name+' minLinearScale(pos) does not match expected value.')
                np.testing.assert_almost_equal(
                        jac2.maxLinearScale(), jac1.maxLinearScale(), 5,
                        'RaDecFunction '+name+' maxLinearScale() does not match expected value.')
                np.testing.assert_almost_equal(
                        wcs2.maxLinearScale(image_pos), jac1.maxLinearScale(), 5,
                        'RaDecFunction '+name+' maxLinearScale(pos) does not match expected value.')

                # The main discrepancy between the jacobians is a rotation term. 
                # The pixels in the projected coordinates do not necessarily point north, since
                # the direction to north changes over the field.  However, we can calculate this
                # expected discrepancy and correct for it to get a comparison of the full
                # jacobian that should be accurate to 5 digits.
                # If A = coord, B = center, and C = the north pole, then the rotation angle is
                # 180 deg - A - B.
                A = coord.angleBetween(north_pole, center)
                B = center.angleBetween(coord, north_pole)
                C = north_pole.angleBetween(center, coord)
                # The angle C should equal coord.ra - cneter.ra, so use this as a unit test of
                # the angleBetween function:
                np.testing.assert_almost_equal(
                        C / galsim.degrees, (coord.ra - center.ra) / galsim.degrees, 6,
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
                        rot_dudx, jac1.dudx, 4,
                        'RaDecFunction '+name+' dudx (rotated) does not match expected value.')
                np.testing.assert_almost_equal(
                        rot_dudy, jac1.dudy, 4,
                        'RaDecFunction '+name+' dudy (rotated) does not match expected value.')
                np.testing.assert_almost_equal(
                        rot_dvdx, jac1.dvdx, 4,
                        'RaDecFunction '+name+' dvdx (rotated) does not match expected value.')
                np.testing.assert_almost_equal(
                        rot_dvdy, jac1.dvdy, 4,
                        'RaDecFunction '+name+' dvdy (rotated) does not match expected value.')

            if abs(center.dec/galsim.degrees) < 45:
                # The projections far to the north or the south don't pass all the tests in 
                # do_celestial because of the high non-linearities in the projection, so just
                # skip them.
                do_celestial_wcs(wcs2, 'RaDecFunc centered at '+str(center.ra/galsim.degrees)+', '+
                                 str(center.dec/galsim.degrees))

    # Check that using a wcs in the context of an image works correctly
    # (Uses the last wcs2 set in the above loops.)
    do_wcs_image(wcs2, 'RaDecFunction')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def do_ref(wcs, ref_list, approx=False, image=None):
    """Test that the given wcs object correctly converts the reference positions
    """
    global digits
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
        dist = ref_coord.distanceTo(coord) / galsim.arcsec
        np.testing.assert_almost_equal(dist, 0, digits, 'wcs.toWorld differed from expected value')

        # Normally, we check the agreement to 1.e-4 arcsec.
        # However, we allow the caller to indicate the that inverse transform is
        # only approximate.  In this case, we only check to 1 digit.
        orig_digits = digits
        if approx:
            digits = 1

        # Check world -> image
        pixel_scale = wcs.minLinearScale(galsim.PositionD(x,y))
        pos = wcs.toImage(galsim.CelestialCoord(ra,dec))
        np.testing.assert_almost_equal((x-pos.x)*pixel_scale, 0, digits,
                                       'wcs.toImage differed from expected value')
        np.testing.assert_almost_equal((y-pos.y)*pixel_scale, 0, digits,
                                       'wcs.toImage differed from expected value')
        digits = orig_digits

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

    # These all work, but it is quite slow, so only test one of the for the regular unit tests.
    # Test all of them when running python test_wcs.py.
    if __name__ == "__main__":
        test_tags = [ 'HPX', 'TAN', 'TSC', 'ZPN', 'SIP', 'REGION' ]
    else:
        test_tags = [ 'SIP' ]

    dir = 'fits_files'
    global digits
    digits = 4
    for tag in test_tags:
        file_name, ref_list = references[tag]
        print tag,' file_name = ',file_name
        wcs = galsim.AstropyWCS(file_name, dir=dir)

        do_ref(wcs, ref_list)

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

    # These all work, but it is quite slow, so only test one of the for the regular unit tests.
    # Test all of them when running python test_wcs.py.
    if __name__ == "__main__":
        test_tags = [ 'HPX', 'TAN', 'TSC', 'ZPN', 'SIP', 'TPV', 'ZPX', 'TAN-PV', 'REGION', 'TNX' ]
    else:
        test_tags = [ 'ZPX' ]

    dir = 'fits_files'
    global digits
    digits = 4
    for tag in test_tags:
        file_name, ref_list = references[tag]
        print tag,' file_name = ',file_name
        wcs = galsim.PyAstWCS(file_name, dir=dir)

        # The PyAst implementation of the SIP type only gets the inverse transformation
        # approximately correct.  So we need to be a bit looser in that check.
        approx = tag in [ 'SIP' , 'ZPX' ]
        do_ref(wcs, ref_list, approx)

        do_celestial_wcs(wcs, 'PyAst file '+file_name)

        do_wcs_image(wcs, 'PyAstWCS_'+tag, approx)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_wcstools():
    """Test the WcsToolsWCS class
    """
    import time
    t1 = time.time()

    # These all work, but it is quite slow, so only test one of the for the regular unit tests.
    # Test all of them when running python test_wcs.py.
    if __name__ == "__main__":
        test_tags = [ 'TAN', 'TSC', 'ZPN', 'SIP', 'TPV', 'ZPX', 'REGION', 'TNX' ]
    else:
        test_tags = [ 'TNX' ]

    dir = 'fits_files'
    try:
        galsim.WcsToolsWCS(references['TAN'][0], dir=dir)
    except OSError:
        print 'Unable to execute xy2sky.  Skipping WcsToolsWCS tests.'
        return

    global digits
    digits = 4
    for tag in test_tags:
        file_name, ref_list = references[tag]
        print tag,' file_name = ',file_name
        wcs = galsim.WcsToolsWCS(file_name, dir=dir)

        # The wcstools implementation of the SIP and TPV types only gets the inverse 
        # transformations approximately correct.  So we need to be a bit looser in those checks.
        approx = tag in [ 'SIP' , 'TPV' ]
        do_ref(wcs, ref_list, approx)

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

    test_tags = [ 'TAN', 'TPV', 'TAN-PV' ]

    dir = 'fits_files'

    global digits
    digits = 4
    for tag in test_tags:
        file_name, ref_list = references[tag]
        print tag,' file_name = ',file_name
        wcs = galsim.GSFitsWCS(file_name, dir=dir)

        do_ref(wcs, ref_list)

        do_celestial_wcs(wcs, 'GSFitsWCS '+file_name)

        do_wcs_image(wcs, 'GSFitsWCS_'+tag)

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

    global digits
    digits = 4
    for tag in test_tags:
        file_name, ref_list = references[tag]
        print tag,' file_name = ',file_name
        wcs = galsim.FitsWCS(file_name, dir=dir)
        print 'FitsWCS is really ',type(wcs)

        if isinstance(wcs, galsim.AffineTransform):
            import warnings
            warnings.warn("None of the existing WCS classes were able to read "+file_name)
        else:
            approx = ( (tag in ['SIP', 'ZPX'] and isinstance(wcs, galsim.PyAstWCS)) or
                       (tag in ['SIP', 'TPV'] and isinstance(wcs, galsim.WcsToolsWCS)) )

            do_ref(wcs, ref_list, approx)
            do_celestial_wcs(wcs, 'FitsWCS '+file_name)
            do_wcs_image(wcs, 'FitsWCS_'+tag, approx)

            # Should also be able to build the file just from a fits.read() call, which 
            # uses FitsWCS behind the scenes.
            im = galsim.fits.read(file_name, dir=dir)
            do_ref(im.wcs, ref_list, approx, im)

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
