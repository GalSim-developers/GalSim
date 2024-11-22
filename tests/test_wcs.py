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
import sys
import warnings
import time
import coord
from unittest import mock

import galsim
from galsim_test_helpers import *


# These positions will be used a few times below, so define them here.
# One of the tests requires that the last pair are integers, so don't change that.
near_x_list = [ 0, 0.242, -1.342, -5 ]
near_y_list = [ 0, -0.173, 2.003, -7 ]

far_x_list = [ 10, -31.7, -183.6, -700 ]
far_y_list = [ 10, 12.5, 103.3, 500 ]

# Make a few different profiles to check.  Make sure to include ones that
# aren't symmetrical so we don't get fooled by symmetries.
prof1 = galsim.Gaussian(sigma = 1.7, flux = 100)
prof2 = prof1.shear(g1=0.3, g2=-0.12)
prof3 = prof2 + galsim.Exponential(scale_radius = 1.3, flux = 20).shift(-0.1,-0.4)
profiles = [ prof1, prof2, prof3 ]

if __name__ != "__main__":
    # Some of the classes we test here are not terribly fast.  WcsToolsWCS in particular.
    # So reduce the number of tests.  Keep the hardest ones, since the easier ones are mostly
    # useful as diagnostics when there are problems.  So they will get run when doing
    # python test_wcs.py.  But not during a pytest run.
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
# I picked the ones that GSFitsWCS can do plus a couple others that struck me as interstingly
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
            [ ('19:39:16.551671', '-63:42:47.346862', 114, 180, 13.59960),
              ('18:19:35.761589', '-63:46:08.860203', 144, 30, 11.49591) ] ),
    'TAN' : ('1904-66_TAN.fits' ,
            [ ('19:39:30.753119', '-63:42:59.217527', 117, 178, 13.43628),
              ('18:19:18.652839', '-63:49:03.833411', 153, 35, 11.44438) ] ),
    'TSC' : ('1904-66_TSC.fits' ,
            [ ('19:39:39.996553', '-63:41:14.585586', 113, 161, 12.48409),
              ('18:19:05.985494', '-63:49:05.781036', 141, 48, 11.65945) ] ),
    'STG' : ('1904-66_STG.fits' ,
            [ ('19:39:14.752140', '-63:44:20.882465', 112, 172, 13.1618),
              ('18:19:37.824461', '-63:46:24.483497', 147, 38, 11.6091) ] ),
    'ZEA' : ('1904-66_ZEA.fits' ,
            [ ('19:39:26.871566', '-63:43:26.059526', 110, 170, 13.253),
              ('18:19:34.480902', '-63:46:40.038427', 144, 39, 11.62) ] ),
    'ARC' : ('1904-66_ARC.fits' ,
            [ ('19:39:28.622018', '-63:41:53.658982', 111, 171, 13.7654),
              ('18:19:47.020701', '-63:46:22.381334', 145, 39, 11.2099) ] ),
    'ZPN' : ('1904-66_ZPN.fits' ,
            [ ('19:39:24.948254', '-63:46:43.636138', 95, 151, 12.84769),
              ('18:19:24.149409', '-63:49:37.453404', 122, 48, 11.01434) ] ),
    'SIP' : ('sipsample.fits' ,
            [ ('13:30:01.474154', '47:12:51.794474', 242, 75, 12.24437),
              ('13:29:43.747626', '47:09:13.879660', 12, 106, 5.30282) ] ),
    'TPV' : ('tpv.fits',
            [ ('03:30:09.340034', '-28:43:50.811107', 418, 78, 2859.53882),
              ('03:30:15.728999', '-28:45:01.488629', 148, 393, 2957.98584) ] ),
    # Strangely, zpx.fits is the same image as tpv.fits, but the WCS-computed RA, Dec
    # values are not anywhere close to TELRA, TELDEC in the header.  It's a bit
    # unfortunate, since my understanding is that ZPX can encode the same function as
    # TPV, so they could have produced the equivalent function.  But instead they just
    # inserted some totally off-the-wall different WCS transformation.
    'ZPX' : ('zpx.fits',
            [ ('21:24:12.094326', '37:10:34.575917', 418, 78, 2859.53882),
              ('21:24:05.350816', '37:11:44.596579', 148, 393, 2957.98584) ] ),
    # Older versions of the new TPV standard just used the TAN wcs name and expected
    # the code to notice the PV values and use them correctly.  This did not become a
    # FITS standard (or even a registered non-standard), but some old FITS files use
    # this, so we want to support it.  I just edited the tpv.fits to change the
    # CTYPE values from TPV to TAN.
    'TAN-PV' : ('tanpv.fits',
            [ ('03:30:09.340034', '-28:43:50.811107', 418, 78, 2859.53882),
              ('03:30:15.728999', '-28:45:01.488629', 148, 393, 2957.98584) ] ),
    # It is apparently valid FITS format to have Dec as the first axis and RA as the second.
    # This is in fact the output of PyAst when writing the file tpv.fits in FITS encoding.
    # It seems worth testing that all the WCS types get this input correct.
    'TAN-FLIP' : ('tanflip.fits',
            [ ('03:30:09.262392', '-28:43:48.697347', 418, 78, 2859.53882),
              ('03:30:15.718834', '-28:44:59.073468', 148, 393, 2957.98584) ] ),
    'REGION' : ('region.fits',
            [ ('14:02:11.202432', '54:30:07.702200', 80, 80, 2241),
              ('14:04:17.341523', '54:16:28.554326', 45, 54, 1227) ] ),
    # Strangely, ds9 seems to get this one wrong.  It differs by about 6 arcsec in dec.
    # But PyAst and wcstools agree on these values, so I'm taking them to be accurate.
    'TNX' : ('tnx.fits',
            [ ('17:46:53.214511', '-30:08:47.895372', 32, 91, 7140),
              ('17:46:58.100741', '-30:07:50.121787', 246, 326, 15022) ] ),

    # Zwicky Transient Facility uses TPV with higher order polynomial than what is in tpv.fits.
    'ZTF' : ('ztf_20180525484722_000600_zg_c05_o_q1_sciimg.fits',
            [ ('0:18:42.475844', '+25:45:00.858971', 1040, 1029, 59344.3),
              ('0:17:03.568150', '+25:32:40.484235', 2340, 1796, 2171.35) ] ),
}
all_tags = references.keys()


def do_wcs_pos(wcs, ufunc, vfunc, name, x0=0, y0=0, color=None):
    # I would call this do_wcs_pos_tests, but pytest takes any function with test
    # _anywhere_ in the name an tries to run it.  So make sure the name doesn't
    # have 'test' in it.  There are a bunch of other do* functions that work similarly.

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
        world_pos2 = wcs.toWorld(image_pos, color=color)
        world_pos3 = wcs.posToWorld(image_pos, color=color)
        np.testing.assert_almost_equal(
                world_pos.x, world_pos2.x, digits2,
                'wcs.toWorld returned wrong world position for '+name)
        np.testing.assert_almost_equal(
                world_pos.y, world_pos2.y, digits2,
                'wcs.toWorld returned wrong world position for '+name)
        np.testing.assert_almost_equal(
                world_pos.x, world_pos3.x, digits2,
                'wcs.posToWorld returned wrong world position for '+name)
        np.testing.assert_almost_equal(
                world_pos.y, world_pos3.y, digits2,
                'wcs.posToWorld returned wrong world position for '+name)

        u1,v1 = wcs.toWorld(x+x0, y+y0, color=color)
        u2,v2 = wcs.xyTouv(x+x0, y+y0, color=color)
        np.testing.assert_almost_equal(
                u1, u, digits2,
                'wcs.toWorld(x,y) returned wrong u position for '+name)
        np.testing.assert_almost_equal(
                v1, v, digits2,
                'wcs.toWorld(x,y) returned wrong v position for '+name)
        np.testing.assert_almost_equal(
                u2, u, digits2,
                'wcs.xyTouv(x,y) returned wrong u position for '+name)
        np.testing.assert_almost_equal(
                v2, v, digits2,
                'wcs.xyTouv(x,y) returned wrong v position for '+name)

        scale = wcs.maxLinearScale(image_pos, color=color)
        try:
            # The reverse transformation is not guaranteed to be implemented,
            # so guard against NotImplementedError being raised:
            image_pos2 = wcs.toImage(world_pos, color=color)
            image_pos3 = wcs.posToImage(world_pos, color=color)
            test_reverse = True
        except NotImplementedError:
            assert_raises(NotImplementedError, wcs._x, world_pos.x, world_pos.y, color=color)
            assert_raises(NotImplementedError, wcs._y, world_pos.x, world_pos.y, color=color)
            test_reverse = False
        else:
            np.testing.assert_almost_equal(
                    image_pos.x*scale, image_pos2.x*scale, digits2,
                    'wcs.toImage returned wrong image position for '+name)
            np.testing.assert_almost_equal(
                    image_pos.y*scale, image_pos2.y*scale, digits2,
                    'wcs.toImage returned wrong image position for '+name)
            np.testing.assert_almost_equal(
                    image_pos.x*scale, image_pos3.x*scale, digits2,
                    'wcs.posToImage returned wrong image position for '+name)
            np.testing.assert_almost_equal(
                    image_pos.y*scale, image_pos3.y*scale, digits2,
                    'wcs.posToImage returned wrong image position for '+name)

            x1,y1 = wcs.toImage(u, v, color=color)
            x2,y2 = wcs.uvToxy(u, v, color=color)
            np.testing.assert_almost_equal(
                    x1, x+x0, digits2,
                    'wcs.toImage(u,v) returned wrong x position for '+name)
            np.testing.assert_almost_equal(
                    y1, y+y0, digits2,
                    'wcs.toImage(u,v) returned wrong y position for '+name)
            np.testing.assert_almost_equal(
                    x2, x+x0, digits2,
                    'wcs.uvToxy(u,v) returned wrong x position for '+name)
            np.testing.assert_almost_equal(
                    y2, y+y0, digits2,
                    'wcs.uvToxy(u,v) returned wrong y position for '+name)

    # Test xyTouv with arrays
    u3,v3 = wcs.toWorld(np.array(x_list)+x0, np.array(y_list)+y0, color=color)
    u4,v4 = wcs.xyTouv(np.array(x_list)+x0, np.array(y_list)+y0, color=color)
    np.testing.assert_almost_equal(
            u3, u_list, digits2,
            'wcs.toWorld(x,y) with arrays returned wrong u positions for '+name)
    np.testing.assert_almost_equal(
            v3, v_list, digits2,
            'wcs.toWorld(x,y) with arrays returned wrong v positions for '+name)
    np.testing.assert_almost_equal(
            u4, u_list, digits2,
            'wcs.xyTouv(x,y) with arrays returned wrong u positions for '+name)
    np.testing.assert_almost_equal(
            v4, v_list, digits2,
            'wcs.xyTouv(x,y) with arrays returned wrong v positions for '+name)

    if test_reverse:
        # Test uvToxy with arrays
        x3,y3 = wcs.toImage(np.array(u_list), np.array(v_list), color=color)
        x4,y4 = wcs.uvToxy(np.array(u_list), np.array(v_list), color=color)
        np.testing.assert_almost_equal(
                x3-x0, x_list, digits2,
                'wcs.toImage(u,v) with arrays returned wrong x positions for '+name)
        np.testing.assert_almost_equal(
                y3-y0, y_list, digits2,
                'wcs.toImage(u,v) with arrays returned wrong y positions for '+name)
        np.testing.assert_almost_equal(
                x4-x0, x_list, digits2,
                'wcs.uvToxy(u,v) with arrays returned wrong x positions for '+name)
        np.testing.assert_almost_equal(
                y4-y0, y_list, digits2,
                'wcs.uvToxy(u,v) with arrays returned wrong y positions for '+name)

    if x0 == 0 and y0 == 0:
        # The last item in list should also work as a PositionI
        image_pos = galsim.PositionI(x,y)
        np.testing.assert_almost_equal(
                world_pos.x, wcs.toWorld(image_pos, color=color).x, digits2,
                'wcs.toWorld gave different value with PositionI image_pos for '+name)
        np.testing.assert_almost_equal(
                world_pos.y, wcs.toWorld(image_pos, color=color).y, digits2,
                'wcs.toWorld gave different value with PositionI image_pos for '+name)

    # Note that this function is only called for EuclideanWCS types.  There is a different
    # set of tests relevant for CelestialWCS done in do_celestial_wcs().
    assert not wcs.isCelestial()
    assert_raises(TypeError, wcs.toWorld)
    assert_raises(TypeError, wcs.toWorld, (3,4))
    assert_raises(TypeError, wcs.toWorld, 3,4, units=galsim.degrees)
    assert_raises(TypeError, wcs.toWorld, 3,4,5)
    assert_raises(TypeError, wcs.toWorld, galsim.CelestialCoord(0*galsim.degrees,0*galsim.degrees))

    assert_raises(TypeError, wcs.posToWorld)
    assert_raises(TypeError, wcs.posToWorld, (3,4))
    assert_raises(TypeError, wcs.posToWorld, 3,4)
    assert_raises(TypeError, wcs.posToWorld, 3,4,5)
    assert_raises(TypeError, wcs.posToWorld,
                  galsim.CelestialCoord(0*galsim.degrees,0*galsim.degrees))

    assert_raises(TypeError, wcs.xyTouv)
    assert_raises(TypeError, wcs.xyTouv, 3)
    assert_raises(TypeError, wcs.xyTouv, 3,4, units=galsim.degrees)
    assert_raises(TypeError, wcs.xyTouv, galsim.PositionD(3,4))

    assert_raises(TypeError, wcs.toImage)
    assert_raises(TypeError, wcs.toImage, (3,4))
    assert_raises(TypeError, wcs.toImage, 3,4, units=galsim.degrees)
    assert_raises(TypeError, wcs.toImage,
                  galsim.CelestialCoord(0*galsim.degrees,0*galsim.degrees))
    assert_raises(TypeError, wcs.toImage, 3,4,5)

    assert_raises(TypeError, wcs.posToImage)
    assert_raises(TypeError, wcs.posToImage, (3,4))
    assert_raises(TypeError, wcs.posToImage,
                  galsim.CelestialCoord(0*galsim.degrees,0*galsim.degrees))
    assert_raises(TypeError, wcs.posToImage, 3,4)
    assert_raises(TypeError, wcs.posToImage, 3,4,5)

    assert_raises(TypeError, wcs.uvToxy, 3,4, units=galsim.degrees)
    assert_raises(TypeError, wcs.uvToxy)
    assert_raises(TypeError, wcs.uvToxy, 3)
    assert_raises(TypeError, wcs.uvToxy, world_pos)


def check_world(pos1, pos2, digits, err_msg):
    if isinstance(pos1, galsim.CelestialCoord):
        np.testing.assert_almost_equal(pos1.distanceTo(pos2) / galsim.arcsec, 0, digits, err_msg)
    else:
        np.testing.assert_almost_equal(pos1.x, pos2.x, digits, err_msg)
        np.testing.assert_almost_equal(pos1.y, pos2.y, digits, err_msg)

def do_wcs_image(wcs, name, run_slow, approx=False):

    print('Start image tests for WCS '+name)

    # Use the "blank" image as our test image.  It's not blank in the sense of having all
    # zeros.  Rather, there are basically random values that we can use to test that
    # the shifted values are correct.  And it is a conveniently small-ish, non-square image.
    dir = 'fits_files'
    file_name = 'blankimg.fits'
    im = galsim.fits.read(file_name, dir=dir)
    np.testing.assert_equal(im.origin.x, 1, "initial origin is not 1,1 as expected")
    np.testing.assert_equal(im.origin.y, 1, "initial origin is not 1,1 as expected")
    im.wcs = wcs
    world1 = im.wcs.toWorld(im.origin)
    value1 = im(im.origin)
    world2 = im.wcs.toWorld(im.center)
    value2 = im(im.center)
    offset = galsim.PositionI(11,13)
    image_pos = im.origin + offset
    world3 = im.wcs.toWorld(image_pos)
    value3 = im(image_pos)

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
    np.testing.assert_equal(im2.origin.x, im.origin.x, "origin changed after write/read")
    np.testing.assert_equal(im2.origin.y, im.origin.y, "origin changed after write/read")
    check_world(im2.wcs.toWorld(im.origin), world1, digits2,
                "World position of origin is wrong after write/read.")
    np.testing.assert_almost_equal(im2(im.origin), value1, digits,
                                   "Image value at origin is wrong after write/read.")
    check_world(im2.wcs.toWorld(im.center), world2, digits2,
                "World position of center is wrong after write/read.")
    np.testing.assert_almost_equal(im2(im.center), value2, digits,
                                   "Image value at center is wrong after write/read.")
    check_world(im2.wcs.toWorld(image_pos), world3, digits2,
                "World position of image_pos is wrong after write/read.")
    np.testing.assert_almost_equal(im2(image_pos), value3, digits,
                                   "Image value at center is wrong after write/read.")

    if wcs.isUniform():
        # Test that the regular CD, CRPIX, CRVAL items that are written to the header
        # describe an equivalent WCS as this one.
        affine = galsim.FitsWCS(test_name, dir=dir)
        check_world(affine.toWorld(im.origin), world1, digits2,
                    "World position of origin is wrong after write/read.")
        check_world(affine.toWorld(im.center), world2, digits2,
                    "World position of center is wrong after write/read.")
        check_world(affine.toWorld(image_pos), world3, digits2,
                    "World position of image_pos is wrong after write/read.")

    # Test that im.shift does the right thing to the wcs
    # Also test parsing a position as x,y args.
    dx = 3
    dy = 9
    im.shift(dx,dy)
    image_pos = im.origin + offset
    np.testing.assert_equal(im.origin.x, 1+dx, "shift set origin to wrong value")
    np.testing.assert_equal(im.origin.y, 1+dy, "shift set origin to wrong value")
    check_world(im.wcs.toWorld(im.origin), world1, digits,
                "World position of origin after shift is wrong.")
    np.testing.assert_almost_equal(im(im.origin), value1, digits,
                                   "Image value at origin after shift is wrong.")
    check_world(im.wcs.toWorld(im.center), world2, digits,
                "World position of center after shift is wrong.")
    np.testing.assert_almost_equal(im(im.center), value2, digits,
                                   "Image value at center after shift is wrong.")
    check_world(im.wcs.toWorld(image_pos), world3, digits,
                "World position of image_pos after shift is wrong.")
    np.testing.assert_almost_equal(im(image_pos), value3, digits,
                                   "image value at center after shift is wrong.")

    # Test that im.setOrigin does the right thing to the wcs
    # Also test parsing a position as a tuple.
    new_origin = (-3432, 1907)
    im.setOrigin(new_origin)
    image_pos = im.origin + offset
    np.testing.assert_equal(im.origin.x, new_origin[0], "setOrigin set origin to wrong value")
    np.testing.assert_equal(im.origin.y, new_origin[1], "setOrigin set origin to wrong value")
    check_world(im.wcs.toWorld(im.origin), world1, digits,
                "World position of origin after setOrigin is wrong.")
    np.testing.assert_almost_equal(im(im.origin), value1, digits,
                                   "Image value at origin after setOrigin is wrong.")
    check_world(im.wcs.toWorld(im.center), world2, digits,
                "World position of center after setOrigin is wrong.")
    np.testing.assert_almost_equal(im(im.center), value2, digits,
                                   "Image value at center after setOrigin is wrong.")
    check_world(im.wcs.toWorld(image_pos), world3, digits,
                "World position of image_pos after setOrigin is wrong.")
    np.testing.assert_almost_equal(im(image_pos), value3, digits,
                                   "Image value at center after setOrigin is wrong.")

    # Test that im.setCenter does the right thing to the wcs.
    # Also test parsing a position as a PositionI object.
    new_center = galsim.PositionI(0,0)
    im.setCenter(new_center)
    image_pos = im.origin + offset
    np.testing.assert_equal(im.center.x, new_center.x, "setCenter set center to wrong value")
    np.testing.assert_equal(im.center.y, new_center.y, "setCenter set center to wrong value")
    check_world(im.wcs.toWorld(im.origin), world1, digits,
                "World position of origin after setCenter is wrong.")
    np.testing.assert_almost_equal(im(im.origin), value1, digits,
                                   "Image value at origin after setCenter is wrong.")
    check_world(im.wcs.toWorld(im.center), world2, digits,
                "World position of center after setCenter is wrong.")
    np.testing.assert_almost_equal(im(im.center), value2, digits,
                                   "Image value at center after setCenter is wrong.")
    check_world(im.wcs.toWorld(image_pos), world3, digits,
                "World position of image_pos after setCenter is wrong.")
    np.testing.assert_almost_equal(im(image_pos), value3, digits,
                                   "Image value at center after setCenter is wrong.")

    # Test makeSkyImage
    if not run_slow:
        # Use a smaller image to speed things up.
        im = im[galsim.BoundsI(im.xmin,im.xmin+5,im.ymin,im.ymin+5)]
    new_origin = (-134, 128)
    im.setOrigin(new_origin)
    sky_level = 177
    wcs.makeSkyImage(im, sky_level)
    for x,y in [ (im.bounds.xmin, im.bounds.ymin),
                 (im.bounds.xmax, im.bounds.ymin),
                 (im.bounds.xmin, im.bounds.ymax),
                 (im.bounds.xmax, im.bounds.ymax),
                 (im.center.x, im.center.y) ]:
        val = im(x,y)
        area = wcs.pixelArea(galsim.PositionD(x,y))
        np.testing.assert_almost_equal(val/(area*sky_level), 1., digits,
                                       "SkyImage at %d,%d is wrong"%(x,y))
    # Check that all values are near the same value.  In particular if ra crosses 0, then this
    # used to be a problem if the wcs returned ra values that jump from 360 to 0.
    # Not very stringent test, since we're just checking that we don't have some pixels
    # that are orders of magnitude different from the average.  So rtol=2 is good.
    # (1 is fine for the pytest runs, but one of the edge cases fails in main runs.  It's not
    # a problem I think, just one of the weird wcs types has a more variable pixel area.)
    np.testing.assert_allclose(im.array, area*sky_level, rtol=2)


def do_local_wcs(wcs, ufunc, vfunc, name):

    print('Start testing local WCS '+name)

    # Check that local and withOrigin work correctly:
    wcs2 = wcs.local()
    assert wcs == wcs2, name+' local() is not == the original'
    new_origin = galsim.PositionI(123,321)
    wcs3 = wcs.withOrigin(new_origin)
    assert wcs != wcs3, name+' is not != wcs.withOrigin(pos)'
    assert wcs3 != wcs, name+' is not != wcs.withOrigin(pos) (reverse)'
    wcs2 = wcs3.local()
    assert wcs == wcs2, name+' is not equal after wcs.withOrigin(pos).local()'
    world_pos1 = wcs.toWorld(galsim.PositionD(0,0))
    world_pos2 = wcs3.toWorld(new_origin)
    np.testing.assert_almost_equal(
            world_pos2.x, world_pos1.x, digits,
            'withOrigin(new_origin) returned wrong world position')
    np.testing.assert_almost_equal(
            world_pos2.y, world_pos1.y, digits,
            'withOrigin(new_origin) returned wrong world position')
    new_world_origin = galsim.PositionD(5352.7, 9234.3)
    wcs4 = wcs.withOrigin(new_origin, new_world_origin)
    world_pos3 = wcs4.toWorld(new_origin)
    np.testing.assert_almost_equal(
            world_pos3.x, new_world_origin.x, digits,
            'withOrigin(new_origin, new_world_origin) returned wrong position')
    np.testing.assert_almost_equal(
            world_pos3.y, new_world_origin.y, digits,
            'withOrigin(new_origin, new_world_origin) returned wrong position')
    wcs5 = wcs.shiftOrigin(new_origin, new_world_origin)
    assert wcs4 == wcs5  # For LocalWCS, shiftOrigin is equivalent to withOrigin

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

    # Check picklability
    check_pickle(wcs)

    # Test the transformation of a GSObject
    # These only work for local WCS projections!

    near_u_list = [ ufunc(x,y) for x,y in zip(near_x_list, near_y_list) ]
    near_v_list = [ vfunc(x,y) for x,y in zip(near_x_list, near_y_list) ]

    im1 = galsim.Image(64,64, wcs=wcs)
    im2 = galsim.Image(64,64, scale=1.)

    for world_profile in profiles:
        # The profiles built above are in world coordinates (as usual)

        # Convert to image coordinates
        image_profile = wcs.toImage(world_profile)

        assert wcs.profileToImage(world_profile) == image_profile

        # Also check round trip (starting with either one)
        world_profile2 = wcs.toWorld(image_profile)
        image_profile2 = wcs.toImage(world_profile2)

        assert wcs.profileToWorld(image_profile) == world_profile2

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
        world_profile.drawImage(im1, method='no_pixel')
        image_profile.drawImage(im2, method='no_pixel')
        np.testing.assert_array_almost_equal(
                im1.array, im2.array, digits,
                'world_profile and image_profile were different when drawn for '+name)

        # Test drawing the profile at a different center
        world_profile.drawImage(im1, method='no_pixel', center=(30.9,34.1))
        image_profile.drawImage(im2, method='no_pixel', center=(30.9,34.1))
        np.testing.assert_array_almost_equal(im1.array, im2.array, digits)

        # Check shear conversion
        world_shear = galsim.Shear(g1=0.2, g2=0.1)
        image_shear = wcs.toImage(world_shear)
        assert image_shear == wcs.shearToImage(world_shear)

        # Check round trip.
        world_shear2 = wcs.toWorld(image_shear)
        assert world_shear2 == wcs.shearToWorld(image_shear)
        assert np.isclose(world_shear2.g1, world_shear.g1)
        assert np.isclose(world_shear2.g2, world_shear.g2)

        # Check that a sheared object in world coordinates gets measured to have the nominal
        # image shear value.
        # This needs to start with a round profile so that the world shape is purely world_shear.
        sigma = wcs.pixelArea(image_pos)**0.5 * 5  # hsm works relatively well with this size.
        shear_profile = galsim.Gaussian(sigma=sigma).shear(world_shear)
        shear_profile.drawImage(im1, method='no_pixel')
        hsm = im1.FindAdaptiveMom()
        assert np.isclose(hsm.observed_shape.g1, image_shear.g1, atol=1.e-3)
        assert np.isclose(hsm.observed_shape.g2, image_shear.g2, atol=1.e-3)

    assert_raises(TypeError, wcs.withOrigin)
    assert_raises(TypeError, wcs.withOrigin, origin=(3,4), color=0.3)
    assert_raises(TypeError, wcs.withOrigin, origin=image_pos, world_origin=(3,4), color=0.3)

def do_jac_decomp(wcs, name):

    scale, shear, theta, flip = wcs.getDecomposition()

    # First see if we can recreate the right matrix from this:
    S = np.array( [ [ 1.+shear.g1, shear.g2 ],
                    [ shear.g2, 1.-shear.g1 ] ] ) / np.sqrt(1.-shear.g1**2-shear.g2**2)
    R = np.array( [ [ np.cos(theta), -np.sin(theta) ],
                    [ np.sin(theta), np.cos(theta) ] ] )
    if flip:
        F = np.array( [ [ 0, 1 ],
                        [ 1, 0 ] ] )
    else:
        F = np.array( [ [ 1, 0 ],
                        [ 0, 1 ] ] )

    M = scale * S.dot(R).dot(F)
    J = wcs.getMatrix()
    np.testing.assert_almost_equal(
            M, J, 8, "Decomposition was inconsistent with jacobian for "+name)

    # The minLinearScale is scale * (1-g) / sqrt(1-g^2)
    import math
    g = shear.g
    min_scale = scale * (1.-g) / math.sqrt(1.-g**2)
    np.testing.assert_almost_equal(wcs.minLinearScale(), min_scale, 6, "minLinearScale")
    # The maxLinearScale is scale * (1+g) / sqrt(1-g^2)
    max_scale = scale * (1.+g) / math.sqrt(1.-g**2)
    np.testing.assert_almost_equal(wcs.maxLinearScale(), max_scale, 6, "minLinearScale")

    # There are some relations between the decomposition and the inverse decomposition that should
    # be true:
    scale2, shear2, theta2, flip2 = wcs.inverse().getDecomposition()
    np.testing.assert_equal(flip, flip2, "inverse flip")
    np.testing.assert_almost_equal(scale, 1./scale2, 6, "inverse scale")
    if flip:
        np.testing.assert_almost_equal(theta.rad, theta2.rad, 6, "inverse theta")
    else:
        np.testing.assert_almost_equal(theta.rad, -theta2.rad, 6, "inverse theta")
    np.testing.assert_almost_equal(shear.g, shear2.g, 6, "inverse shear")
    # There is no simple relation between the directions of the shear in the two cases.
    # The shear direction gets mixed up by the rotation if that is non-zero.

    # Also check that the profile is transformed equivalently as advertised in the docstring
    # for getDecomposition.
    base_obj = galsim.Gaussian(sigma=2)
    # Make sure it doesn't have any initial symmetry!
    base_obj = base_obj.shear(g1=0.1, g2=0.23).shift(0.17, -0.37)

    obj1 = base_obj.transform(wcs.dudx, wcs.dudy, wcs.dvdx, wcs.dvdy)

    if flip:
        obj2 = base_obj.transform(0,1,1,0)
    else:
        obj2 = base_obj
    obj2 = obj2.rotate(theta).shear(shear).expand(scale)

    gsobject_compare(obj1, obj2)



def do_nonlocal_wcs(wcs, ufunc, vfunc, name, test_pickle=True, color=None):

    print('Start testing non-local WCS '+name)

    # Check that shiftOrigin and local work correctly:
    new_origin = galsim.PositionI(123,321)
    wcs3 = wcs.shiftOrigin(new_origin)
    assert wcs != wcs3, name+' is not != wcs.shiftOrigin(pos)'
    wcs4 = wcs.local(wcs.origin, color=color)
    assert wcs != wcs4, name+' is not != wcs.local()'
    assert wcs4 != wcs, name+' is not != wcs.local() (reverse)'
    world_origin = wcs.toWorld(wcs.origin, color=color)
    if wcs.isUniform():
        if wcs.world_origin == galsim.PositionD(0,0):
            wcs2 = wcs.local(wcs.origin, color=color).withOrigin(wcs.origin)
            assert wcs == wcs2, name+' is not equal after wcs.local().withOrigin(origin)'
        wcs2 = wcs.local(wcs.origin, color=color).withOrigin(wcs.origin, wcs.world_origin)
        assert wcs == wcs2, name+' not equal after wcs.local().withOrigin(origin,world_origin)'
    world_pos1 = wcs.toWorld(galsim.PositionD(0,0), color=color)
    wcs3 = wcs.shiftOrigin(new_origin)
    world_pos2 = wcs3.toWorld(new_origin, color=color)
    np.testing.assert_almost_equal(
            world_pos2.x, world_pos1.x, digits,
            'shiftOrigin(new_origin) returned wrong world position')
    np.testing.assert_almost_equal(
            world_pos2.y, world_pos1.y, digits,
            'shiftOrigin(new_origin) returned wrong world position')
    new_world_origin = galsim.PositionD(5352.7, 9234.3)
    wcs5 = wcs.shiftOrigin(new_origin, new_world_origin, color=color)
    world_pos3 = wcs5.toWorld(new_origin, color=color)
    np.testing.assert_almost_equal(
            world_pos3.x, new_world_origin.x, digits,
            'shiftOrigin(new_origin, new_world_origin) returned wrong position')
    np.testing.assert_almost_equal(
            world_pos3.y, new_world_origin.y, digits,
            'shiftOrigin(new_origin, new_world_origin) returned wrong position')

    # Check that (x,y) -> (u,v) and converse work correctly
    # These tests work regardless of whether the WCS is local or not.
    do_wcs_pos(wcs, ufunc, vfunc, name, color=color)

    # Check picklability
    if test_pickle: check_pickle(wcs)

    # The GSObject transformation tests are only valid for a local WCS.
    # But it should work for wcs.local()

    far_u_list = [ ufunc(x,y) for x,y in zip(far_x_list, far_y_list) ]
    far_v_list = [ vfunc(x,y) for x,y in zip(far_x_list, far_y_list) ]

    full_im1 = galsim.Image(galsim.BoundsI(-1023,1024,-1023,1024), wcs=wcs.fixColor(color))
    full_im2 = galsim.Image(galsim.BoundsI(-1023,1024,-1023,1024), scale=1.)

    for x0,y0,u0,v0 in zip(far_x_list, far_y_list, far_u_list, far_v_list):
        local_ufunc = lambda x,y: ufunc(x+x0,y+y0) - u0
        local_vfunc = lambda x,y: vfunc(x+x0,y+y0) - v0
        image_pos = galsim.PositionD(x0,y0)
        world_pos = galsim.PositionD(u0,v0)
        do_wcs_pos(wcs.local(image_pos, color=color), local_ufunc, local_vfunc,
                   name+'.local(image_pos)')
        do_wcs_pos(wcs.jacobian(image_pos, color=color), local_ufunc, local_vfunc,
                   name+'.jacobian(image_pos)')
        do_wcs_pos(wcs.affine(image_pos, color=color), ufunc, vfunc,
                   name+'.affine(image_pos)', x0, y0)

        try:
            # The local call is not guaranteed to be implemented for world_pos.
            # So guard against NotImplementedError.
            do_wcs_pos(wcs.local(world_pos=world_pos, color=color), local_ufunc, local_vfunc,
                       name + '.local(world_pos)')
            do_wcs_pos(wcs.jacobian(world_pos=world_pos, color=color), local_ufunc, local_vfunc,
                       name + '.jacobian(world_pos)')
            do_wcs_pos(wcs.affine(world_pos=world_pos, color=color), ufunc, vfunc,
                       name+'.affine(world_pos)', x0, y0)
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
            image_profile = wcs.toImage(world_profile, image_pos=image_pos, color=color)

            world_profile.drawImage(im1, offset=(dx,dy), method='no_pixel')
            image_profile.drawImage(im2, offset=(dx,dy), method='no_pixel')
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, digits,
                    'world_profile and image_profile differed when drawn for '+name)

            # Equivalent call using center rather than offset
            im1b = world_profile.drawImage(im1.copy(), center=(x0,y0), method='no_pixel')
            np.testing.assert_array_almost_equal(im1b.array, im1.array, digits)
            im2b = image_profile.drawImage(im2.copy(), center=(x0,y0), method='no_pixel')
            np.testing.assert_array_almost_equal(im2b.array, im2.array, digits)

            try:
                # The toImage call is not guaranteed to be implemented for world_pos.
                # So guard against NotImplementedError.
                image_profile = wcs.toImage(world_profile, world_pos=world_pos, color=color)

                world_profile.drawImage(im1, offset=(dx,dy), method='no_pixel')
                image_profile.drawImage(im2, offset=(dx,dy), method='no_pixel')
                np.testing.assert_array_almost_equal(
                        im1.array, im2.array, digits,
                        'world_profile and image_profile differed when drawn for '+name)
            except NotImplementedError:
                pass

            # Since these postage stamps are odd, should get the same answer if we draw
            # using the true center or not.
            world_profile.drawImage(im1, method='no_pixel', use_true_center=False)
            image_profile.drawImage(im2, method='no_pixel', use_true_center=True)
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, digits,
                    'world_profile at center and image_profile differed when drawn for '+name)

            # Can also pass in wcs as a parameter to drawImage.
            world_profile.drawImage(im1, method='no_pixel', wcs=wcs.fixColor(color))
            image_profile.drawImage(im2, method='no_pixel')
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, digits,
                    'world_profile with given wcs and image_profile differed when drawn for '+name)

            # Check shear conversion
            world_shear = galsim.Shear(g1=0.2, g2=0.1)
            image_shear = wcs.toImage(world_shear, image_pos=image_pos, color=color)
            assert image_shear == wcs.shearToImage(world_shear, image_pos=image_pos, color=color)

            # Check round trip.
            world_shear2 = wcs.toWorld(image_shear, image_pos=image_pos, color=color)
            assert world_shear2 == wcs.shearToWorld(image_shear, image_pos=image_pos, color=color)
            assert np.isclose(world_shear2.g1, world_shear.g1)
            assert np.isclose(world_shear2.g2, world_shear.g2)

            # Check that shearing in the respective domains does the same effective action
            # on an initially round profile.
            if image_shear.g > 0.8: continue  # If the image shear is extreme, hsm will fail.
            sigma = wcs.pixelArea(image_pos, color=color)**0.5 * 5
            shear_profile = galsim.Gaussian(sigma=sigma).shear(world_shear)
            shear_profile.drawImage(im1, method='no_pixel')
            hsm = im1.FindAdaptiveMom()
            #print('image_shear = ',image_shear, hsm.observed_shape)
            # Most of these pass at 1.e-3, but some of the more contrived ones need the higher atol.
            assert np.isclose(hsm.observed_shape.g1, image_shear.g1, atol=5.e-3)
            assert np.isclose(hsm.observed_shape.g2, image_shear.g2, atol=5.e-3)

    # Check some properties that should be the same for the wcs and its local jacobian.
    np.testing.assert_allclose(
            wcs.minLinearScale(image_pos=image_pos, color=color),
            wcs.jacobian(image_pos=image_pos, color=color).minLinearScale(color=color))
    np.testing.assert_allclose(
            wcs.maxLinearScale(image_pos=image_pos, color=color),
            wcs.jacobian(image_pos=image_pos, color=color).maxLinearScale(color=color))
    np.testing.assert_allclose(
            wcs.pixelArea(image_pos=image_pos, color=color),
            wcs.jacobian(image_pos=image_pos, color=color).pixelArea(color=color))

    if not wcs.isUniform():
        assert_raises(TypeError, wcs.local)
    assert_raises(TypeError, wcs.local, image_pos=image_pos, world_pos=world_pos, color=color)
    assert_raises(TypeError, wcs.local, image_pos=(3,4), color=color)
    assert_raises(TypeError, wcs.local, world_pos=(3,4), color=color)

    assert_raises(TypeError, wcs.shiftOrigin)
    assert_raises(TypeError, wcs.shiftOrigin, origin=(3,4), color=color)
    assert_raises(TypeError, wcs.shiftOrigin, origin=image_pos, world_origin=(3,4), color=color)


def do_celestial_wcs(wcs, name, test_pickle=True, approx=False):
    # It's a bit harder to test WCS functions that return a CelestialCoord, since
    # (usually) we don't have an exact formula to compare with.  So the tests here
    # are a bit sparer.

    print('Start testing celestial WCS '+name)

    # Check that shiftOrigin and local work correctly:
    new_origin = galsim.PositionI(123,321)
    wcs3 = wcs.shiftOrigin(new_origin)
    assert wcs != wcs3, name+' is not != wcs.shiftOrigin(pos)'
    wcs4 = wcs.local(wcs.origin)
    assert wcs != wcs4, name+' is not != wcs.local()'
    assert wcs4 != wcs, name+' is not != wcs.local() (reverse)'
    world_pos1 = wcs.toWorld(galsim.PositionD(0,0))
    wcs3 = wcs.shiftOrigin(new_origin)
    world_pos2 = wcs3.toWorld(new_origin)
    np.testing.assert_almost_equal(
            world_pos2.distanceTo(world_pos1) / galsim.arcsec, 0, digits,
            'shiftOrigin(new_origin) returned wrong world position')

    world_origin = wcs.toWorld(wcs.origin)

    full_im1 = galsim.Image(galsim.BoundsI(-1023,1024,-1023,1024), wcs=wcs)
    full_im2 = galsim.Image(galsim.BoundsI(-1023,1024,-1023,1024), scale=1.)

    # Some of the FITS images have really huge pixel scales.  Lower the accuracy requirement
    # for them.  2 digits in arcsec corresponds to 4 digits in pixels.
    max_scale = wcs.maxLinearScale(wcs.origin)
    if max_scale > 100:  # arcsec
        digits2 = 2
    else:
        digits2 = digits
    if approx:
        atol = 0.01
    else:
        atol = 1.e-6

    # Check picklability
    if test_pickle: check_pickle(wcs)

    near_ra_list = []
    near_dec_list = []
    for x0,y0 in zip(near_x_list, near_y_list):
        image_pos = galsim.PositionD(x0,y0)
        world_pos = wcs.toWorld(image_pos)
        near_ra_list.append(world_pos.ra.rad)
        near_dec_list.append(world_pos.dec.rad)

        # posToWorld is equivalent
        world_pos2 = wcs.posToWorld(image_pos)
        assert world_pos2 == world_pos

        # xyToradec also works
        ra1, dec1 = wcs.toWorld(x0, y0, units=galsim.radians)
        ra2, dec2 = wcs.xyToradec(x0, y0, galsim.degrees)
        ra3, dec3 = wcs.toWorld(x0, y0, units='arcmin')
        assert np.isclose(ra1, world_pos.ra.rad)
        assert np.isclose(dec1, world_pos.dec.rad)
        assert np.isclose(ra2, world_pos.ra / galsim.degrees)
        assert np.isclose(dec2, world_pos.dec / galsim.degrees)
        assert np.isclose(ra3, world_pos.ra / galsim.arcmin)
        assert np.isclose(dec3, world_pos.dec / galsim.arcmin)

        # toWorld on scalars should not return arrays
        assert_raises(TypeError, len, ra1)
        assert_raises(TypeError, len, dec1)
        assert_raises(TypeError, len, ra2)
        assert_raises(TypeError, len, dec2)
        assert_raises(TypeError, len, ra3)
        assert_raises(TypeError, len, dec3)

        try:
            # The toImage call is not guaranteed to be implemented for world_pos.
            # So guard against NotImplementedError.
            image_pos1 = wcs.toImage(world_pos)
            image_pos2 = wcs.posToImage(world_pos)
            test_reverse = True
        except NotImplementedError:
            test_reverse = False
        else:
            assert np.isclose(image_pos1.x, x0, rtol=1.e-3, atol=atol)
            assert np.isclose(image_pos1.y, y0, rtol=1.e-3, atol=atol)
            assert np.isclose(image_pos2.x, x0, rtol=1.e-3, atol=atol)
            assert np.isclose(image_pos2.y, y0, rtol=1.e-3, atol=atol)

            x1, y1 = wcs.toImage(ra1, dec1, units=galsim.radians)
            x2, y2 = wcs.radecToxy(ra2, dec2, units='deg')
            x3, y3 = wcs.radecToxy(ra3, dec3, units=galsim.arcmin)
            assert_raises(TypeError, len, x1)
            assert_raises(TypeError, len, y1)
            assert_raises(TypeError, len, x2)
            assert_raises(TypeError, len, y2)
            assert_raises(TypeError, len, x3)
            assert_raises(TypeError, len, y3)
            assert np.isclose(x1, x0, rtol=1.e-3, atol=atol)
            assert np.isclose(y1, y0, rtol=1.e-3, atol=atol)
            assert np.isclose(x2, x0, rtol=1.e-3, atol=atol)
            assert np.isclose(y2, y0, rtol=1.e-3, atol=atol)
            assert np.isclose(x3, x0, rtol=1.e-3, atol=atol)
            assert np.isclose(y3, y0, rtol=1.e-3, atol=atol)

        # Check the calculation of the jacobian
        w1 = wcs.toWorld(galsim.PositionD(x0+0.5,y0))
        w2 = wcs.toWorld(galsim.PositionD(x0-0.5,y0))
        w3 = wcs.toWorld(galsim.PositionD(x0,y0+0.5))
        w4 = wcs.toWorld(galsim.PositionD(x0,y0-0.5))
        cosdec = np.cos(world_pos.dec)
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

        # toWorld with projection should be (roughly) equivalent to the local around the
        # projection point.
        origin = galsim.PositionD(0,0)
        uv_pos1 = wcs.toWorld(image_pos, project_center=wcs.toWorld(origin))
        uv_pos2 = wcs.local(origin).toWorld(image_pos)
        uv_pos3 = wcs.posToWorld(image_pos, project_center=wcs.toWorld(origin))
        u3, v3 = wcs.toWorld(origin).project(world_pos, 'gnomonic')
        np.testing.assert_allclose(uv_pos1.x, uv_pos2.x, rtol=1.e-1, atol=1.e-8)
        np.testing.assert_allclose(uv_pos1.y, uv_pos2.y, rtol=1.e-1, atol=1.e-8)
        np.testing.assert_allclose(uv_pos1.x, u3 / galsim.arcsec, rtol=1.e-6, atol=1.e-8)
        np.testing.assert_allclose(uv_pos1.y, v3 / galsim.arcsec, rtol=1.e-6, atol=1.e-8)
        np.testing.assert_allclose(uv_pos3.x, u3 / galsim.arcsec, rtol=1.e-6, atol=1.e-8)
        np.testing.assert_allclose(uv_pos3.y, v3 / galsim.arcsec, rtol=1.e-6, atol=1.e-8)

        origin = galsim.PositionD(x0+0.5, y0-0.5)
        uv_pos1 = wcs.toWorld(image_pos, project_center=wcs.toWorld(origin))
        uv_pos2 = wcs.local(origin).toWorld(image_pos - origin)
        uv_pos3 = wcs.posToWorld(image_pos, project_center=wcs.toWorld(origin))
        u3, v3 = wcs.toWorld(origin).project(world_pos, 'gnomonic')
        np.testing.assert_allclose(uv_pos1.x, uv_pos2.x, rtol=1.e-2, atol=1.e-8)
        np.testing.assert_allclose(uv_pos1.y, uv_pos2.y, rtol=1.e-2, atol=1.e-8)
        np.testing.assert_allclose(uv_pos1.x, u3 / galsim.arcsec, rtol=1.e-6, atol=1.e-8)
        np.testing.assert_allclose(uv_pos1.y, v3 / galsim.arcsec, rtol=1.e-6, atol=1.e-8)
        np.testing.assert_allclose(uv_pos3.x, u3 / galsim.arcsec, rtol=1.e-6, atol=1.e-8)
        np.testing.assert_allclose(uv_pos3.y, v3 / galsim.arcsec, rtol=1.e-6, atol=1.e-8)

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

            world_profile.drawImage(im1, offset=(dx,dy), method='no_pixel')
            image_profile.drawImage(im2, offset=(dx,dy), method='no_pixel')
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, digits,
                    'world_profile and image_profile differed when drawn for '+name)

            # Equivalent call using center rather than offset
            im1b = world_profile.drawImage(im1.copy(), center=(x0,y0), method='no_pixel')
            np.testing.assert_array_almost_equal(im1b.array, im1.array, digits)
            im2b = image_profile.drawImage(im2.copy(), center=(x0,y0), method='no_pixel')
            np.testing.assert_array_almost_equal(im2b.array, im2.array, digits)

            if test_reverse:
                image_profile = wcs.toImage(world_profile, world_pos=world_pos)

                world_profile.drawImage(im1, offset=(dx,dy), method='no_pixel')
                image_profile.drawImage(im2, offset=(dx,dy), method='no_pixel')
                np.testing.assert_array_almost_equal(
                        im1.array, im2.array, digits,
                        'world_profile and image_profile differed when drawn for '+name)

            # Check shear conversion
            world_shear = galsim.Shear(g1=0.2, g2=0.1)
            image_shear = wcs.toImage(world_shear, image_pos=image_pos)
            assert image_shear == wcs.shearToImage(world_shear, image_pos=image_pos)

            # Check round trip.
            world_shear2 = wcs.toWorld(image_shear, image_pos=image_pos)
            assert world_shear2 == wcs.shearToWorld(image_shear, image_pos=image_pos)
            assert np.isclose(world_shear2.g1, world_shear.g1)
            assert np.isclose(world_shear2.g2, world_shear.g2)

            # Check that shearing in the respective domains does the same effective action
            # on an initially round profile.
            sigma = wcs.pixelArea(image_pos)**0.5 * 5
            shear_profile = galsim.Gaussian(sigma=sigma).shear(world_shear)
            shear_profile.drawImage(im1, method='no_pixel')
            hsm = im1.FindAdaptiveMom()
            #print('image_shear = ',image_shear, hsm.observed_shape)
            assert np.isclose(hsm.observed_shape.g1, image_shear.g1, atol=1.e-3)
            assert np.isclose(hsm.observed_shape.g2, image_shear.g2, atol=1.e-3)

    # Test xyToradec with array
    xar = np.array(near_x_list)
    yar = np.array(near_y_list)
    raar = np.array(near_ra_list)
    decar = np.array(near_dec_list)
    ra1, dec1 = wcs.toWorld(xar, yar, units=galsim.radians)
    ra2, dec2 = wcs.xyToradec(xar, yar, galsim.degrees)
    ra3, dec3 = wcs.toWorld(xar, yar, units='arcmin')
    np.testing.assert_allclose(ra1, raar)
    np.testing.assert_allclose(dec1, decar)
    np.testing.assert_allclose(ra2, raar * (galsim.radians / galsim.degrees))
    np.testing.assert_allclose(dec2, decar * (galsim.radians / galsim.degrees))
    np.testing.assert_allclose(ra3, raar * (galsim.radians / galsim.arcmin))
    np.testing.assert_allclose(dec3, decar * (galsim.radians / galsim.arcmin))

    if test_reverse:
        x1, y1 = wcs.toImage(raar, decar, units=galsim.radians)
        x2, y2 = wcs.radecToxy(raar * (galsim.radians / galsim.degrees),
                                  decar * (galsim.radians / galsim.degrees), galsim.degrees)
        x3, y3 = wcs.toImage(raar * (galsim.radians / galsim.arcmin),
                             decar * (galsim.radians / galsim.arcmin), units='arcmin')
        np.testing.assert_allclose(x1, xar, rtol=1.e-3, atol=atol)
        np.testing.assert_allclose(y1, yar, rtol=1.e-3, atol=atol)
        np.testing.assert_allclose(x2, xar, rtol=1.e-3, atol=atol)
        np.testing.assert_allclose(y2, yar, rtol=1.e-3, atol=atol)
        np.testing.assert_allclose(x3, xar, rtol=1.e-3, atol=atol)
        np.testing.assert_allclose(y3, yar, rtol=1.e-3, atol=atol)

    assert_raises(TypeError, wcs.toWorld)
    assert_raises(TypeError, wcs.toWorld, (3,4))
    assert_raises(TypeError, wcs.toWorld, 3,4) # no units
    assert_raises(TypeError, wcs.toWorld, 3,4,5)
    assert_raises(TypeError, wcs.toWorld, galsim.CelestialCoord(0*galsim.degrees,0*galsim.degrees))

    assert_raises(TypeError, wcs.posToWorld)
    assert_raises(TypeError, wcs.posToWorld, (3,4))
    assert_raises(TypeError, wcs.posToWorld, 3,4)
    assert_raises(TypeError, wcs.posToWorld, 3,4,5)
    assert_raises(TypeError, wcs.posToWorld,
                  galsim.CelestialCoord(0*galsim.degrees,0*galsim.degrees))

    assert_raises(TypeError, wcs.xyToradec)
    assert_raises(TypeError, wcs.xyToradec, 3)
    assert_raises(TypeError, wcs.xyToradec, 3,4) # no units
    assert_raises(TypeError, wcs.xyToradec, galsim.PositionD(3,4))
    assert_raises(ValueError, wcs.xyToradec, 3,4,5)  # 5 interpreted as a unit

    assert_raises(TypeError, wcs.toImage)
    assert_raises(TypeError, wcs.toImage, (3,4))
    assert_raises(TypeError, wcs.toImage, 3,4)  # no units
    assert_raises(TypeError, wcs.toImage, galsim.PositionD(3,4))
    assert_raises(TypeError, wcs.toImage, 3,4,5)

    assert_raises(TypeError, wcs.posToImage)
    assert_raises(TypeError, wcs.posToImage, (3,4))
    assert_raises(TypeError, wcs.posToImage, galsim.PositionD(3,4))
    assert_raises(TypeError, wcs.posToImage, 3,4)
    assert_raises(TypeError, wcs.posToImage, 3,4,5)

    assert_raises(TypeError, wcs.radecToxy)
    assert_raises(TypeError, wcs.radecToxy, 3,4)
    assert_raises(TypeError, wcs.radecToxy, units=galsim.degrees)
    assert_raises(TypeError, wcs.radecToxy, 3, units=galsim.degrees)
    assert_raises(TypeError, wcs.radecToxy, 3,4,5, units=galsim.degrees)
    assert_raises(TypeError, wcs.radecToxy, world_pos, units=galsim.degrees)
    assert_raises(ValueError, wcs.radecToxy, 3,4,5)

    assert_raises(TypeError, wcs.local)
    assert_raises(TypeError, wcs.local, image_pos=image_pos, world_pos=world_pos)
    assert_raises(TypeError, wcs.local, image_pos=(3,4))
    assert_raises(TypeError, wcs.local, world_pos=(3,4))

    assert_raises(TypeError, wcs.shiftOrigin)
    assert_raises(TypeError, wcs.shiftOrigin, origin=(3,4))
    assert_raises(TypeError, wcs.shiftOrigin, world_origin=(3,4))
    assert_raises(TypeError, wcs.shiftOrigin, origin=image_pos, world_origin=world_pos)


@timer
def test_pixelscale(run_slow):
    """Test the PixelScale class
    """
    scale = 0.23
    wcs = galsim.PixelScale(scale)
    assert wcs.isPixelScale()
    assert wcs.isLocal()
    assert wcs.isUniform()
    assert not wcs.isCelestial()

    # Check basic copy and == , !=:
    wcs2 = wcs.copy()
    assert wcs == wcs2, 'PixelScale copy is not == the original'
    wcs3 = galsim.PixelScale(scale + 0.1234)
    assert wcs != wcs3, 'PixelScale is not != a different one'
    assert wcs.scale == scale
    assert wcs.origin == galsim.PositionD(0,0)
    assert wcs.world_origin == galsim.PositionD(0,0)

    assert_raises(TypeError, galsim.PixelScale)
    assert_raises(TypeError, galsim.PixelScale, scale=galsim.PixelScale(scale))
    assert_raises(TypeError, galsim.PixelScale, scale=scale, origin=galsim.PositionD(0,0))
    assert_raises(TypeError, galsim.PixelScale, scale=scale, world_origin=galsim.PositionD(0,0))

    ufunc = lambda x,y: x*scale
    vfunc = lambda x,y: y*scale

    # Do generic tests that apply to all WCS types
    do_local_wcs(wcs, ufunc, vfunc, 'PixelScale')

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'PixelScale', run_slow)

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
    origin = galsim.PositionI(x0,y0)
    wcs = galsim.OffsetWCS(scale, origin)
    wcs2 = galsim.PixelScale(scale).withOrigin(origin)
    assert wcs == wcs2, 'OffsetWCS is not == PixelScale.withOrigin(origin)'
    assert wcs.origin == origin
    assert wcs.scale == scale
    assert wcs.isPixelScale()
    assert not wcs.isLocal()
    assert wcs.isUniform()
    assert not wcs.isCelestial()

    # Default origin is (0,0)
    wcs3 = galsim.OffsetWCS(scale)
    assert wcs3.origin == galsim.PositionD(0,0)
    assert wcs3.world_origin == galsim.PositionD(0,0)

    assert_raises(TypeError, galsim.OffsetWCS)
    assert_raises(TypeError, galsim.OffsetWCS, scale=galsim.PixelScale(scale))
    assert_raises(TypeError, galsim.OffsetWCS, scale=scale, origin=5)
    assert_raises(TypeError, galsim.OffsetWCS, scale=scale,
                  origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))
    assert_raises(TypeError, galsim.OffsetWCS, scale=scale,
                  world_origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))

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
    do_wcs_image(wcs, 'OffsetWCS', run_slow)


@timer
def test_shearwcs(run_slow):
    """Test the ShearWCS class
    """
    scale = 0.23
    g1 = 0.14
    g2 = -0.37
    shear = galsim.Shear(g1=g1,g2=g2)
    wcs = galsim.ShearWCS(scale, shear)
    assert wcs.shear == shear
    assert wcs.origin == galsim.PositionD(0,0)
    assert wcs.world_origin == galsim.PositionD(0,0)
    assert not wcs.isPixelScale()
    assert wcs.isLocal()
    assert wcs.isUniform()
    assert not wcs.isCelestial()

    assert_raises(TypeError, galsim.ShearWCS)
    assert_raises(TypeError, galsim.ShearWCS, shear=0.3)
    assert_raises(TypeError, galsim.ShearWCS, shear=shear, origin=galsim.PositionD(0,0))
    assert_raises(TypeError, galsim.ShearWCS, shear=shear, world_origin=galsim.PositionD(0,0))
    assert_raises(TypeError, galsim.ShearWCS, g1=g1, g2=g2)

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
    do_wcs_image(wcs, 'ShearWCS', run_slow)

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
    wcs2 = galsim.ShearWCS(scale, shear).withOrigin(origin)
    assert wcs == wcs2, 'OffsetShearWCS is not == ShearWCS.withOrigin(origin)'
    assert wcs.shear == shear
    assert wcs.origin == origin
    assert wcs.world_origin == galsim.PositionD(0,0)
    assert not wcs.isPixelScale()
    assert not wcs.isLocal()
    assert wcs.isUniform()
    assert not wcs.isCelestial()

    wcs3 = galsim.OffsetShearWCS(scale, shear)
    assert wcs3.origin == galsim.PositionD(0,0)
    assert wcs3.world_origin == galsim.PositionD(0,0)

    assert_raises(TypeError, galsim.OffsetShearWCS)
    assert_raises(TypeError, galsim.OffsetShearWCS, shear=0.3)
    assert_raises(TypeError, galsim.OffsetShearWCS, shear=shear, origin=5)
    assert_raises(TypeError, galsim.OffsetShearWCS, shear=shear,
                  origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))
    assert_raises(TypeError, galsim.OffsetShearWCS, shear=shear,
                  world_origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))

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
    do_wcs_image(wcs, 'OffsetShearWCS', run_slow)


@timer
def test_affinetransform(run_slow):
    """Test the AffineTransform class
    """
    # First a slight tweak on a simple scale factor
    dudx = 0.2342
    dudy = 0.0023
    dvdx = 0.0019
    dvdy = 0.2391

    wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
    assert not wcs.isPixelScale()
    assert wcs.isLocal()
    assert wcs.isUniform()
    assert not wcs.isCelestial()

    assert wcs.dudx == dudx
    assert wcs.dudy == dudy
    assert wcs.dvdx == dvdx
    assert wcs.dvdy == dvdy

    assert_raises(TypeError, galsim.JacobianWCS)
    assert_raises(TypeError, galsim.JacobianWCS, dudx, dudy, dvdx)
    assert_raises(TypeError, galsim.JacobianWCS, dudx, dudy, dvdx, dvdy,
                  origin=galsim.PositionD(0,0))
    assert_raises(TypeError, galsim.JacobianWCS, dudx, dudy, dvdx, dvdy,
                  world_origin=galsim.PositionD(0,0))

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
    wcs2 = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy).withOrigin(origin)
    assert wcs == wcs2, 'AffineTransform is not == JacobianWCS.withOrigin(origin)'
    assert not wcs.isPixelScale()
    assert not wcs.isLocal()
    assert wcs.isUniform()
    assert not wcs.isCelestial()

    assert_raises(TypeError, galsim.AffineTransform)
    assert_raises(TypeError, galsim.AffineTransform, dudx, dudy, dvdx)
    assert_raises(TypeError, galsim.AffineTransform, dudx, dudy, dvdx, dvdy, origin=3)
    assert_raises(TypeError, galsim.AffineTransform, dudx, dudy, dvdx, dvdy,
                  origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))
    assert_raises(TypeError, galsim.AffineTransform, dudx, dudy, dvdx, dvdy,
                  world_origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))

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
    do_wcs_image(wcs, 'JacobianWCS', run_slow)

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
    do_wcs_image(wcs, 'AffineTransform', run_slow)

    # Degenerate transformation should raise some errors
    degen_wcs = galsim.JacobianWCS(0.2, 0.1, 0.2, 0.1)
    assert_raises(galsim.GalSimError, degen_wcs.getDecomposition)

    image_pos = galsim.PositionD(0,0)
    world_pos = degen_wcs.toWorld(image_pos)  # This direction is ok.
    assert_raises(galsim.GalSimError, degen_wcs.toImage, world_pos)  # This is not.
    assert_raises(galsim.GalSimError, degen_wcs._x, 0, 0)
    assert_raises(galsim.GalSimError, degen_wcs._y, 0, 0)
    assert_raises(galsim.GalSimError, degen_wcs.inverse)
    assert_raises(galsim.GalSimError, degen_wcs.toImage, galsim.Gaussian(sigma=2))


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

class Cubic:
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


@timer
def test_uvfunction(run_slow):
    """Test the UVFunction class
    """
    # First make some that are identical to simpler WCS classes:
    # 1. Like PixelScale
    scale = 0.17
    ufunc = lambda x,y: x * scale
    vfunc = lambda x,y: y * scale
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like PixelScale', test_pickle=False)
    assert wcs.ufunc(2.9, 3.7) == ufunc(2.9, 3.7)
    assert wcs.vfunc(2.9, 3.7) == vfunc(2.9, 3.7)
    assert wcs.xfunc is None
    assert wcs.yfunc is None
    assert not wcs.isPixelScale()
    assert not wcs.isLocal()
    assert not wcs.isUniform()
    assert not wcs.isCelestial()

    # Also check with inverse functions.
    xfunc = lambda u,v: u / scale
    yfunc = lambda u,v: v / scale
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like PixelScale with inverse', test_pickle=False)
    assert wcs.ufunc(2.9, 3.7) == ufunc(2.9, 3.7)
    assert wcs.vfunc(2.9, 3.7) == vfunc(2.9, 3.7)
    assert wcs.xfunc(2.9, 3.7) == xfunc(2.9, 3.7)
    assert wcs.yfunc(2.9, 3.7) == yfunc(2.9, 3.7)

    assert_raises(TypeError, galsim.UVFunction)
    assert_raises(TypeError, galsim.UVFunction, ufunc=ufunc)
    assert_raises(TypeError, galsim.UVFunction, vfunc=vfunc)
    assert_raises(TypeError, galsim.UVFunction, ufunc, vfunc, origin=5)
    assert_raises(TypeError, galsim.UVFunction, ufunc, vfunc,
                  origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))
    assert_raises(TypeError, galsim.UVFunction, ufunc, vfunc,
                  world_origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))

    # 2. Like ShearWCS
    scale = 0.23
    g1 = 0.14
    g2 = -0.37
    factor = 1./np.sqrt(1.-g1*g1-g2*g2)
    ufunc = lambda x,y: (x - g1*x - g2*y) * scale * factor
    vfunc = lambda x,y: (y + g1*y - g2*x) * scale * factor
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like ShearWCS', test_pickle=False)

    # Also check with inverse functions.
    xfunc = lambda u,v: (u + g1*u + g2*v) / scale * factor
    yfunc = lambda u,v: (v - g1*v + g2*u) / scale * factor
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like ShearWCS with inverse', test_pickle=False)

    # 3. Like an AffineTransform
    dudx = 0.2342
    dudy = 0.1432
    dvdx = 0.1409
    dvdy = 0.2391

    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction like AffineTransform', test_pickle=False)

    # Check that passing functions as strings works correctly.
    wcs = galsim.UVFunction(ufunc='%r*x + %r*y'%(dudx,dudy), vfunc='%r*x + %r*y'%(dvdx,dvdy))
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction with string funcs', test_pickle=True)

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_string', run_slow)

    # Also check with inverse functions.
    det = dudx*dvdy - dudy*dvdx
    wcs = galsim.UVFunction(
            ufunc='%r*x + %r*y'%(dudx,dudy),
            vfunc='%r*x + %r*y'%(dvdx,dvdy),
            xfunc='(%r*u + %r*v)/(%r)'%(dvdy,-dudy,det),
            yfunc='(%r*u + %r*v)/(%r)'%(-dvdx,dudx,det) )
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction with string inverse funcs', test_pickle=True)

    # The same thing in fact, but nominally takes color as an argument.
    wcsc = galsim.UVFunction(
            ufunc='%r*x + %r*y'%(dudx,dudy),
            vfunc='%r*x + %r*y'%(dvdx,dvdy),
            xfunc='(%r*u + %r*v)/(%r)'%(dvdy,-dudy,det),
            yfunc='(%r*u + %r*v)/(%r)'%(-dvdx,dudx,det), uses_color=True)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction with unused color term', test_pickle=True)

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
    do_nonlocal_wcs(wcs, ufunc2, vfunc2, 'UVFunction with origins in funcs', test_pickle=False)
    wcs = galsim.UVFunction(ufunc, vfunc, origin=origin, world_origin=world_origin)
    do_nonlocal_wcs(wcs, ufunc2, vfunc2, 'UVFunction with origin arguments', test_pickle=False)

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_lambda', run_slow)

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
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Cubic radial UVFunction', test_pickle=False)

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_func', run_slow)

    # 6. Repeat with a function object rather than a regular function.
    # Use a different `a` parameter for u and v to make things more interesting.
    cubic_u = Cubic(2.9e-5, 2000., 'u')
    cubic_v = Cubic(-3.7e-5, 2000., 'v')
    wcs = galsim.UVFunction(cubic_u, cubic_v, origin=galsim.PositionD(x0,y0))
    ufunc = lambda x,y: cubic_u(x-x0, y-y0)
    vfunc = lambda x,y: cubic_v(x-x0, y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Cubic object UVFunction', test_pickle=False)

    # Check that using a wcs in the context of an image works correctly
    do_wcs_image(wcs, 'UVFunction_object', run_slow)

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
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction from demo9', test_pickle=False)

    # Check that passing really long strings works correctly.
    ufuncs = "0.05 * x * (1. + 2.e-6 * (x**2 + y**2))"
    vfuncs = "0.05 * y * (1. + 2.e-6 * (x**2 + y**2))"
    xfuncs = ("(lambda w: ( 0. if w==0. else "
              "   100.*u/w*(( 5*math.sqrt(w**2+5.e3/27.)+5*w )**(1./3.) - "
              "             ( 5*math.sqrt(w**2+5.e3/27.)-5*w )**(1./3.))) )(math.sqrt(u**2+v**2))")
    yfuncs = ("(lambda w: ( 0. if w==0. else "
              "   100.*v/w*(( 5*math.sqrt(w**2+5.e3/27.)+5*w )**(1./3.) - "
              "             ( 5*math.sqrt(w**2+5.e3/27.)-5*w )**(1./3.))) )(math.sqrt(u**2+v**2))")
    wcs = galsim.UVFunction(ufuncs, vfuncs, xfuncs, yfuncs)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction from demo9, string', test_pickle=True)
    do_wcs_image(wcs, 'UVFunction from demo9, string', run_slow)

    # This version doesn't work with numpy arrays because of the math functions.
    # This provides a test of that branch of the makeSkyImage function.
    ufunc = lambda x,y : 0.17 * x * (1. + 1.e-5 * math.sqrt(x**2 + y**2))
    vfunc = lambda x,y : 0.17 * y * (1. + 1.e-5 * math.sqrt(x**2 + y**2))
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'UVFunction with math funcs', test_pickle=False)
    do_wcs_image(wcs, 'UVFunction_math', run_slow)

    # 8. A non-trivial color example
    ufunc = lambda x,y,c: (dudx + 0.1*c)*x + dudy*y
    vfunc = lambda x,y,c: dvdx*x + (dvdy - 0.2*c)*y
    xfunc = lambda u,v,c: ((dvdy - 0.2*c)*u - dudy*v) / ((dudx+0.1*c)*(dvdy-0.2*c)-dudy*dvdx)
    yfunc = lambda u,v,c: (-dvdx*u + (dudx + 0.1*c)*v) / ((dudx+0.1*c)*(dvdy-0.2*c)-dudy*dvdx)
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc, uses_color=True)
    do_nonlocal_wcs(wcs, lambda x,y: ufunc(x,y,-0.3), lambda x,y: vfunc(x,y,-0.3),
                    'UVFunction with color-dependence', test_pickle=False, color=-0.3)

    # Also, check this one as a string
    wcs = galsim.UVFunction(ufunc='(%r+0.1*c)*x + %r*y'%(dudx,dudy),
                            vfunc='%r*x + (%r-0.2*c)*y'%(dvdx,dvdy),
                            xfunc='((%r-0.2*c)*u - %r*v)/((%r+0.1*c)*(%r-0.2*c)-%r)'%(
                                dvdy,dudy,dudx,dvdy,dudy*dvdx),
                            yfunc='(-%r*u + (%r+0.1*c)*v)/((%r+0.1*c)*(%r-0.2*c)-%r)'%(
                                dvdx,dudx,dudx,dvdy,dudy*dvdx),
                            uses_color=True)
    do_nonlocal_wcs(wcs, lambda x,y:  ufunc(x,y,1.7), lambda x,y: vfunc(x,y,1.7),
                    'UVFunction with color-dependence, string', test_pickle=True, color=1.7)

    # 9. A non-trivial color example that fails for arrays
    ufunc = lambda x,y,c : math.exp(c * x)
    vfunc = lambda x,y,c : math.exp(c * y/2)
    xfunc = lambda u,v,c : math.log(u) / c
    yfunc = lambda u,v,c : math.log(v) * 2 / c
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc, uses_color=True)
    do_nonlocal_wcs(wcs, lambda x,y: ufunc(x,y,0.01), lambda x,y: vfunc(x,y,0.01),
                    'UVFunction with math and color-dependence', test_pickle=False, color=0.01)

    # 10. One with invalid functions, which raise errors.  (Just for coverage really.)
    ufunc = lambda x,y : 0.05 * x * math.sqrt(x**2 - y**2)
    vfunc = lambda x,y : 0.05 * y * math.sqrt(3*y**2 - x**2)
    xfunc = lambda u,v : 0.05 * u * math.sqrt(2*u**2 - 7*v**2)
    yfunc = lambda u,v : 0.05 * v * math.sqrt(8*v**2 - u**2)
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc)
    assert_raises(ValueError, wcs.toWorld, galsim.PositionD(5, 6))
    assert_raises(ValueError, wcs.toWorld, galsim.PositionD(8, 2))
    assert_raises(ValueError, wcs.toImage, galsim.PositionD(3, 3))
    assert_raises(ValueError, wcs.toImage, galsim.PositionD(6, 0))
    # Repeat with color
    ufunc = lambda x,y,c : 0.05 * c * math.sqrt(x**2 - y**2)
    vfunc = lambda x,y,c : 0.05 * c * math.sqrt(3*y**2 - x**2)
    xfunc = lambda u,v,c : 0.05 * c * math.sqrt(2*u**2 - 7*v**2)
    yfunc = lambda u,v,c : 0.05 * c * math.sqrt(8*v**2 - u**2)
    wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc, uses_color=True)
    assert_raises(ValueError, wcs.toWorld, galsim.PositionD(5, 6), color=0.2)
    assert_raises(ValueError, wcs.toWorld, galsim.PositionD(8, 2), color=0.2)
    assert_raises(ValueError, wcs.toImage, galsim.PositionD(3, 3), color=0.2)
    assert_raises(ValueError, wcs.toImage, galsim.PositionD(6, 0), color=0.2)


@timer
def test_radecfunction(run_slow):
    """Test the RaDecFunction class
    """
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

    ufunc = lambda x,y: cubic_u(x-x0, y-y0)
    vfunc = lambda x,y: cubic_v(x-x0, y-y0)
    funcs.append( (ufunc, vfunc, 'offset Cubic object') )

    # The last one needs to not have a lambda, since we use it for the image test, which
    # requires the function to be picklable.
    funcs.append( (cubic_u, cubic_v, 'Cubic object') )

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

            scale = galsim.arcsec / galsim.radians
            radec_func = lambda x,y: center.deproject_rad(ufunc(x,y)*scale, vfunc(x,y)*scale,
                                                          projection='lambert')
            wcs2 = galsim.RaDecFunction(radec_func)
            assert not wcs2.isPixelScale()
            assert not wcs2.isLocal()
            assert not wcs2.isUniform()
            assert wcs2.isCelestial()

            # Also test with one that doesn't work with numpy arrays to test that the
            # code does the right thing in that case too, since local and makeSkyImage
            # try the numpy option first and do something else if it fails.
            # This also tests the alternate initialization using separate ra_func, dec_fun.
            ra_func = lambda x,y: center.deproject(
                    ufunc(x,y)*galsim.arcsec, vfunc(x,y)*galsim.arcsec,
                    projection='lambert').ra.rad
            dec_func = lambda x,y: center.deproject(
                    ufunc(x,y)*galsim.arcsec, vfunc(x,y)*galsim.arcsec,
                    projection='lambert').dec.rad
            wcs3 = galsim.RaDecFunction(ra_func, dec_func)

            # The pickle tests need to have a string for ra_func, dec_func, which is
            # a bit tough with the ufunc,vfunc stuff.  So do something simpler for that.
            radec_str = '%r.deproject_rad(x*%f,y*%f,projection="lambert")'%(center,scale,scale)
            wcs4 = galsim.RaDecFunction(radec_str, origin=galsim.PositionD(17.,34.))
            ra_str = '%r.deproject(x*galsim.arcsec,y*galsim.arcsec,projection="lambert").ra.rad'%center
            dec_str = '%r.deproject(x*galsim.arcsec,y*galsim.arcsec,projection="lambert").dec.rad'%center
            wcs5 = galsim.RaDecFunction(ra_str, dec_str, origin=galsim.PositionD(-9.,-8.))

            wcs6 = wcs2.copy()
            assert wcs2 == wcs6, 'RaDecFunction copy is not == the original'
            assert wcs6.radec_func(3,4) == radec_func(3,4)

            # Check that distance, jacobian for some x,y positions match the UV values.
            for x,y in zip(far_x_list, far_y_list):

                # First do some basic checks of project, deproject for the given (u,v)
                u = ufunc(x,y)
                v = vfunc(x,y)
                coord = center.deproject(u*galsim.arcsec, v*galsim.arcsec, projection='lambert')
                ra, dec = radec_func(x,y)
                np.testing.assert_almost_equal(ra, coord.ra.rad, 8,
                                               'rafunc produced wrong value')
                np.testing.assert_almost_equal(dec, coord.dec.rad, 8,
                                               'decfunc produced wrong value')
                pos = center.project(coord, projection='lambert')
                np.testing.assert_almost_equal(pos[0]/galsim.arcsec, u, digits,
                                               'project x was inconsistent')
                np.testing.assert_almost_equal(pos[1]/galsim.arcsec, v, digits,
                                               'project y was inconsistent')
                d1 = np.sqrt(u*u+v*v)
                d2 = center.distanceTo(coord)
                # The distances aren't expected to match.  Instead, for a Lambert projection,
                # d1 should match the straight line distance through the sphere.
                import math
                d2 = 2.*np.sin(d2/2.) * galsim.radians / galsim.arcsec
                np.testing.assert_almost_equal(
                        d2, d1, digits, 'deprojected dist does not match expected value.')

                # Now test the two initializations of RaDecFunction.
                for test_wcs in [ wcs2, wcs3 ]:
                    image_pos = galsim.PositionD(x,y)
                    world_pos1 = wcs1.toWorld(image_pos)
                    world_pos2 = test_wcs.toWorld(image_pos)
                    origin = test_wcs.toWorld(galsim.PositionD(0.,0.))
                    d3 = np.sqrt( world_pos1.x**2 + world_pos1.y**2 )
                    d4 = center.distanceTo(world_pos2)
                    d4 = 2.*np.sin(d4/2) * galsim.radians / galsim.arcsec
                    np.testing.assert_almost_equal(
                            d3, d1, digits, 'UV '+name+' dist does not match expected value.')
                    np.testing.assert_almost_equal(
                            d4, d1, digits, 'RaDec '+name+' dist does not match expected value.')

                    # Calculate the Jacobians for each wcs
                    jac1 = wcs1.jacobian(image_pos)
                    jac2 = test_wcs.jacobian(image_pos)

                    # The pixel area should match pretty much exactly.  The Lambert projection
                    # is an area preserving projection.
                    np.testing.assert_almost_equal(
                            jac2.pixelArea(), jac1.pixelArea(), digits,
                            'RaDecFunction '+name+' pixelArea() does not match expected value.')
                    np.testing.assert_almost_equal(
                            test_wcs.pixelArea(image_pos), jac1.pixelArea(), digits,
                            'RaDecFunction '+name+' pixelArea(pos) does not match expected value.')

                    # The distortion should be pretty small, so the min/max linear scale should
                    # match pretty well.
                    np.testing.assert_almost_equal(
                            jac2.minLinearScale(), jac1.minLinearScale(), digits,
                            'RaDecFunction '+name+' minLinearScale() does not match expected value.')
                    np.testing.assert_almost_equal(
                            test_wcs.minLinearScale(image_pos), jac1.minLinearScale(), digits,
                            'RaDecFunction '+name+' minLinearScale(pos) does not match expected value.')
                    np.testing.assert_almost_equal(
                            jac2.maxLinearScale(), jac1.maxLinearScale(), digits,
                            'RaDecFunction '+name+' maxLinearScale() does not match expected value.')
                    np.testing.assert_almost_equal(
                            test_wcs.maxLinearScale(image_pos), jac1.maxLinearScale(), digits,
                            'RaDecFunction '+name+' maxLinearScale(pos) does not match expected value.')

                    # The main discrepancy between the jacobians is a rotation term.
                    # The pixels in the projected coordinates do not necessarily point north,
                    # since the direction to north changes over the field.  However, we can
                    # calculate this expected discrepancy and correct for it to get a comparison
                    # of the full jacobian that should be accurate to 5 digits.
                    # If A = coord, B = center, and C = the north pole, then the rotation angle is
                    # 180 deg - A - B.
                    A = coord.angleBetween(center, north_pole)
                    B = center.angleBetween(north_pole, coord)
                    C = north_pole.angleBetween(coord, center)
                    # The angle C should equal coord.ra - center.ra, so use this as a unit test of
                    # the angleBetween function:
                    np.testing.assert_almost_equal(
                            C / galsim.degrees, (coord.ra - center.ra) / galsim.degrees, digits,
                            'CelestialCoord calculated the wrong angle between center and coord')
                    angle = 180 * galsim.degrees - A - B

                    # Now we can use this angle to correct the jacobian from test_wcs.
                    s,c = angle.sincos()
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
                                 ', '+str(center.dec/galsim.degrees), test_pickle=False)
                do_celestial_wcs(wcs3, 'RaDecFunc 2 centered at '+str(center.ra/galsim.degrees)+
                                 ', '+str(center.dec/galsim.degrees), test_pickle=False)

                do_celestial_wcs(wcs4, 'RaDecFunc 3 centered at '+str(center.ra/galsim.degrees)+
                                 ', '+str(center.dec/galsim.degrees), test_pickle=True)
                do_celestial_wcs(wcs5, 'RaDecFunc 4 centered at '+str(center.ra/galsim.degrees)+
                                 ', '+str(center.dec/galsim.degrees), test_pickle=True)

    assert_raises(TypeError, galsim.RaDecFunction)
    assert_raises(TypeError, galsim.RaDecFunction, radec_func, origin=5)
    assert_raises(TypeError, galsim.RaDecFunction, radec_func,
                  origin=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees))
    assert_raises(TypeError, galsim.RaDecFunction, radec_func, world_origin=galsim.PositionD(0,0))

    # Check that using a wcs in the context of an image works correctly
    # (Uses the last wcs2, wcs3 set in the above loops.)
    do_wcs_image(wcs2, 'RaDecFunction', run_slow)
    do_wcs_image(wcs3, 'RaDecFunction', run_slow)

    # One with invalid functions, which raise errors.  (Just for coverage really.)
    radec_func = lambda x,y: center.deproject_rad(math.sqrt(x),math.sqrt(y), projection='lambert')
    wcs = galsim.RaDecFunction(radec_func)
    assert_raises(ValueError, wcs.toWorld, galsim.PositionD(-5, 6))
    assert_raises(ValueError, wcs.toWorld, galsim.PositionD(8, -2))

def do_ref(wcs, ref_list, name, approx=False, image=None):
    # Test that the given wcs object correctly converts the reference positions

    # Normally, we check the agreement to 1.e-3 arcsec.
    # However, we allow the caller to indicate the that inverse transform is only approximate.
    # In this case, we only check to 1 digit.  Originally,  this was just for the reverse
    # transformation from world back to image coordinates, since some of the transformations
    # are not analytic, so some routines don't iterate to a very high accuracy.  But older
    # versions of wcstools are slightly (~0.01 arcsec) inaccurate even for the forward
    # transformation for TNX and ZPX.  So now we use digits2 for both toWorld and toImage checks.
    if approx:
        digits2 = 1
    else:
        digits2 = digits

    print('Start reference testing for '+name)
    for ref in ref_list:
        ra = galsim.Angle.from_hms(ref[0])
        dec = galsim.Angle.from_dms(ref[1])
        x = ref[2]
        y = ref[3]
        val = ref[4]

        # Check image -> world
        ref_coord = galsim.CelestialCoord(ra,dec)
        coord = wcs.toWorld(galsim.PositionD(x,y))
        dist = ref_coord.distanceTo(coord) / galsim.arcsec
        print('x, y = ',x,y)
        print('ref_coord = ', ref_coord.ra.hms(), ref_coord.dec.dms())
        print('coord = ', coord.ra.hms(), coord.dec.dms())
        np.testing.assert_almost_equal(dist, 0, digits2, 'wcs.toWorld differed from expected value')

        # Check world -> image
        pixel_scale = wcs.minLinearScale(galsim.PositionD(x,y))
        pos = wcs.toImage(galsim.CelestialCoord(ra,dec))
        np.testing.assert_almost_equal((x-pos.x)*pixel_scale, 0, digits2,
                                       'wcs.toImage differed from expected value')
        np.testing.assert_almost_equal((y-pos.y)*pixel_scale, 0, digits2,
                                       'wcs.toImage differed from expected value')
        if image:
            np.testing.assert_almost_equal(image(x,y), val, digits,
                                           'image(x,y) differed from reference value')


@timer
def test_astropywcs(run_slow):
    """Test the AstropyWCS class
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        import astropy.wcs
        import scipy  # AstropyWCS constructor will do this, so check now.

    # These all work, but it is quite slow, so only test a few of them for the regular unit tests.
    # (1.8 seconds for 4 tags.)
    # Test all of them when running python test_wcs.py.
    if run_slow:
        test_tags = ['HPX', 'TAN', 'TSC', 'STG', 'ZEA', 'ARC', 'ZPN', 'SIP', 'TPV', 'TAN-PV',
                     'TAN-FLIP', 'REGION', 'ZTF']
        # The ones that still don't work are TNX and ZPX.
        # In both cases, astropy thinks it reads them successfully, but fails the tests and/or
        # bombs out with a malloc error:
        # incorrect checksum for freed object - object was probably modified after being freed
    else:
        test_tags = [ 'TAN', 'SIP', 'ZTF', 'TAN-PV' ]

    dir = 'fits_files'
    for tag in test_tags:
        file_name, ref_list = references[tag]
        print(tag,' file_name = ',file_name)
        if tag == 'TAN':
            wcs = galsim.AstropyWCS(file_name, dir=dir, compression='none', hdu=0)
        else:
            wcs = galsim.AstropyWCS(file_name, dir=dir)
        assert not wcs.isPixelScale()
        assert not wcs.isLocal()
        assert not wcs.isUniform()
        assert wcs.isCelestial()

        do_ref(wcs, ref_list, 'AstropyWCS '+tag)

        if tag == 'TAN':
            # Also check origin.  (Now that reference checks are done.)
            wcs = galsim.AstropyWCS(file_name, dir=dir, compression='none', hdu=0,
                                    origin=galsim.PositionD(3,4))

        do_celestial_wcs(wcs, 'Astropy file '+file_name)

        do_wcs_image(wcs, 'AstropyWCS_'+tag, run_slow)

    # Can also use an existing astropy.wcs.WCS instance to construct.
    # This is probably a rare use case, but could aid efficiency if you already build the
    # astropy WCS for other purposes.
    astropy_wcs = wcs.wcs  # Just steal such an object from the last wcs above.
    assert isinstance(astropy_wcs, astropy.wcs.WCS)
    wcs1 = galsim.AstropyWCS(wcs=astropy_wcs)
    do_celestial_wcs(wcs1, 'AstropyWCS from wcs', test_pickle=False)
    repr(wcs1)

    # Can also use a header.  Again steal it from the wcs above.
    wcs2 = galsim.AstropyWCS(header=wcs.header)
    do_celestial_wcs(wcs2, 'AstropyWCS from header', test_pickle=True)

    # Doesn't support LINEAR WCS types.
    with assert_raises(galsim.GalSimError):
        galsim.AstropyWCS('SBProfile_comparison_images/kolmogorov.fits')

    # This file does not have any WCS information in it.
    with assert_raises(galsim.GalSimError):
        galsim.AstropyWCS('fits_files/blankimg.fits')

    assert_raises(TypeError, galsim.AstropyWCS)
    assert_raises(TypeError, galsim.AstropyWCS, file_name, header='dummy')
    assert_raises(TypeError, galsim.AstropyWCS, file_name, wcs=wcs)
    assert_raises(TypeError, galsim.AstropyWCS, wcs=wcs, header='dummy')

    # Astropy thinks it can handle ZPX files, but as of version 2.0.4, they don't work right.
    # It reads it in ok, and even works with it fine.  But it doesn't round trip through
    # its own write and read.  Even worse, it natively gives a fairly obscure error, which
    # we convert into an OSError by hand.
    # This test will let us know when they finally fix it.  If it fails, we can remove this
    # test and add 'ZPX' to the list of working astropy.wcs types above.
    with assert_raises(OSError):
        wcs = galsim.AstropyWCS(references['ZPX'][0], dir=dir)
        do_wcs_image(wcs, 'AstropyWCS_ZPX', run_slow)


@timer
def test_pyastwcs(run_slow):
    """Test the PyAstWCS class
    """
    try:
        import starlink.Ast
    except ImportError:
        print('Unable to import starlink.Ast.  Skipping PyAstWCS tests.')
        return

    # These all work, but it is quite slow, so only test a few of them for the regular unit tests.
    # (2.4 seconds for 6 tags.)
    # Test all of them when running python test_wcs.py.
    if run_slow:
        test_tags = [ 'HPX', 'TAN', 'TSC', 'STG', 'ZEA', 'ARC', 'ZPN', 'SIP', 'TPV', 'ZPX',
                      'TAN-PV', 'TAN-FLIP', 'REGION', 'TNX', 'ZTF' ]
    else:
        test_tags = [ 'TAN', 'ZPX', 'SIP', 'TAN-PV', 'TNX', 'ZTF' ]

    dir = 'fits_files'
    for tag in test_tags:
        file_name, ref_list = references[tag]
        print(tag,' file_name = ',file_name)
        if tag == 'TAN':
            wcs = galsim.PyAstWCS(file_name, dir=dir, compression='none', hdu=0)
        else:
            wcs = galsim.PyAstWCS(file_name, dir=dir)
        assert not wcs.isPixelScale()
        assert not wcs.isLocal()
        assert not wcs.isUniform()
        assert wcs.isCelestial()

        # The PyAst implementation of the SIP type only gets the inverse transformation
        # approximately correct.  So we need to be a bit looser in that check.
        approx = tag in [ 'SIP' ]
        do_ref(wcs, ref_list, 'PyAstWCS '+tag, approx)

        if tag == 'TAN':
            # Also check origin.  (Now that reference checks are done.)
            wcs = galsim.PyAstWCS(file_name, dir=dir, compression='none', hdu=0,
                                  origin=galsim.PositionD(3,4))

        do_celestial_wcs(wcs, 'PyAst file '+file_name, approx=approx)

        # TAN-FLIP has an error of 4mas after write and read here, which I don't really understand.
        # but it's small enough an error that I don't think it's worth worrying about further.
        approx = tag in [ 'ZPX', 'TAN-FLIP', 'ZTF' ]
        do_wcs_image(wcs, 'PyAstWCS_'+tag, run_slow, approx)

    # Can also use an existing startlink.Ast.FrameSet instance to construct.
    # This is probably a rare use case, but could aid efficiency if you already open the
    # fits file with starlink for other purposes.
    wcs = galsim.PyAstWCS(references['TAN'][0], dir=dir)
    wcsinfo = wcs.wcsinfo
    assert isinstance(wcsinfo, starlink.Ast.FrameSet)
    wcs1 = galsim.PyAstWCS(wcsinfo=wcsinfo)
    do_celestial_wcs(wcs1, 'PyAstWCS from wcsinfo', test_pickle=False)
    repr(wcs1)

    # Can also use a header.  Again steal it from the wcs above.
    wcs2 = galsim.PyAstWCS(header=wcs.header)
    do_celestial_wcs(wcs2, 'PyAstWCS from header', test_pickle=True)

    # Doesn't support LINEAR WCS types.
    with assert_raises(galsim.GalSimError):
        galsim.PyAstWCS('SBProfile_comparison_images/kolmogorov.fits')

    # This file does not have any WCS information in it.
    with assert_raises(OSError):
        galsim.PyAstWCS('fits_files/blankimg.fits')

    assert_raises(TypeError, galsim.PyAstWCS)
    assert_raises(TypeError, galsim.PyAstWCS, file_name, header='dummy')
    assert_raises(TypeError, galsim.PyAstWCS, file_name, wcsinfo=wcsinfo)
    assert_raises(TypeError, galsim.PyAstWCS, wcsinfo=wcsinfo, header='dummy')


@timer
def test_wcstools(run_slow):
    """Test the WcsToolsWCS class
    """
    # These all work, but it is very slow, so only test one of them for the regular unit tests.
    # (1.5 seconds for just the one tag.)
    # Test all of them when running python test_wcs.py.
    if run_slow:
        # Note: TPV seems to work, but on one machine, repeated calls to xy2sky with the same
        # x,y values vary between two distinct ra,dec outputs.  I have no idea what's going on,
        # since I thought the calculation ought to be deterministic, but it clearly something
        # isn't working right.  So just skip that test.
        test_tags = [ 'TAN', 'TSC', 'STG', 'ZEA', 'ARC', 'ZPN', 'SIP', 'ZPX', 'TAN-FLIP',
                      'REGION', 'TNX' ]
    else:
        test_tags = [ 'TNX' ]

    dir = 'fits_files'
    try:
        galsim.WcsToolsWCS(references['TAN'][0], dir=dir)
    except OSError:
        print('Unable to execute xy2sky.  Skipping WcsToolsWCS tests.')
        return

    for tag in test_tags:
        file_name, ref_list = references[tag]
        print(tag,' file_name = ',file_name)
        wcs = galsim.WcsToolsWCS(file_name, dir=dir)
        assert not wcs.isPixelScale()
        assert not wcs.isLocal()
        assert not wcs.isUniform()
        assert wcs.isCelestial()

        # The wcstools implementation of the SIP and TPV types only gets the inverse
        # transformations approximately correct.  So we need to be a bit looser in those checks.
        approx = tag in [ 'SIP', 'TPV', 'ZPX', 'TNX' ]
        do_ref(wcs, ref_list, 'WcsToolsWCS '+tag, approx)

        # Recenter (x,y) = (0,0) at the image center to avoid wcstools warnings about going
        # off the image.
        im = galsim.fits.read(file_name, dir=dir)
        wcs = wcs.shiftOrigin(origin = -im.center)

        do_celestial_wcs(wcs, 'WcsToolsWCS '+file_name, approx=approx)

        do_wcs_image(wcs, 'WcsToolsWCS_'+tag, run_slow)

    # HPX is one of the ones that WcsToolsWCS doesn't support.
    with assert_raises(galsim.GalSimError):
        galsim.WcsToolsWCS(references['HPX'][0], dir=dir)

    # This file does not have any WCS information in it.
    with assert_raises(OSError):
        galsim.WcsToolsWCS('fits_files/blankimg.fits')

    # Doesn't support LINEAR WCS types.
    with assert_raises(galsim.GalSimError):
        galsim.WcsToolsWCS('SBProfile_comparison_images/kolmogorov.fits')

    assert_raises(TypeError, galsim.WcsToolsWCS)
    assert_raises(TypeError, galsim.WcsToolsWCS, file_name, header='dummy')


@timer
def test_gsfitswcs(run_slow):
    """Test the GSFitsWCS class
    """
    # These are all relatively fast (total time for all 10 is about 1.1 seconds),
    # so unlike some of the other WCS types, it's not that slow to do all of them.
    # And it's required to get (relatively) complete test coverage.
    test_tags = [ 'TAN', 'STG', 'ZEA', 'ARC', 'TPV', 'TAN-PV', 'TAN-FLIP', 'TNX', 'SIP', 'ZTF' ]

    dir = 'fits_files'

    for tag in test_tags:
        file_name, ref_list = references[tag]
        print(tag,' file_name = ',file_name)
        if tag == 'TAN':
            # For this one, check compression and hdu options.
            wcs = galsim.GSFitsWCS(file_name, dir=dir, compression='none', hdu=0)
        else:
            wcs = galsim.GSFitsWCS(file_name, dir=dir)
        assert not wcs.isPixelScale()
        assert not wcs.isLocal()
        assert not wcs.isUniform()
        assert wcs.isCelestial()

        do_ref(wcs, ref_list, 'GSFitsWCS '+tag)

        if tag == 'TAN':
            # Also check origin.  (Now that reference checks are done.)
            wcs = galsim.GSFitsWCS(file_name, dir=dir, compression='none', hdu=0,
                                   origin=galsim.PositionD(3,4))

        do_celestial_wcs(wcs, 'GSFitsWCS '+file_name)

        do_wcs_image(wcs, 'GSFitsWCS_'+tag, run_slow)

    # tpv_odd.fits is a modified version to have (unsupported) odd powers of r.
    with assert_raises(galsim.GalSimNotImplementedError):
        galsim.GSFitsWCS('tpv_odd.fits', dir=dir)

    # TSC is one of the ones that GSFitsWCS doesn't support.
    with assert_raises(galsim.GalSimValueError):
        galsim.GSFitsWCS(references['TSC'][0], dir=dir)

    # Doesn't support LINEAR WCS types.
    with assert_raises(galsim.GalSimError):
        galsim.GSFitsWCS('SBProfile_comparison_images/kolmogorov.fits')

    # This file does not have any WCS information in it.
    with assert_raises(galsim.GalSimError):
        galsim.GSFitsWCS('fits_files/blankimg.fits')

    assert_raises(TypeError, galsim.GSFitsWCS)
    assert_raises(TypeError, galsim.GSFitsWCS, file_name, header='dummy')
    assert_raises(ValueError, galsim.GSFitsWCS, _data=('invalid',0,0,0,0,0,0))


@timer
def test_inverseab_convergence():
    """Test a SIP file that used to fail to converge.
    cf. #1105, fixed in #1106.
    """
    wcs = galsim.GSFitsWCS(_data = [
        'TAN-SIP',
        np.array([6.643813544313604e-05, 2.167251511173191e-06]),
        np.array([[-0.00019733136932200036, 0.00037674127418905063],
                  [0.0003767412741890354, 0.00019733136932198898]]),
        coord.CelestialCoord(coord.Angle(2.171481673601117, coord.radians),
                             coord.Angle(-0.47508762601580773, coord.radians)),
        None,
        np.array([[[0.0, 0.0, 1.7492391970878894e-11, 5.689175560623694e-12],
                   [1.0, -2.960663840648593e-14, -6.305126324859291e-09, 0.0],
                   [1.6656211233894224e-11, -1.7067526686225035e-11, 0.0, 0.0],
                   [-6.296581950910473e-09, 0.0, 0.0, 0.0]],
                  [[0.0, 1.0, 5.444700521713077e-13, -6.296581950917086e-09],
                   [0.0, -8.377345670352661e-13, 1.7067526700203848e-11, 0.0],
                   [5.69481360769684e-13, -6.305126324862401e-09, 0.0, 0.0],
                   [-5.6891755511980266e-12, 0.0, 0.0, 0.0]]]),
        np.array([[[0.0, 2.0757753968394382e-12, -1.901883278189908e-11, 1.8853670343455113e-11],
                   [0.9988286261206877, 3.8197440624200676e-17, 7.132351432025443e-09, 0.0],
                   [-1.9018950513878596e-11, -5.6561012619600154e-11, 0.0, 0.0],
                   [7.160667102377515e-09, 0.0, 0.0, 0.0]],
                  [[0.0, 0.998828626120979, -6.203305031319767e-13, 7.1606671023315485e-09],
                   [-1.442341346591337e-12, -7.782798728336095e-17, 5.6561012640257335e-11, 0.0],
                   [-6.20409043536687e-13, 7.132351431975042e-09, 0.0, 0.0],
                   [-1.885367062073805e-11, 0.0, 0.0, 0.0]]])
        ])
    ra = 2.1859428247518253
    dec = -0.5001963313433293
    x, y = wcs.radecToxy(ra, dec, units="radians")
    ra1, dec1 = wcs.xyToradec(x, y, units="radians")
    assert np.isclose(ra1, ra)
    assert np.isclose(dec1, dec)

    ra = 2.186179858413897
    dec = -0.49995036220334654
    x, y = wcs.radecToxy(ra, dec, units="radians")
    ra1, dec1 = wcs.xyToradec(x, y, units="radians")
    assert np.isclose(ra1, ra)
    assert np.isclose(dec1, dec)

    # Now one that should fail, since it's well outside the applicable area for the SIP polynomials.
    ra = 2.1
    dec = -0.45
    with assert_raises(galsim.GalSimError):
        x, y = wcs.radecToxy(ra, dec, units="radians")
    try:
        x, y = wcs.radecToxy(ra, dec, units="radians")
    except galsim.GalSimError as e:
        print('Error message is\n',e)
        assert "[0,]" in str(e)

    # Check as part of a longer list (longer than 256 is important)
    ra = np.random.uniform(2.185, 2.186, 1000)
    dec = np.random.uniform(-0.501, -0.499, 1000)
    ra = np.append(ra, [2.1, 2.9])
    dec = np.append(dec, [-0.45, 0.2])
    print('ra = ',ra)
    print('dec = ',dec)
    with assert_raises(galsim.GalSimError):
        x, y = wcs.radecToxy(ra, dec, units="radians")
    try:
        x, y = wcs.radecToxy(ra, dec, units="radians")
    except galsim.GalSimError as e:
        print('Error message is\n',e)
        assert "[1000,1001,]" in str(e)
        # We don't currently do this for the user, but it's not too hard to get a python list
        # of the bad indices.  Included here as an example for users who may need this.
        bad = eval(str(e)[str(e).rfind('['):])
        print('as a python list: ',bad)


@timer
def test_tanwcs(run_slow):
    """Test the TanWCS function, which returns a GSFitsWCS instance.
    """

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
    do_wcs_image(wcs, 'TanWCS 1', run_slow)

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
    do_wcs_image(wcs, 'TanWCS 2', run_slow)

    # Check crossing ra=0.
    # Note: this worked properly even before fixing issue #1030, since GSFitsWCS keeps all
    # the ra values close to the value at the center of the image, so the ra values here
    # span from below 360 to above 360 without wrapping to 0.
    # cf. test_razero below for a test with a wcs that does wrap back to 0.
    dudx = 1.4
    dudy = 0.2
    dvdx = -0.3
    dvdy = 1.5
    u0 = 0.
    v0 = 0.
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, world_origin=galsim.PositionD(u0,v0))
    center = galsim.CelestialCoord(359.99 * galsim.degrees, -37.9 * galsim.degrees)
    wcs = galsim.TanWCS(affine, center)
    do_celestial_wcs(wcs, 'TanWCS 3')
    do_wcs_image(wcs, 'TanWCS 3', run_slow)

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
    do_celestial_wcs(wcs, 'TanWCS 4')
    do_wcs_image(wcs, 'TanWCS 4', run_slow)


@timer
def test_fitswcs(run_slow):
    """Test the FitsWCS factory function
    """
    if run_slow:
        # For more thorough unit tests (when running python test_wcs.py explicitly), this
        # will test everything.  If you don't have everything installed (especially
        # PyAst, then this may fail.
        test_tags = all_tags
    else:
        # These should always work, since GSFitsWCS will work on them.  So this
        # mostly just tests the basic interface of the FitsWCS function.
        test_tags = [ 'TAN', 'TPV' ]
        try:
            import starlink.Ast
            # Useful also to test one that GSFitsWCS doesn't work on.  This works on Travis at
            # least, and helps to cover some of the FitsWCS functionality where the first try
            # isn't successful.
            test_tags.append('HPX')
        except:
            pass

    dir = 'fits_files'

    for tag in test_tags:
        file_name, ref_list = references[tag]
        print(tag,' file_name = ',file_name)
        if tag == 'TAN':
            wcs = galsim.FitsWCS(file_name, dir=dir, compression='none', hdu=0)
        else:
            wcs = galsim.FitsWCS(file_name, dir=dir, suppress_warning=True)
        print('FitsWCS is really ',type(wcs))

        if isinstance(wcs, galsim.AffineTransform):
            import warnings
            warnings.warn("None of the existing WCS classes were able to read "+file_name)
        else:
            approx = tag in ['ZPX','ZTF'] and isinstance(wcs, galsim.PyAstWCS)
            do_ref(wcs, ref_list, 'FitsWCS '+tag)
            do_celestial_wcs(wcs, 'FitsWCS '+file_name)
            do_wcs_image(wcs, 'FitsWCS_'+tag, run_slow, approx)

            # Should also be able to build the file just from a fits.read() call, which
            # uses FitsWCS behind the scenes.
            im = galsim.fits.read(file_name, dir=dir)
            do_ref(im.wcs, ref_list, 'WCS from fits.read '+tag, im)

        # Finally, also check that AffineTransform can read the file.
        # We don't really have any accuracy checks here.  This really just checks that the
        # read function doesn't raise an exception.
        hdu, hdu_list, fin = galsim.fits.readFile(file_name, dir)
        affine = galsim.AffineTransform._readHeader(hdu.header)
        galsim.fits.closeHDUList(hdu_list, fin)

    # This does support LINEAR WCS types.
    linear = galsim.FitsWCS('SBProfile_comparison_images/kolmogorov.fits')
    assert isinstance(linear, galsim.OffsetWCS)

    # This file does not have any WCS information in it.
    with assert_warns(galsim.GalSimWarning):
        pixel = galsim.FitsWCS('fits_files/blankimg.fits')
    assert pixel == galsim.PixelScale(1.0)

    # Can suppress the warning if desired
    pixel = galsim.FitsWCS('fits_files/blankimg.fits', suppress_warning=True)
    assert pixel == galsim.PixelScale(1.0)

    assert_raises(TypeError, galsim.FitsWCS)
    assert_raises(TypeError, galsim.FitsWCS, file_name, header='dummy')

    # If some format can't be handled by one of the installed modules,
    # then FitsWCS can end up at the AffineTransform.
    # The easiest way to mock this up is to adjust the fits_wcs_types list.

    with mock.patch('galsim.fitswcs.fits_wcs_types', [galsim.GSFitsWCS,]):
        file_name, ref_list = references['ZPX']
        with assert_warns(galsim.GalSimWarning):
            wcs = galsim.FitsWCS(file_name, dir=dir)
        print('wcs = ',wcs)
        assert isinstance(wcs, galsim.AffineTransform)

        # Can suppress the warning if desired
        wcs = galsim.FitsWCS(file_name, dir=dir, suppress_warning=True)
        assert isinstance(wcs, galsim.AffineTransform)

def check_sphere(ra1, dec1, ra2, dec2, atol=1):
    # Vectorizing CelestialCoord.distanceTo()
    # ra/dec in rad
    # atol in arcsec
    x1 = np.cos(dec1)*np.cos(ra1)
    y1 = np.cos(dec1)*np.sin(ra1)
    z1 = np.sin(dec1)
    x2 = np.cos(dec2)*np.cos(ra2)
    y2 = np.cos(dec2)*np.sin(ra2)
    z2 = np.sin(dec2)
    dsq = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
    dist = 2*np.arcsin(0.5*np.sqrt(dsq))
    w = dsq >= 3.99
    if np.any(w):
        cross = np.cross(np.array([x1, y1, z1])[w], np.array([x2, y2, z2])[w])
        crosssq = cross[0]**2 + cross[1]**2 + cross[2]**2
        dist[w] = np.pi - np.arcsin(np.sqrt(crosssq))
    dist = np.rad2deg(dist)*3600
    np.testing.assert_allclose(dist, 0.0, rtol=0.0, atol=atol)


@timer
def test_fittedsipwcs(run_slow):
    """Test our ability to construct a WCS from x, y, ra, dec tuples
    """
    import astropy.io.fits as fits

    tol = {  #(arcsec, pixels)
        'HPX': (5.0, 0.02),
        'TAN': (5.0, 0.02),
        'TSC': (6.0, 0.02),
        'STG': (7.0, 0.03),
        'ZEA': (10.0, 0.1),
        'ARC': (5.0, 0.02),
        'ZPN': (3000.0, 6),
        'SIP': (1e-6, 0.01),
        'TPV': (1e-6, 1e-3),
        'ZPX': (1e-5, 3e-5),
        'TAN-PV': (1e-6, 2e-4),
        'TAN-FLIP': (1e-6, 1e-6),
        'REGION': (1e-6, 1e-6),
        'TNX': (1e-6, 2e-5),
        'ZTF': (0.1, 0.1),
    }

    dir = 'fits_files'

    if run_slow:
        test_tags = all_tags
    else:
        # For pytest runs, don't bother with the crazy ones.  We really only care about
        # testing that the code works correctly.  That can be done with just a couple of these.
        test_tags = ['TAN', 'TPV', 'SIP', 'TNX']

    nref = 300  # Use more than 256, since that's the block length in C++.
    nfit = 50

    rng = galsim.UniformDeviate(57721)

    for tag in test_tags:
        file_name, ref_list = references[tag]
        header = fits.getheader(os.path.join(dir, file_name))
        wcs = galsim.FitsWCS(header=header, suppress_warning=True)
        if not wcs.isCelestial():
            continue
        print(tag)
        x = np.empty(nref, dtype=float)
        y = np.empty(nref, dtype=float)
        rng.generate(x)
        rng.generate(y)
        x *= header['NAXIS1']  # rough range of reference WCS image coords
        y *= header['NAXIS2']
        ra, dec = wcs.xyToradec(x, y, units='rad')
        print('pixel scale = ',wcs.pixelArea(galsim.PositionD(1,1))**0.5)

        # First check that we can get a fit without distorions, using appropriate
        # Projections.  Should get very high accuracy here.
        if tag in ('STG', 'ZEA', 'ARC', 'TAN'):
            fittedWCS = galsim.FittedSIPWCS(
                x[:nfit], y[:nfit], ra[:nfit], dec[:nfit],
                order=1, wcs_type=tag, center=wcs.center
            )
            ra_test, dec_test = fittedWCS.xyToradec(x, y, units='rad')
            check_sphere(ra, dec, ra_test, dec_test, atol=1e-9)
            x_test, y_test = fittedWCS.radecToxy(ra, dec, units='rad')
            np.testing.assert_allclose(
                np.hstack([x, y]),
                np.hstack([x_test, y_test]),
                rtol=0,
                atol=1e-9
            )
            header = {}
            fittedWCS.writeToFitsHeader(header, galsim.BoundsI(0, 192, 0, 192))
            assert header['CTYPE1'] == 'RA---'+tag

        # Now try adding in SIP distortions.  We'll just use TAN projection from
        # here forward.
        fittedWCS = galsim.FittedSIPWCS(
            x[:nfit], y[:nfit], ra[:nfit], dec[:nfit]
        )
        ra_test, dec_test = fittedWCS.xyToradec(x, y, units='rad')

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
        # dra = np.rad2deg(((ra-ra_test)+np.pi)%(2*np.pi)-np.pi)*3600
        # ddec = np.rad2deg(dec-dec_test)*3600
        # s0 = axes[0].scatter(
        #     np.rad2deg(ra), np.rad2deg(dec),
        #     c=dra, vmin=np.min(dra[:nfit]), vmax=np.max(dra[:nfit])
        # )
        # plt.colorbar(s0, ax=axes[0])
        # s1 = axes[1].scatter(
        #     np.rad2deg(ra), np.rad2deg(dec),
        #     c=ddec, vmin=np.min(ddec[:nfit]), vmax=np.max(ddec[:nfit])
        # )
        # plt.colorbar(s1, ax=axes[1])
        # axes[0].scatter(
        #     np.rad2deg(ra[:nfit]), np.rad2deg(dec[:nfit]),
        #     facecolors='none', edgecolors='k', s=100
        # )
        # axes[1].scatter(
        #     np.rad2deg(ra[:nfit]), np.rad2deg(dec[:nfit]),
        #     facecolors='none', edgecolors='k', s=100
        # )
        # plt.show()

        check_sphere(ra, dec, ra_test, dec_test, atol=tol[tag][0])

        # Check reverse
        x_test, y_test = fittedWCS.radecToxy(ra, dec, units='rad')

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
        # dx = x-x_test
        # dy = y-y_test
        # s0 = axes[0].scatter(
        #     x, y, c=dx, vmin=np.min(dx[:nfit]), vmax=np.max(dx[:nfit])
        # )
        # plt.colorbar(s0, ax=axes[0])
        # s1 = axes[1].scatter(
        #     x, y, c=dy, vmin=np.min(dy[:nfit]), vmax=np.max(dy[:nfit])
        # )
        # plt.colorbar(s1, ax=axes[1])
        # axes[0].scatter(
        #     x[:nfit], y[:nfit], facecolors='none', edgecolors='k', s=100
        # )
        # axes[1].scatter(
        #     x[:nfit], y[:nfit], facecolors='none', edgecolors='k', s=100
        # )
        # plt.show()

        np.testing.assert_allclose(
            np.hstack([x, y]),
            np.hstack([x_test, y_test]),
            rtol=0,
            atol=tol[tag][1]
        )

        # Check x,y being higher dim than 1
        x_test_2d, y_test_2d = fittedWCS.radecToxy(ra.reshape(30,10), dec.reshape(30,10),
                                                   units='rad')
        assert x_test_2d.shape == (30,10)
        assert y_test_2d.shape == (30,10)
        np.testing.assert_array_equal(x_test_2d.ravel(), x_test)
        np.testing.assert_array_equal(y_test_2d.ravel(), y_test)

        ra_test_3d, dec_test_3d = fittedWCS.xyToradec(x.reshape(5,5,12), y.reshape(5,5,12),
                                                      units='rad')
        assert ra_test_3d.shape == (5,5,12)
        assert dec_test_3d.shape == (5,5,12)
        np.testing.assert_array_equal(ra_test_3d.ravel(), ra_test)
        np.testing.assert_array_equal(dec_test_3d.ravel(), dec_test)

        # Try again, but force a different center
        # Increase tolerance since WCS no longer nicely centered on region of
        # interest.
        center = galsim.CelestialCoord(
            galsim.Angle.from_hms(ref_list[0][0]),
            galsim.Angle.from_dms(ref_list[0][1])
        )
        fittedWCS = galsim.FittedSIPWCS(
            x[:nfit], y[:nfit], ra[:nfit], dec[:nfit],
            order=3, center=center
        )
        ra_test, dec_test = fittedWCS.xyToradec(x, y, units='rad')
        check_sphere(ra, dec, ra_test, dec_test, atol=tol[tag][0])
        # Check reverse
        x_test, y_test = fittedWCS.radecToxy(ra, dec, units='rad')
        np.testing.assert_allclose(
            np.hstack([x, y]),
            np.hstack([x_test, y_test]),
            rtol=0,
            atol=tol[tag][1]
        )
        assert fittedWCS.center == center

        # Check illegal values
        with np.testing.assert_raises(galsim.GalSimValueError):
            galsim.FittedSIPWCS(x, y, ra, dec, order=0)

        # Check that we get a standard TAN WCS if order=1
        header = {}
        wcs = galsim.FittedSIPWCS(x, y, ra, dec, order=1)
        wcs.writeToFitsHeader(header, galsim.BoundsI(0, 192, 0, 192))
        assert "A_ORDER" not in header
        assert header['CTYPE1'] == 'RA---TAN'
        # and a TAN-SIP WCS if order > 1
        header = {}
        wcs = galsim.FittedSIPWCS(x, y, ra, dec, order=2)
        wcs.writeToFitsHeader(header, galsim.BoundsI(0, 192, 0, 192))
        assert "A_ORDER" in header
        assert header['CTYPE1'] == 'RA---TAN-SIP'

    # Check that error is raised if not enough stars are supplied.
    # 3 stars is enough for order=1
    wcs = galsim.FittedSIPWCS(x[:3], y[:3], ra[:3], dec[:3], order=1)
    # but 2 is not.
    with np.testing.assert_raises(galsim.GalSimError):
        wcs = galsim.FittedSIPWCS(x[:2], y[:2], ra[:2], dec[:2], order=1)
    # For order=3, there are 2*(3+4) ab coefficiens, 2 crpix, and 4 cd.
    # 2 constraints per star means we need at least 10 stars
    wcs = galsim.FittedSIPWCS(x[:10], y[:10], ra[:10], dec[:10], order=3)
    with np.testing.assert_raises(galsim.GalSimError):
        wcs = galsim.FittedSIPWCS(x[:9], y[:9], ra[:9], dec[:9], order=3)

    if not run_slow:
        return

    # Finally, the ZPN fit isn't very good with a TAN projection.
    # The native projection for ZPN is ARC (postel), and there are radial
    # polynomial corrections up to 7th order in this particular WCS. Our 2D SIP
    # corrections can't reproduce an odd-powered radial correction, (would need
    # a sqrt to do so) so we never really get that close here.  Best ARC-SIP I
    # could find is order=4, which gives an error of ~0.5 degree over an image a
    # few 10s of degrees across.  Not great, but I think this is just an
    # exceptional WCS, and not a problem with the code.
    file_name, ref_list = references['ZPN']
    header = fits.getheader(os.path.join(dir, file_name))
    wcs = galsim.FitsWCS(header=header, suppress_warning=True)
    x = np.empty(nref, dtype=float)
    y = np.empty(nref, dtype=float)
    rng.generate(x)
    rng.generate(y)
    x *= header['NAXIS1']
    y *= header['NAXIS2']
    ra, dec = wcs.xyToradec(x, y, units='rad')
    # Use the same center of projection as the source wcs.
    center = galsim.CelestialCoord(
        ra=header['LONPOLE']*galsim.degrees,
        dec=header['LATPOLE']*galsim.degrees
    )
    fittedWCS = galsim.FittedSIPWCS(
        x[:nfit], y[:nfit], ra[:nfit], dec[:nfit], wcs_type='ARC', order=4,
        center=center
    )
    ra_test, dec_test = fittedWCS.xyToradec(x, y, units='rad')
    check_sphere(ra, dec, ra_test, dec_test, atol=2400)
    x_test, y_test = fittedWCS.radecToxy(ra, dec, units='rad')
    np.testing.assert_allclose(
        np.hstack([x, y]),
        np.hstack([x_test, y_test]),
        rtol=0,
        atol=10.0
    )
    # We can at least confirm we made an ARC-SIP WCS
    header = {}
    fittedWCS.writeToFitsHeader(header, galsim.BoundsI(0, 192, 0, 192))
    assert header['CTYPE1'] == 'RA---ARC-SIP'

@timer
def test_fittedsipwcs_singular():
    # This test is in response to a case in imsim where the FittedSIPWCS could hit a
    # singular matrix error in the solver.

    data = np.load('input/singular.npz')
    x = data['x']
    y = data['y']
    ra = data['ra']
    dec = data['dec']

    wcs = galsim.FittedSIPWCS(x, y, ra, dec, order=3)

    # Mostly the test is that that doesn't throw a SingularMatrix exceptions,
    # but let's test for accuracy.
    print('wcs = ',wcs)

    test_ra, test_dec = wcs.xyToradec(x, y, units='radians')

    np.testing.assert_allclose(test_ra, ra, rtol=1.e-6)
    np.testing.assert_allclose(test_dec, dec, rtol=1.e-6)


@timer
def test_scamp():
    """Test that we can read in a SCamp .head file correctly
    """
    dir = 'fits_files'
    file_name = 'scamp.head'

    wcs = galsim.FitsWCS(file_name, dir=dir, text_file=True)
    print('SCamp FitsWCS is really ',type(wcs))

    # These are just random points that I checked on one machine with this file.
    # For this test, we don't care much about an independent accuracy test, since that should
    # be covered by the other tests.  We are mostly testing that the above syntax works
    # correctly, and that different machines (with different pyfits versions perhaps) end
    # up reading in the same GSFitsWCS object.
    ref_list = [ ('01:04:44.197307', '-03:39:07.588000', 123, 567, 0),
                 ('01:04:36.022067', '-03:39:33.900586', 789, 432, 0) ]
    # This also checks that the dms parser works with : separators, which I'm not sure if
    # I test anywhere else...

    do_ref(wcs, ref_list, 'Scamp FitsWCS')


@timer
def test_compateq():
    """Test that WCS equality vs. compatibility work as physically expected.
    """
    # First check that compatible works properly for two WCS that are actually equal
    assert galsim.wcs.compatible(galsim.PixelScale(0.23), galsim.PixelScale(0.23))
    # Now for a simple offset: check they are compatible but not equal
    assert galsim.wcs.compatible(
        galsim.PixelScale(0.23), galsim.OffsetWCS(0.23, galsim.PositionD(12,34)))
    assert galsim.PixelScale(0.23) != galsim.OffsetWCS(0.23, galsim.PositionD(12,34))
    # Further examples of compatible but != below.
    assert galsim.wcs.compatible(
        galsim.JacobianWCS(0.2,0.01,-0.02,0.23),
        galsim.AffineTransform(0.2,0.01,-0.02,0.23,
                               galsim.PositionD(12,34),
                               galsim.PositionD(45,54))
        )
    assert galsim.JacobianWCS(0.2,0.01,-0.02,0.23) != \
        galsim.AffineTransform(0.2,0.01,-0.02,0.23, galsim.PositionD(12,34), galsim.PositionD(45,54))
    assert galsim.wcs.compatible(
        galsim.PixelScale(0.23),
        galsim.AffineTransform(0.23,0.0,0.0,0.23,
                               galsim.PositionD(12,34),
                               galsim.PositionD(45,54))
        )
    assert galsim.PixelScale(0.23) != \
        galsim.AffineTransform(0.23,0.0,0.0,0.23, galsim.PositionD(12,34), galsim.PositionD(45,54))

    # Finally, some that are truly incompatible.
    assert not galsim.wcs.compatible(galsim.PixelScale(0.23), galsim.PixelScale(0.27))
    assert not galsim.wcs.compatible(
        galsim.PixelScale(0.23), galsim.JacobianWCS(0.23,0.01,-0.02,0.27))
    assert not galsim.wcs.compatible(
        galsim.JacobianWCS(0.2,-0.01,0.02,0.23),
        galsim.AffineTransform(0.2,0.01,-0.02,0.23,
                               galsim.PositionD(12,34),
                               galsim.PositionD(45,54))
        )

    # Non-uniform WCSs are considered compatible if their jacobians are everywhere the same.
    # It (obviously) doesn't actually check this -- it relies on the functional part being
    # the same, and maybe just resetting the origin(s).
    uv1 = galsim.UVFunction('0.2*x + 0.01*x*y - 0.03*y**2',
                            '0.2*y - 0.01*x*y + 0.04*x**2',
                            origin=galsim.PositionD(12,34),
                            world_origin=galsim.PositionD(45,54))
    uv2 = galsim.UVFunction('0.2*x + 0.01*x*y - 0.03*y**2',
                            '0.2*y - 0.01*x*y + 0.04*x**2',
                            origin=galsim.PositionD(23,56),
                            world_origin=galsim.PositionD(11,22))
    uv3 = galsim.UVFunction('0.2*x - 0.01*x*y + 0.03*y**2',
                            '0.2*y + 0.01*x*y - 0.04*x**2',
                            origin=galsim.PositionD(23,56),
                            world_origin=galsim.PositionD(11,22))
    affine = galsim.AffineTransform(0.2,0.01,-0.02,0.23,
                                    galsim.PositionD(12,34),
                                    galsim.PositionD(45,54))
    assert galsim.wcs.compatible(uv1,uv2)
    assert galsim.wcs.compatible(uv2,uv1)
    assert not galsim.wcs.compatible(uv1,uv3)
    assert not galsim.wcs.compatible(uv2,uv3)
    assert not galsim.wcs.compatible(uv3,uv1)
    assert not galsim.wcs.compatible(uv3,uv2)
    assert not galsim.wcs.compatible(uv1,affine)
    assert not galsim.wcs.compatible(uv2,affine)
    assert not galsim.wcs.compatible(uv3,affine)
    assert not galsim.wcs.compatible(affine,uv1)
    assert not galsim.wcs.compatible(affine,uv2)
    assert not galsim.wcs.compatible(affine,uv3)


@timer
def test_coadd():
    """
    This mostly serves as an example of how to treat the WCSs properly when using
    galsim.InterpolatedImages to make a coadd.  Not exactly what this class was designed
    for, but since people have used it that way, it's useful to have a working example.
    """
    # Make three "observations" of an object on images with different WCSs.

    # Three different local jacobaians.  (Even different relative flips to make the differences
    # more obvious than just the relative rotations and distortions.)
    jac = [
        (0.26, 0.05, -0.08, 0.24),  # Normal orientation
        (0.25, -0.02, 0.01, -0.24), # Flipped on y axis (e2 -> -e2)
        (0.03, 0.27, 0.29, 0.07)    # Flipped on x=y axis (e1 -> -e1)
        ]

    # Three different centroid positions
    pos = [
        (123.23, 743.12),
        (772.11, 444.61),
        (921.37, 382.82)
        ]

    # All the same sky position
    sky_pos = galsim.CelestialCoord(5 * galsim.hours, -25 * galsim.degrees)

    # Calculate the appropriate bounds to use
    N = 32
    bounds = [ galsim.BoundsI(int(p[0])-N/2+1, int(p[0])+N/2,
                              int(p[1])-N/2+1, int(p[1])+N/2) for p in pos ]

    # Calculate the offset from the center
    offset = [ galsim.PositionD(*p) - b.true_center for (p,b) in zip(pos,bounds) ]

    # Construct the WCSs
    wcs = [ galsim.TanWCS(affine=galsim.AffineTransform(*j, origin=galsim.PositionD(*p)),
                          world_origin=sky_pos) for (j,p) in zip(jac,pos) ]

    # All the same galaxy profile.  (NB: I'm ignoring the PSF here.)
    gal = galsim.Exponential(half_light_radius=1.3, flux=456).shear(g1=0.4,g2=0.3)

    # Draw the images
    # NB: no_pixel here just so it's easier to check the shear values at the end without having
    #     to account for the dilution by the pixel convolution.
    images = [ gal.drawImage(image=galsim.Image(b, wcs=w), offset=o, method='no_pixel')
               for (b,w,o) in zip(bounds,wcs,offset) ]

    # Measured moments should have very different shears, and accurate centers
    mom0 = images[0].FindAdaptiveMom()
    print('im0: observed_shape = ',mom0.observed_shape,'  center = ',mom0.moments_centroid)
    assert mom0.observed_shape.e1 > 0
    assert mom0.observed_shape.e2 > 0
    np.testing.assert_almost_equal(mom0.moments_centroid.x, pos[0][0], decimal=1)
    np.testing.assert_almost_equal(mom0.moments_centroid.y, pos[0][1], decimal=1)

    mom1 = images[1].FindAdaptiveMom()
    print('im1: observed_shape = ',mom1.observed_shape,'  center = ',mom1.moments_centroid)
    assert mom1.observed_shape.e1 > 0
    assert mom1.observed_shape.e2 < 0
    np.testing.assert_almost_equal(mom1.moments_centroid.x, pos[1][0], decimal=1)
    np.testing.assert_almost_equal(mom1.moments_centroid.y, pos[1][1], decimal=1)

    mom2 = images[2].FindAdaptiveMom()
    print('im2: observed_shape = ',mom2.observed_shape,'  center = ',mom2.moments_centroid)
    assert mom2.observed_shape.e1 < 0
    assert mom2.observed_shape.e2 > 0
    np.testing.assert_almost_equal(mom2.moments_centroid.x, pos[2][0], decimal=1)
    np.testing.assert_almost_equal(mom2.moments_centroid.y, pos[2][1], decimal=1)

    # Make an empty image for the coadd
    coadd_image = galsim.Image(48,48, scale=0.2)

    for p, im in zip(pos,images):
        # Make sure we tell the profile where we think the center of the object is on the image.
        offset = galsim.PositionD(*p) - im.true_center
        interp = galsim.InterpolatedImage(im, offset=offset)
        # Here the no_pixel is required.  The InterpolatedImage already has pixels so we
        # don't want to convovle by a pixel response again.
        interp.drawImage(coadd_image, add_to_image=True, method='no_pixel')

    mom = coadd_image.FindAdaptiveMom()
    print('coadd: observed_shape = ',mom.observed_shape,'  center = ',mom.moments_centroid)
    np.testing.assert_almost_equal(mom.observed_shape.g1, 0.4, decimal=2)
    np.testing.assert_almost_equal(mom.observed_shape.g2, 0.3, decimal=2)
    np.testing.assert_almost_equal(mom.moments_centroid.x, 24.5, decimal=2)
    np.testing.assert_almost_equal(mom.moments_centroid.y, 24.5, decimal=2)


@timer
def test_lowercase():
    # The WCS parsing should be insensitive to the case of the header key values.
    # Matt Becker ran into a problem when his wcs dict had lowercase keys.
    wcs_dict = {
        'simple': True,
        'bitpix': -32,
        'naxis': 2,
        'naxis1': 10000,
        'naxis2': 10000,
        'extend': True,
        'gs_xmin': 1,
        'gs_ymin': 1,
        'gs_wcs': 'GSFitsWCS',
        'ctype1': 'RA---TAN',
        'ctype2': 'DEC--TAN',
        'crpix1': 5000.5,
        'crpix2': 5000.5,
        'cd1_1': -7.305555555556e-05,
        'cd1_2': 0.0,
        'cd2_1': 0.0,
        'cd2_2': 7.305555555556e-05,
        'cunit1': 'deg     ',
        'cunit2': 'deg     ',
        'crval1': 86.176841,
        'crval2': -22.827778}
    wcs = galsim.FitsWCS(header=wcs_dict)
    print('wcs = ',wcs)
    assert isinstance(wcs, galsim.GSFitsWCS)
    print(wcs.local(galsim.PositionD(0,0)))
    np.testing.assert_allclose(wcs.local(galsim.PositionD(0,0)).getMatrix().ravel(),
                               [0.26298, 0.00071,
                                -0.00072, 0.26298], atol=1.e-4)


@timer
def test_int_args(run_slow):
    """Test that integer arguments for various things work correctly.
    """
    # Some of these used to trigger
    # TypeError: Cannot cast ufunc subtract output from dtype('float64') to dtype('int64')
    #            with casting rule 'same_kind'
    # This started with numpy v1.10.

    test_tags = all_tags

    dir = 'fits_files'

    for tag in test_tags:
        file_name, ref_list = references[tag]
        wcs = galsim.FitsWCS(file_name, dir=dir, suppress_warning=True)

        posi = galsim.PositionI(5,6)
        posd = galsim.PositionD(5,6)

        local_wcs1 = wcs.local(posd)
        local_wcs2 = wcs.local(posi)
        assert local_wcs1 == local_wcs2

        wposi = wcs.toWorld(posi)
        posi_roundtrip = wcs.toImage(wposi)
        print('posi_roundtrip = ',posi_roundtrip)
        assert np.isclose(posi_roundtrip.x, posi.x)
        assert np.isclose(posi_roundtrip.y, posi.y)

    # Also check a DES file.
    # Along the way, check issue #1024 where Erin noticed that reading the WCS from the
    # header of a compressed file was spending lots of time decompressing the data, which
    # is unnecessary.
    dir = 'des_data'
    file_name = 'DECam_00158414_01.fits.fz'
    with Profile():
        t0 = time.time()
        wcs = galsim.FitsWCS(file_name, dir=dir)
        t1 = time.time()
    # Before fixing #1024, this took about 0.5 sec.
    # Now it usually takes about 0.04 sec.  Testing at 0.25 seems like a reasonable midpoint.
    print('Time = ',t1-t0)
    if run_slow:
        # Don't include this in regular unit tests, since it's not really something we need
        # to guarantee.  This timing estimate is appropriate for my laptop, but maybe not
        # all systems.  It also fails for pypy on GHA for some reason.
        assert t1-t0 < 0.25

    posi = galsim.PositionI(5,6)
    posd = galsim.PositionD(5,6)

    local_wcs1 = wcs.local(posd)
    local_wcs2 = wcs.local(posi)
    assert local_wcs1 == local_wcs2

    wposi = wcs.toWorld(posi)
    posi_roundtrip = wcs.toImage(wposi)
    print('posi_roundtrip = ',posi_roundtrip)
    assert np.isclose(posi_roundtrip.x, posi.x)
    assert np.isclose(posi_roundtrip.y, posi.y)


@timer
def test_razero(run_slow):
    """Test the makeSkyImage function near ra=0.
    """
    # This test reproduces the problem Chris Walter found when using the LSST WCS backend
    # (with imsim) near ra=0.  The WCS radec function would return number slightly less than
    # 360 and then cross over to just over 0, rather than vary smoothly across the image.

    # Note: GSFitsWCS was never a problem.  It always returns ra values near the value of the
    # center of the image.  So if the center is just below 360, it will return values > 360
    # when crossing the ra=0 line.  Or if the center is just above 0, then it will return
    # negative values when crossing the other way.

    # However, astropy has this "feature" of always wrapping the angles to 0..2pi, so we can
    # use that to test that our makeSkyImage function works properly for wcs functions that
    # do this.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        import astropy.wcs
        import scipy  # AstropyWCS constructor will do this, so check now.

    dir = 'fits_files'
    # This file is based in sipsample.fits, but with the CRVAL1 changed to 0.002322805429
    file_name = 'razero.fits'
    wcs = galsim.AstropyWCS(file_name, dir=dir)

    print('0,0 -> ',wcs.toWorld(galsim.PositionD(0,0)))
    # The makeSkyImage test uses an image with bounds (-135,9,127,288)
    # So these tests just make sure that we'll see the jump in ra values when we test this.
    print('-135,127 -> ',wcs.toWorld(galsim.PositionD(-135,127)))
    print('9,127 -> ',wcs.toWorld(galsim.PositionD(9,127)))
    print('-135,288 -> ',wcs.toWorld(galsim.PositionD(-135,288)))
    print('9,288 -> ',wcs.toWorld(galsim.PositionD(9,288)))
    assert 359 < wcs.toWorld(galsim.PositionD(-135,127)).ra / galsim.degrees < 360
    assert 0 < wcs.toWorld(galsim.PositionD(9,288)).ra / galsim.degrees < 1

    do_celestial_wcs(wcs, 'Astropy file '+file_name)
    do_wcs_image(wcs, 'Astropy near ra=0', run_slow)

    # Test that the local wcs comes out right where ra crosses 0.
    image_pos = wcs.toImage(galsim.CelestialCoord(ra=0.*galsim.degrees, dec=47*galsim.degrees))
    print('0,47 is at ',image_pos)
    local_wcs = wcs.local(image_pos)
    print('local_wcs = ',local_wcs)

    # Should be very close to the same jacobian a spot a few arcsec over.
    image_pos2 = wcs.toImage(galsim.CelestialCoord(ra=3*galsim.arcsec, dec=47*galsim.degrees))
    print('3 arcsec east = ',image_pos2)
    local_wcs2 = wcs.local(image_pos2)
    print('local_wcs2 = ',local_wcs2)
    np.testing.assert_allclose(local_wcs.getMatrix(), local_wcs2.getMatrix(), rtol=1.e-3)

    # This file is similar, but adjusted to be located near the south pole.
    file_name = 'pole.fits'
    wcs = galsim.AstropyWCS(file_name, dir=dir)

    print('0,0 -> ',wcs.toWorld(galsim.PositionD(0,0)))
    # This time we get a whole range of ra values going around the perimeter, so a simple
    # wrapping wouldn't work.
    print('-135,127 -> ',wcs.toWorld(galsim.PositionD(-135,127)))
    print('-63,127 -> ',wcs.toWorld(galsim.PositionD(-63,127)))
    print('9,127 -> ',wcs.toWorld(galsim.PositionD(9,127)))
    print('9,208 -> ',wcs.toWorld(galsim.PositionD(9,208)))
    print('9,288 -> ',wcs.toWorld(galsim.PositionD(9,288)))
    print('-63, 288 -> ',wcs.toWorld(galsim.PositionD(-63,288)))
    print('-135,288 -> ',wcs.toWorld(galsim.PositionD(-135,288)))
    print('-135, 208 -> ',wcs.toWorld(galsim.PositionD(-135,208)))
    # The center is pretty close to the south pole.
    # If it gets any closer, then the precise test at the center of the image fails at 3 d.p.
    # I think we just need to accept that at the pole itself, our finite difference calculation
    # won't be super accurate.  With this, the pole is only a few pixels from the center of the
    # image, and the center pixel passes our test.
    print('-63, 208 -> ',wcs.toWorld(galsim.PositionD(-63,208)))

    do_celestial_wcs(wcs, 'Astropy file '+file_name)
    do_wcs_image(wcs, 'Astropy near pole', run_slow)


if __name__ == "__main__":
    runtests(__file__)
