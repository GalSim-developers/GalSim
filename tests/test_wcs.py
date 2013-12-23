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

def do_wcs(wcs, ufunc, vfunc, name):
    # I would call this do_wcs_tests, but nosetests takes any function with test 
    # _anywhere_ in the name an tries to run it.  So make sure the name doesn't 
    # have 'test' in it.

    print 'Start testing '+name

    # Check that (x,y) -> (u,v) and converse work correctly
    x_list = [ 0, 0.242, -1.342, 5 ]
    y_list = [ 0, -0.173, 2.003, 7 ]

    u_list = [ ufunc(x,y) for x,y in zip(x_list,y_list) ]
    v_list = [ vfunc(x,y) for x,y in zip(x_list,y_list) ]

    for x,y,u,v in zip(x_list, y_list, u_list, v_list):
        print 'x,y = ',x,y
        print 'u,v = ',u,v
        image_pos = galsim.PositionD(x,y)
        world_pos = galsim.PositionD(u,v)
        print 'image_pos = ',image_pos
        print 'world_pos = ',world_pos
        print 'toWorld(image_pos) = ',wcs.toWorld(image_pos)
        print 'toImage(world_pos) = ',wcs.toImage(world_pos)

        np.testing.assert_almost_equal(
                world_pos.x, wcs.toWorld(image_pos).x, 6,
                err_msg='wcs.toWorld returned wrong world position for '+name)
        np.testing.assert_almost_equal(
                world_pos.y, wcs.toWorld(image_pos).y, 6,
                err_msg='wcs.toWorld returned wrong world position for '+name)
        np.testing.assert_almost_equal(
                image_pos.x, wcs.toImage(world_pos).x, 6,
                err_msg='wcs.toImage returned wrong image position for '+name)
        np.testing.assert_almost_equal(
                image_pos.y, wcs.toImage(world_pos).y, 6,
                err_msg='wcs.toImage returned wrong image position for '+name)

    # Test the transformation of a GSObject
    # Make a few different profiles to check.  Make sure to include ones that 
    # aren't symmetrical so we don't get fooled by symmetries.
    profiles = []
    prof = galsim.Gaussian(sigma = 1.7, flux = 100)
    profiles.append(prof)
    prof = prof.createSheared(g1=0.3, g2=-0.12)
    profiles.append(prof)
    prof = prof + galsim.Exponential(scale_radius = 1.3, flux = 20).createShifted(-0.1,-0.4)
    profiles.append(prof)

    for world_profile in profiles:
        # The profiles build above are in world coordinates (as usual)
    
        # Convert to image coordinates
        image_profile = wcs.toImage(world_profile)

        # Also check round trip (starting with either one)
        world_profile2 = wcs.toWorld(image_profile)
        image_profile2 = wcs.toImage(world_profile2)

        for x,y,u,v in zip(x_list, y_list, u_list, v_list):
            print 'x,y = ',x,y
            print 'u,v = ',u,v
            image_pos = galsim.PositionD(x,y)
            world_pos = galsim.PositionD(u,v)
            pixel_area = wcs.pixelArea(image_pos=image_pos)
            print 'image xValue(x,y) = ',image_profile.xValue(image_pos)
            print 'with pixelArea = ',image_profile.xValue(image_pos) / pixel_area
            print 'world xValue(x,y) = ',world_profile.xValue(world_pos)

            np.testing.assert_almost_equal(
                    image_profile.xValue(image_pos) / pixel_area,
                    world_profile.xValue(world_pos), 6,
                    err_msg='xValue for image_profile and world_profile differ for '+name)
            np.testing.assert_almost_equal(
                    image_profile.xValue(image_pos),
                    image_profile2.xValue(image_pos), 6,
                    err_msg='image_profile not equivalent after round trip through world for '+name)
            np.testing.assert_almost_equal(
                    world_profile.xValue(world_pos),
                    world_profile2.xValue(world_pos), 6,
                    err_msg='world_profile not equivalent after round trip through image for '+name)


def test_pixelscale():
    """Test the PixelScale class
    """
    import time
    t1 = time.time()

    scale = 0.23
    wcs = galsim.PixelScale(scale)
    ufunc = lambda x,y: x*scale
    vfunc = lambda x,y: y*scale

    print 'PixelScale: scale = ',scale

    # Do generic tests that apply to all WCS types
    do_wcs(wcs, ufunc, vfunc, 'PixelScale')

    # Check localAffine()
    affine = wcs.localAffine()
    np.testing.assert_almost_equal(affine.dudx, scale,
                                   err_msg = 'PixelScale dudx does not match expected value.')
    np.testing.assert_almost_equal(affine.dudy, 0.,
                                   err_msg = 'PixelScale dudy does not match expected value.')
    np.testing.assert_almost_equal(affine.dvdx, 0.,
                                   err_msg = 'PixelScale dvdx does not match expected value.')
    np.testing.assert_almost_equal(affine.dvdy, scale,
                                   err_msg = 'PixelScale dvdy does not match expected value.')
    np.testing.assert_almost_equal(affine.image_origin.x, 0.,
                                   err_msg = 'PixelScale x0 does not match expected value.')
    np.testing.assert_almost_equal(affine.image_origin.y, 0.,
                                   err_msg = 'PixelScale y0 does not match expected value.')
    np.testing.assert_almost_equal(affine.world_origin.x, 0.,
                                   err_msg = 'PixelScale u0 does not match expected value.')
    np.testing.assert_almost_equal(affine.world_origin.y, 0.,
                                   err_msg = 'PixelScale v0 does not match expected value.')

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
    factor = 1./np.sqrt(1.-g1*g1-g2*g2)
    ufunc = lambda x,y: (x + g1*x + g2*y) * scale * factor
    vfunc = lambda x,y: (y - g1*y + g2*x) * scale * factor

    print 'ShearWCS: scale,g1,g2,shear = ',scale,g1,g2,shear

    # Do generic tests that apply to all WCS types
    do_wcs(wcs, ufunc, vfunc, 'ShearWCS')

    # Check localAffine()
    affine = wcs.localAffine()
    np.testing.assert_almost_equal(affine.dudx, (1.+g1) * scale * factor, 
                                   err_msg = 'ShearWCS dudx does not match expected value.')
    np.testing.assert_almost_equal(affine.dudy, g2 * scale * factor, 
                                   err_msg = 'ShearWCS dudy does not match expected value.')
    np.testing.assert_almost_equal(affine.dvdx, g2 * scale * factor, 
                                   err_msg = 'ShearWCS dvdx does not match expected value.')
    np.testing.assert_almost_equal(affine.dvdy, (1.-g1) * scale * factor, 
                                   err_msg = 'ShearWCS dvdy does not match expected value.')
    np.testing.assert_almost_equal(affine.image_origin.x, 0.,
                                   err_msg = 'PixelScale x0 does not match expected value.')
    np.testing.assert_almost_equal(affine.image_origin.y, 0.,
                                   err_msg = 'PixelScale y0 does not match expected value.')
    np.testing.assert_almost_equal(affine.world_origin.x, 0.,
                                   err_msg = 'PixelScale u0 does not match expected value.')
    np.testing.assert_almost_equal(affine.world_origin.y, 0.,
                                   err_msg = 'PixelScale v0 does not match expected value.')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_affinetransform():
    """Test the AffineTransform class
    """
    import time
    t1 = time.time()

    # First a simple tweak on a simple scale factor
    dudx = 0.2342
    dudy = 0.0023
    dvdx = 0.0019
    dvdy = 0.2391

    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y

    print 'AffineTransform: ',dudx, dudy, dvdx, dvdy
    do_wcs(wcs, ufunc, vfunc, 'AffineTransform 1')

    # Next one with a flip and significant rotation and a large (u,v) offset
    dudy = 0.1432
    dudx = 0.2342
    dvdy = 0.2391
    dvdx = 0.1409

    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y

    print 'AffineTransform: ',dudx, dudy, dvdx, dvdy
    do_wcs(wcs, ufunc, vfunc, 'AffineTransform 2')

    # Finally a really crazy one that isn't remotely regular
    dudy = -0.1432
    dudx = 0.2342
    dvdy = -0.3013
    dvdx = 0.0924

    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y

    print 'AffineTransform: ',dudx, dudy, dvdx, dvdy
    do_wcs(wcs, ufunc, vfunc, 'AffineTransform 3')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_pixelscale()
    test_shearwcs()
    test_affinetransform()

