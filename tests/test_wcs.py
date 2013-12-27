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

def do_wcs_pos(wcs, ufunc, vfunc, name):

    # Check that (x,y) -> (u,v) and converse work correctly
    if 'local' in name:
        # If the "local" is really a non-local WCS which has been localized, then we cannot
        # count on the far positions to be sufficiently accurate. Just use near positions.
        x_list = near_x_list
        y_list = near_y_list
    else:
        x_list = all_x_list
        y_list = all_y_list
    u_list = [ ufunc(x,y) for x,y in zip(x_list, y_list) ]
    v_list = [ vfunc(x,y) for x,y in zip(x_list, y_list) ]

    for x,y,u,v in zip(x_list, y_list, u_list, v_list):
        print 'x,y = ',x,y
        print 'u,v = ',u,v
        image_pos = galsim.PositionD(x,y)
        world_pos = galsim.PositionD(u,v)
        print 'image_pos = ',image_pos
        print 'world_pos = ',world_pos

        # The transformations are not guaranteed to be implemented in both directions,
        # so guard against NotImplementedError being raised:
        try:
            print 'toWorld(image_pos) = ',wcs.toWorld(image_pos)
            np.testing.assert_almost_equal(
                    world_pos.x, wcs.toWorld(image_pos).x, digits,
                    err_msg='wcs.toWorld returned wrong world position for '+name)
            np.testing.assert_almost_equal(
                    world_pos.y, wcs.toWorld(image_pos).y, digits,
                    err_msg='wcs.toWorld returned wrong world position for '+name)
        except NotImplementedError:
            pass
        try:
            print 'toImage(world_pos) = ',wcs.toImage(world_pos)
            np.testing.assert_almost_equal(
                    image_pos.x, wcs.toImage(world_pos).x, digits,
                    err_msg='wcs.toImage returned wrong image position for '+name)
            np.testing.assert_almost_equal(
                    image_pos.y, wcs.toImage(world_pos).y, digits,
                    err_msg='wcs.toImage returned wrong image position for '+name)
        except NotImplementedError:
            pass

    # The last item in list should also work as a PositionI
    image_pos = galsim.PositionI(x,y)
    np.testing.assert_almost_equal(
            world_pos.x, wcs.toWorld(image_pos).x, digits,
            err_msg='wcs.toWorld gave different value with PositionI image_pos for '+name)
    np.testing.assert_almost_equal(
            world_pos.y, wcs.toWorld(image_pos).y, digits,
            err_msg='wcs.toWorld gave different value with PositionI image_pos for '+name)



def do_local_wcs(wcs, ufunc, vfunc, name):
    # I would call this do_local_wcs_tests, but nosetests takes any function with test 
    # _anywhere_ in the name an tries to run it.  So make sure the name doesn't 
    # have 'test' in it.

    print 'Start testing local WCS '+name

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
                    world_profile.xValue(world_pos), digits,
                    err_msg='xValue for image_profile and world_profile differ for '+name)
            np.testing.assert_almost_equal(
                    image_profile.xValue(image_pos),
                    image_profile2.xValue(image_pos), digits,
                    err_msg='image_profile not equivalent after round trip through world for '+name)
            np.testing.assert_almost_equal(
                    world_profile.xValue(world_pos),
                    world_profile2.xValue(world_pos), digits,
                    err_msg='world_profile not equivalent after round trip through image for '+name)

        # The last item in list should also work as a PositionI
        image_pos = galsim.PositionI(x,y)
        np.testing.assert_almost_equal(
                pixel_area, wcs.pixelArea(image_pos=image_pos), digits,
                err_msg='pixelArea gave different result for PositionI image_pos for '+name)
        np.testing.assert_almost_equal(
                image_profile.xValue(image_pos) / pixel_area,
                world_profile.xValue(world_pos), digits,
                err_msg='xValue for image_profile gave different result for PositionI for '+name)
        np.testing.assert_almost_equal(
                image_profile.xValue(image_pos),
                image_profile2.xValue(image_pos), digits,
                err_msg='round trip xValue gave different result for PositionI for '+name)

        # Test drawing the profile on an image with the given wcs
        world_profile.draw(im1)
        image_profile.draw(im2)
        np.testing.assert_array_almost_equal(
                im1.array, im2.array, digits,
                err_msg='world_profile and image_profile were different when drawn for '+name)



def do_nonlocal_wcs(wcs, ufunc, vfunc, name):

    print 'Start testing non-local WCS '+name

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
        print 'x0,y0 = ',x0,y0
        print 'u0,v0 = ',u0,v0
        local_ufunc = lambda x,y: ufunc(x+x0,y+y0) - u0
        local_vfunc = lambda x,y: vfunc(x+x0,y+y0) - v0
        image_pos = galsim.PositionD(x0,y0)
        world_pos = galsim.PositionD(u0,v0)
        # The local call is not guaranteed to be implemented with both image_pos and 
        # world_pos.  So guard against NotImplementedError
        try:
            do_local_wcs(wcs.local(image_pos=image_pos), local_ufunc, local_vfunc,
                                name + '.local(image_pos)')
            do_local_wcs(wcs.localAffine(image_pos=image_pos), local_ufunc, local_vfunc,
                                name + '.localAffine(image_pos)')
        except NotImplementedError:
            pass
        try:
            do_local_wcs(wcs.local(world_pos=world_pos), local_ufunc, local_vfunc,
                                name + '.local(world_pos)')
            do_local_wcs(wcs.localAffine(world_pos=world_pos), local_ufunc, local_vfunc,
                                name + '.localAffine(world_pos)')
        except NotImplementedError:
            pass

        # Test drawing the profile on an image with the given wcs
        ix0 = int(x0)
        iy0 = int(y0)
        dx = x0 = ix0
        dy = y0 = iy0
        b = galsim.BoundsI(ix0-31, ix0+32, iy0-31, iy0+32)
        im1 = full_im1[b]
        im2 = full_im2[b]

        for world_profile in profiles:
            # The toImage call is not guaranteed to be implemented with both image_pos and 
            # world_pos.  So guard against NotImplementedError
            try:
                image_profile = wcs.toImage(world_profile, image_pos=image_pos)

                world_profile.draw(im1, offset=(dx,dy))
                image_profile.draw(im2, offset=(dx,dy))
                np.testing.assert_array_almost_equal(
                        im1.array, im2.array, digits,
                        err_msg='world_profile and image_profile differed when drawn for '+name)
            except NotImplementedError:
                pass

            try:
                image_profile = wcs.toImage(world_profile, world_pos=world_pos)

                world_profile.draw(im1, offset=(dx,dy))
                image_profile.draw(im2, offset=(dx,dy))
                np.testing.assert_array_almost_equal(
                        im1.array, im2.array, digits,
                        err_msg='world_profile and image_profile differed when drawn for '+name)
            except NotImplementedError:
                pass



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
    do_local_wcs(wcs, ufunc, vfunc, 'PixelScale')

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
    do_local_wcs(wcs, ufunc, vfunc, 'ShearWCS')

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
    do_local_wcs(wcs, ufunc, vfunc, 'Local AffineTransform 1')

    # Add an image origin offset
    x0 = 1
    y0 = 1
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, image_origin=galsim.PositionD(x0,y0))
    ufunc = lambda x,y: dudx*(x-x0) + dudy*(y-y0)
    vfunc = lambda x,y: dvdx*(x-x0) + dvdy*(y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Non-local AffineTransform 1')

    # Next one with a flip and significant rotation and a large (u,v) offset
    dudy = 0.1432
    dudx = 0.2342
    dvdy = 0.2391
    dvdx = 0.1409

    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y

    print 'AffineTransform: ',dudx, dudy, dvdx, dvdy
    do_local_wcs(wcs, ufunc, vfunc, 'AffineTransform 2')

    # Add a world origin offset
    u0 = 124.3
    v0 = -141.9
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, world_origin=galsim.PositionD(u0,v0))
    ufunc = lambda x,y: dudx*x + dudy*y + u0
    vfunc = lambda x,y: dvdx*x + dvdy*y + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Non-local AffineTransform 2')

    # Finally a really crazy one that isn't remotely regular
    dudy = -0.1432
    dudx = 0.2342
    dvdy = -0.3013
    dvdx = 0.0924

    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy)
    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y

    print 'AffineTransform: ',dudx, dudy, dvdx, dvdy
    do_local_wcs(wcs, ufunc, vfunc, 'AffineTransform 3')

    # Add both kinds of offsets
    x0 = -3
    y0 = 104
    u0 = 1423.9
    v0 = 8242.7
    wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, 
                                 image_origin=galsim.PositionI(x0,y0),
                                 world_origin=galsim.PositionD(u0,v0))
    ufunc = lambda x,y: dudx*(x-x0) + dudy*(y-y0) + u0
    vfunc = lambda x,y: dvdx*(x-x0) + dvdy*(y-y0) + v0
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Non-local AffineTransform 3')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def radial_u(x, y):
    """A cubic radial function used for a u(x,y) function """
    # Note: This is designed to be smooth enough that the local approximates are accurate
    # to 5 decimal places when we do the local tests.
    #
    # We will use a functional form of rho/r0 = r/r0 + a (r/r0)^3 
    # To be accurate to < 1.e-6 at an offset of 7 pixels at r = 700/2000, we need:
    #     rho(r+dr) - rho(r) - drho/dr(r) * dr < 1.e-6
    #     1/2 d2(rho)/dr^2 * dr^2  < 1.e-6
    #     rho = r + a/r0^2 r^3 
    #     rho' = 1 + 3a/r0^2 r^2 
    #     rho'' = 6a/r0^2 r 
    #     1/2 6a / 2000^2 * 700 * 7^2 < 1.e-6 
    #     |a| < 3.8e-5

    import numpy
    r0 = 2000.  # scale factor for function
    r = numpy.sqrt(x*x+y*y) / r0
    a = 2.3e-5
    rho = r + a * r**3 
    return x * rho / r

def radial_v(x, y):
    """A radial function used for a u(x,y) function """
    import numpy
    r0 = 2000.  # scale factor for function
    r = numpy.sqrt(x*x+y*y) / r0
    a = 2.3e-5
    rho = r + a * r**3 
    return y * rho / r

class Cubic(object):
    """A class that can act as a function, implementing a cubic radial function.  """
    def __init__(self, a, r0, whichuv): 
        self._a = a
        self._r0 = r0
        self._uv = whichuv

    def __call__(self, x, y): 
        import numpy
        r = numpy.sqrt(x*x+y*y) / self._r0
        rho = r + self._a * r**3 
        if self._uv == 'u':
            return x * rho / r
        else:
            return y * rho / r


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
    do_local_wcs(wcs, ufunc, vfunc, 'UVFunction like PixelScale')
 
    # 2. Like ShearWCS
    scale = 0.23
    g1 = 0.14
    g2 = -0.37
    factor = 1./np.sqrt(1.-g1*g1-g2*g2)
    ufunc = lambda x,y: (x + g1*x + g2*y) * scale * factor
    vfunc = lambda x,y: (y - g1*y + g2*x) * scale * factor
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_local_wcs(wcs, ufunc, vfunc, 'UVFunction like ShearWCS')

    # 3. Like an AffineTransform
    dudy = 0.1432
    dudx = 0.2342
    dvdy = 0.2391
    dvdx = 0.1409

    ufunc = lambda x,y: dudx*x + dudy*y
    vfunc = lambda x,y: dvdx*x + dvdy*y
    wcs = galsim.UVFunction(ufunc, vfunc)
    do_local_wcs(wcs, ufunc, vfunc, 'UVFunction like AffineTransform')

    # 4. Next some UVFunctions with non-trivial offsets
    x0 = 1.3
    y0 = -0.9
    u0 = 124.3
    v0 = -141.9
    image_origin = galsim.PositionD(x0,y0)
    world_origin = galsim.PositionD(u0,v0)
    wcs = wcs.atOrigin(image_origin, world_origin)
    ufunc2 = lambda x,y: dudx*(x-x0) + dudy*(y-y0) + u0
    vfunc2 = lambda x,y: dvdx*(x-x0) + dvdy*(y-y0) + v0
    do_nonlocal_wcs(wcs, ufunc2, vfunc2, 'UVFunction with atOrigin')
    wcs = galsim.UVFunction(ufunc, vfunc, image_origin, world_origin)
    do_nonlocal_wcs(wcs, ufunc2, vfunc2, 'UVFunction with explicit origins')
    wcs = galsim.UVFunction(ufunc2, vfunc2)
    do_nonlocal_wcs(wcs, ufunc2, vfunc2, 'UVFunction with origins in funcs')

    # 5. Now some non-trivial 3rd order radial function.
    global digits
    orig_digits = digits
    # This is only designed to be accurate to 5 digits, rather than usual 6.
    digits = 5
    x0 = 1024
    y0 = 1024
    wcs = galsim.UVFunction(radial_u, radial_v, image_origin=galsim.PositionI(x0,y0))
    ufunc = lambda x,y: radial_u(x-x0, y-y0)
    vfunc = lambda x,y: radial_v(x-x0, y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Cubic radial UVFunction')

    # Repeat with a function object rather than a regular function.
    # Use a different `a` parameter for u and v to make things more interesting.
    cubic_u = Cubic(2.9e-5, 2000., 'u')
    cubic_v = Cubic(-3.7e-5, 2000., 'v')
    wcs = galsim.UVFunction(cubic_u, cubic_v, image_origin=galsim.PositionI(x0,y0))
    ufunc = lambda x,y: cubic_u(x-x0, y-y0)
    vfunc = lambda x,y: cubic_v(x-x0, y-y0)
    do_nonlocal_wcs(wcs, ufunc, vfunc, 'Cubic object UVFunction')
    # Return digits to original value
    digits = orig_digits

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

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

    # The 1904_66*.fits files were downloaded from the web site:
    # 
    # http://www.atnf.csiro.au/people/mcalabre/WCS/example_data.html
    # 
    # From that page: "The example maps and spectra are offered mainly as test
    # material for software that deals with FITS WCS."
    #
    # I picked 4 that looked rather different in ds9, but there are a bunch more
    # on that web page as well.  In particular, I included ZPN, since that uses
    # PV values, so it is a bit different from the others.
    # 
    # The sipsample.fits file was downloaded from the web site:
    #
    # http://fits.gsfc.nasa.gov/registry/sip.html
    #
    # It's important to have at least one file that has some non-trivial telescope
    # distortion term, so this seemed a good choice.
    
    file_names = [ '1904-66_HPX.fits',
                   '1904-66_TAN.fits',
                   '1904-66_TSC.fits',
                   '1904-66_ZPN.fits',
                   'sipsample.fits' ]

    # For each file, I use ds9 to pick out two reference points.
    # For the 1904-66 files, the two reference points are the brightest pixels in the same two 
    # stars.  They don't have exactly the same ra, dec, due to the finite pixel size, but they
    # are close.  The x,y values though are rather different.
    # For sipsample, it is just two bright spots on opposite sides of the galaxy.
    references = [
        # HPX
        [ ('193919.953', '-634341.90', 113.75, 180, 13.5996),
          ('181936.455', '-634708.64', 143.75, 30, 11.4959) ],
        # TAN
        [ ('193934.162', '-634342.98', 116.75, 178, 13.4363),
          ('181917.969', '-634815.69', 153.25, 35, 11.4444) ],
        # TSC
        [ ('193944.339', '-634210.26', 112.75, 161, 12.4841),
          ('181906.848', '-635007.10', 140.75, 48, 11.6595) ],
        # ZPN
        [ ('193927.305', '-634747.56', 94.875, 151, 12.8477),
          ('181924.624', '-635047.54', 121.875, 48, 11.0143) ],
        # SIP
        [ ('133001.463', '471251.69', 241.875, 75, 12.2444 ),
          ('132943.737', '470913.78', 11.875, 106, 5.30282 ) ]
    ]

    dir = 'fits_files'
    for file_name, ref_list in zip(file_names, references):
        print 'file_name = ',file_name
        print 'ref_list = ',ref_list
        wcs = galsim.AstropyWCS(file_name, dir=dir)
        print 'wcs = ',wcs

        for ref in ref_list:
            ra = galsim.HMS_Angle(ref[0])
            dec = galsim.DMS_Angle(ref[1])
            x = ref[2]
            y = ref[3]
            val = ref[4]
            print ra,dec,x,y,val

            # Check image -> world
            ref_coord = galsim.CelestialCoord(ra,dec)
            coord = wcs.toWorld(galsim.PositionD(x,y))
            print 'ra = ', ra / galsim.degrees, coord.ra / galsim.degrees
            print 'dec = ', dec / galsim.degrees, coord.dec / galsim.degrees
            print 'dist = ', ref_coord.distanceTo(coord) / galsim.arcsec,' arcsec'
            # The conversions should be accurate to at least 1.e-2 pixels.
            scale = wcs.minLinearScale(galsim.PositionD(x,y))
            print 'scale = ',scale
            np.testing.assert_almost_equal(ref_coord.distanceTo(coord)/galsim.arcsec/scale, 0, 2)

            # Check world -> image
            pos = wcs.toImage(galsim.CelestialCoord(ra,dec))
            print 'x = ', x, pos.x
            print 'y = ', y, pos.y
            np.testing.assert_almost_equal(x, pos.x, 2)
            np.testing.assert_almost_equal(y, pos.y, 2)

        #do_celestial_wcs(wcs, 'Astropy file '+file_name)

if __name__ == "__main__":
    test_astropywcs()
    test_pixelscale()
    test_shearwcs()
    test_affinetransform()
    test_uvfunction()

