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

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def funcname():
    import inspect
    return inspect.stack()[1][3]

# TODO: Add more tests of the higher level config items.
# So far, I only added two tests related to bugs that David Kirkby found in issues
# #380 and #391.  But clearly more deserve to be added to our test suite.

def test_scattered():
    """Test aspects of building an Scattered image
    """
    import time
    t1 = time.time()

    # Check that the stamps are centered at the correct locations:
    config = {
        'gal' : { 'type' : 'Gaussian', 
                  'half_light_radius' : 2 },
        'image' : { 'type' : 'Scattered',
                    'size' : 8,
                    'stamp_size' : 7,
                    'pixel_scale' : 1,
                    'center' : { 'type' : 'XY', 'x' : 4, 'y' : 1 },
                    'nobjects' : 1 },
    }

    image, _, _, _  = galsim.config.BuildImage(config)
    image.write('junk.fits')

    # Check that stamp_xsize, stamp_ysize, center use the object count, rather than the 
    # image count.
    dx = 1
    hlr = 4
    flux = 10
    size = 40

    config = {
        'gal' : { 'type' : 'Gaussian', 
                  'half_light_radius' : hlr,
                  'flux' : flux 
                },
        'image' : { 'type' : 'Scattered',
                    'size' : size,
                    'stamp_xsize' : { 'type': 'Sequence', 'first' : 20 },
                    'stamp_ysize' : { 'type': 'Sequence', 'first' : 20 },
                    'pixel_scale' : dx,
                    'center' : { 'type' : 'List',
                                 'items' : [ galsim.PositionD(14,14),
                                             galsim.PositionD(33,18),
                                             galsim.PositionD(11,29) ] 
                               },
                    'nobjects' : 3 
                  }
    }

    image, _, _, _  = galsim.config.BuildImage(config)
    image2 = galsim.ImageF(size,size)
    image2.setZero()
    image2.scale = dx
    gal = galsim.Gaussian(half_light_radius=hlr, flux=flux)

    stamp = galsim.ImageF(20,20)
    stamp.scale = dx
    gal.draw(stamp)
    stamp.setCenter(14,14)
    b = image2.bounds & stamp.bounds
    image2[b] += stamp[b]

    stamp = galsim.ImageF(21,21)
    stamp.scale = dx
    gal.draw(stamp)
    stamp.setCenter(33,18)
    b = image2.bounds & stamp.bounds
    image2[b] += stamp[b]

    stamp = galsim.ImageF(22,22)
    stamp.scale = dx
    gal.draw(stamp)
    stamp.setCenter(11,29)
    b = image2.bounds & stamp.bounds
    image2[b] += stamp[b]

    image.write('junk.fits')
    image2.write('junk2.fits')
    np.testing.assert_almost_equal(image.array, image2.array)


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_scattered()


