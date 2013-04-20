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

    config = {
        'gal' : { 'type' : 'Gaussian', 
                  'half_light_radius' : 2 },
        'image' : { 'type' : 'Scattered',
                    'size' : 8,
                    'stamp_size' : 7,
                    'pixel_scale' : 1,
                    'center' : { 'type' : 'XY', 'x' : 4, 'y' : 1 },
                    'nobjects' : 1 }
    }

    image, _, _, _  = galsim.config.BuildImage(config)
    image.write('junk.fits')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_scattered()


