#!/usr/bin/env python

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

"""An example script to evaluate timing for shooting photons through a Exponential distribution with
the GalSim library.
"""

import sys
import logging
import time

# This machinery lets us run Python examples even though they aren't positioned
# properly to find galsim as a package in the current directory.
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

NIMAGES = 100
NPHOTONS = 500000        # Number of photons per draw
PIXEL_SCALE = 1.0        # arcsec  (size units in input catalog are pixels)
IMAGE_XMAX = 64          # pixels
IMAGE_YMAX = 64          # pixels
EXPONENTIAL_SCALE_RADIUS = 5.

RANDOM_SEED = 3231139901

def time_exponential_shoot():
    """Shoot photons through an Exponential profile recording times for comparison between 
    USE_COS_SIN  method in SBProfile.cpp and the unit circle rejection method, and for comparing
    the customized radius sampling method against the use of the OneDimensionalDeviate with a
    Sersic(n=1) profile.
    """
    logger = logging.getLogger("time_exponential")

    # Initialize the random number generator we will be using.
    rng = galsim.UniformDeviate(RANDOM_SEED)

    # Build the image for drawing the galaxy into
    image = galsim.ImageF(IMAGE_XMAX, IMAGE_YMAX, PIXEL_SCALE)

    # Start the timer
    t1 = time.time()

    for i in range(NIMAGES):
        # Build the galaxy
        gal = galsim.Exponential(scale_radius=EXPONENTIAL_SCALE_RADIUS)
        # Build the image for drawing the galaxy into
        image = galsim.ImageF(IMAGE_XMAX, IMAGE_YMAX)
        # Shoot the galaxy
        gal.drawShoot(image, NPHOTONS) 

    # Get the time
    t2 = time.time()
    logger.info(
        'time_exponential_shoot: NIMAGES = %d, NPHOTONS = %d, total time = %f sec', NIMAGES, 
        NPHOTONS, t2-t1
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=sys.stdout
    )
    time_exponential_shoot()
