
#!/usr/bin/env python
"""
Some example scripts to evaluate timing for shooting photons through a Gaussian distribution 
with the GalSim library.
"""

import sys
import os
import math
import numpy
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

NDRAWS = 100
NPHOTONS = 10000         # Number of photons per draw
PIXEL_SCALE = 1.0        # arcsec  (size units in input catalog are pixels)
IMAGE_XMAX = 64          # pixels
IMAGE_YMAX = 64          # pixels
GAUSSIAN_SIGMA = 5.

def time_gaussian():
    """Shoot photons through a Gaussian profile recording times for comparison between USE_COS_SIN
    method in SBProfile.cpp and the unit circle rejection method.
    """
    logger = logging.getLogger("time_gaussian")

    # Initialize the random number generator we will be using.
    rng = galsim.UniformDeviate(random_seed)

    t1 = time.time()

    for i in range(NDRAWS):
        # Build the galaxy
        gal = galsim.Gaussian(sigma=GAUSSIAN_SIGMA)
        # Build the image for drawing the galaxy into
        image = galsim.ImageF(IMAGE_XMAX, IMAGE_YMAX)
        # Shoot the galaxy
        gal.drawShoot(image, NPHOTONS) 





