# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

"""An example script to evaluate timing for shooting photons through a Gaussian distribution with 
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
NPHOTONS = 500000         # Number of photons per draw
PIXEL_SCALE = 1.0        # arcsec  (size units in input catalog are pixels)
IMAGE_XMAX = 64          # pixels
IMAGE_YMAX = 64          # pixels
GAUSSIAN_SIGMA = 5.

RANDOM_SEED = 3231139901

def time_gaussian_shoot():
    """Shoot photons through a Gaussian profile recording times for comparison between USE_COS_SIN
    method in SBProfile.cpp and the unit circle rejection method.
    """
    logger = logging.getLogger("time_gaussian")

    # Initialize the random number generator we will be using.
    rng = galsim.UniformDeviate(RANDOM_SEED)

    # Build the image for drawing the galaxy into
    image = galsim.ImageF(IMAGE_XMAX, IMAGE_YMAX, PIXEL_SCALE)

    # Start the timer
    t1 = time.time()

    for i in range(NIMAGES):
        # Build the galaxy
        gal = galsim.Gaussian(sigma=GAUSSIAN_SIGMA)
        # Build the image for drawing the galaxy into
        image = galsim.ImageF(IMAGE_XMAX, IMAGE_YMAX)
        # Shoot the galaxy
        gal.drawShoot(image, NPHOTONS) 

    # Get the time
    t2 = time.time()
    logger.info(
        'time_gaussian_shoot: NIMAGES = %d, NPHOTONS = %d, total time = %f sec', NIMAGES, 
        NPHOTONS, t2-t1
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=sys.stdout
    )
    time_gaussian_shoot()
