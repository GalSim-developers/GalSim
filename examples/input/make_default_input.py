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

"""@brief Example script for generating the examples/input/galsim_default_input.asc default input
catalog file.

Generates a 10 x 10 grid of galaxy postage stamps, each of size 48 pixels.
"""
import os
import numpy as np
# This machinery lets us run Python examples even though they aren't positioned
# properly to find galsim as a package in the current directory.
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..", "..")))
    import galsim

# Some fixed parameter values for the catalogue

STAMPSIZE = 48  # Postage stamp size in pixels
GRIDSIZE = 10   # Number of postage stamps per square side of image

MOFFAT_BETA = 3.5             # } Adopt GREAT08 PSF values for historicity!
MOFFAT_FWHM = 2.85            # }
MOFFAT_E1 = -0.019            # }
MOFFAT_E2 = -0.007            # }
MOFFAT_TRUNCATIONFWHM = 2.    # }

EXPONENTIAL_HLR = 0.82 * MOFFAT_FWHM    # } Again, things are slightly more complex than this for
DEVAUCOULEURS_HLR = 1.59 * MOFFAT_FWHM  # } actual GREAT08 images, but this is a starting example
                                        # } to adopt.

EXPONENTIAL_DEVAUCOULEURS_SIGMA_E = 0.3  # } Approximate the ellipticity distribition as a Gaussian
                                         # } with this sigma.

GAL_CENTROID_SHIFT_RADIUS = 1.0  # Set the radius of centroid shifts (uniform within unit circle).
GAL_CENTROID_SHIFT_RADIUS_SQUARED = GAL_CENTROID_SHIFT_RADIUS**2

RNG_SEED = 1848

NOBJECTS = GRIDSIZE * GRIDSIZE

def make_default_input():

    # Set the PSF catalogue values
    moffat_beta = np.zeros(NOBJECTS) + MOFFAT_BETA
    moffat_fwhm = np.zeros(NOBJECTS) + MOFFAT_FWHM
    moffat_e1 = np.zeros(NOBJECTS) + MOFFAT_E1
    moffat_e2 = np.zeros(NOBJECTS) + MOFFAT_E2
    moffat_trunc = np.zeros(NOBJECTS) + MOFFAT_TRUNCATIONFWHM * MOFFAT_FWHM
    # Then set the exponential disc catalogue fixed values
    exponential_hlr = np.zeros(NOBJECTS) + EXPONENTIAL_HLR
    # Then set the dVc bulge catalogue fixed values
    devaucouleurs_hlr = np.zeros(NOBJECTS) + DEVAUCOULEURS_HLR
    # Then set up the Gaussian RNG for making the ellipticity values
    urng = galsim.UniformDeviate(RNG_SEED)
    edist = galsim.GaussianDeviate(urng, sigma=EXPONENTIAL_DEVAUCOULEURS_SIGMA_E)
    # Slightly hokey way of making vectors of Gaussian deviates, using images... No direct NumPy
    # array-filling with galsim RNGs at the moment.
    #
    # In GREAT08 these galaxy ellipticies were made in rotated pairs to reduce shape noise, but for
    # this illustrative default file we do not do this.
    ime1 = galsim.ImageD(NOBJECTS, 1)
    ime1.addNoise(edist)
    exponential_e1 = ime1.array.flatten()
    ime2 = galsim.ImageD(NOBJECTS, 1)
    ime2.addNoise(edist)
    exponential_e2 = ime2.array.flatten()
    # Make galaxies co-elliptical
    devaucouleurs_e1 = exponential_e1
    devaucouleurs_e2 = exponential_e2

    # Add a centroid shift in drawn uniform randomly from the unit circle around (0., 0.)
    dx = np.empty(NOBJECTS)
    dy = np.empty(NOBJECTS)
    for i in xrange(NOBJECTS):
        # Apply a random centroid shift:
        rsq = 2 * GAL_CENTROID_SHIFT_RADIUS_SQUARED
        while (rsq > GAL_CENTROID_SHIFT_RADIUS_SQUARED):
            dx[i] = (2. * urng() - 1.) * GAL_CENTROID_SHIFT_RADIUS
            dy[i] = (2. * urng() - 1.) * GAL_CENTROID_SHIFT_RADIUS
            rsq = dx[i]**2 + dy[i]**2

    # Then write this to file
    path, modfile = os.path.split(__file__)
    outfile = os.path.join(path, "galsim_default_input.asc")
    # Make a nice header with the default fields described
    header = ("# psf.beta  psf.fwhm  psf.e1  psf.e2  psf.trunc"+
              "  disk.hlr  disk.e1  disk.e2"+
              "  bulge.hlr  bulge.e1  bulge.e2"+
              "  gal.shift.dx  gal.shift.dy \n")
    # Open the file and output the columns in the correct order, row-by-row
    output = open(outfile, "w")
    output.write("#  galsim_default_input.asc : illustrative default input catalog for GalSim\n")
    output.write("#\n")
    output.write(header)
    for i in xrange(NOBJECTS):
        outline = (" %6.2f  %6.2f  %7.3f  %7.3f  %6.2f  %6.2f  %14.7f  %14.7f "+
                   "%6.2f  %14.7f  %14.7f  %14.7f  %14.7f\n") % \
            (moffat_beta[i], moffat_fwhm[i], moffat_e1[i], moffat_e2[i], moffat_trunc[i],
             exponential_hlr[i], exponential_e1[i], exponential_e2[i],
             devaucouleurs_hlr[i], devaucouleurs_e1[i], devaucouleurs_e2[i], dx[i], dy[i])
        output.write(outline)
    output.close()
    
if __name__ == "__main__":
    make_default_input()

