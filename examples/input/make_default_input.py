#!/usr/bin/env python

"""Example script for generating the examples/input/galsim_default_input.asc default input catalog
file, which by default generates a 10 x 10 grid of galaxy postage stamps, each of size 48 pixels.

The galsim_default configuration file is expecting the following columns in this ascii catalogue:

config.input_cat.ascii_fields = ["BoundsI.xMin", "BoundsI.yMin", "BoundsI.xMax", "BoundsI.yMax",
                                 "Moffat.beta", "Moffat.FWHM", "Moffat.e1", "Moffat.e2",
                                 "Moffat.truncationFWHM",
                                 "Exponential.re", "Exponential.e1", "Exponential.e2",
                                 "DeVaucouleurs.re", "DeVaucouleurs.e1", "DeVaucouleurs.e2"]

...and so these will be the columns of examples/input/galsim_default_input.asc
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
MOFFAT_G1 = -0.019            # }
MOFFAT_G2 = -0.007            # }
MOFFAT_TRUNCATIONFWHM = 100.  # } ... well, almost.  This mixes conventions, so just make it large.

EXPONENTIAL_RE = 0.82         # } Again, things are slightly more complex than this for actual
DEVAUCOULEURS_RE = 1.59       # } GREAT08 images, but this is a enough reasonable example to adopt.

EXPONENTIAL_DEVAUCOULEURS_SIGMA_G = 0.3  # } Approximate the ellipticity distribition as a Gaussian
                                         # } with this sigma.
RNG_SEED = 1848

NOBJECTS = GRIDSIZE * GRIDSIZE

def make_default_input():
    # Set the bounds
    bounds_xmin, bounds_ymin = np.meshgrid(np.arange(GRIDSIZE) * STAMPSIZE,
                                           np.arange(GRIDSIZE) * STAMPSIZE)
    bounds_xmax, bounds_ymax = np.meshgrid((np.arange(GRIDSIZE) + 1) * STAMPSIZE - 1,
                                           (np.arange(GRIDSIZE) + 1) * STAMPSIZE - 1)
    bounds_xmin = bounds_xmin.flatten()
    bounds_ymin = bounds_ymin.flatten()
    bounds_xmax = bounds_xmax.flatten()
    bounds_ymax = bounds_ymax.flatten()
    # Then set the PSF catalogue values
    moffat_beta = np.zeros(NOBJECTS) + MOFFAT_BETA
    moffat_fwhm = np.zeros(NOBJECTS) + MOFFAT_FWHM
    moffat_g1 = np.zeros(NOBJECTS) + MOFFAT_G1
    moffat_g2 = np.zeros(NOBJECTS) + MOFFAT_G2
    moffat_truncationfwhm = np.zeros(NOBJECTS) + MOFFAT_TRUNCATIONFWHM
    # Then set the exponential disc catalogue fixed values
    exponential_re = np.zeros(NOBJECTS) + EXPONENTIAL_RE
    # Then set the dVc bulge catalogue fixed values
    devaucouleurs_re = np.zeros(NOBJECTS) + DEVAUCOULEURS_RE
    # Then set up the Gaussian RNG for making the ellipticity values
    gdist = galsim.GaussianDeviate(galsim.UniformDeviate(RNG_SEED),
                                   sigma=EXPONENTIAL_DEVAUCOULEURS_SIGMA_G)
    # Slightly hokey way of making vectors of Gaussian deviates, using images... No direct NumPy
    # array-filling with galsim RNGs at the moment.
    #
    # In GREAT08 these galaxy ellipticies were made in rotated pairs to reduce shape noise, but for
    # this illustrative default file we do not do this.
    ime1 = galsim.ImageD(NOBJECTS, 1)
    ime1.addNoise(gdist)
    exponential_g1 = ime1.array.flatten()
    ime2 = galsim.ImageD(NOBJECTS, 1)
    ime2.addNoise(gdist)
    exponential_g2 = ime2.array.flatten()
    # Make galaxies co-elliptical
    devaucouleurs_g1 = exponential_g1
    devaucouleurs_g2 = exponential_g2

    # Then write this to file
    path, modfile = os.path.split(__file__)
    outfile = os.path.join(path, "galsim_default_input.asc")
    # Make a nice header with the default ascii_fields described
    config = galsim.config.load()
    header = "#  "+("  ".join(config.input_cat.ascii_fields))+"\n"
    # Open the file and output the columns in the correct order, row-by-row
    output = open(outfile, "w")
    output.write("#  galsim_default_input.asc : illustrative default input catalog for GalSim\n")
    output.write("#\n")
    output.write(header)
    for i in xrange(NOBJECTS):
        outline = ("%4d  %4d  %4d  %4d  %6.2f  %6.2f  %7.3f  %7.3f  %6.2f  %6.2f  %14.7f  %14.7f "+
                   "%6.2f  %14.7f  %14.7f\n") % \
            (bounds_xmin[i], bounds_ymin[i], bounds_xmax[i], bounds_ymax[i],
             moffat_beta[i], moffat_fwhm[i], moffat_g1[i], moffat_g2[i], moffat_truncationfwhm[i],
             exponential_re[i], exponential_g1[i], exponential_g2[i],
             devaucouleurs_re[i], devaucouleurs_g1[i], devaucouleurs_g2[i])
        output.write(outline)
    output.close()
    
if __name__ == "__main__":
    make_default_input()

