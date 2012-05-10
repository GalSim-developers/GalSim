#!/usr/bin/env python
"""
A script that can be run from the command-line to carry out PSF correction on images, parallel to
MeasShape as compiled from the .cpp
"""

import sys
import os

# This machinery lets us run Python examples even though they aren't positioned properly to find
# galsim as a package in the current directory. 
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# We want to take arguments from the command-line, i.e.
# MeasShape.py gal_image_file PSF_image_file [sky_var shear_estimator sky_level]
# where the ones in brackets are optional.  This will make the behavior nearly parallel to the
# executable from the C++, so those who are used to its behavior can simply replace it with a call
# to this python script.

numArg = len(sys.argv)-1 # first entry in sys.argv is the name of the script
if (numArg < 2 or numArg > 6):
    raise RuntimeError("Wrong number of command-line arguments: should be in the range 2...6!")

# process args
gal_image_file = sys.argv[1]
PSF_image_file = sys.argv[2]
sky_var = 0.0
shear_estimator = "REGAUSS"
sky_level = 0.0
sky_level_psf = 0.0
if (numArg >= 3):
    sky_var = float(sys.argv[3])
if (numArg >= 4):
    shear_estimator = sys.argv[4]
if (numArg >= 5):
    sky_level = float(sys.argv[5])
if (numArg == 6):
    sky_level_psf = float(sys.argv[6])

# prepare and read in inputs
gal_image = galsim.fits.read(gal_image_file)
if sky_level != 0.:
    gal_image -= sky_level
PSF_image = galsim.fits.read(PSF_image_file)
if sky_level_psf != 0.:
    PSF_image -= sky_level_psf

# measure shape, output results
result = galsim.EstimateShearHSM(gal_image, PSF_image, sky_var = sky_var,
                                 shear_est = shear_estimator)
meas_type = 'e'
if (shear_estimator == "KSB"):
    meas_type = 'g'
print '%d   %12.6f   %12.6f   %c  %12.6f   %12.6f   %12.6f   %12.6f   %12.6f' % \
    (result.correction_status, result.corrected_shape.getE1(), result.corrected_shape.getE2(),
     meas_type, 1.0, result.resolution_factor, result.corrected_shape_err, result.moments_sigma,
     result.moments_amp)
