#!/usr/bin/env python
"""
A script that can be run from the command-line to check the properties of images, parallel to
MeasMoments as compiled from the .cpp
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
# MeasMoments.py image_file [guess_sigma sky_level]
# where the ones in brackets are optional.  This will make the behavior nearly parallel to the
# executable from the C++, so those who are used to its behavior can simply replace it with a call
# to this python script.

numArg = len(sys.argv)-1 # first entry in sys.argv is the name of the script
if (numArg < 1 or numArg > 5):
    raise RuntimeError("Wrong number of command-line arguments: should be in the range 1...5!")

image_file = sys.argv[1]
guess_sigma = 5.0
sky_level = 0.0
if (numArg == 2):
    guess_sigma = float(sys.argv[2])
if (numArg == 3):
    sky_level = float(sys.argv[3])

image = galsim.fits.read(image_file)
if sky_level > 0.:
    image -= sky_level
result = galsim.FindAdaptiveMom(image, guess_sig = guess_sigma)
print '%d   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %03d    %12.6f   %12.6f %12.6f' % (result.moments_status, result.getMxx(), result.getMyy(), result.getMxy(), result.observed_shape.getE1(), result.observed_shape.getE2(), result.moments_n_iter, result.moments_amp, result.moments_centroid.x-result.image_bounds.getXMin(), result.moments_centroid.y-result.image_bounds.getYMin()) 
