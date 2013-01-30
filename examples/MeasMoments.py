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
"""
This script can be run from the command-line to check the properties of images, in particular, the
adaptive moments that come from iteratively determining the best-fit Gaussian for the object at the
center of an image.

It takes arguments from the command line, i.e.
MeasMoments.py image_file [guess_sigma sky_level]
where the ones in brackets are optional:
image_file: A file containing an image for which the moments should be measured
[guess_sigma]: An initial guess for the Gaussian sigma of the image
[sky_level]: If the image contains a non-zero sky level, it must be specified

Results will be printed to stdout:
Status (0 = success), Mxx, Myy, Mxy, e1, e2, number of iterations, total flux in best-fit elliptical
Gaussian, x centroid, y centroid

Here we use the e1 = (Mxx - Myy)/(Mxx + Myy) and e2 = 2*Mxy/(Mxx + Mxy) definition of ellipticity.
"""

import sys
import os
import numpy as np

# This machinery lets us run Python examples even though they aren't positioned properly to find
# galsim as a package in the current directory. 
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# properly handle command-line arguments
numArg = len(sys.argv)-1
if (numArg < 1 or numArg > 3):
    raise RuntimeError("Wrong number of command-line arguments: should be in the range 1...3!")
image_file = sys.argv[1]
guess_sigma = 5.0
sky_level = 0.0
if (numArg >= 2):
    guess_sigma = float(sys.argv[2])
if (numArg == 3):
    sky_level = float(sys.argv[3])

# read in image
image = galsim.fits.read(image_file)
if sky_level > 0.:
    image -= sky_level

# measure adaptive moments
result = galsim.FindAdaptiveMom(image, guess_sig = guess_sigma)

# manipulate results to get moments
e1_val = result.observed_shape.e1
e2_val = result.observed_shape.e2
a_val = (1.0 + e1_val) / (1.0 - e1_val)
b_val = np.sqrt(a_val - (0.5*(1.0+a_val)*e2_val)**2)
mxx = a_val * (result.moments_sigma**2) / b_val
myy = (result.moments_sigma**2) / b_val
mxy = 0.5 * e2_val * (mxx + myy)

# output results
print '%d   %12.6f   %12.6f   %12.6f   %12.6f   %12.6f   %03d    %12.6f   %12.6f %12.6f' % \
        (result.moments_status, mxx, myy, mxy, e1_val, e2_val, result.moments_n_iter,
         result.moments_amp, result.moments_centroid.x-result.image_bounds.getXMin(),
         result.moments_centroid.y-result.image_bounds.getYMin())
