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
A simple script based originally on script 4 from MultiObjectDemo.py.  
The point of the test is to which interpolant is the best choice for OpticalPSF.
We draw an image with oversampling = 5 and also with the normal oversampling = 1.5
and compare the rms difference in the resulting images.
The conclusion was that Linear is not very good (rms = 1.1e-5) compared to the rest
which are all around 1.7-1.8e-6.
Technically, Quintic is the best at 1.70e-6, then Cubic at 1.78e-6, then the 
various Lanczos varieties.
"""

import sys
import os
import math
import numpy as np

# This machinery lets us run Python examples even though they aren't positioned
# properly to find galsim as a package in the current directory.
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def Script4():

    pixel_scale = 0.28      # arcsec
    nx = 64
    ny = 64

    psf_fwhm = 0.65         # arcsec

    pix = galsim.Pixel(xw = pixel_scale)

    atmos = galsim.Gaussian(fwhm=psf_fwhm)

    i1d_list = [ "galsim.Linear(tol=1.e-4)",
                 "galsim.Cubic(tol=1.e-4)",
                 "galsim.Quintic(tol=1.e-4)",
                 "galsim.Lanczos(3, conserve_dc=False, tol=1.e-4)",
                 "galsim.Lanczos(5, conserve_dc=False, tol=1.e-4)",
                 "galsim.Lanczos(7, conserve_dc=False, tol=1.e-4)",
                 "galsim.Lanczos(3, conserve_dc=True, tol=1.e-4)",
                 "galsim.Lanczos(5, conserve_dc=True, tol=1.e-4)",
                 "galsim.Lanczos(7, conserve_dc=True, tol=1.e-4)" ]

    for i1d_name in i1d_list:
        print 'i1d = ',i1d_name
        # A workaround for the fact that the interpolants don't have good __repr__'s yet.
        i1d = eval(i1d_name)
        #print 'i1d = ',i1d

        # Make the PSF profile:
        i2d = galsim.InterpolantXY(i1d)
        optics = galsim.OpticalPSF(
                interpolantxy = i2d,
                lam_over_diam = 0.6 * psf_fwhm,
                obscuration = 0.4,
                defocus = 0.1,
                astig1 = 0.3, astig2 = -0.2,
                coma1 = 0.2, coma2 = 0.1,
                spher = -0.3) 
        psf1 = galsim.Convolve([atmos,optics])
        #print 'Made psf1'

        # Also make the same thing using oversampling = 5
        optics = galsim.OpticalPSF(
                interpolantxy = i2d,
                oversampling = 5,
                lam_over_diam = 0.6 * psf_fwhm,
                obscuration = 0.4,
                defocus = 0.1,
                astig1 = 0.3, astig2 = -0.2,
                coma1 = 0.2, coma2 = 0.1,
                spher = -0.3) 
        psf2 = galsim.Convolve([atmos,optics])
        #print 'Made psf2'

        # build final profile
        epsf1 = galsim.Convolve([psf1, pix])
        epsf2 = galsim.Convolve([psf2, pix])
        #print 'Made epsf1, epsf2'

        # Create the large, double width output image
        image1 = galsim.ImageF(nx,ny, scale=pixel_scale)
        image2 = galsim.ImageF(nx,ny, scale=pixel_scale)
        #print 'Made images'

        # Draw the profile
        epsf1.draw(image1)
        epsf2.draw(image2)
        #print 'Done drawing images for i1d = ',i1d

        print 'rms diff = ',math.sqrt(np.mean((image1.array - image2.array)**2))


if __name__ == "__main__":
    Script4()
