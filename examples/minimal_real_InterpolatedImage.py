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
A basic script demonstrating usage of the RealGalaxy functionality.
"""
# NOTE: if catalog and image files for real galaxies are not in examples/data/, this cannot be run -
# and, even if they are there, it must be run while sitting in examples/ !!!

import sys
import os
import math

# This machinery lets us run Python examples even though they aren't positioned
# properly to find galsim as a package in the current directory.
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# define some variables etc.
real_catalog_filename = 'real_galaxy_catalog_example.fits'
image_dir = 'data'
output_dir = 'output'
good_psf_central_fwhm = 0.6 # arcsec; FWHM of smaller Gaussian in the double Gaussian for good 
                            # seeing
bad_psf_central_fwhm = 1.3 # arcsec; FWHM of smaller Gaussian in the double Gaussian for bad seeing
central_psf_amp = 0.8 # relative contribution of inner Gaussian in the double Gaussian PSF
outer_fwhm_mult = 2.0 # ratio of (outer)/(inner) Gaussian FWHM for double Gaussian PSF
pixel_scale = 0.2 # arcsec
wmult = 1.0 # oversampling to use in intermediate steps of calculations

for i in range(1000):
    # read in a random galaxy from the training data
    rgc = galsim.RealGalaxyCatalog(real_catalog_filename, dir=image_dir)
    real_galaxy = galsim.RealGalaxy(rgc, index=1)
    print 'Fluxes:', real_galaxy.getFlux()

    # make a target PSF object
    good_psf_inner = galsim.Gaussian(flux=central_psf_amp, fwhm = good_psf_central_fwhm)
    good_psf_outer = galsim.Gaussian(flux=1.0-central_psf_amp,
                                 fwhm=outer_fwhm_mult*good_psf_central_fwhm)
    good_psf = good_psf_inner + good_psf_outer

    pixel = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
    good_epsf = galsim.Convolve(good_psf, pixel)

    # simulate some high-quality ground-based data, e.g., Subaru/CFHT with good seeing
    sim_image_good = galsim.simReal(real_galaxy, good_epsf, pixel_scale, rand_rotate=False)

    orig_gal_img = real_galaxy.original_image.draw(dx=pixel_scale)
    orig_gal_img.write(os.path.join(output_dir, 'demoreal.orig_gal.'+str(i)+'.fits'), clobber=True)

    sim_image_good.write(os.path.join(output_dir, 
                         'demoreal.good_simulated_image.noshear.'+str(i)+'fits'),clobber=True)

