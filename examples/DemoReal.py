#!/usr/bin/env python
"""
A basic script demonstrating usage of the RealGalaxy functionality.
"""
# NOTE: if catalog and image files for real galaxies are not in examples/data/, this cannot be run
# (and, even if they are there, it must be run while sitting in examples/)!!!

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
real_catalog_filename = 'data/real_galaxy_catalog_example.fits'
image_dir = 'data'
psf_beta = 4
good_psf_fwhm = 0.6 # arcsec
bad_psf_fwhm = 1.5
pixel_scale = 0.2 # arcsec
g1 = 0.05
g2 = 0.00

# read in a random galaxy from the training data
rgc = galsim.RealGalaxyCatalog(real_catalog_filename, image_dir)
real_galaxy = galsim.RealGalaxy(rgc, random = True)
print 'Made real galaxy from catalog index ',real_galaxy.index

# make a target PSF object
good_psf = galsim.Moffat(psf_beta, fwhm = good_psf_fwhm)
bad_psf = galsim.Moffat(psf_beta, fwhm = bad_psf_fwhm)

# simulate some nice ground-based data, e.g., Subaru/CFHT with good seeing; with and without shear
sim_image_good_noshear = galsim.simReal(real_galaxy, good_psf, pixel_scale, rand_rotate = False)
sim_image_good_shear = galsim.simReal(real_galaxy, good_psf, pixel_scale, g1 = g1, g2 = g2, rand_rotate = False)

# simulate some crappy ground-based data, e.g., a bad night at SDSS; with and without shear
sim_image_bad_noshear = galsim.simReal(real_galaxy, bad_psf, pixel_scale, rand_rotate = False)
sim_image_bad_shear = galsim.simReal(real_galaxy, bad_psf, pixel_scale, g1 = g1, g2 = g2, rand_rotate = False)

# write to files: original galaxy, original PSF, 2 target PSFs, 4 simulated images
# note: will differ each time it is run, because we chose a random image
orig_gal_img = real_galaxy.draw(dx = real_galaxy.pixel_scale)
orig_gal_img.write(os.path.join(image_dir, 'demoreal.orig_gal.fits', clobber = True) 


orig_psf_img = real_galaxy.PSF.draw(dx = real_galaxy.pixel_scale)
orig_psf_img.write(os.path.join(image_dir, 'demoreal.orig_PSF.fits', clobber = True) 

good_psf_img = good_psf.draw(dx = pixel_scale)
good_psf_img.write(os.path.join(image_dir, 'demoreal.good_target_PSF.fits', clobber = True) 

bad_psf_img = bad_psf.draw(dx = pixel_scale)
bad_psf_img.write(os.path.join(image_dir, 'demoreal.bad_target_PSF.fits', clobber = True) 

sim_image_good_noshear.write(os.path.join(image_dir, 'demoreal.good_simulated_image.noshear.fits', clobber = True) 
sim_image_good_shear.write(os.path.join(image_dir, 'demoreal.good_simulated_image.shear.fits', clobber = True) 
sim_image_bad_noshear.write(os.path.join(image_dir, 'demoreal.bad_simulated_image.noshear.fits', clobber = True) 
sim_image_bad_shear.write(os.path.join(image_dir, 'demoreal.bad_simulated_image.shear.fits', clobber = True) 
