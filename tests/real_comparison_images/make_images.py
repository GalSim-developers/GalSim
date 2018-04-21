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

# this script was used to generate the idealized Gaussian test cases used in ../test_real.py
import galsim

fake_gal_fwhm = 0.7 # arcsec
fake_gal_shear1 = 0.29 # shear representing intrinsic shape component 1
fake_gal_shear2 = -0.21 # shear representing intrinsic shape component 2; note non-round, to detect
              # possible issues with x<->y or others that might not show up using circular galaxy
fake_gal_flux = 1000.0
fake_gal_orig_PSF_fwhm = 0.1 # arcsec
fake_gal_orig_PSF_shear1 = 0.0
fake_gal_orig_PSF_shear2 = -0.07

pixel_scale = 0.03 # high-resolution data

orig_gal = galsim.Gaussian(flux = fake_gal_flux, fwhm = fake_gal_fwhm)
orig_gal.applyShear(fake_gal_shear1, fake_gal_shear2)
orig_PSF = galsim.Gaussian(flux = 1.0, fwhm = fake_gal_orig_PSF_fwhm)
orig_PSF.applyShear(fake_gal_orig_PSF_shear1, fake_gal_orig_PSF_shear2)
orig_observed = galsim.Convolve(orig_gal, orig_PSF)

obs_image = galsim.ImageF(200, 200)
orig_observed.draw(obs_image, dx = pixel_scale)
obs_PSF_image = galsim.ImageF(30, 30)
orig_PSF.draw(obs_PSF_image, dx = pixel_scale)

obs_image.write('tmp_obs_image.fits', clobber=True)
obs_PSF_image.write('tmp_obs_PSF_image.fits', clobber = True)

