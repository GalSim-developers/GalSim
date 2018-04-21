; Copyright (c) 2012-2018 by the GalSim developers team on GitHub
; https://github.com/GalSim-developers
;
; This file is part of GalSim: The modular galaxy image simulation toolkit.
; https://github.com/GalSim-developers/GalSim
;
; GalSim is free software: redistribution and use in source and binary forms,
; with or without modification, are permitted provided that the following
; conditions are met:
;
; 1. Redistributions of source code must retain the above copyright notice, this
;    list of conditions, and the disclaimer given in the accompanying LICENSE
;    file.
; 2. Redistributions in binary form must reproduce the above copyright notice,
;    this list of conditions, and the disclaimer given in the documentation
;    and/or other materials provided with the distribution.
;
PRO make_test_images_real

;; Script used by Rachel to put together fake images for unit tests of the
;; RealGalaxy base class and associated functions in real.py

outfile = 'test_images.fits'
fakegfile = 'tmp_obs_image.fits'
fakepfile = 'tmp_obs_PSF_image.fits'

real_image_file = '../data/real_galaxy_images.fits'
real_PSF_file = '../data/real_galaxy_PSF_images.fits'
hdu = 2

; read in real galaxy and PSF images
realg = mrdfits(real_image_file, hdu)
realp = mrdfits(real_PSF_file, hdu)

; read in fake galaxy and PSF images
fakeg = mrdfits(fakegfile, 0)
fakep = mrdfits(fakepfile, 0)

; put in the right order (make new file with first one), then write to
; successive HDUs
mwrfits, realg, outfile, /create
mwrfits, fakeg, outfile
mwrfits, realp, outfile
mwrfits, fakep, outfile

mwrfits, realg, 'tmpreal.fits', /create
mwrfits, realp, 'tmprealpsf.fits', /create

END
