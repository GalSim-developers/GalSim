; Copyright 2012, 2013 The GalSim developers:
; https://github.com/GalSim-developers
;
; This file is part of GalSim: The modular galaxy image simulation toolkit.
;
; GalSim is free software: you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation, either version 3 of the License, or
; (at your option) any later version.
;
; GalSim is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.
;
; You should have received a copy of the GNU General Public License
; along with GalSim.  If not, see <http://www.gnu.org/licenses/>
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
