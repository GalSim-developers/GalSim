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
PRO make_test_catalog

;; Script used by Rachel to make fake catalogs for unit tests (tests/test_real.py) of the
;; RealGalaxy base class and associated functions in real.py

infile = '../data/real_galaxy_catalog_example.fits'
outfile = 'test_catalog.fits'
induse = 2 ; the index of the one to use
imgfile = 'test_images.fits'

; read input catalog from before
cat = mrdfits(infile,1)
catuse = cat[induse]

newcat = replicate(catuse, 2)
newcat[*].gal_filename = imgfile
newcat[*].PSF_filename = imgfile
newcat[0].gal_hdu = 0
newcat[1].gal_hdu = 1
newcat[0].PSF_hdu = 2
newcat[1].PSF_hdu = 3

mwrfits,newcat,outfile,/create

END
