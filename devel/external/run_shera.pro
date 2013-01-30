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
PRO run_shera

;; script used by Rachel to generate a comparison image from SHERA to
;; use in unit tests of the RealGalaxy base class and associated
;; functions in real.py

totflux = 1000.0/(0.03*0.03)
orig_file = '/scr1/rmandelb/cosmos/Aug2010/Variable/100533.0_150.117017_2.513069_masknoise.fits'
orig_PSFfile = '/scr1/rmandelb/cosmos/Aug2010/Variable/100533.0_150.117017_2.513069.psf.fits'
orig_targPSFfile = '/scr1/rmandelb/cosmos/Aug2010/Variable/100533.0_150.117017_2.513069.sdsspsf.fits'

; put target PSF image in proper format (no soft bias)
targpsf = mrdfits(orig_targPSFfile,0,/dscale)
targpsf = targpsf - 1000
mwrfits,targpsf,'shera_target_PSF.fits',/create

; stick a header with WCS in the real galaxy image, since SHERA
; requires one
img = mrdfits('tmpreal.fits',0)
junk = mrdfits(orig_file, 0, hdr)
mwrfits, img, 'tmpreal_hdr.fits', hdr

imgpsf = mrdfits('tmprealpsf.fits',0)
junk = mrdfits(orig_PSFfile, 0, hdr)
mwrfits, imgpsf, 'tmprealpsf_hdr.fits', hdr

; run shera with the right parameters for this unit test
shera, 'tmpreal_hdr.fits', 'tmprealpsf_hdr.fits', 'shera_target_PSF.fits', 'shera_result.fits', 'junk.fits', totflux, shear1 = 0.06, shear2 = -0.04, idealnoisevar = 0.0, softbias = 0.0, targpixelsize = 0.24, /nopsfmargin

END
