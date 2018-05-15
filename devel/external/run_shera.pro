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
