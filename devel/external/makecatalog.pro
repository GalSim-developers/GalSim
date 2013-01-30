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
PRO makecatalog

;; program used by Rachel to generate the catalogs for the first 100
;; real galaxies, as an example use case

; define input catalog, output file
incatw = '/u/rmandelb/svn/simimage/release/catalogs/shera_sdss_catalog_v1.fits'
outcat = './real_galaxy_catalog_example.fits'
galaxy_img_file = 'real_galaxy_images.fits'
psf_img_file = 'real_galaxy_PSF_images.fits'
nuse = 100

; read input
mycat = mrdfits(incatw, 1)
nlines = n_elements(mycat)
print,'Read in ',nlines,' from file ',incatw

; select galaxies
mycatuse = mycat[0:(nuse-1 < nlines)]
ngal = n_elements(mycatuse)
print,'Using ',ngal

mycatuse.ps_wt = mycatuse.ps_wt/max(mycatuse.ps_wt)

indices = lindgen(ngal)

; make new data structure
datastr = {ident: 0L, $
           mag: 0.D, $
           band: '', $
           weight: 0.D, $
           gal_filename: '', $
           PSF_filename: '', $
           gal_hdu: 0L, $
           PSF_hdu: 0L, $
           pixel_scale: 0.D} ; arcsec
data = replicate(datastr, ngal)
           
; populate
data.ident = mycatuse.ident
data.mag = mycatuse.F814W
data.band[*] = 'F814W'
data.weight = mycatuse.ps_wt
data.gal_filename[*] = galaxy_img_file
data.PSF_filename[*] = psf_img_file
data.gal_hdu = indices
data.PSF_hdu = indices
data.pixel_scale = 0.03
           
; write out
print,'Writing to file ',outcat
mwrfits,data,outcat,/create

END
