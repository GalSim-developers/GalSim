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
