import galsim
import numpy as np
import math

# define the linear size for the images, in the hopes that this is always big enough
# otherwise some images get drawn way too big and take up huge amounts of memory
im_size = 512

psf_fwhm = 0.7 # arcsec
pix_scale = 0.2 # arcsec
sky_level = 1.e6
gal_flux = 1000.0 # initial guess for flux, will refine to get desired S/N

psf = galsim.Kolmogorov(fwhm = psf_fwhm)
pix = galsim.Pixel(pix_scale)
epsf = galsim.Convolve(psf, pix)

# make list of galaxies
gal_list = []

## loop over all 100 real galaxies in example catalog
cat_file_name = 'real_galaxy_catalog_example.fits'
image_dir = 'data' # only works on Rachel's Mac!
real_galaxy_catalog = galsim.RealGalaxyCatalog(cat_file_name, image_dir)
real_galaxy_catalog.preload()
for i in range(len(real_galaxy_catalog.ident)):
    real_gal = galsim.RealGalaxy(real_galaxy_catalog, index=i, flux=gal_flux)
    gal_list.append(real_gal)

n_gal = len(gal_list)
print "Doing all calculations for ",n_gal," galaxies!"

# check intrinsic shape - except not really, let's do a simpler case where we just draw images over
# and over into the same galsim.ImageF()
print "Sanity check of intrinsic shapes of galaxies:"
g_intrinsic = []
im = galsim.ImageF(im_size, im_size)
for gal_ind in range(len(gal_list)):
    gal = gal_list[gal_ind]
    obj = galsim.Convolve(gal, epsf)
    im = obj.draw(image = im, dx = pix_scale)
#    result = galsim.EstimateShearHSM(im, epsf_img, strict=False)
#    g_intrinsic.append(result.corrected_shape.g)
#    print "   Galaxy ",gal_ind," has intrinsic shear ",result.corrected_shape.g

print "Finished loop to get intrinsic shapes."
