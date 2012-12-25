import galsim
import numpy as np

# make a galsim Gaussian image, get moments
imsize = 513
g_sigma = 4.
pix_scale = 0.2
s_hlr = 2.0
g = galsim.Gaussian(sigma=g_sigma)
g_im = galsim.ImageF(imsize, imsize)
g_im = g.draw(image=g_im, dx=pix_scale)
res = g_im.FindAdaptiveMom()
print "Results for Gaussian without masking:"
print "e1, e2, sigma: ",res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma

# now mask center pixel, make sure moments are same
mask_im = galsim.ImageI(g_im.bounds)+1
cent_pix = int(round(0.5*(g_im.xmax+g_im.xmin)))
mask_im.setValue(cent_pix, cent_pix, 0)
res = g_im.FindAdaptiveMom(object_mask_image = mask_im)
print "\nResults after masking central pixel at ",cent_pix
print "e1, e2, sigma: ",res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma

# now mask central 9x9 pixels, make sure moments are same
mask_im.setValue(cent_pix-1, cent_pix-1, 0)
mask_im.setValue(cent_pix-1, cent_pix, 0)
mask_im.setValue(cent_pix-1, cent_pix+1, 0)
mask_im.setValue(cent_pix, cent_pix-1, 0)
mask_im.setValue(cent_pix, cent_pix+1, 0)
mask_im.setValue(cent_pix+1, cent_pix-1, 0)
mask_im.setValue(cent_pix+1, cent_pix, 0)
mask_im.setValue(cent_pix+1, cent_pix+1, 0)
res = g_im.FindAdaptiveMom(object_mask_image = mask_im)
print "\nResults after masking central 9x9 pixels:"
print "e1, e2, sigma: ",res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma

mask_im = galsim.ImageI(g_im.bounds)+1
mask_im.setValue(cent_pix-10, cent_pix, 0)
mask_im.setValue(cent_pix+10, cent_pix, 0)
mask_im.setValue(cent_pix, cent_pix-10, 0)
mask_im.setValue(cent_pix, cent_pix+10, 0)
res = g_im.FindAdaptiveMom(object_mask_image=mask_im)
print "\nResults after masking 10 pixels from center in 4 directions:"
print "e1, e2, sigma: ",res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma

# make deVauc convolved with Pixel, get moments
s = galsim.Sersic(4., half_light_radius = s_hlr)
p = galsim.Pixel(pix_scale)
s_im = galsim.ImageF(imsize, imsize)
obj = galsim.Convolve(s, p)
s_im = obj.draw(image=s_im, dx=pix_scale)
res = s_im.FindAdaptiveMom()
print "\nResults for deVauc without masking:"
print "e1, e2, sigma: ",res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma

# mask some pixels near center, make sure moments reflect wings more than core
# now mask central 9x9 pixels, make sure moments are same
mask_im = galsim.ImageI(s_im.bounds)+1
mask_im.setValue(cent_pix-1, cent_pix-1, 0)
mask_im.setValue(cent_pix-1, cent_pix, 0)
mask_im.setValue(cent_pix-1, cent_pix+1, 0)
mask_im.setValue(cent_pix, cent_pix-1, 0)
mask_im.setValue(cent_pix, cent_pix, 0)
mask_im.setValue(cent_pix, cent_pix+1, 0)
mask_im.setValue(cent_pix+1, cent_pix-1, 0)
mask_im.setValue(cent_pix+1, cent_pix, 0)
mask_im.setValue(cent_pix+1, cent_pix+1, 0)
res = s_im.FindAdaptiveMom(object_mask_image = mask_im)
print "\nResults for deVauc after masking central 9x9 pixels:"
print "e1, e2, sigma: ",res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma

# mask some pixels near edge, make sure moments reflect core more than wings
mask_im = galsim.ImageI(s_im.bounds)
square_size = 9
lower_marg = int(0.5*square_size)
upper_marg = lower_marg + 1
for ind1 in range(cent_pix-lower_marg, cent_pix+upper_marg):
    for ind2 in range(cent_pix-lower_marg, cent_pix+upper_marg):
        mask_im.setValue(ind1, ind2, 1)
s_im.write("test_sersic_im.fits")
mask_im.write("test_mask_im.fits")
res = s_im.FindAdaptiveMom(object_mask_image = mask_im)
print "\nResults for deVauc after masking out all but central 9x9:"
print "e1, e2, sigma: ",res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma

# do shear estimation with, without masked bits
gal = galsim.Gaussian(fwhm=1.0)
psf = galsim.Gaussian(fwhm=1.0)
pix = galsim.Pixel(0.2)
gal.applyShear(e1=0.3, e2=-0.15)
obj = galsim.Convolve(gal, psf, pix)
epsf = galsim.Convolve(psf, pix)
gal_im = galsim.ImageF(imsize, imsize)
psf_im = galsim.ImageF(imsize, imsize)
gal_im = obj.draw(image=gal_im, dx=0.2)
psf_im = epsf.draw(image=psf_im, dx=0.2)
res = galsim.EstimateShearHSM(gal_im, psf_im)
print "\nResults for shear estimation using regauss without masking: ",res.corrected_shape.e1, res.corrected_shape.e2

mask_im = galsim.ImageI(imsize, imsize)+1
mask_im.setValue(cent_pix-10, cent_pix, 0)
mask_im.setValue(cent_pix+10, cent_pix, 0)
mask_im.setValue(cent_pix, cent_pix-10, 0)
mask_im.setValue(cent_pix, cent_pix+10, 0)
res = galsim.EstimateShearHSM(gal_im, psf_im, mask_im)
print "\nResults for shear estimation using regauss with a few masked pixels: ",res.corrected_shape.e1, res.corrected_shape.e2

# try with masks that are invalid in various ways, make sure exceptions get raised
#mask_im = galsim.ImageI(imsize+3, imsize)
#mask_im = galsim.ImageF(imsize, imsize)+1.0
#mask_im = galsim.ImageI(imsize, imsize)-1
#res = galsim.EstimateShearHSM(gal_im, psf_im, mask_im)
