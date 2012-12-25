import galsim
import time
import math
import numpy

imsize = 48
psf_fwhm = 2.5
pixel_scale = 1.0
ntest = 20
sky_level = 1.0e6
gal_flux = 1.0e4
gal_e1 = 0.2
gal_e2 = 0.1
sn_noisy = 25.0 # desired S/N for noisy case
seed = 1234

# for 48x48 image, Gaussian PSF with FWHM=2.5 pix and Gaussian galaxy with same size, make noiseless image
psf = galsim.Gaussian(fwhm = psf_fwhm)
gal = galsim.Gaussian(fwhm = psf_fwhm, flux=gal_flux)
exp_sigma = math.sqrt(2.)*gal.getSigma()
gal.applyShear(e1 = gal_e1, e2 = gal_e2)
pix = galsim.Pixel(pixel_scale)
obj = galsim.Convolve(gal, psf, pix)
epsf = galsim.Convolve(psf, pix)
im_obj = galsim.ImageF(imsize, imsize)
im_epsf = galsim.ImageF(imsize, imsize)
im_obj = obj.draw(image = im_obj, dx=pixel_scale)
im_epsf = epsf.draw(image = im_epsf, dx=pixel_scale)

# get adaptive moments some number of times so we can average over the calls to get an average speed
t1 = time.time()
for i in range(ntest):
    res = im_obj.FindAdaptiveMom(strict=False)
t2 = time.time()
time_per_call = (t2-t1)/ntest
# check results
print "For image size ",imsize," per side, no noise, time to get moments was ",time_per_call," per call"
print "Results for sigma: ",res.moments_sigma
print "Expected (ignoring pixel): ",exp_sigma
print "Error message: ",res.error_message

# for 48x48 image, Gaussian PSF with FWHM=2.5 pix and Gaussian galaxy with same size, make noisy image
# get adaptive moments some number of times so we can average over the calls to get an average speed
sky_level_pix = sky_level * pixel_scale**2
sn_meas = math.sqrt( numpy.sum(im_obj.array**2) / sky_level_pix )
flux_ratio = sn_noisy / sn_meas
im_obj *= flux_ratio
im_obj += sky_level_pix
ud = galsim.UniformDeviate(seed)
tot_time_meas = 0.0
for i in range(ntest):
    tmp_im = im_obj
    tmp_im.addNoise(galsim.CCDNoise(ud))
    tmp_im -= sky_level_pix
    t1 = time.time()
    res = tmp_im.FindAdaptiveMom(strict=False)
    t2 = time.time()
    tot_time_meas += (t2-t1)
time_per_call = tot_time_meas / ntest
# check results
print "For image size ",imsize," per side, with noise, time to get moments was ",time_per_call," per call"
print "Final results for e1, e2, sigma: ",res.moments_sigma
print "Error message: ",res.error_message

# check how it scales if the image size is larger

# do shear estimation




