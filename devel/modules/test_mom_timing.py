# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#

import galsim
import time
import math
import numpy

imsize = 48
nmult = 5 # i.e., consider 1*imsize linear scale, 2*imsize linear scale, up to (nmult-1)*imsize linear scale
psf_fwhm = 3.5
pixel_scale = 1.0
ntest = 200
sky_level = 1.0e6
gal_flux = 1.0e4
gal_e1 = 0.2
gal_e2 = 0.1
sn_noisy = 25.0 # desired S/N for noisy case
seed = 1234
save_im = 0

print "Doing all tests with ",ntest," trials"

# for 48x48 image, Gaussian PSF with FWHM=2.5 pix and Gaussian galaxy with same size, make objects
epsf = galsim.Gaussian(fwhm = psf_fwhm) # let's say this is the epsf
gal = galsim.Gaussian(fwhm = psf_fwhm, flux=gal_flux)
exp_sigma = math.sqrt(2.)*gal.getSigma()
gal_hlr = gal.getHalfLightRadius()
gal.applyShear(e1 = gal_e1, e2 = gal_e2)
obj = galsim.Convolve(gal, epsf)

# do tests with noiseless images of various sizes
for sizemult in range(1, nmult):
    this_imsize = imsize*sizemult
    im_obj = galsim.ImageF(this_imsize, this_imsize)
    im_epsf = galsim.ImageF(this_imsize, this_imsize)
    im_obj = obj.draw(image = im_obj, dx=pixel_scale)
    im_epsf = epsf.draw(image = im_epsf, dx=pixel_scale)
    if save_im:
        im_obj.write('im_obj.fits')
        im_epsf.write('im_epsf.fits')

# get adaptive moments some number of times so we can average over the calls to get an average speed
    t1 = time.time()
    for i in range(ntest):
        res = im_obj.FindAdaptiveMom(strict=False)
    t2 = time.time()
    time_per_call = (t2-t1)/ntest
    # check results
    print "\nFor image size ",this_imsize," per side, no noise, time to get moments was ",time_per_call," per call"
    print "Results for e1, e2, sigma: ",res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma
    print "Expected: ",0.5*gal_e1, 0.5*gal_e2, exp_sigma

# for 48x48 image, Gaussian PSF with FWHM=2.5 pix and Gaussian galaxy with same size, make noisy image
# get adaptive moments some number of times so we can average over the calls to get an average speed
im_obj = galsim.ImageF(imsize, imsize)
im_epsf = galsim.ImageF(imsize, imsize)
im_obj = obj.draw(image = im_obj, dx=pixel_scale)
im_epsf = epsf.draw(image = im_epsf, dx=pixel_scale)
sky_level_pix = sky_level * pixel_scale**2
sn_meas = math.sqrt( numpy.sum(im_obj.array**2) / sky_level_pix )
flux_ratio = sn_noisy / sn_meas
ud = galsim.UniformDeviate(seed)
tot_time_meas = 0.0
mean_sigma = 0.
mean_e1 = 0.
mean_e2 = 0.
for i in range(ntest):
    tmp_im = flux_ratio*im_obj+sky_level_pix
    tmp_im.addNoise(galsim.CCDNoise(ud))
    tmp_im -= sky_level_pix
    if save_im==1 and i==0:
        tmp_im.write('tmp_im_first.fits')
    if save_im==1 and i==ntest-1:
        tmp_im.write('tmp_im_last.fits')
    t1 = time.time()
    res = tmp_im.FindAdaptiveMom(strict=False)
    t2 = time.time()
    tot_time_meas += (t2-t1)
    mean_sigma += res.moments_sigma
    mean_e1 += res.observed_shape.e1
    mean_e2 += res.observed_shape.e2
time_per_call = tot_time_meas / ntest
mean_sigma /= ntest
mean_e1 /= ntest
mean_e2 /= ntest
# check results
print "\nFor image size ",imsize," per side, with noise, time to get moments was ",time_per_call," per call"
print "Final results for e1, e2, sigma: ",mean_e1, mean_e2, mean_sigma

# do shear estimation in the noiseless case
t1 = time.time()
for i in range(ntest):
    res = galsim.hsm.EstimateShear(im_obj, im_epsf)
t2 = time.time()
time_per_call = (t2-t1)/ntest
# check results
print "\nFor image size ",imsize," per side, no noise, time to estimate shear was ",time_per_call," per call"
print "Results for e1, e2: ",res.corrected_e1, res.corrected_e2
print "Expected: ",gal_e1, gal_e2

# do shear estimation in the noisy case
tot_time_meas = 0.0
mean_e1 = 0.
mean_e2 = 0.
for i in range(ntest):
    tmp_im = flux_ratio*im_obj+sky_level_pix
    tmp_im.addNoise(galsim.CCDNoise(ud))
    tmp_im -= sky_level_pix
    if save_im==1 and i==0:
        tmp_im.write('tmp_im_first.fits')
    if save_im==1 and i==ntest-1:
        tmp_im.write('tmp_im_last.fits')
    t1 = time.time()
    res = galsim.hsm.EstimateShear(tmp_im, im_epsf)
    t2 = time.time()
    tot_time_meas += (t2-t1)
    mean_e1 += res.corrected_e1
    mean_e2 += res.corrected_e2
time_per_call = tot_time_meas / ntest
mean_e1 /= ntest
mean_e2 /= ntest
# check results
print "\nFor image size ",imsize," per side, with noise, time to estimate shear was ",time_per_call," per call"
print "Final results for e1, e2: ",mean_e1, mean_e2

# let's try something more complicated, like Sersic n=3 (same HLR as the Gaussian)
# and Kolmogorov with same FWHM of 2.5 pix, convolved with pixel
# do our conclusions still hold?
psf = galsim.Kolmogorov(fwhm = psf_fwhm)
gal = galsim.Sersic(3, half_light_radius = gal_hlr, flux=gal_flux)
pix = galsim.Pixel(pixel_scale)
gal.applyShear(e1 = gal_e1, e2 = gal_e2)
obj = galsim.Convolve(gal, psf, pix)
epsf = galsim.Convolve(psf, pix)
im_obj = galsim.ImageF(imsize, imsize)
im_epsf = galsim.ImageF(imsize, imsize)
im_obj = obj.draw(image = im_obj, dx=pixel_scale)
im_epsf = epsf.draw(image = im_epsf, dx=pixel_scale)

# get adaptive moments some number of times so we can average over the calls to get an average speed
time_mom = 0.0
time_shear = 0.0
for i in range(ntest):
    t1 = time.time()
    res1 = im_obj.FindAdaptiveMom(strict=False)
    t2 = time.time()
    res2 = galsim.hsm.EstimateShear(im_obj, im_epsf, strict=False)
    t3 = time.time()
    time_mom += (t2-t1)
    time_shear += (t3-t2)
time_mom /= ntest
time_shear /= ntest
# check results
print "\nFor Kolmogorov, Pixel, Sersic, ",imsize," per side, no noise, time to get moments was ",time_mom," per call"
print "time to estimate shear was ",time_shear," per call"
print "Results for e1, e2 (corrected): ",res2.corrected_e1, res2.corrected_e2
print "Results for sigma observed: ",res1.moments_sigma

# and also Sersic n=3 with an Airy
psf = galsim.Airy(lam_over_diam = psf_fwhm) # lam_over_diam is pretty close to FWHM
gal = galsim.Sersic(3, half_light_radius = gal_hlr, flux=gal_flux)
pix = galsim.Pixel(pixel_scale)
gal.applyShear(e1 = gal_e1, e2 = gal_e2)
obj = galsim.Convolve(gal, psf, pix)
epsf = galsim.Convolve(psf, pix)
im_obj = galsim.ImageF(imsize, imsize)
im_epsf = galsim.ImageF(imsize, imsize)
im_obj = obj.draw(image = im_obj, dx=pixel_scale)
im_epsf = epsf.draw(image = im_epsf, dx=pixel_scale)

# get adaptive moments some number of times so we can average over the calls to get an average speed
time_mom = 0.0
time_shear = 0.0
for i in range(ntest):
    t1 = time.time()
    res1 = im_obj.FindAdaptiveMom(strict=False)
    t2 = time.time()
    res2 = galsim.hsm.EstimateShear(im_obj, im_epsf, strict=False)
    t3 = time.time()
    time_mom += (t2-t1)
    time_shear += (t3-t2)
time_mom /= ntest
time_shear /= ntest
# check results
print "\nFor Airy, Pixel, Sersic, ",imsize," per side, no noise, time to get moments was ",time_mom," per call"
print "time to estimate shear was ",time_shear," per call"
print "Results for e1, e2 (corrected): ",res2.corrected_e1, res2.corrected_e2
print "Results for sigma observed: ",res1.moments_sigma
