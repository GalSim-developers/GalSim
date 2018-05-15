# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import galsim

def radial_integrate(prof, minr, maxr, dr):
    """A simple helper that calculates int 2pi r f(r) dr, from rmin to rmax
       for an axially symmetric profile.
    """
    import math
    assert prof.isAxisymmetric()
    r = minr
    sum = 0.
    while r < maxr:
        sum += r * prof.xValue(galsim.PositionD(r,0)) 
        r += dr
    sum *= 2. * math.pi * dr
    return sum

test_hlr = 0.5
sersic_n = 6.
print 'Creating Sersic profile with n=',sersic_n,' and hlr=',test_hlr
s = galsim.Sersic(sersic_n, half_light_radius = test_hlr)
print 'Checking radial integration of profile....'
hlr_sum = radial_integrate(s, 0., test_hlr, 1.e-4)
print 'Sum of profile to half-light-radius: ',hlr_sum

# make Sersic convolved with some ground-based PSF (Kolmogorov x optical PSF)
print "Testing sersic ground-based sim..."
ground_psf_fwhm = 0.7
ground_pix_scale = 0.2
space_psf_fwhm = 0.1
imsize = 512
n_photons = 1000000
psf = galsim.Kolmogorov(fwhm = ground_psf_fwhm)
opt_psf = galsim.Airy(space_psf_fwhm)
pix = galsim.Pixel(ground_pix_scale)
epsf = galsim.Convolve(psf, opt_psf, pix)
obj_epsf = galsim.Convolve(s, psf, opt_psf, pix)
obj_psf = galsim.Convolve(s, psf, opt_psf)
# compare photon-shot vs. Fourier draw image, directly and with moments
im_draw = galsim.ImageF(imsize, imsize)
im_shoot = galsim.ImageF(imsize, imsize)
im_epsf = galsim.ImageF(imsize, imsize)
im_draw = obj_epsf.draw(image = im_draw, dx = ground_pix_scale, wmult=4.)
im_shoot, _ = obj_psf.drawShoot(image = im_shoot, dx = ground_pix_scale, n_photons = n_photons)
im_epsf = epsf.draw(image = im_epsf, dx = ground_pix_scale)
res_draw = im_draw.FindAdaptiveMom()
res_shoot = im_shoot.FindAdaptiveMom()
print 'Flux in drawn image with linear size ',imsize,' is ',im_draw.array.sum()
print 'Moments of drawn image: e1, e2, sigma = ',res_draw.observed_shape.e1, res_draw.observed_shape.e2, res_draw.moments_sigma
print 'Moments of shot image: e1, e2, sigma = ',res_shoot.observed_shape.e1, res_shoot.observed_shape.e2, res_shoot.moments_sigma
file_draw_ground = 'test_highn_ground_draw.fits'
file_shoot_ground = 'test_highn_ground_shoot.fits'
file_diff_ground = 'test_highn_ground_diff.fits'
file_psf_ground = 'test_highn_ground_psf.fits'
print 'Writing drawn, shot, and diff image to files: ',file_draw_ground, file_shoot_ground, file_diff_ground, file_psf_ground
im_draw.write(file_draw_ground)
im_shoot.write(file_shoot_ground)
im_diff = im_draw-im_shoot
im_diff.write(file_diff_ground)
im_epsf.write(file_psf_ground)

# do the same test as previous for some space-based image
print "Testing sersic space-based sim"
space_psf_fwhm = 0.1
space_pix_scale = 0.03
imsize = 512
n_photons = 10000000
psf = galsim.Airy(space_psf_fwhm) # since lam/diam ~ fwhm
pix = galsim.Pixel(space_pix_scale)
epsf = galsim.Convolve(psf, pix)
obj_epsf = galsim.Convolve(s, psf, pix)
obj_psf = galsim.Convolve(s, psf)
# compare photon-shot vs. Fourier draw image, directly and with moments
im_draw = galsim.ImageF(imsize, imsize)
im_shoot = galsim.ImageF(imsize, imsize)
im_epsf = galsim.ImageF(imsize, imsize)
im_draw = obj_epsf.draw(image = im_draw, dx = space_pix_scale)
im_shoot, _ = obj_psf.drawShoot(image = im_shoot, dx = space_pix_scale, n_photons = n_photons)
im_epsf = epsf.draw(image = im_epsf, dx = space_pix_scale)
res_draw = im_draw.FindAdaptiveMom()
res_shoot = im_shoot.FindAdaptiveMom()
print 'Flux in drawn image with linear size ',imsize,' is ',im_draw.array.sum()
print 'Moments of drawn image: e1, e2, sigma = ',res_draw.observed_shape.e1, res_draw.observed_shape.e2, res_draw.moments_sigma
print 'Moments of shot image: e1, e2, sigma = ',res_shoot.observed_shape.e1, res_shoot.observed_shape.e2, res_shoot.moments_sigma
file_draw_space = 'test_highn_space_draw.fits'
file_shoot_space = 'test_highn_space_shoot.fits'
file_diff_space = 'test_highn_space_diff.fits'
file_psf_space = 'test_highn_space_psf.fits'
print 'Writing drawn, shot, and diff image to files: ',file_draw_space, file_shoot_space, file_diff_space, file_psf_space
im_draw.write(file_draw_space)
im_shoot.write(file_shoot_space)
im_diff = im_draw-im_shoot
im_diff.write(file_diff_space)
im_epsf.write(file_psf_space)

# check shearing

# check rotation

# check flux normalization
