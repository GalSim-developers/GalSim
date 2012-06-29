#!/usr/bin/env python
"""
Some example scripts to make multi-object images using the GalSim library.
"""

import galsim
import numpy

# Define some parameters we'll use below.

file_name = 'test.fits'

random_seed = 1512413
pixel_scale = 0.28      # arcsec
nx = 64
ny = 64

gal_flux = 1.26e4
gal_hlr = 0.59          # arcsec
gal_e1 = 0.01
gal_e2 = -0.01

psf_fwhm = 0.65         # arcsec

# Initialize the random number generator we will be using.
rng = galsim.UniformDeviate(random_seed)

all_images = []
for i in range(500):
    print 'i = ',i

    # Make the profiles:
    pix = galsim.Pixel(xw=pixel_scale)
    psf = galsim.AtmosphericPSF(fwhm = psf_fwhm)
    gal = galsim.Gaussian(flux=gal_flux, half_light_radius=gal_hlr)
    gal.applyShear(e1=gal_e1, e2=gal_e2)

    final = galsim.Convolve([gal,psf,pix])
    final_nopix = galsim.Convolve([gal,psf])

    # Setup the image: left half is FFT, right half is Photon Shooting
    image = galsim.ImageF(2*nx+2,ny)
    image.setScale(pixel_scale)
    fft_image = image[galsim.BoundsI(1,nx,1,nx)]
    phot_image = image[galsim.BoundsI(nx+3,2*nx+2,1,nx)]

    # Draw the profile using FFT
    final.draw(fft_image, dx=pixel_scale)

    # Repeat for photon shooting image.
    final_nopix.drawShoot(phot_image, uniform_deviate=rng, poisson_flux=False)

    # Add Poisson noise
    fft_image.addNoise(galsim.CCDNoise(rng))
    
    # For photon shooting, galaxy already has poisson noise, and not adding sky_noise here.

    # Add this image to our list
    all_images += [image]

# Now write the images to disk.
#galsim.fits.writeCube(all_images, file_name, clobber=True)

# Check some statistics for some of the pixels near the middle:
n = len(all_images)
# I had been looking at a range like -3,4, but for now just do the central pixel.
# (Similar behavior happens in all of them.)
for x in range(0,1):  
    for y in range(0,1):
        fft_ar = numpy.zeros(n)
        phot_ar = numpy.zeros(n)
        for i in range(len(all_images)):
            image = all_images[i]
            fft_image = image[galsim.BoundsI(1,nx,1,nx)]
            phot_image = image[galsim.BoundsI(nx+3,2*nx+2,1,nx)]
            fft_image.setCenter(0,0)
            phot_image.setCenter(0,0)
            fft_ar[i] = fft_image(x,y)
            phot_ar[i] = phot_image(x,y)
        fft_ar.sort()
        phot_ar.sort()
        #print 'fft flux values for %d,%d = '%(x,y), fft_ar
        #print 'phot flux values for %d,%d = '%(x,y), phot_ar
        mean = fft_ar.sum()/n
        var = ((fft_ar-mean)**2).sum()/(n-1)
        print 'fft mean = %f, variance = %f'%(mean, var)
        print '    range = %f, %f'%(fft_ar.min(),fft_ar.max())
        print '    quartiles = %f, %f'%(fft_ar[n/4],fft_ar[3*n/4])
        mean = phot_ar.sum()/n
        var = ((phot_ar-mean)**2).sum()/(n-1)
        print 'phot mean = %f, variance = %f'%(mean, var)
        print '     range = %f, %f'%(phot_ar.min(),phot_ar.max())
        print '     quartiles = %f, %f'%(phot_ar[n/4],phot_ar[3*n/4])


