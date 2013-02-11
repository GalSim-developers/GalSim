import galsim
import numpy as np

sigma = 1.
dx = 0.1
pad_var = 2.e-6
add_flux = 0.04
noise_file = '../tests/blankimg.fits'
im_file = 'data/147246.0_150.416558_1.998697_masknoise.fits'

# make some simple object and draw its image
g = galsim.Gaussian(sigma=sigma)
im = g.draw(dx=dx)

# make into II with no noise_pad and no pad_image
ii = galsim.InterpolatedImage(im)
new_im1 = galsim.ImageF(ii.x_size, ii.y_size)
new_im1 = ii.draw(new_im1, dx=dx)

# make into II with no pad_image, and: noise_pad=float, string
ii = galsim.InterpolatedImage(im, noise_pad = pad_var)
new_im2 = galsim.ImageF(ii.x_size, ii.y_size)
new_im2 = ii.draw(new_im2, dx=dx)

ii = galsim.InterpolatedImage(im, noise_pad = noise_file)
new_im3 = galsim.ImageF(ii.x_size, ii.y_size)
new_im3 = ii.draw(new_im3, dx=dx)

# make into II with no noise_pad, and: pad_image=float, string
ii = galsim.InterpolatedImage(im, pad_image = add_flux)
new_im4 = galsim.ImageF(ii.x_size, ii.y_size)
new_im4 = ii.draw(new_im4, dx=dx)

ii = galsim.InterpolatedImage(im, pad_image=im_file, pad_factor=1.)
new_im5 = galsim.ImageF(ii.x_size, ii.y_size)
new_im5 = ii.draw(new_im5, dx=dx)

# make into II with noise_pad, pad_image = float, float
ii = galsim.InterpolatedImage(im, noise_pad = pad_var, pad_image = add_flux)
new_im6 = galsim.ImageF(ii.x_size, ii.y_size)
new_im6 = ii.draw(new_im6, dx=dx)

# make into II with noise_pad, pad_image = string, float
ii = galsim.InterpolatedImage(im, noise_pad = noise_file, pad_image = add_flux)
new_im7 = galsim.ImageF(ii.x_size, ii.y_size)
new_im7 = ii.draw(new_im7, dx=dx)

# make into II with noise_pad, pad_image = float, string
ii = galsim.InterpolatedImage(im, noise_pad = pad_var, pad_image = im_file)
new_im8 = galsim.ImageF(ii.x_size, ii.y_size)
new_im8 = ii.draw(new_im8, dx=dx)

# make into II with noise_pad, pad_image = string, string
ii = galsim.InterpolatedImage(im, noise_pad = noise_file, pad_image = im_file)
new_im9 = galsim.ImageF(ii.x_size, ii.y_size)
new_im9 = ii.draw(new_im9, dx=dx)

# write all images to file for testing purposes
im.write('test_im.fits')
new_im1.write('new_im1.fits')
new_im2.write('new_im2.fits')
new_im3.write('new_im3.fits')
new_im4.write('new_im4.fits')
new_im5.write('new_im5.fits')
new_im6.write('new_im6.fits')
new_im7.write('new_im7.fits')
new_im8.write('new_im8.fits')
new_im9.write('new_im9.fits')
