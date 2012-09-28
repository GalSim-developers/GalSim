"""
Demo #1

This is the first script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

Each of these demo*.py files are designed to be equivalent to the corresponding
demo*.yaml file.  If you are new to python, you should probably look at those files first
as they will probably have a quicker learning curve for you.  Then you can look through these 
python scripts, which show how to do the same thing.  Of course, experienced pythonistas
may prefer to start with these scripts and then look at the corresponding YAML files.

If you run `scons examples`, executable versions of these demo scripts will be created in 
the GalSim bin directory.  So if that has been done, you can write either of the following
to run this script:

    ../bin/demo1
    python demo1.py


This first script is about as simple as it gets.  We draw an image of a single galaxy convolved 
with a PSF and write it to disk.  We use a circular Gaussian profile for both the PSF and the 
galaxy.  And we add a constant level of Gaussian noise to the image.

New features introduced in this demo:

- obj = galsim.Gaussian(flux, sigma)
- obj = galsim.Pixel(pixel_scale)
- obj = galsim.Convolve([list of objects])
- image = obj.draw(dx)
- noise = galsim.GaussianDeviate(sigma)
- image.addNoise(noise)
- image.write(file_name)
- image.FindAdaptiveMom()
"""

import sys
import os
import math
import logging
import galsim

def main(argv):
    """
    About as simple as it gets:
      - Use a circular Gaussian profile for the galaxy.
      - Convolve it by a circular Gaussian PSF.
      - Add Gaussian noise to the image.
    """
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo1") 

    gal_flux = 1.e5    # total counds on the image
    gal_sigma = 2.     # arcsec
    psf_sigma = 1.     # arcsec
    pixel_scale = 0.2  # arcsec / pixel
    noise = 30.        # standard deviation of the counts in each pixel

    logger.info('Starting demo script 1 using:')
    logger.info('    - circular Gaussian galaxy (flux = %.1e, sigma = %.1f),',gal_flux,gal_sigma)
    logger.info('    - circular Gaussian PSF (sigma = %.1f),',psf_sigma)
    logger.info('    - pixel scale = %.2f,',pixel_scale)
    logger.info('    - Gaussian noise (sigma = %.2f).',noise)

    # Define the galaxy profile
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
    logger.debug('Made galaxy profile')

    # Define the PSF profile
    psf = galsim.Gaussian(flux=1., sigma=psf_sigma) # PSF flux should always = 1
    logger.debug('Made PSF profile')

    # Define the pixel size
    pix = galsim.Pixel(pixel_scale)
    logger.debug('Made pixel profile')

    # Final profile is the convolution of these
    # Can include any number of things in the list, all of which are convolved 
    # together to make the final flux profile.
    final = galsim.Convolve([gal, psf, pix])
    logger.debug('Convolved components into final profile')

    # Draw the image with a particular pixel scale
    image = final.draw(dx=pixel_scale)
    logger.debug('Made image of the profile')

    # Add Gaussian noise to the image with specified sigma
    image.addNoise(galsim.GaussianDeviate(sigma=noise))
    logger.debug('Added Gaussian noise')

    # Write the image to a file
    if not os.path.isdir('output'):
        os.mkdir('output')
    file_name = os.path.join('output','demo1.fits')
    # Note: if the file already exists, this will overwrite it.
    image.write(file_name)
    logger.info('Wrote image to %r' % file_name)  # using %r adds quotes around filename for us

    results = image.FindAdaptiveMom()

    logger.info('HSM reports that the image has observed shape and size:')
    logger.info('    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)', results.observed_shape.getE1(),
                results.observed_shape.getE2(), results.moments_sigma)
    logger.info('Expected values in the limit that pixel response and noise are negligible:')
    logger.info('    e1 = %.3f, e2 = %.3f, sigma = %.3f', 0.0, 0.0, 
                math.sqrt(gal_sigma**2 + psf_sigma**2)/pixel_scale) 

if __name__ == "__main__":
    main(sys.argv)
