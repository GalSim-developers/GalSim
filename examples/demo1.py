"""
Some example scripts to see some basic usage of the GalSim library.
"""

import sys
import os
import math
import logging
import galsim

# Simple Gaussian for both galaxy and psf, with Gaussian noise
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

    gal_flux = 1.e5    # ADU
    gal_sigma = 2.     # arcsec
    psf_sigma = 1.     # arcsec
    pixel_scale = 0.2  # arcsec / pixel
    noise = 30.        # ADU / pixel

    logger.info('Starting demo script 1 using:')
    logger.info('    - circular Gaussian galaxy (flux = %.1e, sigma = %.1f),',gal_flux,gal_sigma)
    logger.info('    - circular Gaussian PSF (sigma = %.1f),',psf_sigma)
    logger.info('    - pixel scale = %.2f,',pixel_scale)
    logger.info('    - Gaussian noise (sigma = %.2f).',noise)

    # Define the galaxy profile
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
    logger.info('Made galaxy profile')

    # Define the PSF profile
    psf = galsim.Gaussian(flux=1., sigma=psf_sigma) # psf flux should always = 1
    logger.info('Made PSF profile')

    # Define the pixel size
    # The pixels could be rectangles, but normally xw = yw = pixel_scale
    pix = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
    logger.info('Made pixel profile')

    # Final profile is the convolution of these
    # Can include any number of things in the list, all of which are convolved 
    # together to make the final flux profile.
    final = galsim.Convolve([gal, psf, pix])
    logger.info('Convolved components into final profile')

    # Draw the image with a particular pixel scale
    image = final.draw(dx=pixel_scale)
    logger.info('Made image of the profile')

    # Add Gaussian noise to the image with specified sigma
    image.addNoise(galsim.GaussianDeviate(sigma=noise))
    logger.info('Added Gaussian noise')

    # Write the image to a file
    if not os.path.isdir('output'):
        os.mkdir('output')
    file_name = os.path.join('output','demo1.fits')
    image.write(file_name, clobber=True)
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
