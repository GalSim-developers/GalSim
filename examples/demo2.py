"""
Demo #2

The second script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(Script is designed to be viewed in a window 100 characters wide.)
 
This script is a bit more sophisticated, but still pretty basic.  We're still only making
a single image, but now the galaxy has an exponential radial profile and is sheared.
The PSF is a circular Moffat profile.  And the noise is Poisson using the flux from both
the object and a background sky level to determine the variance in each pixel.

New features introduced in this demo:

- obj = galsim.Exponential(flux, scale_radius)
- obj = galsim.Moffat(beta, flux, half_light_radius)
- obj.applyShear(g1,g2)  -- with explanation of other ways to specify shear
- image = galsim.ImageF(image_size,image_size)
- image += constant
- image -= constant
- noise = galsim.CCDNoise(seed)
- obj.draw(image, dx)
- galsim.EstimateShearHSM(image, image_epsf)
"""

import sys
import os
import math
import logging
import galsim

def main(argv):
    """
    A little bit more sophisticated, but still pretty basic:
      - Use a sheared, exponential profile for the galaxy.
      - Convolve it by a circular Moffat PSF.
      - Add Poisson noise to the image.
    """
    # In non-script code, use getLogger(__name__) at module scope instead.    
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo2") 

    gal_flux = 1.e5    # ADU
    gal_r0 = 2.7       # arcsec
    g1 = 0.1           #
    g2 = 0.2           #
    psf_beta = 5       #
    psf_re = 1.0       # arcsec
    pixel_scale = 0.2  # arcsec / pixel
    sky_level = 2.5e4  # ADU / arcsec^2
    gain = 1.0         # ADU / e-

    # This time use a particular seed, so the image is deterministic.
    random_seed = 1534225

    logger.info('Starting demo script 2 using:')
    logger.info('    - sheared (%.2f,%.2f) exponential galaxy (flux = %.1e, scale radius = %.2f),',
            g1, g2, gal_flux, gal_r0)
    logger.info('    - circular Moffat PSF (beta = %.1f, re = %.2f),', psf_beta,psf_re)
    logger.info('    - pixel scale = %.2f,', pixel_scale)
    logger.info('    - Poisson noise (sky level = %.1e).', sky_level)

    # Define the galaxy profile.
    gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

    # Shear the galaxy by some value.
    # There are quite a few ways you can use to specify a shape.
    # q, beta      Axis ratio and position angle: q = b/a, 0 < q < 1
    # e, beta      Ellipticity and position angle: |e| = (1-q^2)/(1+q^2)
    # g, beta      ("Reduced") Shear and position angle: |g| = (1-q)/(1+q)
    # eta, beta    Conformal shear and position angle: eta = ln(1/q)
    # e1,e2        Ellipticity components: e1 = e cos(2 beta), e2 = e sin(2 beta)
    # g1,g2        ("Reduced") shear components: g1 = g cos(2 beta), g2 = g sin(2 beta)
    # eta1,eta2    Conformal shear components: eta1 = eta cos(2 beta), eta2 = eta sin(2 beta)
    gal.applyShear(g1=g1, g2=g2)
    logger.debug('Made galaxy profile')

    # Define the PSF profile.
    psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)
    logger.debug('Made PSF profile')

    # Define the pixel size
    pix = galsim.Pixel(pixel_scale)
    logger.debug('Made pixel profile')

    # Final profile is the convolution of these.
    final = galsim.Convolve([gal, psf, pix])
    final_epsf = galsim.Convolve([psf, pix])
    logger.debug('Convolved components into final profile')

    # Draw the image with a particular pixel scale.
    image = final.draw(dx=pixel_scale)
    image_epsf = final_epsf.draw(dx=pixel_scale)
    logger.debug('Made image of the profile')

    # Add a constant sky level to the image.
    image += sky_level * pixel_scale**2

    # Use this to add Poisson noise using the CCDNoise class.
    image.addNoise(galsim.CCDNoise(random_seed))

    # Subtract off the sky.
    image -= sky_level * pixel_scale**2
    logger.debug('Added Poisson noise')

    # Write the image to a file.
    if not os.path.isdir('output'):
        os.mkdir('output')
    file_name = os.path.join('output', 'demo2.fits')
    file_name_epsf = os.path.join('output','demo2_epsf.fits')
    image.write(file_name, clobber=True)
    image_epsf.write(file_name_epsf, clobber=True)
    logger.info('Wrote image to %r',file_name)
    logger.info('Wrote effective PSF image to %r',file_name_epsf)

    results = galsim.EstimateShearHSM(image, image_epsf)

    logger.info('HSM reports that the image has observed shape and size:')
    logger.info('    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)', results.observed_shape.getE1(),
                results.observed_shape.getE2(), results.moments_sigma)
    logger.info('When carrying out Regaussianization PSF correction, HSM reports')
    e_temp = results.corrected_shape.getE()
    if e_temp > 0.:
        gfac = results.corrected_shape.getG()/e_temp
    else:
        gfac = 0.
    logger.info('    g1, g2 = %.3f, %.3f', 
            gfac*results.corrected_shape.getE1(), gfac*results.corrected_shape.getE2())
    logger.info('Expected values in the limit that noise and non-Gaussianity are negligible:')
    logger.info('    g1, g2 = %.3f, %.3f', g1,g2)

if __name__ == "__main__":
    main(sys.argv)
