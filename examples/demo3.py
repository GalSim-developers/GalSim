#!/usr/bin/env python
"""
Some example scripts to see some basic usage of the GalSim library.
"""

import sys
import os
import math
import logging
import galsim

# Sheared, Sersic galaxy, Gaussian + OpticalPSF (atmosphere + optics) PSF, Poisson noise 
def main(argv):
    """
    Getting reasonably close to including all the principle features of a 
    ground-based telescope:
      - Use a sheared, Sersic profile for the galaxy 
        (n = 3.5, half_light_radius=3.7).
      - Let the PSF have both atmospheric and optical components.
      - The atmospheric component is the sum of two non-circular Gaussians.
      - The optical component has some defocus, coma, and astigmatism.
      - Add both Poisson noise to the image and Gaussian read noise.
      - Let the pixels be slightly distorted relative to the sky.
    """
    # We do some fancier logging for demo3, just to demonstrate that we can:
    # - we log to both stdout and to a log file
    # - the log file has a lot more (mostly redundant) information
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    if not os.path.isdir('output'):
        os.mkdir('output')
    logFile = logging.FileHandler(os.path.join("output", "script3.log"))
    logFile.setFormatter(logging.Formatter("%(name)s[%(levelname)s] %(asctime)s: %(message)s"))
    logging.getLogger("demo3").addHandler(logFile)
    logger = logging.getLogger("demo3") 

    gal_flux = 1.e6        # ADU
    gal_n = 3.5            #
    gal_re = 3.7           # arcsec
    gal_q = 0.73           # (axis ratio 0 < q < 1)
    gal_beta = 23          # degrees (position angle on the sky)
    atmos_outer_sigma=2.1      # arcsec
    atmos_outer_e = 0.13       # (ellipticity of "outer" Gaussian)
    atmos_outer_beta = 0.81    # radians
    atmos_fouter=0.2           # (fraction of flux in "outer" Gaussian)
    atmos_inner_sigma=0.9      # arcsec
    atmos_inner_e = 0.04       # (ellipticity of "inner" Gaussian)
    atmos_inner_beta = -0.17   # radians
    opt_defocus=0.53       # wavelengths
    opt_a1=-0.29           # wavelengths
    opt_a2=0.12            # wavelengths
    opt_c1=0.64            # wavelengths
    opt_c2=-0.33           # wavelengths
    opt_obscuration=0.3    # linear scale size of secondary mirror obscuration
    lam = 800              # nm    NB: don't use lambda - that's a reserved word.
    tel_diam = 4.          # meters 
    pixel_scale = 0.23     # arcsec / pixel
    image_size = 64        # n x n pixels
    wcs_g1 = -0.02         #
    wcs_g2 = 0.01          #
    sky_level = 2.5e4      # ADU / arcsec^2
    gain = 1.7             # e- / ADU
    read_noise = 0.3       # ADU / pixel

    random_seed = 1314662  

    logger.info('Starting demo script 3 using:')
    logger.info('    - q,beta (%.2f,%.2f) Sersic galaxy (flux = %.1e, n = %.1f, re = %.2f),', 
            gal_q, gal_beta, gal_flux, gal_n, gal_re)
    logger.info('    - elliptical double-Gaussian atmospheric PSF')
    logger.info('          Outer component: sigma = %.2f, e,beta = (%.2f,%.2f), frac = %.2f',
            atmos_outer_sigma, atmos_outer_e, atmos_outer_beta, atmos_fouter)
    logger.info('          Inner component: sigma = %.2f, e,beta = (%.2f,%.2f), frac = %.2f',
            atmos_inner_sigma, atmos_inner_e, atmos_inner_beta, 1-atmos_fouter)
    logger.info('    - optical PSF with defocus = %.2f, astigmatism = (%.2f,%.2f),',
            opt_defocus, opt_a1, opt_a2)
    logger.info('          coma = (%.2f,%.2f), lambda = %.0f nm, D = %.1f m', 
            opt_c1, opt_c2, lam, tel_diam)
    logger.info('          obscuration linear size = %.1f',opt_obscuration)
    logger.info('    - pixel scale = %.2f,',pixel_scale)
    logger.info('    - WCS distortion = (%.2f,%.2f),',wcs_g1,wcs_g2)
    logger.info('    - Poisson noise (sky level = %.1e, gain = %.1f).',sky_level, gain)
    logger.info('    - Gaussian read noise (sigma = %.2f).',read_noise)

 
    # Define the galaxy profile.
    gal = galsim.Sersic(gal_n, flux=gal_flux, half_light_radius=gal_re)

    # Set the shape of the galaxy according to axis ration and position angle
    gal_shape = galsim.Shear(q=gal_q, beta=gal_beta*galsim.degrees)
    gal.applyShear(gal_shape)
    logger.info('Made galaxy profile')

    # Define the atmospheric part of the PSF.
    atmos_outer = galsim.Gaussian(sigma=atmos_outer_sigma)
    # For the PSF shape here, we use ellipticity rather than axis ratio.
    # And the position angle can be either degrees or radians.  Here we chose radians.
    atmos_outer.applyShear(e=atmos_outer_e , beta=atmos_outer_beta*galsim.radians)
    atmos_inner = galsim.Gaussian(sigma=atmos_inner_sigma)
    atmos_inner.applyShear(e=atmos_inner_e , beta=atmos_inner_beta*galsim.radians)
    atmos = atmos_fouter * atmos_outer + (1. - atmos_fouter) * atmos_inner
    # Could also have written either of the following, which do the same thing:
    # atmos = galsim.Add(atmos_outer.setFlux(fouter), atmos_inner.setFlux(1. - fouter))
    # atmos = galsim.Add([atmos_outer.setFlux(fouter), atmos_inner.setFlux(1. - fouter)])
    # For more than two summands, you can either string together +'s or use the list version.
    logger.info('Made atmospheric PSF profile')

    # Define the optical part of the PSF.
    # The first argument of OpticalPSF below is lambda/diam,
    # which needs to be in arcsec, so do the calculation:
    lam_over_diam = lam * 1.e-9 / tel_diam # radians
    lam_over_diam *= 206265  # arcsec
    logger.info('Calculated lambda over diam = %f arcsec', lam_over_diam)
    # The rest of the values here should be given in units of the 
    # wavelength of the incident light.
    optics = galsim.OpticalPSF(lam_over_diam, 
                               defocus = opt_defocus,
                               coma1 = opt_c1, coma2 = opt_c2,
                               astig1 = opt_a1, astig2 = opt_a2,
                               obscuration = opt_obscuration)
    logger.info('Made optical PSF profile')

    # Now apply the wcs shear to the profile without the pix
    nopix = galsim.Convolve([atmos, optics, gal])
    psf = galsim.Convolve([atmos, optics])
    nopix.applyShear(g1=wcs_g1, g2=wcs_g2)
    psf.applyShear(g1=wcs_g1, g2=wcs_g2)
    logger.info('Applied WCS distortion')

    # Start with square pixels
    pix = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
    logger.info('Made pixel profile')

    # Final profile is the convolution of these.
    final = galsim.Convolve([nopix, pix])
    final_epsf = galsim.Convolve([psf, pix])
    logger.info('Convolved components into final profile')

    # This time we specify a particular size for the image rather than let galsim 
    # choose the size automatically.
    image = galsim.ImageF(image_size,image_size)
    # Draw the image with a particular pixel scale.
    final.draw(image=image, dx=pixel_scale)

    # Also draw the effective PSF by itself and the optical PSF component alone.
    image_epsf = galsim.ImageF(image_size,image_size)
    final_epsf.draw(image_epsf, dx=pixel_scale)
    image_opticalpsf = optics.draw(dx=lam_over_diam/2.)
    logger.info('Made image of the profile')

    # Add a constant sky level to the image.
    image += sky_level * pixel_scale**2

    # Add Poisson noise and Gaussian read noise to the image using the CCDNoise class.
    image.addNoise(galsim.CCDNoise(random_seed, gain=gain, read_noise=read_noise))

    # Subtract off the sky.
    image -= sky_level * pixel_scale**2
    logger.info('Added Gaussian and Poisson noise')

    # Write the image to a file
    file_name = os.path.join('output', 'demo3.fits')
    file_name_epsf = os.path.join('output','demo3_epsf.fits')
    file_name_opticalpsf = os.path.join('output','demo3_opticalpsf.fits')
    
    image.write(file_name, clobber=True)
    image_epsf.write(file_name_epsf, clobber=True)
    image_opticalpsf.write(file_name_opticalpsf, clobber=True)
    logger.info('Wrote image to %r', file_name)
    logger.info('Wrote effective PSF image to %r', file_name_epsf)
    logger.info('Wrote optics-only PSF image (Nyquist sampled) to %r', file_name_opticalpsf)

    results = galsim.EstimateShearHSM(image, image_epsf)

    logger.info('HSM reports that the image has observed shape and size:')
    logger.info('    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)', results.observed_shape.getE1(),
                results.observed_shape.getE2(), results.moments_sigma)
    logger.info('When carrying out Regaussianization PSF correction, HSM reports')
    logger.info('    e1, e2 = %.3f, %.3f',
            results.corrected_shape.getE1(), results.corrected_shape.getE2())
    logger.info('Expected values in the limit that noise and non-Gaussianity are negligible:')
    # Convention for shear addition is to apply the second (RHS) term initially followed by the
    # first (LHS).
    # So wcs needs to be LHS and galaxy shape RHS.
    total_shape = galsim.Shear(g1=wcs_g1, g2=wcs_g2) + gal_shape
    logger.info('    e1, e2 = %.3f, %.3f', total_shape.getE1(), total_shape.getE2())

if __name__ == "__main__":
    main(sys.argv)
