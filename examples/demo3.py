"""
Demo #3

The third script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script is getting reasonably close to including all the principal features of an image
from a ground-based telescope.  The galaxy is a bulge plus disk, where each component is 
a sheared Sersic profile (with different Sersic indices).  The PSF has both atmospheric and 
optical components.  The atmospheric component is a Kolmogorov turbulent spectrum.
The optical component includes defocus, coma and astigmatism, as well as obscuration from
a secondary mirror.  The noise model includes both a gain and read noise.  And finally,
we include the effect of a slight telescope distortion.

New features introduced in this demo:

- obj = galsim.Sersic(n, flux, half_light_radius)
- obj = galsim.Kolmogorov(fwhm)
- obj = galsim.OpticalPSF(lam_over_diam, defocus, coma1, coma2, astig1, astig2, obscuration)
- obj = galsim.Pixel(xw,yw)
- obj.applyShear(e, beta)  -- including how to specify an angle in GalSim
- shear = galsim.Shear(q, beta)
- obj.applyShear(shear)
- obj3 = x1 * obj1 + x2 * obj2
- shear3 = shear1 + shear2
- noise = galsim.CCDNoise(seed, gain, read_noise)
"""

import sys
import os
import math
import logging
import galsim

def main(argv):
    """
    Getting reasonably close to including all the principle features of an image from a
    ground-based telescope:
      - Use a bulge plus disk model for the galaxy 
      - Both galaxy components are Sersic profiles (n=3.5 and n=1.5 respectively)
      - Let the PSF have both atmospheric and optical components.
      - The atmospheric component is a Kolmogorov spectrum.
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
    bulge_n = 3.5          #
    bulge_re = 2.3         # arcsec
    disk_n = 1.5           #
    disk_re = 3.7          # arcsec
    bulge_frac = 0.3       #
    gal_q = 0.73           # (axis ratio 0 < q < 1)
    gal_beta = 23          # degrees (position angle on the sky)
    atmos_fwhm=2.1         # arcsec
    atmos_e = 0.13         # 
    atmos_beta = 0.81      # radians
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
    logger.info('    - Galaxy is bulge plus disk, flux = %.1e',gal_flux)
    logger.info('       - Bulge is Sersic (n = %.1f, re = %.2f), frac = %.1f',
            bulge_n,bulge_re,bulge_frac)
    logger.info('       - Disk is Sersic (n = %.1f, re = %.2f), frac = %.1f',
            disk_n,disk_re,1-bulge_frac)
    logger.info('       - Shape is q,beta (%.2f,%.2f deg)', gal_q, gal_beta)
    logger.info('    - Atmospheric PSF is Kolmogorov with fwhm = %.2f',atmos_fwhm)
    logger.info('       - Shape is e,beta (%.2f,%.2f rad)', atmos_e, atmos_beta)
    logger.info('    - Optical PSF has defocus = %.2f, astigmatism = (%.2f,%.2f),',
            opt_defocus, opt_a1, opt_a2)
    logger.info('          coma = (%.2f,%.2f), lambda = %.0f nm, D = %.1f m', 
            opt_c1, opt_c2, lam, tel_diam)
    logger.info('          obscuration linear size = %.1f',opt_obscuration)
    logger.info('    - pixel scale = %.2f,',pixel_scale)
    logger.info('    - WCS distortion = (%.2f,%.2f),',wcs_g1,wcs_g2)
    logger.info('    - Poisson noise (sky level = %.1e, gain = %.1f).',sky_level, gain)
    logger.info('    - Gaussian read noise (sigma = %.2f).',read_noise)

 
    # Define the galaxy profile.
    bulge = galsim.Sersic(bulge_n, half_light_radius=bulge_re)
    disk = galsim.Sersic(disk_n, half_light_radius=disk_re)

    # Objects may be mutliplied by a scalar (which means scaling the flux) and also
    # added to each other.
    gal = bulge_frac * bulge + (1-bulge_frac) * disk
    # Could also have written the following, which does the same thing:
    # gal = galsim.Add([ bulge.setFlux(bulge_frac) , disk.setFlux(1-bulge_frac) ])
    # Both syntaxes work with more than two summands as well.

    # Set the overall flux of the combined object.
    gal.setFlux(gal_flux)

    # Set the shape of the galaxy according to axis ration and position angle
    # Note: All angles in GalSim must have explicit units.  Options are:
    #       galsim.radians
    #       galsim.degrees
    #       galsim.arcmin
    #       galsim.arcsec
    #       galsim.hours
    gal_shape = galsim.Shear(q=gal_q, beta=gal_beta*galsim.degrees)
    gal.applyShear(gal_shape)
    logger.debug('Made galaxy profile')

    # Define the atmospheric part of the PSF.
    # Note: the flux here is the default flux=1.
    atmos = galsim.Kolmogorov(fwhm=atmos_fwhm)
    # For the PSF shape here, we use ellipticity rather than axis ratio.
    # And the position angle can be either degrees or radians.  Here we chose radians.
    atmos.applyShear(e=atmos_e , beta=atmos_beta*galsim.radians)
    logger.debug('Made atmospheric PSF profile')

    # Define the optical part of the PSF.
    # The first argument of OpticalPSF below is lambda/diam, which needs to be in arcsec,
    # so do the calculation:
    lam_over_diam = lam * 1.e-9 / tel_diam # radians
    lam_over_diam *= 206265  # arcsec
    logger.debug('Calculated lambda over diam = %f arcsec', lam_over_diam)
    # The rest of the values should be given in units of the wavelength of the incident light.
    optics = galsim.OpticalPSF(lam_over_diam, 
                               defocus = opt_defocus,
                               coma1 = opt_c1, coma2 = opt_c2,
                               astig1 = opt_a1, astig2 = opt_a2,
                               obscuration = opt_obscuration)
    logger.debug('Made optical PSF profile')

    # Now apply the wcs shear to the profile without the pix
    nopix = galsim.Convolve([atmos, optics, gal])
    psf = galsim.Convolve([atmos, optics])
    nopix.applyShear(g1=wcs_g1, g2=wcs_g2)
    psf.applyShear(g1=wcs_g1, g2=wcs_g2)
    logger.debug('Applied WCS distortion')

    # While Pixels are usually square, you can make them rectangular if you want
    # by specifying xw and yw separately.  Here they are still the same value, but if you
    # wanted non-square pixels, just specify a different value for xw and yw.
    pix = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
    logger.debug('Made pixel profile')

    # Final profile is the convolution of these.
    final = galsim.Convolve([nopix, pix])
    final_epsf = galsim.Convolve([psf, pix])
    logger.debug('Convolved components into final profile')

    # This time we specify a particular size for the image rather than let galsim 
    # choose the size automatically.
    image = galsim.ImageF(image_size,image_size)
    # Draw the image with a particular pixel scale.
    final.draw(image=image, dx=pixel_scale)

    # Also draw the effective PSF by itself and the optical PSF component alone.
    image_epsf = galsim.ImageF(image_size,image_size)
    final_epsf.draw(image_epsf, dx=pixel_scale)
    image_opticalpsf = optics.draw(dx=lam_over_diam/2.)
    logger.debug('Made image of the profile')

    # Add a constant sky level to the image.
    image += sky_level * pixel_scale**2

    # Add Poisson noise and Gaussian read noise to the image using the CCDNoise class.
    image.addNoise(galsim.CCDNoise(random_seed, gain=gain, read_noise=read_noise))

    # Subtract off the sky.
    image -= sky_level * pixel_scale**2
    logger.debug('Added Gaussian and Poisson noise')

    # Write the image to a file
    file_name = os.path.join('output', 'demo3.fits')
    file_name_epsf = os.path.join('output','demo3_epsf.fits')
    file_name_opticalpsf = os.path.join('output','demo3_opticalpsf.fits')
    
    image.write(file_name)
    image_epsf.write(file_name_epsf)
    image_opticalpsf.write(file_name_opticalpsf)
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
