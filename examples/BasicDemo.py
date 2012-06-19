#!/usr/bin/env python
"""
Some example scripts to see some basic usage of the GalSim library.
"""

import sys
import os
import subprocess
import math
import logging

# This machinery lets us run Python examples even though they aren't positioned
# properly to find galsim as a package in the current directory.
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# Script 1: Simple Gaussian for both galaxy and psf, with Gaussian noise
def Script1():
    """
    About as simple as it gets:
      - Use a circular Gaussian profile for the galaxy.
      - Convolve it by a circular Gaussian PSF.
      - Add Gaussian noise to the image.
    """
    # In non-script code, use getLogger(__name__) at module scope instead.
    logger = logging.getLogger("Script1") 

    gal_flux = 1.e5    # ADU
    gal_sigma = 2.     # arcsec
    psf_sigma = 1.     # arcsec
    pixel_scale = 0.2  # arcsec / pixel
    noise = 300.       # ADU / pixel

    logger.info('Starting script 1 using:')
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

    # Add some noise to the image
    # First we need to set up a random number generator:
    # Defaut seed is set from the current time.
    rng = galsim.UniformDeviate()
    # Use this to add Gaussian noise with specified sigma
    image.addNoise(galsim.GaussianDeviate(rng, sigma=noise))
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
    print

# Script 2: Sheared, exponential galaxy, Moffat PSF, Poisson noise
def Script2():
    """
    A little bit more sophisticated, but still pretty basic:
      - Use a sheared, exponential profile for the galaxy.
      - Convolve it by a circular Moffat PSF.
      - Add Poisson noise to the image.
    """
    # In non-script code, use getLogger(__name__) at module scope instead.    
    logger = logging.getLogger("Script2") 

    gal_flux = 1.e5    # ADU
    gal_r0 = 2.7       # arcsec
    g1 = 0.1           #
    g2 = 0.2           #
    psf_beta = 5       #
    psf_re = 1.0       # arcsec
    pixel_scale = 0.2  # arcsec / pixel
    sky_level = 1.e3   # ADU / pixel
    gain = 1.0         # ADU / e-

    logger.info('Starting script 2 using:')
    logger.info('    - sheared (%.2f,%.2f) exponential galaxy (flux = %.1e, scale radius = %.2f),',
            g1, g2, gal_flux, gal_r0)
    logger.info('    - circular Moffat PSF (beta = %.1f, re = %.2f),', psf_beta,psf_re)
    logger.info('    - pixel scale = %.2f,', pixel_scale)
    logger.info('    - Poisson noise (sky level = %.1e, gain = %.1f).', sky_level, gain)

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
    logger.info('Made galaxy profile')

    # Define the PSF profile.
    psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)
    logger.info('Made PSF profile')

    # Define the pixel size
    pix = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
    logger.info('Made pixel profile')

    # Final profile is the convolution of these.
    final = galsim.Convolve([gal, psf, pix])
    final_epsf = galsim.Convolve([psf, pix])
    logger.info('Convolved components into final profile')

    # Draw the image with a particular pixel scale.
    image = final.draw(dx=pixel_scale)
    image_epsf = final_epsf.draw(dx=pixel_scale)
    logger.info('Made image of the profile')

    # Add a constant sky level to the image.
    # Create an image with the same bounds as image, with a constant
    # sky level.
    sky_image = galsim.ImageF(bounds=image.getBounds(), init_value=sky_level)
    image += sky_image

    # This time use a particular seed, so it the image is deterministic.
    rng = galsim.UniformDeviate(1534225)
    # Use this to add Poisson noise using the CCDNoise class.
    image.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=0.))

    # Subtract off the sky.
    image -= sky_image
    logger.info('Added Poisson noise')

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
    print


# Script 3: Sheared, Sersic galaxy, Gaussian + OpticalPSF (atmosphere + optics) PSF, Poisson noise 
def Script3():
    """
    Getting reasonably close to including all the principle features of a 
    ground-based telescope:
      - Use a sheared, Sersic profile for the galaxy 
        (n = 3., half_light_radius=4.).
      - Let the PSF have both atmospheric and optical components.
      - The atmospheric component is the sum of two non-circular Gaussians.
      - The optical component has some defocus, coma, and astigmatism.
      - Add both Poisson noise to the image and Gaussian read noise.
      - Let the pixels be slightly distorted relative to the sky.
    """
    # In non-script code, use getLogger(__name__) at module scope instead.
    logger = logging.getLogger("Script3") 
    gal_flux = 1.e6        # ADU
    gal_n = 3.5            #
    gal_re = 3.7           # arcsec
    gal_q = 0.73           # (axis ratio 0 < q < 1)
    gal_beta = 23          # degrees (position angle on the sky)
    atmos_a_sigma=2.1      # arcsec
    atmos_a_e = 0.13       # (ellipticity of "a")
    atmos_a_beta = 0.81    # radians
    atmos_fa=0.2           # (fraction of flux in "a")
    atmos_b_sigma=0.9      # arcsec
    atmos_b_e = 0.04       # (ellipticity of "b")
    atmos_b_beta = -0.17   # radians
    opt_defocus=0.53       # wavelengths
    opt_a1=-0.29           # wavelengths
    opt_a2=0.12            # wavelengths
    opt_c1=0.64            # wavelengths
    opt_c2=-0.33           # wavelengths
    opt_obscuration=0.3    # linear scale size of secondary mirror obscuration
    opt_padfactor=2        # multiples of Airy padding required to avoid folding for aberrated PSFs
    lam = 800              # nm    NB: don't use lambda - that's a reserved word.
    tel_diam = 4.          # meters 
    pixel_scale = 0.23     # arcsec / pixel
    wcs_g1 = -0.02         #
    wcs_g2 = 0.01          #
    sky_level = 1.e3       # ADU / pixel
    gain = 1.7             # ADU / e-
    read_noise = 0.3       # ADU / pixel

    logger.info('Starting script 3 using:')
    logger.info('    - q,beta (%.2f,%.2f) Sersic galaxy (flux = %.1e, n = %.1f, re = %.2f),', 
            gal_q, gal_beta, gal_flux, gal_n, gal_re)
    logger.info('    - elliptical double-Gaussian atmospheric PSF')
    logger.info('          First component: sigma = %.2f, e,beta = (%.2f,%.2f), frac = %.2f',
            atmos_a_sigma, atmos_a_e, atmos_a_beta, atmos_fa)
    logger.info('          Second component: sigma = %.2f, e,beta = (%.2f,%.2f), frac = %.2f',
            atmos_b_sigma, atmos_b_e, atmos_b_beta, 1-atmos_fa)
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
    atmos_a = galsim.Gaussian(sigma=atmos_a_sigma)
    # For the PSF shape here, we use ellipticity rather than axis ratio.
    # And the position angle can be either degrees or radians.  Here we chose radians.
    atmos_a.applyShear(e=atmos_a_e , beta=atmos_a_beta*galsim.radians)
    atmos_b = galsim.Gaussian(sigma=atmos_b_sigma)
    atmos_b.applyShear(e=atmos_b_e , beta=atmos_b_beta*galsim.radians)
    atmos = atmos_fa * atmos_a + (1-atmos_fa) * atmos_b
    # Could also have written either of the following, which do the same thing:
    # atmos = galsim.Add(atmos_a, atmos_b)
    # atmos = galsim.Add([atmos_a, atmos_b])
    # For more than two summands, you can either string together +'s or use the list version.
    logger.info('Made atmospheric PSF profile')

    # Define the optical part of the PSF.
    # The first argument of OpticalPSF below is lambda/D,
    # which needs to be in arcsec, so do the calculation:
    lam_over_D = lam * 1.e-9 / tel_diam # radians
    lam_over_D *= 206265 # arcsec
    logger.info('Calculated lambda over D = %f arcsec', lam_over_D)
    # The rest of the values here should be given in units of the 
    # wavelength of the incident light.
    # pad_factor is used to here to reduce 'folding' in the fourier transform.
    optics = galsim.OpticalPSF(lam_over_D, 
                               defocus = opt_defocus,
                               coma1 = opt_c1, coma2 = opt_c2,
                               astig1 = opt_a1, astig2 = opt_a2,
                               obscuration = opt_obscuration,
                               pad_factor = opt_padfactor)
    logger.info('Made optical PSF profile')

    # Start with square pixels
    pix = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
    # Then shear them slightly by the negative of the wcs shear.
    # This way the later distortion of the full image will bring them back to square.
    pix.applyShear(g1=-wcs_g1, g2=-wcs_g2)
    logger.info('Made pixel profile')

    # Final profile is the convolution of these.
    final = galsim.Convolve([gal, atmos, optics, pix])
    final_epsf = galsim.Convolve([atmos, optics, pix])
    logger.info('Convolved components into final profile')

    # Now apply the wcs shear to the final image.
    final.applyShear(g1=wcs_g1, g2=wcs_g2)
    final_epsf.applyShear(g1=wcs_g1, g2=wcs_g2)
    logger.info('Applied WCS distortion')

    # Draw the image with a particular pixel scale.
    image = final.draw(dx=pixel_scale)
    image_epsf = final_epsf.draw(dx=pixel_scale)
    # Draw the optical PSF component at its Nyquist sample rate
    image_opticalpsf = optics.draw(dx=lam_over_D/2.)
    logger.info('Made image of the profile')

    # Add a constant sky level to the image.
    sky_image = galsim.ImageF(bounds=image.getBounds(), init_value=sky_level)
    image += sky_image

    # Add Poisson noise and Gaussian read noise to the image using the CCDNoise class.
    rng = galsim.UniformDeviate(1314662)
    image.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise))

    # Subtract off the sky.
    image -= sky_image
    logger.info('Added Gaussian and Poisson noise')

    # Write the image to a file
    if not os.path.isdir('output'):
        os.mkdir('output')
    file_name = os.path.join('output', 'demo3.fits')
    file_name_opticalpsf = os.path.join('output','demo3_opticalpsf.fits')
    file_name_epsf = os.path.join('output','demo3_epsf.fits')
    
    image.write(file_name, clobber=True)
    image_opticalpsf.write(file_name_opticalpsf, clobber=True)
    image_epsf.write(file_name_epsf, clobber=True)
    logger.info('Wrote image to %r', file_name)
    logger.info('Wrote optics-only PSF image (Nyquist sampled) to %r', file_name_opticalpsf)
    logger.info('Wrote effective PSF image to %r', file_name_epsf)

    results = galsim.EstimateShearHSM(image, image_epsf)

    logger.info('HSM reports that the image has observed shape and size:')
    logger.info('    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)', results.observed_shape.getE1(),
                results.observed_shape.getE2(), results.moments_sigma)
    logger.info('When carrying out Regaussianization PSF correction, HSM reports')
    logger.info('    e1, e2 = %.3f, %.3f',
            results.corrected_shape.getE1(), results.corrected_shape.getE2())
    logger.info('Expected values in the limit that noise and non-Gaussianity are negligible:')
    # Convention for shear addition is to apply the second term and then the first.
    # So wcs needs to be first and galaxy shape second.
    total_shape = galsim.Shear(g1=wcs_g1, g2=wcs_g2) + gal_shape
    logger.info('    e1, e2 = %.3f, %.3f', total_shape.getE1(), total_shape.getE2())
    print

def main(argv):
    try:
        # If no argument, run all scripts (indicated by scriptNum = 0)
        scriptNum = int(argv[1]) if len(argv) > 1 else 0
    except Exception as err:
        print __doc__
        raise err
    
    # Setup logging here, rather than at module scope, so the user can do it
    # differently if they import the module and run the scripts as functions.
    # If this isn't called at all, no logging is done.
    # For fancy logging setups (e.g. when running on a big cluster) we could
    # use logging.fileConfig to use a config file to control logging.
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=sys.stdout
    )
    # We do some fancier logging for Script3, just to demonstrate that we can:
    # - we log to both stdout and to a log file
    # - the log file has a lot more (mostly redundant) information
    if not os.path.isdir('output'):
        os.mkdir('output')
    logFile = logging.FileHandler(os.path.join("output", "script3.log"))
    logFile.setFormatter(logging.Formatter("%(name)s[%(levelname)s] %(asctime)s: %(message)s"))
    logging.getLogger("Script3").addHandler(logFile)

    # Script 1: Gaussian galaxy, Gaussian PSF, Gaussian noise.
    if scriptNum == 0 or scriptNum == 1:
        Script1()

    # Script 2: Sheared exponential galaxy, Moffat PSF, Poisson noise.
    if scriptNum == 0 or scriptNum == 2:
        Script2()

    # Script 3: Essentially fully realistic ground-based image.
    if scriptNum == 0 or scriptNum == 3:
        Script3()
        

if __name__ == "__main__":
    main(sys.argv)
