"""
Demo #5

The fifth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script is intended to mimic a Great08 challenge LowNoise image.  We produce a single image
made up of tiles of postage stamps for each individual object.  (We only do 10 x 10 
postage stamps rather than 100 x 100 as they did in the interest of time.)  Each postage
stamp is 40 x 40 pixels.  One image is all stars.  A second image is all galaxies.
The stars are truncated Moffat profiles.  The galaxies are Exponential profiles.
(Great08 mixed pure bulge and pure disk for its LowNoise run.  We're just doing disks to 
make things simpler.)  The galaxies are oriented randomly, but in 90 degree-rotated pairs to 
cancel the effect of shape noise.  The applied shear is the same for each galaxy.  

New features introduced in this demo:

- ud = galsim.UniformDeviate(seed)
- gd = galsim.GaussianDeviate(ud, sigma)
- ccdnoise = galsim.CCDNoise(ud)
- image *= scalar
- bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
- pos = bounds.center()
- pos.x, pos.y
- sub_image = image[bounds]
- obj2 = obj.createSheared(e,beta)

- Building a single large image, and access sub-images within it
- Set the galaxy size based on the psf size and a resolution factor.
- Set the object's flux according to a target S/N vale.
- Use 90 degree-rotated pairs for the intrinsic galaxy shapes
- Shift by a random (dx,dy) drawn from a unit circle top hat.
"""

import sys
import os
import math
import numpy
import logging
import time
import galsim

def main(argv):
    """
    Make images similar to that done for the Great08 challenge:
      - Each fits file is 10 x 10 postage stamps.
        (The real Great08 images are 100x100, but in the interest of making the Demo
         script a bit quicker, we only build 100 stars and 100 galaxies.)
      - Each postage stamp is 40 x 40 pixels.
      - One image is all stars.
      - A second image is all galaxies.
      - Applied shear is the same for each galaxy.
      - Galaxies are oriented randomly, but in pairs to cancel shape noise.
      - Noise is poisson using a nominal sky value of 1.e6.
      - Galaxies are Exponential profiles.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo5")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    nx_stamps = 10                  #
    ny_stamps = 10                  #
    nx_pixels = 40                  #
    ny_pixels = 40                  #

    random_seed = 6424512           #

    pixel_scale = 1.0               # arcsec / pixel
    sky_level = 1.e6                # ADU / arcsec^2

    psf_file_name = os.path.join('output','g08_psf.fits')
    psf_beta = 3                    #
    psf_fwhm = 2.85                 # arcsec (=pixels)
    psf_trunc = 2.*psf_fwhm         # arcsec (=pixels)
    psf_e1 = -0.019                 #
    psf_e2 = -0.007                 #

    gal_file_name = os.path.join('output','g08_gal.fits')
    gal_signal_to_noise = 200       # Great08 "LowNoise" run
    gal_resolution = 0.98           # r_gal / r_psf (use r = half_light_radius)
    # Note: Great08 defined their resolution as r_obs / r_psf, using the convolved 
    #       size rather than the pre-convolved size.  
    #       Therefore, our r_gal/r_psf = 0.98 approximately corresponds to 
    #       their r_obs / r_psf = 1.4.

    gal_ellip_rms = 0.2             # using "shear" definition of ellipticity.
    gal_ellip_max = 0.6             #
    gal_g1 = 0.013                  #
    gal_g2 = -0.008                 #

    centroid_shift = 1.0            # arcsec (=pixels)

    logger.info('Starting demo script 5 using:')
    logger.info('    - image with %d x %d postage stamps',nx_stamps,ny_stamps)
    logger.info('    - postage stamps of size %d x %d pixels',nx_pixels,ny_pixels)
    logger.info('    - Moffat PSF (beta = %.1f, FWHM = %.2f, trunc = %.2f),',
            psf_beta,psf_fwhm,psf_trunc)
    logger.info('    - PSF ellip = (%.3f,%.3f)',psf_e1,psf_e2)
    logger.info('    - Exponential galaxies')
    logger.info('    - Resolution (r_gal / r_psf) = %.2f',gal_resolution)
    logger.info('    - Ellipticities have rms = %.1f, max = %.1f',
            gal_ellip_rms, gal_ellip_max)
    logger.info('    - Applied gravitational shear = (%.3f,%.3f)',gal_g1,gal_g2)
    logger.info('    - Poisson noise (sky level = %.1e).', sky_level)
    logger.info('    - Centroid shifts up to = %.2f pixels',centroid_shift)


    # Define the PSF profile
    psf = galsim.Moffat(beta=psf_beta, fwhm=psf_fwhm, trunc=psf_trunc)

    # When something can be constructed from multiple sizes, e.g. Moffat, then
    # you can get any size out even if it wasn't the way the object was constructed.
    # In this case, we extract the half-light radius, even though we built it with fwhm.
    # We'll use this later to set the galaxy's half-light radius in terms of a resolution.
    psf_re = psf.getHalfLightRadius()

    psf.applyShear(e1=psf_e1,e2=psf_e2)
    logger.debug('Made PSF profile')

    pix = galsim.Pixel(pixel_scale)
    logger.debug('Made pixel profile')

    final_psf = galsim.Convolve([psf,pix])
    logger.debug('Made final_psf profile')


    # Define the galaxy profile

    # First figure out the size we need from the resolution
    gal_re = psf_re * gal_resolution

    # Make the galaxy profile starting with flux = 1.
    gal = galsim.Exponential(flux=1., half_light_radius=gal_re)
    logger.debug('Made galaxy profile')

    # This profile is placed with different orientations and noise realizations
    # at each postage stamp in the gal image.
    gal_image = galsim.ImageF(nx_pixels * nx_stamps-1 , ny_pixels * ny_stamps-1)
    psf_image = galsim.ImageF(nx_pixels * nx_stamps-1 , ny_pixels * ny_stamps-1)

    centroid_shift_sq = centroid_shift**2

    first_in_pair = True  # Make pairs that are rotated by 45 degrees

    k = 0
    for ix in range(nx_stamps):
        for iy in range(ny_stamps):
            # The normal procedure for setting random numbers in GalSim is to start a new
            # random number generator for each object using sequential seed values.
            # This sounds weird at first (especially if you were indoctrinated by Numerical 
            # Recipes), but for the boost random number generator we use, the "random" 
            # number sequences produced from sequential initial seeds are highly uncorrelated.
            # 
            # The reason for this procedure is that when we use multiple processes to build
            # our images, we want to make sure that the results are deterministic regardless
            # of the way the objects get parcelled out to the different processes. 
            #
            # Of course, this script isn't using multiple processes, so it isn't required here.
            # However, we do it nonetheless in order to get the same results as the config
            # version of this demo script (demo5.yaml).
            ud = galsim.UniformDeviate(random_seed+k)

            # Any kind of random number generator can take another rng as its first 
            # argument rather than a seed value.  This makes both objects use the same
            # underlying generator for their pseudo-random values.
            gd = galsim.GaussianDeviate(ud, sigma=gal_ellip_rms)

            # The -1's in the next line are to provide a border of
            # 1 pixel between postage stamps
            b = galsim.BoundsI(ix*nx_pixels+1 , (ix+1)*nx_pixels-1, 
                               iy*ny_pixels+1 , (iy+1)*ny_pixels-1)
            sub_gal_image = gal_image[b]
            sub_psf_image = psf_image[b]

            # Great08 randomized the locations of the two galaxies in each pair,
            # but for simplicity, we just do them in sequential postage stamps.
            if first_in_pair:
                # Use a random orientation:
                beta = ud() * 2. * math.pi * galsim.radians
                #print 'beta = ',beta

                # Determine the ellipticity to use for this galaxy.
                ellip = 1
                while (ellip > gal_ellip_max):
                    # Don't do `ellip = math.fabs(gd())`
                    # Python basically implements this as a macro, so gd() is called twice!
                    val = gd()
                    ellip = math.fabs(val)
                #print 'ellip = ',ellip

                first_in_pair = False
            else:
                # Use the previous ellip and beta + 90 degrees
                beta += 90 * galsim.degrees
                #print 'ring beta = ',beta
                #print 'ring ellip = ',ellip
                first_in_pair = True

            # Make a new copy of the galaxy with an applied e1/e2-type distortion 
            # by specifying the ellipticity and a real-space position angle
            this_gal = gal.createSheared(e=ellip, beta=beta)

            # Apply the gravitational reduced shear by specifying g1/g2
            this_gal.applyShear(g1=gal_g1, g2=gal_g2)
            #print 'g1,g2 = ',gal_g1,gal_g2

            # Apply a random centroid shift:
            rsq = 2 * centroid_shift_sq
            while (rsq > centroid_shift_sq):
                dx = (2*ud()-1) * centroid_shift
                dy = (2*ud()-1) * centroid_shift
                rsq = dx**2 + dy**2
            #print 'dx,dy = ',dx,dy

            this_gal.applyShift(dx,dy)
            this_psf = final_psf.createShifted(dx,dy)

            # Make the final image, convolving with psf and pixel
            final_gal = galsim.Convolve([psf,pix,this_gal])

            # Draw the image
            #print 'pixel_scale = ',pixel_scale
            final_gal.draw(sub_gal_image, dx=pixel_scale)

            # Now determine what we need to do to get our desired S/N
            # There are lots of definitions of S/N, but here is the one used by Great08
            # We use a weighted integral of the flux:
            # S = sum W(x,y) I(x,y) / sum W(x,y)
            # N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
            # Now we assume that Var(I(x,y)) is dominated by the sky noise, so
            # Var(I(x,y)) = sky_level
            # We also assume that we are using a matched filter for W, so W(x,y) = I(x,y).
            # Then a few things cancel and we find that
            # S/N = sqrt( sum I(x,y)^2 / sky_level )
            sky_level_pix = sky_level * pixel_scale**2
            sn_meas = math.sqrt( numpy.sum(sub_gal_image.array**2) / sky_level_pix )
            flux = gal_signal_to_noise / sn_meas
            # Now we rescale the flux to get our desired S/N
            #print 'noise_var = ',sky_level_pix
            #print 'sn_meas = ',sn_meas
            #print 'flux = ',flux
            sub_gal_image *= flux

            # Add Poisson noise -- the CCDNoise can also take another rng as its argument
            # so it will be part of the same stream of random numbers as ud and gd.
            sub_gal_image += sky_level_pix
            sub_gal_image.addNoise(galsim.CCDNoise(ud))
            sub_gal_image -= sky_level_pix

            # Draw the PSF image
            # No noise on PSF images.  Just draw it as is.
            this_psf.draw(sub_psf_image, dx=pixel_scale)

            # for first instance, measure moments
            if ix==0 and iy==0:
                psf_shape = sub_psf_image.FindAdaptiveMom()
                g_to_e = psf_shape.observed_shape.getG() / psf_shape.observed_shape.getE()
                logger.info('Measured best-fit elliptical Gaussian for first PSF image: ')
                logger.info('  g1, g2, sigma = %7.4f, %7.4f, %7.4f (pixels)',
                            g_to_e*psf_shape.observed_shape.getE1(),
                            g_to_e*psf_shape.observed_shape.getE2(), psf_shape.moments_sigma)

            x = b.center().x
            y = b.center().y
            logger.info('Galaxy (%d,%d): center = (%.0f,%0.f)  (e,beta) = (%.4f,%.3f)',
                    ix,iy,x,y,ellip,beta/galsim.radians)
            k = k+1

    logger.info('Done making images of postage stamps')

    # Now write the images to disk.
    psf_image.write(psf_file_name, clobber=True)
    logger.info('Wrote PSF file %s',psf_file_name)

    gal_image.write(gal_file_name, clobber=True)
    logger.info('Wrote image to %r',gal_file_name)  # using %r adds quotes around filename for us


if __name__ == "__main__":
    main(sys.argv)
