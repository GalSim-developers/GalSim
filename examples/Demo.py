#!/usr/bin/env python
"""
Some example scripts to see some basic usage of the GalSim library.
"""

import sys
import os

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

    # Define the galaxy profile
    gal = galsim.Gaussian(flux = 100, sigma = 2.)

    # Define the PSF profile
    psf = galsim.Gaussian(flux = 1., sigma = 1.) # psf flux should always = 1

    # Final profile is the convolution of these
    # TODO: Are we ever going to convolve more than 2 things at a time?
    #       The syntax would be cleaner without the brackets [].
    #       Also, should remove the GS prefix.
    final = galsim.GSConvolve([gal,psf])

    # Draw the image with a particular pixel scale
    # TODO: How do we specify the size of the image that gets created?
    image = final.draw(dx = 0.2)

    # Add some noise to the image
    # First we need to set up a random number generator:
    # Defaut seed is set from the current time.  Can also speficy seed as argument.
    rng = galsim.UniformDeviate()
    # Use this to add Gaussian noise with specified sigma
    # TODO: Not sure if we want mean as an option to addGaussian.
    #       Furthermore, we probably want to have sigma be required so we don't 
    #       accidentally forget to specify it.
    # TODO: I can't get this to work.  This seems to be the right signature 
    #       from noise.py, but here is what I get from running it:
    # Traceback (most recent call last):
    #   File "./Demo.py", line 123, in <module>
    #     main(sys.argv)
    #   File "./Demo.py", line 113, in main
    #     Script1()
    #   File "./Demo.py", line 60, in Script1
    #     addGaussian(image, rng, sigma=0.1)
    #   NameError: global name 'addGaussian' is not defined
    #addGaussian(image, rng, sigma=0.1)

    # Write the image to a file
    fileName = os.path.join('output','demo1.fits')
    # TODO: It should be easier than this to specify that we want to overwrite
    #       an existing file.
    if os.path.exists(fileName):
        os.remove(fileName)
    image.write(fileName)


# Script 2: Sheared, exponential galaxy, Moffat PSF, Poisson noise
def Script2():
    """
    A little bit more sophisticated, but still pretty basic:
      - Use a sheared, exponential profile for the galaxy.
      - Convolve it by a circular Moffat PSF.
      - Add Poisson noise to the image.
    """

    # Define the galaxy profile
    gal = galsim.Exponential(flux = 1.e5, r0 = 2.7)

    # Shear the galaxy by some value:
    # TODO: Double check.  These are called e1,e2 in the signature.
    #       Are they distortions or shears?
    g1 = 0.1
    g2 = 0.2
    gal.applyShear(g1,g2)

    # Define the PSF profile
    # TODO: re is not very clear for the half-light radius.
    #       Should switch this to something more verbose, like half_light_radius
    psf = galsim.Moffat(beta = 5, flux = 1., re = 1.)

    # Final profile is the convolution of these
    final = galsim.GSConvolve([gal,psf])

    # Draw the image with a particular pixel scale
    # TODO: Again, how do we specify the size of the image that gets created?
    #       This image is much larger than the one for script 1.
    image = final.draw(dx = 0.2)

    # Add some noise to the image

    # For Poisson noise, we want to have a sky level as well.
    skyLevel = 1.e4
    # TODO: How do we add a constant sky level to the image?

    # This time use a particular seed, so it the image is deterministic
    rng = galsim.UniformDeviate(1534225)
    # Use this to add Poisson noise
    # TODO: This also doesn't work.  Same problem as above for addGaussian.
    #addPoisson(image, rng, gain = 1.)

    # Write the image to a file
    fileName = os.path.join('output','demo2.fits')
    if os.path.exists(fileName):
        os.remove(fileName)
    image.write(fileName)

    
def main(argv):
    try:
        # If no argument, run all scripts (indicated by scriptNum = 0)
        scriptNum = int(argv[1]) if len(argv) > 1 else 0
    except Exception as err:
        print __doc__
        raise err

    # Script 1: Gaussian galaxy, Gaussian PSF, Gaussian noise:
    if scriptNum == 0 or scriptNum == 1:
        Script1()

    # Script 2: Sheared exponential galaxy, Moffat PSF, Poisson noise:
    if scriptNum == 0 or scriptNum == 2:
        Script2()
        



if __name__ == "__main__":
    main(sys.argv)
