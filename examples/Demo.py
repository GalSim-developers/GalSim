#!/usr/bin/env python
"""
Some example scripts to see some basic usage of the GalSim library.
"""

# List is issues that came up while making this.  These should be spawned into 
# issues on GitHub.
#
# - Need a way to easily specify the size of the output image.
#   This should work to write to a particular portion of a larger image as 
#   well, since that will be the normal thing we'll be doing.
#
# - Probably want to use the name Pixel for what is now Boxcar.  
#   This means that SBPixel should probably also be renamed.
#
# - Probably don't want noise.addGaussian to have mean as a variable, since 
#   it doesn't really make sense.  Also, sigma should be a required variable, 
#   not one with a default value of 1.
#
# - Should we move addGaussian to main galsim namespace?  
#   If so, probably change the name to addGaussianNoise.
#   Maybe even make it a method of image, rather than a free function:
#   image.addGaussianNoise(rng, sigma = 0.3)
#   This seems clearer than galsim.noise.addGaussian(image,rng,simga)
#   Likewise:
#   image.addPoissonNoise(rng, gain = 1.7)
# 
# - For Moffat, Sersic (others?) constructors: 
#   re is not a very clear name for the half-light radius.
#   Should switch this to something more verbose, like half_light_radius.
#
# - Could we overload the + operator for GSObject?
#   So we could define a double gaussian as (atmos1 + atmos2)?
#   Then each gaussian could be separately sheared for example.
#   Or you might have an Airy + Gaussian.  Or a triple Gaussian.
#   Better than defining specific new classes for every combination someone 
#   might want.
#   
# - If not the above, then we need to make GSAdd more user-friendly.
#   Currenly, its only constructor takes an SBAdd object, and we don't 
#   really want the users to have to deal with that.  (Or anything that
#   starts with SB I think.)
#
# - GSConvolve should drop the GS.  GSAdd too.
#
# - Should get python versions of hsm, so we can do the hsm checks
#   directly here in the python code.
#
# - The applyShear function has the "wrong" conventions for the shear.
#   The arguments are taken to be (e1,e2) which are a distortion, not
#   a shear.  Need to fix that.  We can also have applyDistortion if we 
#   want to have something that uses this convention, but it shouldn't 
#   be what applyShear does.


import sys
import os
import subprocess
import math

# This machinery lets us run Python examples even though they aren't positioned
# properly to find galsim as a package in the current directory.
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

class HSM_Moments:
    """
    A class that runs the meas_moments program on an image
    and stores the results.
    This is temporary.  This functionality should be python wrapped.
    """
    
    def __init__(self, file_name):
        proc = subprocess.Popen('../bin/meas_moments %s'%file_name,
            stdout=subprocess.PIPE, shell=True)
        buf = os.read(proc.stdout.fileno(),1000)
        while proc.poll() == None:
            pass
        if proc.returncode != 0:
            raise RuntimeError("meas_moments exited with an error code")

        results = buf.split()
        if results[0] is not '0':
            raise RuntimeError("meas_moments returned an error status")
        self.mxx = float(results[1])
        self.myy = float(results[2])
        self.mxy = float(results[3])
        self.e1 = float(results[4])
        self.e2 = float(results[5])
        # These are distortions e1,e2
        # Find the corresponding shear:
        esq = self.e1*self.e1 + self.e2*self.e2
        e = math.sqrt(esq)
        g = math.tanh(0.5 * math.atanh(e))
        self.g1 = self.e1 * (g/e)
        self.g2 = self.e2 * (g/e)




# Script 1: Simple Gaussian for both galaxy and psf, with Gaussian noise
def Script1():
    """
    About as simple as it gets:
      - Use a circular Gaussian profile for the galaxy.
      - Convolve it by a circular Gaussian PSF.
      - Add Gaussian noise to the image.
    """

    print 'Script 1:'
    print '    Starting script'

    # Define the galaxy profile
    gal = galsim.Gaussian(flux=1000, sigma=2.)

    # Define the PSF profile
    psf = galsim.Gaussian(flux=1., sigma=1.) # psf flux should always = 1

    # Define the pixel size
    pixel_scale = 0.2  # arcsec / pixel
    # Boxcar function to represent this pixellation
    # The pixels could be rectangles, but normally xw = yw = pixel_scale
    pix = galsim.Boxcar(xw=pixel_scale, yw=pixel_scale)

    # Final profile is the convolution of these
    # Can include any number of things in the list, all of which are convolved 
    # together to make the final flux profile.
    final = galsim.GSConvolve([gal, psf, pix])

    # Draw the image with a particular pixel scale
    image = final.draw(dx=pixel_scale)

    # Add some noise to the image
    # First we need to set up a random number generator:
    # Defaut seed is set from the current time.
    rng = galsim.UniformDeviate()
    # Use this to add Gaussian noise with specified sigma
    galsim.noise.addGaussian(image, rng, sigma=0.01)

    # Write the image to a file
    if not os.path.isdir('output'):
        os.mkdir('output')
    file_name = os.path.join('output','demo1.fits')
    image.write(file_name, clobber=True)
    print '    Wrote image to',file_name

    moments = HSM_Moments(file_name)
    print '    HSM reports that the image has measured moments:'
    print '       ',moments.mxx,moments.myy,moments.mxy
    print '    e1,e2 = ',moments.e1,moments.e2
    print '    g1,g2 = ',moments.g1,moments.g2



# Script 2: Sheared, exponential galaxy, Moffat PSF, Poisson noise
def Script2():
    """
    A little bit more sophisticated, but still pretty basic:
      - Use a sheared, exponential profile for the galaxy.
      - Convolve it by a circular Moffat PSF.
      - Add Poisson noise to the image.
    """
    
    print 'Script 2:'
    print '    Starting script'

    # Define the galaxy profile.
    gal = galsim.Exponential(flux=1.e5, r0=2.7)

    # Shear the galaxy by some value.
    g1 = 0.1
    g2 = 0.2
    gal.applyShear(g1, g2)

    # Define the PSF profile.
    psf = galsim.Moffat(beta=5, flux=1., re=1.0)

    # Define the pixel size
    pixel_scale = 0.2  # arcsec / pixel
    pix = galsim.Boxcar(xw=pixel_scale, yw=pixel_scale)

    # Final profile is the convolution of these.
    final = galsim.GSConvolve([gal, psf, pix])

    # Draw the image with a particular pixel scale.
    image = final.draw(dx=pixel_scale)

    # Add a constant sky level to the image.
    sky_level = 1.e4
    # Create an image with the same bounds as image, with a constant
    # sky level.
    sky_image = galsim.ImageF(bounds=image.getBounds(), initValue=sky_level)
    image += sky_image

    # This time use a particular seed, so it the image is deterministic.
    rng = galsim.UniformDeviate(1534225)
    # Use this to add Poisson noise.
    galsim.noise.addPoisson(image, rng, gain=1.)

    # Subtract off the sky.
    image -= sky_image

    # Write the image to a file.
    if not os.path.isdir('output'):
        os.mkdir('output')
    file_name = os.path.join('output', 'demo2.fits')
    image.write(file_name, clobber=True)
    print '    Wrote image to',file_name

    moments = HSM_Moments(file_name)
    print '    HSM reports that the image has measured moments:'
    print '       ',moments.mxx,moments.myy,moments.mxy
    print '    e1,e2 = ',moments.e1,moments.e2
    print '    g1,g2 = ',moments.g1,moments.g2



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

    print 'Script 3:'
    print '    Starting script'

    # Define the galaxy profile.
    gal = galsim.Sersic(3.5, flux=1.e5, re=4.)

    # Shear the galaxy by some value.
    g1 = 0.1
    g2 = -0.2
    gal.applyShear(g1, g2)

    # Define the atmospheric part of the PSF.
    atmos = galsim.atmosphere.DoubleGaussian(
            flux1=0.3, sigma1=2.1, flux2=0.7, sigma2=0.9)
    atmos_g1 = -0.13
    atmos_g2 = -0.09
    atmos.applyShear(atmos_g1 , atmos_g2)

    # Define the pixel scale here, since we need it for the optical PSF
    pixel_scale = 0.23  # arcsec / pixel

    # Define the optical part of the PSF.
    # The first argument of OpticalPSF below is lambda/D,
    # which needs to be in pixel units, so do the calculation:
    lam = 800 * 1.e-9 # 800 nm.  NB: don't use lambda - that's a reserved word.
    D = 4.  # 4 meter telescope, say.
    lam_over_D = lam / D # radians
    lam_over_D *= 206265 # arcsec
    lam_over_D *= pixel_scale # pixels
    print '    lambda over D = ',lam_over_D
    # The rest of the values here should be given in units of the 
    # wavelength of the incident light.
    optics = galsim.OpticalPSF(lam_over_D, 
            defocus=15.0, coma1=6.4, coma2=-3.3, astig1=-2.9, astig2=1.2)

    # Start with square pixels
    pix = galsim.Boxcar(xw=pixel_scale, yw=pixel_scale)
    # Then shear them slightly
    wcs_g1 = -0.02
    wcs_g2 = 0.01
    pix.applyShear(wcs_g1, wcs_g2)

    # Final profile is the convolution of these.
    final = galsim.GSConvolve([gal, atmos, optics, pix])

    # Draw the image with a particular pixel scale.
    image = final.draw(dx=pixel_scale)

    # Add a constant sky level to the image.
    sky_level = 1.e4
    sky_image = galsim.ImageF(bounds=image.getBounds(), initValue=sky_level)
    image += sky_image

    # Add Poisson noise to the image.
    gain = 1.7
    rng = galsim.UniformDeviate(1314662)
    galsim.noise.addPoisson(image, rng, gain=gain)

    # Also add (Gaussian) read noise.
    read_noise = 0.3
    galsim.noise.addGaussian(image, rng, sigma=read_noise)

    # Subtract off the sky.
    image -= sky_image

    # Write the image to a file
    if not os.path.isdir('output'):
        os.mkdir('output')
    file_name = os.path.join('output', 'demo3.fits')
    image.write(file_name, clobber=True)
    print '    Wrote image to',file_name

    moments = HSM_Moments(file_name)
    print '    HSM reports that the image has measured moments:'
    print '       ',moments.mxx,moments.myy,moments.mxy
    print '    e1,e2 = ',moments.e1,moments.e2
    print '    g1,g2 = ',moments.g1,moments.g2



def main(argv):
    try:
        # If no argument, run all scripts (indicated by scriptNum = 0)
        scriptNum = int(argv[1]) if len(argv) > 1 else 0
    except Exception as err:
        print __doc__
        raise err

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
