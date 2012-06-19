"""
A script to make some example images for Claire to use, to test how her bulge/disk decompositions
compare with our simulated bulge/disk galaxies with an ACS PSF.  The purpose of this comparison is
(a) to validate against a known and well-tested galaxy simulation code, (b) to make sure that if
we have galaxy model fits to the COSMOS images, that we can reconstruct what the best-fit galaxy
images must look like (i.e., that our conventions for radii definitions and so on are correct), and
(c) to understand if there are any regimes where bulge/disk decompositions have problematic
parameter degeneracies.  For more information, see
https://github.com/GalSim-developers/GalSim/issues/96

The comparison did not reveal any major systematic issues.
"""

import sys
import os
import numpy as np
import math

# This machinery lets us run Python examples even though they aren't positioned
# properly to find galsim as a package in the current directory.
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# Set up input parameter values
if not os.path.isdir('output'):
    os.mkdir('output')
PSFFile = os.path.join('data','147246.0_150.416558_1.998697.psf.fits')
outDir = os.path.join('output','testImage.')

# galaxy parameters
bulge2Total = [0.0, 1.0/3, 2.0/3, 1.0]
bulgeEllip = [0.2]
diskEllip = [0.2, 0.45, 0.7]
invSN = [0.00, 0.01, 0.02]
nRealization = [1, 10, 10]
if len(invSN) is not len(nRealization):
    raise RuntimeError("Grids in inverse S/N and number of noise realizations do not have same size!")
diskRe = [0.5, 1.0]
bulgeRe = [0.25, 0.5]
if len(diskRe) is not len(bulgeRe):
    raise RuntimeError("Grids in bulge and disk scale lengths do not have same size!")
nBulge = 4.0
nDisk = 1.0
pixelScale = 0.03 # we are simulating ACS data
totFlux = 1000.0 # total flux for galaxy

#### note: the following parameters were not used ####
sigmaBulge = 1.0 # dispersion in Sersic n for bulge
sigmaDisk = 0.2 # dispersion in Sersic n for disk

# read in ePSF and normalize (note: already includes pixel response, don't have to do separately)
l3 = galsim.Lanczos(3, True, 1.0E-4)
l32d = galsim.InterpolantXY(l3)
PSFImage = galsim.fits.read(PSFFile)
PSF = galsim.SBInterpolatedImage(PSFImage, l32d, pixelScale, 2.)
PSF.setFlux(1.0)

# Loop over the various grids: values of bulge-to-total ratio, bulge ellipticity, disk ellipticity,
# S/N and number of noise realizations, radii
rng = galsim.UniformDeviate()
for bt in bulge2Total:
    print "Beginning bulge-to-total ratio of ",bt
    for bell in bulgeEllip:
        print "Beginning bulge ellipticity of ",bell
        for dell in diskEllip:
            print "Beginning disk ellipticity of ",dell
            for dreind in range(len(diskRe)):
                print "Beginning disk half-light radius of ",diskRe[dreind]

                # Make the bulge: use a Sersic rather than a DeVauc specifically, because we want to
                # allow more general bulges with other values of n
                bulge = galsim.Sersic(nBulge, flux=totFlux, half_light_radius=bulgeRe[dreind])
                
                # make it non-circular; choose a random position angle for this galaxy, and shear
                posAngle = np.pi*rng()
                be1 = bell*np.cos(2.0*posAngle)
                be2 = bell*np.sin(2.0*posAngle)
                besq = be1*be1 + be2*be2
                if besq > 0.:
                    be = np.sqrt(besq)
                    bg = math.tanh(0.5 * math.atanh(be))
                    bg1 = be1 * (bg/be)
                    bg2 = be2 * (bg/be)
                else:
                    bg1 = 0.0
                    bg2 = 0.0
                bulge.applyShear(g1=bg1, g2=bg2)

                # Make the disk
                disk = galsim.Sersic(nDisk, flux=totFlux, half_light_radius=diskRe[dreind])

                # make it non-circular, using the same position angle as for the bulge
                de1 = dell*np.cos(2.0*posAngle)
                de2 = dell*np.sin(2.0*posAngle)
                desq = de1*de1 + de2*de2
                if desq > 0.:
                    de = np.sqrt(desq)
                    dg = math.tanh(0.5 * math.atanh(de))
                    dg1 = de1 * (dg/de)
                    dg2 = de2 * (dg/de)
                else:
                    dg1 = 0.0
                    dg2 = 0.0
                disk.applyShear(g1=dg1, g2=dg2)
  
                # Rescale fluxes and add: use the overloaded multiplication and addition operators
                galaxy = bt*bulge + (1.0-bt)*disk

                # Convolve with PSF, and draw image
                convGalaxy = galsim.SBConvolve([PSF,galaxy.SBProfile])
                convGalaxyImg = convGalaxy.draw(dx = pixelScale)

                # More noise realizations?
                for invsnind in range(len(invSN)):
                    print "Beginning inverse S/N of ",invSN[invsnind]
                    if invSN[invsnind] > 0.0:
                        # Choose Gaussian sigma per pixel based on GREAT08-style S/N definition
                        gaussSig = invSN[invsnind] * np.sqrt(np.sum(convGalaxyImg.array**(2.0)))

                        # Add noise the appropriate number of times, and write each one to file
                        for iRealization in range(nRealization[invsnind]):
                            tmpImg = convGalaxyImg.copy()
                            tmpImg.addNoise(galsim.GaussianDeviate(rng, mean=0.0, sigma=gaussSig))
                            outFile = outDir + ('BT%5.3f.' % bt)
                            outFile += 'bulgeellip%5.3f.' % bell
                            outFile += 'diskellip%5.3f.' % dell
                            outFile += 'diskRe%5.3f.' % diskRe[dreind]
                            outFile += 'invSNR%5.3f.' % invSN[invsnind]
                            outFile += 'image%02d.fits' % iRealization
                            tmpImg.write(outFile, clobber=True)
                            print 'Wrote image to file %s' % outFile
                    else:
                        # Just write to file without adding noise
                            outFile = outDir + 'BT%5.3f.' % bt
                            outFile += 'bulgeellip%5.3f.' % bell
                            outFile += 'diskellip%5.3f.' % dell
                            outFile += 'diskRe%5.3f.' % diskRe[dreind]
                            outFile += 'invSNR%5.3f.' % invSN[invsnind]
                            outFile += 'image00.fits'
                            convGalaxyImg.write(outFile, clobber=True)
                            print 'Wrote image to file %s' % outFile
                        
# For infinite S/N case, make some dispersion around the bulge and disk Sersic n values: later?
