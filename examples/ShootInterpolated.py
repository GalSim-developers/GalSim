#!/usr/bin/env python
"""
Simple test of shooting photons to sample from an existing pixelized image.
Usage: ShootInterpolated.py <input FITS> <output FITS> <dx> <dim> <nPhotons> [g1] [g2]
<input FITS> is name of the input FITS image, whose pixel scale will be treated as 1 unit.
<output FITS> is name of FITS file image to be produced by sampling input
dx is pixel scale for output image
dim is number of pixels per side for output image
nPhotons is number of photons to shoot through the image
g1 and g2 are reduced shears to be applied while shooting
"""

# Example usage:
# python ShootInterpolated.py data/147246.0_150.416558_1.998697_masknoise.fits tmp.fits 1 500 1000000


# This machinery lets us run Python examples even though they aren't positioned
# properly to find galsim as a package in the current directory.
try:
    import galsim
except ImportError:
    import sys
    import os
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def main(argv):


    # Try different interpolants!
    #interp1d = galsim.Linear();
    #interp1d = galsim.Delta();
    interp1d = galsim.Lanczos(5,conserve_flux=True);
    #interp1d = galsim.Quintic();
    interp2d = galsim.InterpolantXY(interp1d)

    try:
        inname = argv[1]
        outname = argv[2]
        dxOut = float(argv[3])
        dim = int(argv[4])
        nPhotons = int(argv[5])
        g1 = float(argv[6]) if len(argv) > 6 else 0.
        g2 = float(argv[7]) if len(argv) > 7 else 0.
    except Exception as err:
        print __doc__
        raise err

    galaxyImg = galsim.fits.read(inname)
    galaxy = galsim.SBInterpolatedImage(galaxyImg, interp2d, dx=1., pad_factor=1.0)
    galaxy.applyShear(galsim.Shear(g1=g1,g2=g2)._shear)

    rng = galsim.UniformDeviate(1534225)
    bounds = galsim.BoundsI(-dim/2, dim/2+1, -dim/2, dim/2+1)
    img = galsim.ImageF(bounds)
    img.setScale(dxOut)
    galaxy.drawShoot(img, nPhotons, rng)
    img.write(outname)

if __name__ == "__main__":
    import sys
    main(sys.argv)
