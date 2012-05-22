#!/usr/bin/env python
"""
Simple test of shooting photons to sample from an existing pixelized image.
Usage: ShootInterpolated.py <input FITS> <output FITS> <dx> <dim> <nPhotons> [e1] [e2]
<input FITS> is name of the input FITS image, whose pixel scale will be treated as 1 unit.
<output FITS> is name of FITS file image to be produced by sampling input
dx is pixel scale for output image
dim is number of pixels per side for output image
nPhotons is number of photons to shoot through the image
e1 and e2 are optional shear to be applied while shooting (specified as e1/e2-type distortions)
"""

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
    interp1d = galsim.Delta();
    #interp1d = galsim.Lanczos(5,true);
    #interp1d = galsim.Quintic();
    interp2d = galsim.InterpolantXY(interp1d)

    try:
        inname = argv[1]
        outname = argv[2]
        dxOut = float(argv[3])
        dim = int(argv[4])
        nPhotons = int(argv[5])
        e1 = float(argv[6]) if len(argv) > 6 else 0.
        e2 = float(argv[7]) if len(argv) > 7 else 0.
    except Exception as err:
        print __doc__
        raise err

    galaxyImg = galsim.fits.read(inname)
    galaxy = galsim.SBInterpolatedImage(galaxyImg, interp2d, 1., 1.0)
    shearedGalaxy = galaxy.shear(e1,e2)

    rng = galsim.UniformDeviate(1534225)
    bounds = galsim.BoundsI(-dim/2, dim/2+1, -dim/2, dim/2+1)
    img = galsim.ImageF(bounds)
    img.setScale(dxOut)
    shearedGalaxy.drawShoot(img, nPhotons, rng)
    img.write(outname)

if __name__ == "__main__":
    import sys
    main(sys.argv)

if __name__ == "__main__":
    import sys
    main(sys.argv)
