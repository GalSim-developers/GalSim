#!/usr/bin/env python
"""
Simple test of shooting photons to sample from an existing pixelized image.
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

    # translation from C++ by Jim; comments after this one are in Gary's voice

    #linear = galsim.Lanczos(5);
    linear = galsim.Linear();
    bilinear = galsim.InterpolantXY(linear)

    inname = argv[1]
    outname = argv[2]
    dxOut = float(argv[3])
    dim = int(argv[4])
    nPhotons = int(argv[5])
    e1 = float(argv[6]) if len(argv) > 6 else 0.
    e2 = float(argv[7]) if len(argv) > 7 else 0.

    galaxyImg = galsim.fits.read(inname)
    galaxy = galsim.SBInterpolatedImage(galaxyImg, bilinear, 1., 1.0)
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
