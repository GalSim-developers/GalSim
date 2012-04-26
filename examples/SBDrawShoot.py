#!/usr/bin/env python
"""
Make a FITS file of a surface brightness pattern using photon shooting
Usage:  SBDraw <sb_string> <fitsname> <dx> <dim> <nPhotons>
sb_string is parsed to define the pattern, enclose in quotes on cmd line
fitsname  is name of output FITS file
dx is pixel scale for output image
dim is number of pixels per side for FITS image
nPhotons is number of photons to shoot through the image
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
    try:
        sbs = argv[1]
        fitsname = argv[2]
        dx = float(argv[3]) 
        dim = int(argv[4]) 
        nPhotons = int(argv[5]) 
    except Exception as err:
        print __doc__
        raise err
    rng = galsim.UniformDeviate(1534225)
    sbp = galsim.SBParse(sbs)
    bounds = galsim.BoundsI(-dim/2, dim/2+1, -dim/2, dim/2+1)
    img = galsim.ImageF(bounds)
    img.setScale(dx)
    sbp.drawShoot(img, nPhotons, rng)
    img.write(fitsname)

if __name__ == "__main__":
    import sys
    main(sys.argv)
