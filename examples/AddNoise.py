#!/usr/bin/env python
"""
Make a square FITS image of constant value, then add noise to it
Usage:  AddNoise <dimension> <fitsname> <value> <gain> <readNoise>
dimension is number of pixels per side in image
fitsname  is name of output FITS file
value     is (pre-noise) value of image
gain      is number of e per ADU in image
readNoise is standard deviation of Gaussian noise, in e, (or in ADU, if gain<=0).
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
        dim = int(argv[1])
        fitsname = argv[2]
        value = float(argv[3])
        gain = float(argv[4])
        rn = float(argv[5])
    except Exception as err:
        print __doc__
        raise err
    img = galsim.Image<float>(dim, dim);
    img.fill(value)
    img.write(fitsname)

if __name__ == "__main__":
    import sys
    main(sys.argv)
