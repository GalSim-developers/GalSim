#!/usr/bin/env python
"""
Make a FITS file of a surface brightness pattern
Usage:  SBDraw <sb_string> <fitsname> [dx] [wmult=1]
sb_string is parsed to define the pattern, enclose in quotes on cmd line
fitsname  is name of output FITS file
dx is pixel scale for output image, default is to choose automatically
wmult is optional integral factor by which to expand image size beyond default
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
    
import pyfits

def main(argv):
    try:
        sbs = argv[1]
        dx = float(argv[3]) if len(argv) > 3 else 0.
        wmult = int(argv[4]) if len(argv) > 4 else 1
        fitsname = argv[2]
    except Exception as err:
        print __doc__
        raise err
    sbp = galsim.SBParse(sbs)
    img = sbp.draw(dx, wmult)
    img.write(fitsname)

if __name__ == "__main__":
    import sys
    main(sys.argv)
