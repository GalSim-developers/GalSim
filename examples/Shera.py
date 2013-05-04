# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""
SBProfile-based implementation of SHERA, with comparison to actual
SHERA results.  Currently, can only be run in the examples/ directory as 
python Shera.py data/147246.0_150.416558_1.998697 

Note that a better example of how to manipulate real galaxy training data with GalSim can be found
in RealDemo.py or in demos 6 and 10.  In particular, those demos use the RealGalaxy base class which
was has functions that automatically carry out many of the steps below.
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

    l3 = galsim.Lanczos(3, True, 1.0E-4)
    l32d = galsim.InterpolantXY(l3)

    dxHST = 0.03
    dxSDSS = 0.396
    g1 = 0.02
    g2 = 0.0
    psfSky = 1000.0
    
    rootname = argv[1]
    xshift = float(argv[2]) if len(argv) > 2 else 0.
    yshift = float(argv[3]) if len(argv) > 3 else 0.
    s = galsim.Shear(g1=g1, g2=g2)

    # Rachel is probably using the (1+g, 1-g) form of shear matrix,
    # which means there is some (de)magnification, by my definition:
    #e = galsim.Ellipse(s, -(g1*g1+g2*g2), galsim.PositionD(xshift,yshift));

    galaxyImg = galsim.fits.read(rootname + "_masknoise.fits")
    galaxy = galsim.InterpolatedImage(galaxyImg, x_interpolant=l32d, dx=dxHST, flux=0.804*1000.*dxSDSS*dxSDSS)

    psf1Img = galsim.fits.read(rootname + ".psf.fits")
    psf1 = galsim.InterpolatedImage(psf1Img, x_interpolant=l32d, dx=dxHST, flux=1.)

    psf2Img = galsim.fits.read(rootname + ".sdsspsf.fits")
    psf2 = galsim.InterpolatedImage(psf2Img, x_interpolant=l32d, dx=dxSDSS, flux=1.)

    outImg = galsim.fits.read(rootname + ".g1_0.02.g2_0.00.fits")
    result = outImg.copy()

    psfInv = galsim.Deconvolve(psf1)
    deconv = galsim.Convolve(galaxy, psfInv)
    sheared = deconv.createTransformed(e)
    out = galsim.Convolve(sheared, psf2)

    test_outImg = out.draw(result, dx=dxSDSS)
    test_outImg.write(rootname + ".gary.fits")
    result += psfSky
    result -= test_outImg
    result.write(rootname + ".diff.fits")

if __name__ == "__main__":
    import sys
    main(sys.argv)
