"""
SBProfile-based implementation of SHERA, with comparison to actual
SHERA results.  Currently, can only be run in the examples/ directory as 
python Shera.py data/147246.0_150.416558_1.998697 
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
    s = galsim.Shear()
    s.setG1G2(g1, g2)
    # Rachel is probably using the (1+g, 1-g) form of shear matrix,
    # which means there is some (de)magnification, by my definition:
    e = galsim.Ellipse(s, -(g1*g1+g2*g2), galsim.PositionD(xshift,yshift));

    galaxyImg = galsim.fits.read(rootname + "_masknoise.fits")
    galaxy = galsim.SBInterpolatedImage(galaxyImg, l32d, dxHST, 1.0)
    galaxy.setFlux(0.804*1000.*dxSDSS*dxSDSS)

    psf1Img = galsim.fits.read(rootname + ".psf.fits")
    psf1 = galsim.SBInterpolatedImage(psf1Img, l32d, dxHST, 2.)
    psf1.setFlux(1.)

    psf2Img = galsim.fits.read(rootname + ".sdsspsf.fits")
    psf2 = galsim.SBInterpolatedImage(psf2Img, l32d, dxSDSS, 2.)
    psf2.setFlux(1.)

    outImg = galsim.fits.read(rootname + ".g1_0.02.g2_0.00.fits")
    result = outImg.duplicate()

    psfInv = galsim.SBDeconvolve(psf1)
    deconv = galsim.SBConvolve([galaxy, psfInv])
    sheared = deconv.distort(e)
    out = galsim.SBConvolve([sheared, psf2])

    out.draw(result, dxSDSS)
    result.write(rootname + ".gary.fits")
    result += psfSky
    result -= outImg
    result.write(rootname + ".diff.fits")

if __name__ == "__main__":
    import sys
    main(sys.argv)
