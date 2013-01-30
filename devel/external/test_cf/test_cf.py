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

# Just some temporary testing stuff
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import galsim

    IMSIZE=40
    
    # First make a noise field with an even number of elements
    enoise = galsim.ImageViewD(np.random.randn(IMSIZE, IMSIZE))
    encf = galsim.correlatednoise.CorrFunc(enoise)
    print encf.SBProfile.xValue(galsim.PositionD(0, 0))

    # Then make a noise field with an odd number of elements
    onoise = galsim.ImageViewD(np.random.randn(IMSIZE + 1, IMSIZE + 1))
    oncf = galsim.correlatednoise.CorrFunc(onoise)
    print oncf.SBProfile.xValue(galsim.PositionD(0, 0))

    testim = galsim.ImageD(10, 10)
    cv = encf.SBProfile.getCovarianceMatrix(testim.view(), dx=1.)
    #plt.pcolor(cv.array); plt.colorbar()

    # Construct an image with noise correlated in the y direction
    plt.figure()
    ynoise = galsim.ImageViewD(enoise.array[:, :] + np.roll(enoise.array, 1, axis=0))
    plt.pcolor(ynoise.array); plt.colorbar()
    plt.title('Noise correlated in y direction')
    plt.savefig('ynoise.png')

    yncf = galsim.correlatednoise.CorrFunc(ynoise)
    yim = galsim.ImageD(IMSIZE, IMSIZE)
    yncf.draw(yim, dx=1.)
    plt.figure()
    plt.pcolor(yim.array); plt.colorbar()
    plt.title('CorrFunc with noise correlated in y direction')
    plt.savefig('ycorrfunc.png')
    ycv = yncf.SBProfile.getCovarianceMatrix(testim.view(), dx=1.)
    plt.figure()
    plt.pcolor(ycv.array); plt.colorbar()
    plt.title('CorrFunc.getCovarianceMatrix() with noise correlated in y direction')
    plt.savefig('ycovmatrix.png')
    
    # Then construct an image with noise correlated in the x direction
    plt.figure()
    xnoise = galsim.ImageViewD(enoise.array[:, :] + np.roll(enoise.array, 1, axis=1))
    plt.pcolor(xnoise.array); plt.colorbar()
    plt.title('Noise correlated in x direction')
    plt.savefig('xnoise.png')

    xncf = galsim.correlatednoise.CorrFunc(xnoise)
    xim = galsim.ImageD(IMSIZE, IMSIZE)
    xncf.draw(xim, dx=1.)
    plt.figure()
    plt.pcolor(xim.array); plt.colorbar()
    plt.title('CorrFunc with noise correlated in x direction')
    plt.savefig('xcorrfunc.png')
    xcv = xncf.SBProfile.getCovarianceMatrix(testim.view(), dx=1.)
    plt.figure()
    plt.pcolor(xcv.array); plt.colorbar()
    plt.title('CorrFunc.getCovarianceMatrix() with noise correlated in x direction')
    plt.savefig('xcovmatrix.png')

    # Plot a hires, rotated, ycf
    yim2 = galsim.ImageD(IMSIZE * 3, IMSIZE * 3)
    yncf2 = yncf.createRotated(20. * galsim.degrees)
    yncf2.draw(yim2, dx=0.4)
    plt.figure()
    plt.pcolor(yim2.array); plt.colorbar()
    plt.title('CorrFunc with noise correlated in y direction (at 0.4 pix resolution, rotated by 20 deg)')
    plt.savefig('ycorrfunc_hires_rot20.png')

