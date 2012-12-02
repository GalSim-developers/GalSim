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

    yncf = galsim.correlatednoise.CorrFunc(ynoise)
    yim = galsim.ImageD(IMSIZE, IMSIZE)
    yncf.draw(yim, dx=1.)
    plt.figure()
    plt.pcolor(yim.array); plt.colorbar()
    plt.title('CorrFunc with noise correlated in y direction')
    ycv = yncf.SBProfile.getCovarianceMatrix(testim.view(), dx=1.)
    plt.figure()
    plt.pcolor(ycv.array); plt.colorbar()
    plt.title('CorrFunc.getCovarianceMatrix() with noise correlated in y direction')

    # Then construct an image with noise correlated in the x direction
    plt.figure()
    xnoise = galsim.ImageViewD(enoise.array[:, :] + np.roll(enoise.array, 1, axis=1))
    plt.pcolor(xnoise.array); plt.colorbar()
    plt.title('Noise correlated in x direction')

    xncf = galsim.correlatednoise.CorrFunc(xnoise)
    xim = galsim.ImageD(IMSIZE, IMSIZE)
    xncf.draw(xim, dx=1.)
    plt.figure()
    plt.pcolor(xim.array); plt.colorbar()
    plt.title('CorrFunc with noise correlated in x direction')
    xcv = xncf.SBProfile.getCovarianceMatrix(testim.view(), dx=1.)
    plt.figure()
    plt.pcolor(xcv.array); plt.colorbar()
    plt.title('CorrFunc.getCovarianceMatrix() with noise correlated in x direction')
    plt.show()


    
