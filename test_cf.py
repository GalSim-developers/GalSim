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
    out = encf.SBProfile.getCovarianceMatrix(testim.view())
    plt.pcolor(out.array); plt.colorbar()
    
    plt.show()


    
