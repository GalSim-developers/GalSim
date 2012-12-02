# Just some temporary testing stuff
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import galsim

    IMSIZE=40
    
    # First make a noise field with an even number of elements
    enoise = galsim.ImageViewD(np.random.randn(IMSIZE, IMSIZE))
    encf = galsim.correlatednoise.CorrFunc(enoise)
    plt.figure()
    plt.pcolor(encf.cf_array)
    plt.colorbar()
    print encf.SBProfile.xValue(galsim.PositionD(0, 0))

    # Then make a noise field with an odd number of elements
    onoise = galsim.ImageViewD(np.random.randn(IMSIZE + 1, IMSIZE + 1))
    oncf = galsim.correlatednoise.CorrFunc(onoise)
    plt.figure()
    plt.pcolor(oncf.cf_array)
    plt.colorbar()
    print oncf.SBProfile.xValue(galsim.PositionD(0, 0))
    plt.show()


    
