import galsim
import numpy as np
import time


def test_CRG_noise(args):
    """Test noise propagation in ChromaticRealGalaxy
    """
    t0 = time.time()
    print "Constructing simplified HST PSF"
    HST_PSF = galsim.ChromaticAiry(lam=700., diam=2.4)

    print "Constructing simplified Euclid PSF"
    Euclid_PSF = galsim.ChromaticAiry(lam=700., diam=1.2)

    print "Constructing simple filters and SEDs"
    waves = np.arange(550.0, 830.1, 10.0)

    # Construct some simple filters.
    visband = galsim.Bandpass(galsim.LookupTable(waves, np.ones_like(waves), interpolant='linear'))
    rband = visband.truncate(blue_limit=550.0, red_limit=700.0)
    iband = visband.truncate(blue_limit=700.0, red_limit=825.0)

    maxk = max([Euclid_PSF.evaluateAtWavelength(waves[0]).maxK(),
                Euclid_PSF.evaluateAtWavelength(waves[-1]).maxK()])

    const_SED = (galsim.SED(galsim.LookupTable(waves, np.ones_like(waves),
                                               interpolant='linear'))
                 .withFluxDensity(1.0, 700.0))
    linear_SED = (galsim.SED(galsim.LookupTable(waves, (waves-550.0)/(825-550),
                                                interpolant='linear'))
                  .withFluxDensity(1.0, 700.0))

    print "Creating noise field images"
    HST_images = [galsim.Image(args.nx, args.nx, scale=0.03, dtype=np.float64),
                  galsim.Image(args.nx, args.nx, scale=0.03, dtype=np.float64)]

    # Use COSMOS correlated noise
    rng = galsim.BaseDeviate(args.seed)
    xi1 = galsim.getCOSMOSNoise(rng=rng)
    xi2 = xi1.dilate(0.8)

    HST_images[0].addNoise(xi1)
    HST_images[1].addNoise(xi2)

    print "Constructing ChromaticRealGalaxy"
    crg = galsim.ChromaticRealGalaxy((HST_images,
                                      [rband, iband],
                                      [const_SED, linear_SED],
                                      [xi1, xi2],
                                      HST_PSF),
                                     maxk=maxk)

    crg2 = crg.shear(g1=0.2)
    noise = crg.noiseWithPSF(visband, Euclid_PSF, wcs=galsim.Pixel(0.03))
    noise2 = crg2.noiseWithPSF(visband, Euclid_PSF, wcs=galsim.Pixel(0.03))

    print "Convolving by Euclid PSF"
    Euclid_obj = galsim.Convolve(crg, Euclid_PSF)
    Euclid_obj2 = galsim.Convolve(crg2, Euclid_PSF)

    print "Drawing through Euclid filter"
    Euclid_im = Euclid_obj.drawImage(visband, nx=30, ny=30, scale=args.scale, iimult=args.iimult)
    Euclid_im2 = Euclid_obj2.drawImage(visband, nx=30, ny=30, scale=args.scale, iimult=args.iimult)

    print "Get CorrelatedNoise from image"
    xi_obs = galsim.CorrelatedNoise(Euclid_im, correct_periodicity=False)
    xi_obs2 = galsim.CorrelatedNoise(Euclid_im2, correct_periodicity=False)

    for im in HST_images+[Euclid_im, Euclid_im2]:
        im.setCenter(0, 0)
    bd = galsim.BoundsI(-20, 20, -20, 20)

    print "predicted variance: ", noise.getVariance()
    print "observed variance: ", xi_obs.getVariance()
    print "ratio: ", noise.getVariance()/xi_obs.getVariance()
    print "predicted variance: ", noise2.getVariance()
    print "observed variance: ", xi_obs2.getVariance()
    print "ratio: ", noise2.getVariance()/xi_obs2.getVariance()

    print "Took {} seconds".format(time.time()-t0)

    if args.plot:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("r-band image")
        im = ax.imshow(HST_images[0][bd].array)
        plt.colorbar(im)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("i-band image")
        im = ax.imshow(HST_images[1][bd].array)
        plt.colorbar(im)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Euclid image")
        im = ax.imshow(Euclid_im.array)
        plt.colorbar(im)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Observed noise correlation function")
        im = ax.imshow(xi_obs.drawImage().array)
        plt.colorbar(im)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("sheared Euclid image")
        im = ax.imshow(Euclid_im2.array)
        plt.colorbar(im)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Observed sheared noise correlation function")
        im = ax.imshow(xi_obs2.drawImage().array)
        plt.colorbar(im)

        plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nx', type=int, default=256)
    parser.add_argument('--iimult', type=int, default=2)
    args = parser.parse_args()

    test_CRG_noise(args)
