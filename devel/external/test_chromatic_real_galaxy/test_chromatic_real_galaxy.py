import galsim
import numpy as np


def test_CRG_noise(plot=False):
    """Test noise propagation in ChromaticRealGalaxy
    """
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
    HST_images = [galsim.Image(192, 192, scale=0.03, dtype=np.float64),
                  galsim.Image(192, 192, scale=0.03, dtype=np.float64)]
    # Use COSMOS correlated noise
    rng = galsim.BaseDeviate(1)
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
    noise = crg.noiseWithPSF(visband, Euclid_PSF)
    noise2 = crg2.noiseWithPSF(visband, Euclid_PSF)

    print "Convolving by Euclid PSF"
    Euclid_obj = galsim.Convolve(crg, Euclid_PSF)
    Euclid_obj2 = galsim.Convolve(crg2, Euclid_PSF)

    print "Drawing through Euclid filter"
    Euclid_im = Euclid_obj.drawImage(visband, nx=64, ny=64, scale=0.03, iimult=2)
    Euclid_im2 = Euclid_obj2.drawImage(visband, nx=64, ny=64, scale=0.03, iimult=2)

    print "Get CorrelatedNoise from image"
    xi_obs = galsim.CorrelatedNoise(Euclid_im, correct_periodicity=False)
    xi_obs2 = galsim.CorrelatedNoise(Euclid_im2, correct_periodicity=False)

    for im in HST_images+[Euclid_im, Euclid_im2]:
        im.setCenter(0, 0)
    bd = galsim.BoundsI(-20, 20, -20, 20)

    if plot:
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
        im = ax.imshow(Euclid_im[bd].array)
        plt.colorbar(im)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Observed noise correlation function")
        im = ax.imshow(xi_obs.drawImage().array)
        plt.colorbar(im)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("sheared Euclid image")
        im = ax.imshow(Euclid_im2[bd].array)
        plt.colorbar(im)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Observed sheared noise correlation function")
        im = ax.imshow(xi_obs2.drawImage().array)
        plt.colorbar(im)

        plt.show()


def test_CRG(plot=False):
    """Use some simplified simulated HST-like observations around r and i band to predict
    Euclid-ish visual band observations."""

    print "Constructing simplified HST PSF"
    HST_PSF = galsim.ChromaticAiry(lam=700, diam=2.4)

    print "Constructing simplified Euclid PSF"
    Euclid_PSF = galsim.ChromaticAiry(lam=700, diam=1.2)

    print "Constructing simple filters and SEDs"
    waves = np.arange(550.0, 830.1, 10.0)

    # Construct some simple filters.
    visband = galsim.Bandpass(galsim.LookupTable(waves, np.ones_like(waves), interpolant='linear'))
    rband = visband.truncate(blue_limit=550.0, red_limit=700.0)
    iband = visband.truncate(blue_limit=700.0, red_limit=825.0)

    const_SED = (galsim.SED(galsim.LookupTable(waves, np.ones_like(waves),
                                               interpolant='linear'))
                 .withFluxDensity(1.0, 700.0))
    linear_SED = (galsim.SED(galsim.LookupTable(waves, (waves-550.0)/(825-550),
                                                interpolant='linear'))
                  .withFluxDensity(1.0, 700.0))

    print "Constructing galaxy"
    gal1 = galsim.Gaussian(half_light_radius=0.45).shear(e1=0.1, e2=0.2).shift(0.1, 0.2)
    gal2 = galsim.Gaussian(half_light_radius=0.35).shear(e1=-0.1, e2=0.4).shift(-0.3, 0.5)
    gal = gal1 * const_SED + gal2 * linear_SED

    HST_prof = galsim.Convolve(gal, HST_PSF)
    Euclid_prof = galsim.Convolve(gal, Euclid_PSF)

    print "Drawing HST images"
    # Draw HST images
    HST_images = [HST_prof.drawImage(rband, nx=128, ny=128, scale=0.03),
                  HST_prof.drawImage(iband, nx=128, ny=128, scale=0.03)]
    cn1 = galsim.getCOSMOSNoise()
    cn2 = cn1.rotate(45*galsim.degrees)
    # wcs = galsim.PixelScale(0.03)
    # wcs = None
    # corrfunc1 = galsim.Gaussian(sigma=0.06).shear(g1=0.3, g2=0.3)
    # corrfunc2 = galsim.Gaussian(sigma=0.1).shear(g1=-0.2, g2=-0.2)
    # cn1 = galsim.correlatednoise._BaseCorrelatedNoise(rng=None, gsobject=corrfunc1, wcs=wcs)
    # cn2 = galsim.correlatednoise._BaseCorrelatedNoise(rng=None, gsobject=corrfunc2, wcs=wcs)
    var1 = HST_images[0].addNoiseSNR(cn1, 150, preserve_flux=True)
    var2 = HST_images[1].addNoiseSNR(cn2, 150, preserve_flux=True)
    cn1 = cn1.withVariance(var1)
    cn2 = cn2.withVariance(var2)

    print "Drawing Euclid image"
    Euclid_image = Euclid_prof.drawImage(visband, nx=30, ny=30, scale=0.1)

    # Now "deconvolve" the chromatic HST PSF while asserting the correct SEDs.
    print "Constructing ChromaticRealGalaxy"
    crg = galsim.ChromaticRealGalaxy((HST_images,
                                      [rband, iband],
                                      [const_SED, linear_SED],
                                      [cn1, cn2],
                                      HST_PSF))
    # crg should be effectively the same thing as gal now.  Let's test.

    Euclid_recon_image = (galsim.Convolve(crg, Euclid_PSF)
                          .drawImage(visband, nx=30, ny=30, scale=0.1))

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(231)
        im = ax.imshow(HST_images[0].array)
        plt.colorbar(im)
        ax.set_title('rband')
        ax = fig.add_subplot(232)
        im = ax.imshow(HST_images[1].array)
        plt.colorbar(im)
        ax.set_title('iband')
        ax = fig.add_subplot(234)
        im = ax.imshow(Euclid_image.array)
        plt.colorbar(im)
        ax.set_title('Euclid')
        ax = fig.add_subplot(235)
        im = ax.imshow(Euclid_recon_image.array)
        plt.colorbar(im)
        ax.set_title('Euclid reconstruction')
        ax = fig.add_subplot(236)
        resid = Euclid_recon_image.array - Euclid_image.array
        vmin, vmax = np.percentile(resid, [5.0, 95.0])
        im = ax.imshow(resid, cmap='seismic',
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        ax.set_title('residual')
        plt.tight_layout()
        plt.show()

    print "Max comparison:", Euclid_image.array.max(), Euclid_recon_image.array.max()
    print "Sum comparison:", Euclid_image.array.sum(), Euclid_recon_image.array.sum()

    # Other tests:
    #     - draw the same image as origin?
    #     - compare intermediate products: does aj match the input spatial profiles?
    #       (are there degeneracies?)
    #     - stupid tests like using only one filter should perform similarly to RealGalaxy?
    #     - ellipticity tests like those above for RealGalaxy?

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    test_CRG_noise(plot=args.plot)
