import galsim
import numpy as np
import time


def test_CRG(args):
    """Use some simplified simulated HST-like observations around r and i band to predict
    Euclid-ish visual band observations."""
    t0 = time.time()
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
    var1 = HST_images[0].addNoiseSNR(cn1, args.SNR, preserve_flux=True)
    var2 = HST_images[1].addNoiseSNR(cn2, args.SNR, preserve_flux=True)
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

    print "Took {} seconds".format(time.time()-t0)

    if args.plot:
        import matplotlib.pyplot as plt
        hst_extent = 0.03*128 * np.array([-0.5, 0.5, -0.5, 0.5])
        fig = plt.figure(figsize=(13, 8))
        ax = fig.add_subplot(231)
        im = ax.imshow(HST_images[0].array, extent=hst_extent)
        plt.colorbar(im)
        ax.set_title('rband')
        ax = fig.add_subplot(232)
        im = ax.imshow(HST_images[1].array, extent=hst_extent)
        plt.colorbar(im)
        ax.set_title('iband')
        ax = fig.add_subplot(234)
        euclid_extent = 30*0.1*np.array([-0.5, 0.5, -0.5, 0.5])
        im = ax.imshow(Euclid_image.array, extent=euclid_extent)
        plt.colorbar(im)
        ax.set_title('Euclid')
        ax = fig.add_subplot(235)
        im = ax.imshow(Euclid_recon_image.array, extent=euclid_extent)
        plt.colorbar(im)
        ax.set_title('Euclid reconstruction')
        ax = fig.add_subplot(236)
        resid = Euclid_recon_image.array - Euclid_image.array
        vmin, vmax = np.percentile(resid, [5.0, 95.0])
        im = ax.imshow(resid, cmap='seismic',
                       vmin=vmin, vmax=vmax, extent=euclid_extent)
        plt.colorbar(im)
        ax.set_title('residual')
        plt.tight_layout()
        plt.show()

    print "Max comparison:", Euclid_image.array.max(), Euclid_recon_image.array.max()
    print "Sum comparison:", Euclid_image.array.sum(), Euclid_recon_image.array.sum()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--SNR', type=float, default=100.0,
                        help="Input signal-to-noise ratio")
    args = parser.parse_args()
    test_CRG(args)
