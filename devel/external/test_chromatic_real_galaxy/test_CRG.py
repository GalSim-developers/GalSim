import galsim
import numpy as np
import time


def test_CRG(args):
    """Predict a Euclid-like image given HST-like images of a galaxy with color gradients."""
    t0 = time.time()

    print "Constructing chromatic PSFs"
    in_PSF = galsim.ChromaticAiry(lam=700, diam=2.4)
    out_PSF = galsim.ChromaticAiry(lam=700, diam=1.2)

    print "Constructing filters and SEDs"
    waves = np.arange(550.0, 900.1, 10.0)
    visband = galsim.Bandpass(galsim.LookupTable(waves, np.ones_like(waves), interpolant='linear'))
    split_points = np.linspace(550.0, 900.1, args.Nim+1, endpoint=True)
    bands = [visband.truncate(blue_limit=blim, red_limit=rlim)
             for blim, rlim in zip(split_points[:-1], split_points[1:])]

    maxk = max([out_PSF.evaluateAtWavelength(waves[0]).maxK(),
                out_PSF.evaluateAtWavelength(waves[-1]).maxK()])

    SEDs = [galsim.SED(galsim.LookupTable(waves, waves**i, interpolant='linear'),
                       flux_type='fphotons').withFlux(1.0, visband)
            for i in xrange(args.NSED)]

    print "Construction input noise correlation functions"
    rng = galsim.BaseDeviate(args.seed)
    in_xis = [galsim.getCOSMOSNoise(cosmos_scale=args.in_scale, rng=rng)
              .dilate(1 + i * 0.05)
              .rotate(5 * i * galsim.degrees)
              for i in xrange(args.Nim)]

    print "Constructing galaxy"
    components = [galsim.Gaussian(half_light_radius=0.3).shear(e1=0.1)]
    for i in xrange(1, args.Nim):
        components.append(
            galsim.Gaussian(half_light_radius=0.3+0.1*np.cos(i))
            .shear(e=0.4+np.cos(i)*0.4, beta=i*galsim.radians)
            .shift(0.4*i, -0.4*i)
        )
    gal = galsim.Add([c*s for c, s in zip(components, SEDs)])
    gal = gal.shift(-gal.centroid(visband))

    in_prof = galsim.Convolve(gal, in_PSF)
    out_prof = galsim.Convolve(gal, out_PSF)

    print "Drawing input images"
    in_imgs = [in_prof.drawImage(band, nx=args.in_Nx, ny=args.in_Nx, scale=args.in_scale)
               for band in bands]
    [img.addNoiseSNR(xi, args.SNR, preserve_flux=True) for xi, img in zip(in_xis, in_imgs)]

    print "Drawing true output image"
    out_img = out_prof.drawImage(visband, nx=args.out_Nx, ny=args.out_Nx, scale=args.out_scale)

    # Now "deconvolve" the chromatic HST PSF while asserting the correct SEDs.
    print "Constructing ChromaticRealGalaxy"
    crg = galsim.ChromaticRealGalaxy((in_imgs, bands, SEDs, in_xis, in_PSF), maxk=maxk)
    # crg should be effectively the same thing as gal now.  Let's test.

    crg_prof = galsim.Convolve(crg, out_PSF)
    crg_img = crg_prof.drawImage(visband, nx=args.out_Nx, ny=args.out_Nx, scale=args.out_scale)
    print "Max comparison:", out_img.array.max(), crg_img.array.max()
    print "Sum comparison:", out_img.array.sum(), crg_img.array.sum()

    print "Took {} seconds".format(time.time()-t0)

    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        in_extent = [-args.in_Nx*args.in_scale/2,
                     args.in_Nx*args.in_scale/2,
                     -args.in_Nx*args.in_scale/2,
                     args.in_Nx*args.in_scale/2]
        out_extent = [-args.out_Nx*args.out_scale/2,
                      args.out_Nx*args.out_scale/2,
                      -args.out_Nx*args.out_scale/2,
                      args.out_Nx*args.out_scale/2]

        fig = plt.figure(figsize=(10, 5))
        outer_grid = gridspec.GridSpec(2, 1)

        # Input images
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, args.Nim, outer_grid[0])
        for i, img in enumerate(in_imgs):
            ax = plt.Subplot(fig, inner_grid[i])
            im = ax.imshow(img.array, extent=in_extent)
            ax.set_title("band[{}] input".format(i))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            plt.colorbar(im)

        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3, outer_grid[1])
        # Output image, truth, and residual
        ax = plt.Subplot(fig, inner_grid[0])
        ax.set_title("True output")
        im = ax.imshow(out_img.array, extent=out_extent)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
        plt.colorbar(im)

        ax = plt.Subplot(fig, inner_grid[1])
        ax.set_title("Reconstructed output")
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(crg_img.array, extent=out_extent)
        fig.add_subplot(ax)
        plt.colorbar(im)

        ax = plt.Subplot(fig, inner_grid[2])
        ax.set_title("Residual")
        ax.set_xticks([])
        ax.set_yticks([])
        resid = crg_img.array - out_img.array
        vmin, vmax = np.percentile(resid, [5.0, 95.0])
        v = np.max([np.abs(vmin), np.abs(vmax)])
        im = ax.imshow(resid, extent=out_extent, cmap='seismic', vmin=-v, vmax=v)
        fig.add_subplot(ax)
        plt.colorbar(im)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--Nim', type=int, default=2, help="[Default: 2]")
    parser.add_argument('--NSED', type=int, default=2, help="[Default: 2]")
    parser.add_argument('--in_Nx', type=int, default=128, help="[Default: 128]")
    parser.add_argument('--in_scale', type=float, default=0.03, help="[Default: 0.03]")
    parser.add_argument('--out_scale', type=float, default=0.1, help="[Default: 0.1]")
    parser.add_argument('--out_Nx', type=int, default=30, help="[Default: 30]")
    parser.add_argument('--seed', type=int, default=1, help="[Default: 1]")
    parser.add_argument('--iimult', type=int, default=1, help="[Default: 1]")
    parser.add_argument('--SNR', type=float, default=100.0, help="[Default: 100]")
    args = parser.parse_args()
    test_CRG(args)
