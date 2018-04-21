# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from __future__ import print_function
import galsim
import numpy as np
import time
from astropy.utils.console import ProgressBar


def test_CRG_noise(args):
    """Test noise propagation in ChromaticRealGalaxy
    """
    t0 = time.time()

    print("Constructing chromatic PSFs")
    in_PSF = galsim.ChromaticAiry(lam=700., diam=2.4)
    out_PSF = galsim.ChromaticAiry(lam=700., diam=1.2)

    print("Constructing filters and SEDs")
    waves = np.arange(550.0, 900.1, 10.0)
    visband = galsim.Bandpass(galsim.LookupTable(waves, np.ones_like(waves), interpolant='linear'),
                              wave_type='nm')
    split_points = np.linspace(550.0, 900.0, args.Nim+1, endpoint=True)
    bands = [visband.truncate(blue_limit=blim, red_limit=rlim)
             for blim, rlim in zip(split_points[:-1], split_points[1:])]

    maxk = max([out_PSF.evaluateAtWavelength(waves[0]).maxK(),
                out_PSF.evaluateAtWavelength(waves[-1]).maxK()])

    SEDs = [galsim.SED(galsim.LookupTable(waves, waves**i, interpolant='linear'),
                       flux_type='fphotons', wave_type='nm').withFlux(1.0, visband)
            for i in range(args.NSED)]

    print("Constructing input noise correlation functions")
    rng = galsim.BaseDeviate(args.seed)
    in_xis = [galsim.getCOSMOSNoise(cosmos_scale=args.in_scale, rng=rng)
              .dilate(1 + i * 0.05)
              .rotate(5 * i * galsim.degrees)
              for i in range(args.Nim)]

    print("Creating noise images")
    img_sets = []
    for i in range(args.Ntrial):
        imgs = []
        for j, xi in enumerate(in_xis):
            img = galsim.Image(args.in_Nx, args.in_Nx, scale=args.in_scale)
            img.addNoise(xi)
            imgs.append(img)
        img_sets.append(imgs)

    print("Constructing `ChromaticRealGalaxy`s")
    crgs = []
    with ProgressBar(len(img_sets)) as bar:
        for imgs in img_sets:
            crgs.append(galsim.ChromaticRealGalaxy.makeFromImages(
                    imgs, bands, in_PSF, in_xis, SEDs=SEDs, maxk=maxk))
            bar.update()

    print("Convolving by output PSF")
    objs = [galsim.Convolve(crg, out_PSF) for crg in crgs]

    print("Drawing through output filter")
    out_imgs = [obj.drawImage(visband, nx=args.out_Nx, ny=args.out_Nx, scale=args.out_scale,
                              iimult=args.iimult)
                for obj in objs]

    noise = objs[0].noise

    print("Measuring images' correlation functions")
    xi_obs = galsim.correlatednoise.CorrelatedNoise(out_imgs[0])
    for img in out_imgs[1:]:
        xi_obs += galsim.correlatednoise.CorrelatedNoise(img)
    xi_obs /= args.Ntrial
    xi_obs_img = galsim.Image(args.out_Nx, args.out_Nx, scale=args.out_scale)
    xi_obs.drawImage(xi_obs_img)

    print("Observed image variance: ", xi_obs.getVariance())
    print("Predicted image variance: ", noise.getVariance())
    print("Predicted/Observed variance:", noise.getVariance()/xi_obs.getVariance())

    print("Took {} seconds".format(time.time()-t0))

    if args.plot:
        import matplotlib.pyplot as plt
        out_array = (np.arange(args.out_Nx) - args.out_Nx/2) * args.out_scale
        out_extent = [-args.out_Nx*args.out_scale/2,
                      args.out_Nx*args.out_scale/2,
                      -args.out_Nx*args.out_scale/2,
                      args.out_Nx*args.out_scale/2]

        fig = plt.figure(figsize=(5, 5))

        # Sample image
        ax = fig.add_subplot(111)
        ax.imshow(out_imgs[0].array, extent=out_extent)
        ax.set_title("sample output image")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.colorbar()
        fig.show()

        # 2D correlation functions
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(221)
        noise_img = galsim.Image(args.out_Nx, args.out_Nx, scale=args.out_scale)
        noise.drawImage(noise_img)
        ax1.imshow(np.log10(np.abs(noise_img.array)), extent=out_extent)
        ax1.set_title("predicted covariance function")
        ax1.set_xlabel(r"$\Delta x$")
        ax1.set_ylabel(r"$\Delta y$")
        ax2 = fig.add_subplot(222)
        ax2.imshow(np.log10(np.abs(xi_obs_img.array)), extent=out_extent)
        ax2.set_title("observed covariance function")
        ax2.set_xlabel(r"$\Delta x$")
        ax2.set_ylabel(r"$\Delta y$")

        # 1D slide through correlation functions
        ax3 = fig.add_subplot(223)
        ax3.plot(out_array, noise_img.array[args.out_Nx/2, :], label="prediction", color='red')
        ax3.plot(out_array, xi_obs_img.array[args.out_Nx/2, :], label="observation", color='blue')
        ax3.legend(loc='best')
        ax3.set_xlabel(r"$\Delta x$")
        ax3.set_ylabel(r"$\xi$")

        ax4 = fig.add_subplot(224)
        ax4.plot(out_array, noise_img.array[args.out_Nx/2, :], label="prediction", color='red')
        ax4.plot(out_array, xi_obs_img.array[args.out_Nx/2, :], label="observation", color='blue')
        ax4.plot(out_array, -noise_img.array[args.out_Nx/2, :], ls=':', color='red')
        ax4.plot(out_array, -xi_obs_img.array[args.out_Nx/2, :], ls=':', color='blue')
        ax4.legend(loc='best')
        ax4.set_yscale('log')
        ax4.set_xlabel(r"$\Delta x$")
        ax4.set_ylabel(r"$\xi$")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--Ntrial', type=int, default=100, help="[Default: 100]")
    parser.add_argument('--Nim', type=int, default=2, help="[Default: 2]")
    parser.add_argument('--NSED', type=int, default=2, help="[Default: 2]")
    parser.add_argument('--in_Nx', type=int, default=128, help="[Default: 128]")
    parser.add_argument('--in_scale', type=float, default=0.03, help="[Default: 0.03]")
    parser.add_argument('--out_scale', type=float, default=0.1, help="[Default: 0.1]")
    parser.add_argument('--out_Nx', type=int, default=30, help="[Default: 30]")
    parser.add_argument('--seed', type=int, default=1, help="[Default: 1]")
    parser.add_argument('--iimult', type=int, default=1, help="[Default: 1]")
    args = parser.parse_args()

    test_CRG_noise(args)
