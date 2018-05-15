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

"""Script to investigate the dependence of the first kick on the resolution of the phase screens.
Produces a plot in which each column uses progressively less resolution screens.  The first row
shows a FFT PSF and the second row shows a first kick geometric PSF.

The screen_scale.py script also in this directory demonstrates that FFT PSFs require ~cm resolution
before their sizes converge, but this script shows that first kick PSFs seem to be reasonably
converged with only ~10 cm resolution.  This is important in that it allows us to use memory to
make physically larger screens instead of high resolution screens.  With a screen_scale of 0.2m, the
first kick sizes (with kcrit=0.2) only change ~1.5% wrt to 1cm resolution.
"""


import os
import numpy as np
import galsim

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from astropy.utils.console import ProgressBar


def shrink_atm(atm, factor):
    """Shrink the resolution of an atmosphere by `factor`.
    """
    layers = atm._layers
    ret = galsim.PhaseScreenList.__new__(galsim.PhaseScreenList)
    ret._layers = [shrink_layer(l, factor) for l in layers]
    ret.rng = atm.rng
    ret.dynamic = atm.dynamic
    ret.reversible = atm.reversible
    ret._pending = atm._pending
    return ret


def shrink_layer(layer, factor):
    """Shrink the resolution of single atmospheric layer by `factor`.
    """
    tab2d = layer._tab2d
    orig = tab2d.f[:-1, :-1]

    new = orig[::factor, ::factor]

    ret = galsim.AtmosphericScreen.__new__(galsim.AtmosphericScreen)
    ret.npix = new.shape[0]
    ret.screen_scale = layer.screen_scale*factor
    ret.screen_size = layer.screen_size
    ret.altitude = layer.altitude
    ret.time_step = layer.time_step
    ret.r0_500 = layer.r0_500
    ret.L0 = layer.L0
    ret.vx = layer.vx
    ret.vy = layer.vy
    ret.alpha = layer.alpha
    ret._time = layer._time
    ret._orig_rng = layer._orig_rng.duplicate()
    ret.dynamic = layer.dynamic
    ret.reversible = layer.reversible
    ret.rng = layer.rng.duplicate()
    ret._xs = layer._xs[::factor]
    ret._ys = layer._ys[::factor]
    ret._tab2d = galsim.LookupTable2D(
        ret._xs, ret._ys, new, interpolant='linear', edge_mode='wrap')
    ret.kmin = layer.kmin
    ret.kmax = layer.kmax

    return ret


def make_plot(args):
    # Initiate some GalSim random number generators.
    rng = galsim.BaseDeviate(args.seed)
    u = galsim.UniformDeviate(rng)

    # The GalSim atmospheric simulation code describes turbulence in the 3D atmosphere as a series
    # of 2D turbulent screens.  The galsim.Atmosphere() helper function is useful for constructing
    # this screen list.

    # First, we estimate a weight for each screen, so that the turbulence is dominated by the lower
    # layers consistent with direct measurements.  The specific values we use are from SCIDAR
    # measurements on Cerro Pachon as part of the 1998 Gemini site selection process
    # (Ellerbroek 2002, JOSA Vol 19 No 9).

    Ellerbroek_alts = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
    Ellerbroek_weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    Ellerbroek_interp = galsim.LookupTable(Ellerbroek_alts, Ellerbroek_weights,
                                           interpolant='linear')

    # Use given number of uniformly spaced altitudes
    alts = np.max(Ellerbroek_alts)*np.arange(args.nlayers)/(args.nlayers-1)
    weights = Ellerbroek_interp(alts)  # interpolate the weights
    weights /= sum(weights)  # and renormalize

    # Each layer can have its own turbulence strength (roughly inversely proportional to the Fried
    # parameter r0), wind speed, wind direction, altitude, and even size and scale (though note that
    # the size of each screen is actually made infinite by "wrapping" the edges of the screen.)  The
    # galsim.Atmosphere helper function is useful for constructing this list, and requires lists of
    # parameters for the different layers.

    spd = []  # Wind speed in m/s
    dirn = [] # Wind direction in radians
    r0_500 = [] # Fried parameter in m at a wavelength of 500 nm.
    for i in range(args.nlayers):
        spd.append(u()*args.max_speed)  # Use a random speed between 0 and max_speed
        dirn.append(u()*360*galsim.degrees)  # And an isotropically distributed wind direction.
        # The turbulence strength of each layer is specified by through its Fried parameter r0_500,
        # which can be thought of as the diameter of a telescope for which atmospheric turbulence
        # and unaberrated diffraction contribute equally to image resolution (at a wavelength of
        # 500nm).  The weights above are for the refractive index structure function (similar to a
        # variance or covariance), however, so we need to use an appropriate scaling relation to
        # distribute the input "net" Fried parameter into a Fried parameter for each layer.  For
        # Kolmogorov turbulence, this is r0_500 ~ (structure function)**(-3/5):
        r0_500.append(args.r0_500*weights[i]**(-3./5))
        print("Adding layer at altitude {:5.2f} km with velocity ({:5.2f}, {:5.2f}) m/s, "
              "and r0_500 {:5.3f} m."
              .format(alts[i], spd[i]*dirn[i].cos(), spd[i]*dirn[i].sin(), r0_500[i]))

    # Generate atmosphere, set the initial screen size and scale.
    atmRng = galsim.BaseDeviate(args.seed+1)
    fineAtm = galsim.Atmosphere(r0_500=r0_500, L0=args.L0,
                                speed=spd, direction=dirn, altitude=alts, rng=atmRng,
                                screen_size=args.screen_size, screen_scale=args.screen_scale)
    r0 = args.r0_500*(args.lam/500.0)**(6./5)
    with ProgressBar(args.nlayers) as bar:
        fineAtm.instantiate(kmax=args.kcrit/r0, _bar=bar)
    # `fineAtm` is now an instance of a galsim.PhaseScreenList object.

    # Construct an Aperture object for computing the PSF.  The Aperture object describes the
    # illumination pattern of the telescope pupil, and chooses good sampling size and resolution
    # for representing this pattern as an array.
    aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                           screen_list=fineAtm, pad_factor=args.pad_factor,
                           oversampling=args.oversampling)
    print(repr(aper))

    # Start output
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(12, 5))
    FigureCanvasAgg(fig)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    for icol, shrinkFactor in enumerate([1,2,4,8,16,32,64]):

        if shrinkFactor == 1:
            shrunkenAtm = fineAtm
        else:
            shrunkenAtm = shrink_atm(fineAtm, shrinkFactor)
        print("Drawing with Fourier optics")
        with ProgressBar(args.exptime/args.time_step) as bar:
            psf = shrunkenAtm.makePSF(lam=args.lam, aper=aper, exptime=args.exptime,
                                      time_step=args.time_step, second_kick=False, _bar=bar)
            img = psf.drawImage(nx=args.nx, ny=args.nx, scale=args.scale)

        try:
            mom = galsim.hsm.FindAdaptiveMom(img)
        except RuntimeError:
            mom = None

        axes[0,icol].imshow(img.array)
        axes[0,icol].set_title("scale = {}".format(shrunkenAtm[0].screen_scale))
        if mom is not None:
            axes[0,icol].text(0.5, 0.9, "{:6.3f}".format(mom.moments_sigma),
                              transform=axes[0,icol].transAxes, color='w')

        airy = galsim.Airy(lam=args.lam, diam=args.diam, obscuration=args.obscuration)
        firstKick = galsim.Convolve(psf, airy)
        firstKickImg = firstKick.drawImage(nx=args.nx, ny=args.nx, scale=args.scale,
                                           method='phot', n_photons=args.nphot)
        try:
            firstKickMom = galsim.hsm.FindAdaptiveMom(firstKickImg)
        except RuntimeError:
            firstKickMom = None

        axes[1,icol].imshow(img.array)
        if mom is not None:
            axes[1,icol].text(0.5, 0.9, "{:6.3f}".format(firstKickMom.moments_sigma),
                              transform=axes[1,icol].transAxes, color='w')

    axes[0, 0].set_ylabel("FFT")
    axes[1, 0].set_ylabel("1st Kick")

    fig.tight_layout()

    dirname, filename = os.path.split(args.outfile)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    fig.savefig(args.outfile)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=1,
                        help="Random number seed for generating turbulence.  Default: 1")
    parser.add_argument("--r0_500", type=float, default=0.15,
                        help="Fried parameter at wavelength 500 nm in meters.  Default: 0.15")
    parser.add_argument("--L0", type=float, default=25.0,
                        help="Outer scale in meters.  Default: 25.0")
    parser.add_argument("--kcrit", type=float, default=0.2,
                        help="Critical Fourier scale in units of 1/r0.  Default: 0.2")
    parser.add_argument("--nlayers", type=int, default=6,
                        help="Number of atmospheric layers.  Default: 6")
    parser.add_argument("--time_step", type=float, default=0.025,
                        help="Incremental time step for advancing phase screens and accumulating "
                             "instantaneous PSFs in seconds.  Default: 0.025")
    parser.add_argument("--exptime", type=float, default=0.5,
                        help="Total amount of time to integrate in seconds.  Default: 0.5")
    parser.add_argument("--screen_size", type=float, default=102.4,
                        help="Size of atmospheric screen in meters.  Note that the screen wraps "
                             "with periodic boundary conditions.  Default: 102.4")
    parser.add_argument("--screen_scale", type=float, default=0.0125,
                        help="Resolution of atmospheric screen in meters.  Default: 0.0125")
    parser.add_argument("--max_speed", type=float, default=20.0,
                        help="Maximum wind speed in m/s.  Default: 20.0")
    parser.add_argument("--nphot", type=int, default=int(3e6),
                        help="Number of photons to shoot.  Default: 3e6")

    parser.add_argument("--lam", type=float, default=700.0,
                        help="Wavelength in nanometers.  Default: 700.0")
    parser.add_argument("--diam", type=float, default=8.36,
                        help="Size of circular telescope pupil in meters.  Default: 8.36")
    parser.add_argument("--obscuration", type=float, default=0.61,
                        help="Linear fractional obscuration of telescope pupil.  Default: 0.61")

    parser.add_argument("--nx", type=int, default=64,
                        help="Output PSF image dimensions in pixels.  Default: 64")
    parser.add_argument("--scale", type=float, default=0.04,
                        help="Scale of PSF output pixels in arcseconds.  Default: 0.04")

    parser.add_argument("--pad_factor", type=float, default=1.0,
                        help="Factor by which to pad PSF InterpolatedImage to avoid aliasing. "
                             "Default: 1.0")
    parser.add_argument("--oversampling", type=float, default=1.0,
                        help="Factor by which to oversample the PSF InterpolatedImage. "
                             "Default: 1.0")

    parser.add_argument("--outfile", type=str, default="output/firstKick_screen_scale.png",
                        help="Output filename.  Default: output/firstKick_screen_scale.png")

    args = parser.parse_args()
    make_plot(args)
