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

"""Script to investigate the FFT PSF dependence on phase screen resolution.  Produces a plot of 4
different PSFs using progressively decreasing screen resolution.  We find that the size of the PSF
produced increases as resolution increases, with the difference between 1.25 cm and 2.5 cm
resolution being less than 1%, but the difference between 1.25 cm and 10 cm resolution being about
6%.
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
    layers = atm._layers
    ret = galsim.PhaseScreenList.__new__(galsim.PhaseScreenList)
    ret._layers = [shrink_layer(l, factor) for l in layers]
    ret.rng = atm.rng
    ret.dynamic = atm.dynamic
    ret.reversible = atm.reversible
    ret._pending = atm._pending
    return ret


def shrink_layer(layer, factor):
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
    with ProgressBar(args.nlayers) as bar:
        fineAtm.instantiate(_bar=bar)
    # `fineAtm` is now an instance of a galsim.PhaseScreenList object.

    # Construct an Aperture object for computing the PSF.  The Aperture object describes the
    # illumination pattern of the telescope pupil, and chooses good sampling size and resolution
    # for representing this pattern as an array.
    aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                           screen_list=fineAtm, pad_factor=args.pad_factor,
                           oversampling=args.oversampling)
    print(repr(aper))

    # Start output
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
    FigureCanvasAgg(fig)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Coarse
    print("Drawing with Fourier optics")
    with ProgressBar(args.exptime/args.time_step) as bar:
        psf = fineAtm.makePSF(lam=args.lam, aper=aper, exptime=args.exptime,
                              time_step=args.time_step, _bar=bar)
        img = psf.drawImage(nx=args.nx, ny=args.nx, scale=args.scale)

    try:
        mom = galsim.hsm.FindAdaptiveMom(img)
    except RuntimeError:
        mom = None

    axes[0,0].imshow(img.array)
    axes[0,0].set_title("{}".format(fineAtm[0].screen_scale))
    if mom is not None:
        axes[0,0].text(0.5, 0.9, "{:6.3f}".format(mom.moments_sigma),
                       transform=axes[0,0].transAxes, color='w')

    # Factor of 2
    shrunkenAtm = shrink_atm(fineAtm, 2)
    print("Drawing with shrink scale 2")
    with ProgressBar(args.exptime/args.time_step) as bar:
        psf = shrunkenAtm.makePSF(lam=args.lam, aper=aper, exptime=args.exptime,
                                  time_step=args.time_step, _bar=bar)
        img = psf.drawImage(nx=args.nx, ny=args.nx, scale=args.scale)
    try:
        mom = galsim.hsm.FindAdaptiveMom(img)
    except RuntimeError:
        mom = None

    axes[0,1].imshow(img.array)
    axes[0,1].set_title("{}".format(shrunkenAtm[0].screen_scale))
    if mom is not None:
        axes[0,1].text(0.5, 0.9, "{:6.3f}".format(mom.moments_sigma),
                       transform=axes[0,1].transAxes, color='w')

    # Factor of 4
    shrunkenAtm = shrink_atm(fineAtm, 4)
    print("Drawing with shrink scale 4")
    with ProgressBar(args.exptime/args.time_step) as bar:
        psf = shrunkenAtm.makePSF(lam=args.lam, aper=aper, exptime=args.exptime,
                                  time_step=args.time_step, _bar=bar)
        img = psf.drawImage(nx=args.nx, ny=args.nx, scale=args.scale)
    try:
        mom = galsim.hsm.FindAdaptiveMom(img)
    except RuntimeError:
        mom = None

    axes[1,0].imshow(img.array)
    axes[1,0].set_title("{}".format(shrunkenAtm[0].screen_scale))
    if mom is not None:
        axes[1,0].text(0.5, 0.9, "{:6.3f}".format(mom.moments_sigma),
                       transform=axes[1,0].transAxes, color='w')

    # Factor of 8
    shrunkenAtm = shrink_atm(fineAtm, 8)
    print("Drawing with shrink scale 8")
    with ProgressBar(args.exptime/args.time_step) as bar:
        psf = shrunkenAtm.makePSF(lam=args.lam, aper=aper, exptime=args.exptime,
                                  time_step=args.time_step, _bar=bar)
        img = psf.drawImage(nx=args.nx, ny=args.nx, scale=args.scale)
    try:
        mom = galsim.hsm.FindAdaptiveMom(img)
    except RuntimeError:
        mom = None

    axes[1,1].imshow(img.array)
    axes[1,1].set_title("{}".format(shrunkenAtm[0].screen_scale))
    if mom is not None:
        axes[1,1].text(0.5, 0.9, "{:6.3f}".format(mom.moments_sigma),
                       transform=axes[1,1].transAxes, color='w')

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
    parser.add_argument("--nlayers", type=int, default=6,
                        help="Number of atmospheric layers.  Default: 6")
    parser.add_argument("--time_step", type=float, default=0.025,
                        help="Incremental time step for advancing phase screens and accumulating "
                             "instantaneous PSFs in seconds.  Default: 0.025")
    parser.add_argument("--exptime", type=float, default=30.0,
                        help="Total amount of time to integrate in seconds.  Default: 30.0")
    parser.add_argument("--screen_size", type=float, default=102.4,
                        help="Size of atmospheric screen in meters.  Note that the screen wraps "
                             "with periodic boundary conditions.  Default: 102.4")
    parser.add_argument("--screen_scale", type=float, default=0.0125,
                        help="Resolution of atmospheric screen in meters.  Default: 0.0125")
    parser.add_argument("--max_speed", type=float, default=20.0,
                        help="Maximum wind speed in m/s.  Default: 20.0")

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

    parser.add_argument("--outfile", type=str, default="output/screen_scale.png",
                        help="Output filename.  Default: output/screen_scale.png")

    args = parser.parse_args()
    make_plot(args)
