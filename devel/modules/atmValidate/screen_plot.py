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

"""Script that generates plots of phase screens with upper and lower truncations in their power
spectra.  This is mostly just to sanity check that we're able to generate screens with the same
turbulent phases, but with different amplitudes for different k-modes.
"""


import os
import numpy as np
import galsim

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from astropy.utils.console import ProgressBar


def save_plot(img, fullpath):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    fig.tight_layout()

    dirname, filename = os.path.split(fullpath)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    fig.savefig(fullpath)


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
        spd.append(u()*20)  # Use a random speed between 0 and max_speed
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

    # Make sure to use a consistent seed for the atmosphere when varying kcrit
    # Additionally, we set the screen size and scale.
    atmRng = galsim.BaseDeviate(args.seed+1)
    print("Inflating atmosphere")
    atm = galsim.Atmosphere(r0_500=r0_500, L0=args.L0,
                            speed=spd, direction=dirn, altitude=alts, rng=atmRng,
                            screen_size=args.screen_size, screen_scale=args.screen_scale)
    with ProgressBar(args.nlayers) as bar:
        atm.instantiate(_bar=bar)
    print(atm[0].screen_scale, atm[0].screen_size)
    print(atm[0]._tab2d.f.shape)
    # `atm` is now an instance of a galsim.PhaseScreenList object.

    x = np.linspace(-0.5*args.nx*args.scale, 0.5*args.nx*args.scale, args.nx)
    x, y = np.meshgrid(x, x)
    img = atm.wavefront(x, y, 0)

    save_plot(img, args.outprefix+"full.png")

    del atm

    kcrits = np.logspace(np.log10(args.kmin), np.log10(args.kmax), 4)
    r0 = args.r0_500*(args.lam/500.0)**(6./5)
    for icol, kcrit in enumerate(kcrits):
        atmRng = galsim.BaseDeviate(args.seed+1)
        atmLowK = galsim.Atmosphere(r0_500=r0_500, L0=args.L0,
                                    speed=spd, direction=dirn, altitude=alts, rng=atmRng,
                                    screen_size=args.screen_size, screen_scale=args.screen_scale)
        with ProgressBar(args.nlayers) as bar:
            atmLowK.instantiate(kmax=kcrit/r0, _bar=bar)

        img = atmLowK.wavefront(x, y, 0)
        save_plot(img, "{}{}_{}".format(args.outprefix, icol, "low.png"))
        del atmLowK

        atmRng = galsim.BaseDeviate(args.seed+1)
        atmHighK = galsim.Atmosphere(r0_500=r0_500, L0=args.L0,
                                     speed=spd, direction=dirn, altitude=alts, rng=atmRng,
                                     screen_size=args.screen_size, screen_scale=args.screen_scale)
        with ProgressBar(args.nlayers) as bar:
            atmHighK.instantiate(kmin=kcrit/r0, _bar=bar)

        img = atmHighK.wavefront(x, y, 0)
        save_plot(img, "{}{}_{}".format(args.outprefix, icol, "high.png"))
        del atmHighK


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
    parser.add_argument("--screen_size", type=float, default=102.4,
                        help="Size of atmospheric screen in meters.  Note that the screen wraps "
                             "with periodic boundary conditions.  Default: 102.4")
    parser.add_argument("--screen_scale", type=float, default=0.05,
                        help="Resolution of atmospheric screen in meters.  Default: 0.0125")
    parser.add_argument("--kmin", type=float, default=0.1,
                        help="Minimum kcrit to plot.  Default: 0.1")
    parser.add_argument("--kmax", type=float, default=1.0,
                        help="Maximum kcrit to plot.  Default: 1.0")

    parser.add_argument("--lam", type=float, default=700.0,
                        help="Wavelength in nanometers.  Default: 700.0")
    parser.add_argument("--diam", type=float, default=8.36,
                        help="Size of circular telescope pupil in meters.  Default: 8.36")
    parser.add_argument("--obscuration", type=float, default=0.61,
                        help="Linear fractional obscuration of telescope pupil.  Default: 0.61")

    parser.add_argument("--nx", type=int, default=1024,
                        help="Output screen image dimensions in pixels.  Default: 1024")
    parser.add_argument("--scale", type=float, default=0.01,
                        help="Output screen image resolution in meters.  Default: 0.01")

    parser.add_argument("--outprefix", type=str, default="output/screen_",
                        help="Output filename prefix.  Default: output/screen_")

    args = parser.parse_args()
    make_plot(args)
