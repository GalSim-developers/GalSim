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

"""@file psf_wf_movie.py
Script to visualize the build up of an atmospheric PSF due to a frozen-flow Kolmogorov atmospheric
phase screens.  Note that the ffmpeg command line tool is required to run this script.
"""

import warnings
import numpy as np
import galsim

try:
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.animation as anim
except ImportError:
    raise ImportError("This demo requires matplotlib!")
from distutils.version import LooseVersion
if LooseVersion(matplotlib.__version__) < LooseVersion('1.2'):
    raise RuntimeError("This demo requires matplotlib version 1.2 or greater!")

try:
    from astropy.utils.console import ProgressBar
except ImportError:
    raise ImportError("This demo requires astropy!")

def make_movie(args):
    """Actually make the movie of the atmosphere given command line arguments stored in `args`.
    """

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

    # Additionally, we set the screen size and scale.
    atm = galsim.Atmosphere(r0_500=r0_500, speed=spd, direction=dirn, altitude=alts, rng=rng,
                            screen_size=args.screen_size, screen_scale=args.screen_scale)
    # `atm` is now an instance of a galsim.PhaseScreenList object.

    # Place to store the cumulative PSF image if args.accumulate is set.
    psf_img_sum = galsim.ImageD(args.psf_nx, args.psf_nx, scale=args.psf_scale)

    # Field angle (angle on the sky wrt the telescope boresight) at which to compute the PSF.
    theta = (args.x*galsim.arcmin, args.y*galsim.arcmin)

    # Construct an Aperture object for computing the PSF.  The Aperture object describes the
    # illumination pattern of the telescope pupil, and chooses good sampling size and resolution
    # for representing this pattern as an array.
    aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                           nstruts=args.nstruts, strut_thick=args.strut_thick,
                           strut_angle=args.strut_angle*galsim.degrees,
                           screen_list=atm, pad_factor=args.pad_factor,
                           oversampling=args.oversampling)

    # Code to setup the Matplotlib animation.
    metadata = dict(title='Wavefront Movie', artist='Matplotlib')
    writer = anim.FFMpegWriter(fps=15, bitrate=5000, metadata=metadata)

    # For the animation code, we essentially draw a single figure first, and then use various
    # `set_XYZ` methods to update each successive frame.
    fig = Figure(facecolor='k', figsize=(11, 6))
    FigureCanvasAgg(fig)

    # Axis for the PSF image on the left.
    psf_ax = fig.add_axes([0.08, 0.15, 0.35, 0.7])
    psf_ax.set_xlabel("Arcsec")
    psf_ax.set_ylabel("Arcsec")
    psf_im = psf_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                           vmin=0.0, vmax=args.psf_vmax, cmap='hot',
                           extent=np.r_[-1, 1, -1, 1]*0.5*args.psf_nx*args.psf_scale)

    # Axis for the wavefront image on the right.
    wf_ax = fig.add_axes([0.51, 0.15, 0.35, 0.7])
    wf_ax.set_xlabel("Meters")
    wf_ax.set_ylabel("Meters")
    wf_im = wf_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                         vmin=-args.wf_vmax, vmax=args.wf_vmax, cmap='YlGnBu',
                         extent=np.r_[-1, 1, -1, 1]*0.5*aper.pupil_plane_size)
    cbar_ax = fig.add_axes([0.88, 0.175, 0.03, 0.65])
    cbar_ax.set_ylabel("Radians")
    fig.colorbar(wf_im, cax=cbar_ax)

    # Overlay an alpha-mask on the wavefront image showing which parts are actually illuminated.
    ilum = np.ma.masked_greater(aper.illuminated, 0.5)
    wf_ax.imshow(ilum, alpha=0.4, extent=np.r_[-1, 1, -1, 1]*0.5*aper.pupil_plane_size)

    # Color items white to show up on black background
    for ax in [psf_ax, wf_ax, cbar_ax]:
        for _, spine in ax.spines.items():
            spine.set_color('w')
        ax.title.set_color('w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='both', colors='w')

    etext = psf_ax.text(0.05, 0.92, '', transform=psf_ax.transAxes)
    etext.set_color('w')

    nstep = int(args.exptime / args.time_step)
    t0 = 0.0
    # Use astropy ProgressBar to keep track of progress and show an estimate for time to completion.
    with ProgressBar(nstep) as bar:
        with writer.saving(fig, args.outfile, 100):
            for i in range(nstep):
                # The wavefront() method accepts pupil plane coordinates `u` and `v` in meters, a
                # time `t` in seconds, and possibly a field angle `theta`.  It returns the wavefront
                # lag or lead in nanometers with respect to the "perfect" planar wavefront at the
                # specified location angle and time.  In normal use for computing atmospheric PSFs,
                # this is just an implementation detail.  In this script, however, we include the
                # wavefront in the visualization.
                wf = atm.wavefront(aper.u, aper.v, t0, theta=theta) * 2*np.pi/args.lam  # radians
                # To make an actual PSF GSObject, we use the makePSF() method, including arguments
                # for the wavelength `lam`, the field angle `theta`, the aperture `aper`, the
                # starting time t0, and the exposure time `exptime`.  Here, since we're making a
                # movie, we set the exptime equal to just a single timestep, though normally we'd
                # want to set this to the full exposure time.
                psf = atm.makePSF(lam=args.lam, theta=theta, aper=aper,
                                  t0=t0, exptime=args.time_step)
                # `psf` is now just like an any other GSObject, ready to be convolved, drawn, or
                # transformed.  Here, we just draw it into an image to add to our movie.
                psf_img0 = psf.drawImage(nx=args.psf_nx, ny=args.psf_nx, scale=args.psf_scale)

                if args.accumulate:
                    psf_img_sum += psf_img0
                    psf_img = psf_img_sum/(i+1)
                else:
                    psf_img = psf_img0

                # Calculate simple estimate of size and ellipticity
                e = galsim.utilities.unweighted_shape(psf_img)

                # Update t0 for the next movie frame.
                t0 += args.time_step

                # Matplotlib code updating plot elements
                wf_im.set_array(wf)
                wf_ax.set_title("t={:5.2f} s".format(i*args.time_step))
                psf_im.set_array(psf_img.array)
                etext.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                        e['e1'], e['e2'], e['rsqr']*args.psf_scale**2))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    writer.grab_frame(facecolor=fig.get_facecolor())
                bar.update()


if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
"""
Script to visualize the build up of an atmospheric PSF due to a frozen-flow Kolmogorov atmospheric
phase screens.  Note that the ffmpeg command line tool is required to run this script.
"""), formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("--seed", type=int, default=1,
                        help="Random number seed for generating turbulence.  Default: 1")
    parser.add_argument("--r0_500", type=float, default=0.2,
                        help="Fried parameter at wavelength 500 nm in meters.  Default: 0.2")
    parser.add_argument("--nlayers", type=int, default=6,
                        help="Number of atmospheric layers.  Default: 6")
    parser.add_argument("--time_step", type=float, default=0.03,
                        help="Incremental time step for advancing phase screens and accumulating "
                             "instantaneous PSFs in seconds.  Default: 0.03")
    parser.add_argument("--exptime", type=float, default=3.0,
                        help="Total amount of time to integrate in seconds.  Default: 3.0")
    parser.add_argument("--screen_size", type=float, default=102.4,
                        help="Size of atmospheric screen in meters.  Note that the screen wraps "
                             "with periodic boundary conditions.  Default: 102.4")
    parser.add_argument("--screen_scale", type=float, default=0.1,
                        help="Resolution of atmospheric screen in meters.  Default: 0.1")
    parser.add_argument("--max_speed", type=float, default=20.0,
                        help="Maximum wind speed in m/s.  Default: 20.0")
    parser.add_argument("-x", "--x", type=float, default=0.0,
                        help="x-coordinate of PSF in arcmin.  Default: 0.0")
    parser.add_argument("-y", "--y", type=float, default=0.0,
                        help="y-coordinate of PSF in arcmin.  Default: 0.0")

    parser.add_argument("--lam", type=float, default=700.0,
                        help="Wavelength in nanometers.  Default: 700.0")
    parser.add_argument("--diam", type=float, default=4.0,
                        help="Size of circular telescope pupil in meters.  Default: 4.0")
    parser.add_argument("--obscuration", type=float, default=0.0,
                        help="Linear fractional obscuration of telescope pupil.  Default: 0.0")
    parser.add_argument("--nstruts", type=int, default=0,
                        help="Number of struts supporting secondary obscuration.  Default: 0")
    parser.add_argument("--strut_thick", type=float, default=0.05,
                        help="Thickness of struts as fraction of aperture diameter.  Default: 0.05")
    parser.add_argument("--strut_angle", type=float, default=0.0,
                        help="Starting angle of first strut in degrees.  Default: 0.0")

    parser.add_argument("--psf_nx", type=int, default=512,
                        help="Output PSF image dimensions in pixels.  Default: 512")
    parser.add_argument("--psf_scale", type=float, default=0.005,
                        help="Scale of PSF output pixels in arcseconds.  Default: 0.005")
    parser.add_argument("--accumulate", action='store_true',
                        help="Set to accumulate flux over exposure, as opposed to displaying the "
                             "instantaneous PSF.  Default: False")

    parser.add_argument("--pad_factor", type=float, default=1.0,
                        help="Factor by which to pad PSF InterpolatedImage to avoid aliasing. "
                             "Default: 1.0")
    parser.add_argument("--oversampling", type=float, default=1.0,
                        help="Factor by which to oversample the PSF InterpolatedImage. "
                             "Default: 1.0")

    parser.add_argument("--psf_vmax", type=float, default=0.0003,
                        help="Matplotlib imshow vmax kwarg for PSF image.  Sets value that "
                             "maxes out the colorbar range.  Default: 0.0003")
    parser.add_argument("--wf_vmax", type=float, default=50.0,
                        help="Matplotlib imshow vmax kwarg for wavefront image.  Sets value "
                             "that maxes out the colorbar range.  Default: 50.0")

    parser.add_argument("--outfile", type=str, default="output/psf_wf_movie.mp4",
                        help="Output filename.  Default: output/psf_wf_movie.mp4")

    args = parser.parse_args()
    make_movie(args)
