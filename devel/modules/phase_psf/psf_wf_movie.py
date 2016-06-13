# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import galsim
from astropy.utils.console import ProgressBar
import warnings

def simple_moments(img):
    """Compute unweighted 0th, 1st, and 2nd moments of image.  Return result as a dictionary.
    """
    array = img.array
    scale = img.scale
    x = y = np.arange(array.shape[0])*scale
    y, x = np.meshgrid(y, x)
    I0 = np.sum(array)
    Ix = np.sum(x*array)/I0
    Iy = np.sum(y*array)/I0
    Ixx = np.sum((x-Ix)**2*array)/I0
    Iyy = np.sum((y-Iy)**2*array)/I0
    Ixy = np.sum((x-Ix)*(y-Iy)*array)/I0
    return dict(I0=I0, Ix=Ix, Iy=Iy, Ixx=Ixx, Iyy=Iyy, Ixy=Ixy)

def ellip(mom):
    """Convert moments dictionary into dictionary with ellipticity (e1, e2) and size (rsqr).
    """
    rsqr = mom['Ixx'] + mom['Iyy']
    return dict(rsqr=rsqr, e1=(mom['Ixx']-mom['Iyy'])/rsqr, e2=2*mom['Ixy']/rsqr)

def make_movie(args):
    """Function that actually makes the movie of the atmosphere, given command line arguments stored
    in `args`.
    """

    # Use GalSim random number generators.
    rng = galsim.BaseDeviate(args.seed)
    u = galsim.UniformDeviate(rng)

    # Interpolate atmospheric weights using Jee+Tyson (2011) values.
    JT_alts = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
    JT_weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]  # weight
    JT_interp = galsim.LookupTable(JT_alts, JT_weights, interpolant='linear')
    weights = JT_interp(15.46*np.arange(args.nlayers)/(args.nlayers-1))
    weights /= sum(weights)

    # GalSim Atmospheric PSF code.  We start by assembling a set of phase screens representing
    # different layers of turbulence in the atmosphere.  In principle, each layer can have its own
    # turbulence strength, boiling parameter, wind speed and direction, altitude, and even size and
    # resolution (though note that the size of each screen is actually made infinite by "wrapping"
    # the edges of the screen.)  The galsim.Atmosphere helper function is useful for constructing
    # this list, and requires lists of parameters for the different layers.

    spd = []  # Wind speed in m/s
    dirn = [] # Wind direction in radians
    alt = []  # Layer altitude in km
    r0_500 = [] # Fried parameter in m.
    for i in xrange(args.nlayers):
        spd.append(u()*20.0)  # Use a random speed between 0 and 20 m/s for each layer
        dirn.append(u()*360*galsim.degrees)  # And an isotropically distributed direction.
        alt.append(15.*i/(args.nlayers-1))  # And spread out the altitudes between 0 and 15 km.
        r0_500.append(args.r0_500*weights[i]**(-3./5))
        print ("Adding layer at altitude {:5.2f} km with velocity: ({:5.2f}, {:5.2f}) m/s, "
               "and r0_500: {:5.3f} m."
               .format(alt[-1], spd[-1]*dirn[-1].cos(), spd[-1]*dirn[-1].sin(), r0_500[-1]))

    # Additionally, we set the turbulence strength of the entire set of phase screens with `r0_500`
    # the Fried parameter at a wavelength of 500 nm, the screen temporal evolution `time_step`,
    # and the screen size and scale.
    atm = galsim.Atmosphere(r0_500=r0_500, speed=spd, direction=dirn, altitude=alt, rng=rng,
                            time_step=args.time_step, screen_size=args.screen_size,
                            screen_scale=args.screen_scale)

    # Place to store the cumulative PSF image if args.accumulate is set.
    psf_img_sum = galsim.ImageD(args.psf_nx, args.psf_nx, scale=args.psf_scale)

    # Field angle (angle on the sky wrt the telescope boresight) at which to compute the PSF.
    theta = (args.x*galsim.arcmin, args.y*galsim.arcmin)

    # Construct an Aperture object for computing the PSF.
    psf_aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                               nstruts=args.nstruts, strut_thick=args.strut_thick,
                               strut_angle=args.strut_angle*galsim.degrees,
                               screen_list=atm, pad_factor=args.pad_factor,
                               oversampling=args.oversampling)

    # Setup an Aperture for the wavefront.  We can ignore any warnings here since  this is just for
    # visualization.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wf_aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                                  nstruts=args.nstruts, strut_thick=args.strut_thick,
                                  strut_angle=args.strut_angle*galsim.degrees,
                                  pupil_plane_size=args.wf_scale*args.wf_nx,
                                  pupil_plane_scale=args.wf_scale)

    # Code to setup the Matplotlib animation.
    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title='Wavefront Movie', artist='Matplotlib')
    writer = FFMpegWriter(fps=15, bitrate=5000, metadata=metadata)

    # For the animation code, we essentially draw a single figure first, and then use various
    # `set_XYZ` methods to update each successive frame.
    fig = plt.figure(facecolor='k', figsize=(11, 6))

    psf_ax = fig.add_axes([0.08, 0.15, 0.35, 0.7])
    psf_im = psf_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                           vmin=0.0, vmax=args.psf_vmax, cmap='hot',
                           extent=np.r_[-1, 1, -1, 1]*0.5*args.psf_nx*args.psf_scale)

    wf_ax = fig.add_axes([0.51, 0.15, 0.35, 0.7])
    wf_im = wf_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                         vmin=-args.wf_vmax, vmax=args.wf_vmax, cmap='YlGnBu',
                         extent=np.r_[-1, 1, -1, 1]*0.5*wf_aper.pupil_plane_size)

    ilum = np.ma.masked_greater(wf_aper.illuminated, 0.5)
    wf_ax.imshow(ilum, alpha=0.4, extent=np.r_[-1, 1, -1, 1]*0.5*wf_aper.pupil_plane_size)

    cbar_ax = fig.add_axes([0.88, 0.175, 0.03, 0.65])
    plt.colorbar(wf_im, cax=cbar_ax)

    for ax in [psf_ax, wf_ax, cbar_ax]:
        for _, spine in ax.spines.iteritems():
            spine.set_color('w')
        ax.title.set_color('w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='both', colors='w')

    psf_ax.set_xlabel("Arcsec")
    psf_ax.set_ylabel("Arcsec")

    wf_ax.set_xlabel("Meters")
    wf_ax.set_ylabel("Meters")

    cbar_ax.set_ylabel("Radians")

    etext = psf_ax.text(0.05, 0.05, '', transform=psf_ax.transAxes)
    etext.set_color('w')

    nstep = int(args.exptime / args.time_step)
    with ProgressBar(nstep) as bar:
        with writer.saving(fig, args.outfile, 100):
            for i in xrange(nstep):
                # GalSim wavefront code
                wf = atm.wavefront(wf_aper, theta, compact=False) * 2*np.pi / args.lam  # radians
                wf = galsim.InterpolatedImage(galsim.Image(wf, scale=wf_aper.pupil_plane_scale),
                                              normalization='sb')
                wf_img = wf.drawImage(nx=args.wf_nx, ny=args.wf_nx, scale=args.wf_scale,
                                      method='sb')
                # GalSim PSF code
                psf = atm.makePSF(lam=args.lam, theta=theta, aper=psf_aper, exptime=args.time_step)
                psf_img0 = psf.drawImage(nx=args.psf_nx, ny=args.psf_nx, scale=args.psf_scale)

                if args.accumulate:
                    psf_img_sum += psf_img0
                    psf_img = psf_img_sum/(i+1)
                else:
                    psf_img = psf_img0

                # Calculate and display simple estimate of ellipticity
                e = ellip(simple_moments(psf_img))
                etext.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                        e['e1'], e['e2'], e['rsqr']))

                # Matplotlib code
                wf_im.set_array(wf_img.array)
                wf_ax.set_title("t={:5.2f} s".format(i*args.time_step))
                psf_im.set_array(psf_img.array)
                with warnings.catch_warnings(FutureWarning):
                    warnings.simplefilter("ignore")
                    writer.grab_frame(facecolor=fig.get_facecolor())
                bar.update()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="Random number seed for generating turbulence.  Default: 1")
    parser.add_argument("--r0_500", type=float, default=0.2,
                        help="Fried parameter at wavelength 500 nm in meters.  Default: 0.2")
    parser.add_argument("--nlayers", type=int, default=3,
                        help="Number of atmospheric layers.  Default: 3")
    parser.add_argument("--lam", type=float, default=700.0,
                        help="Wavelength in nanometers.  Default: 700.0")
    parser.add_argument("--time_step", type=float, default=0.03,
                        help="Incremental time step for advancing phase screens and accumulating "
                             "instantaneous PSFs in seconds.  Default: 0.03")
    parser.add_argument("--exptime", type=float, default=3.0,
                        help="Total amount of time to integrate in seconds.  Default: 3.0")
    parser.add_argument("-x", "--x", type=float, default=0.0,
                        help="x-coordinate of PSF in arcmin.  Default: 0.0")
    parser.add_argument("-y", "--y", type=float, default=0.0,
                        help="y-coordinate of PSF in arcmin.  Default: 0.0")
    parser.add_argument("--psf_nx", type=int, default=512,
                        help="Output PSF image dimensions in pixels.  Default: 128")
    parser.add_argument("--psf_scale", type=float, default=0.005,
                        help="Scale of PSF output pixels in arcseconds.  Default: 0.03")
    parser.add_argument("--wf_nx", type=int, default=128,
                        help="Output wavefront image dimensions in pixels.  Default: 128")
    parser.add_argument("--wf_scale", type=float, default=0.05,
                        help="Scale of wavefront output pixels in meters.  Default: 0.05")
    parser.add_argument("--accumulate", action='store_true',
                        help="Set to accumulate flux over exposure, as opposed to displaying the "
                             "instantaneous PSF.  Default: False")
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
    parser.add_argument("--screen_size", type=float, default=30.0,
                        help="Size of atmospheric screen in meters.  Note that the screen wraps "
                             "with periodic boundary conditions.  Default: 30")
    parser.add_argument("--screen_scale", type=float, default=0.1,
                        help="Resolution of atmospheric screen in meters.  Default: 0.1")
    parser.add_argument("--pad_factor", type=float, default=1.0,
                        help="Factor by which to pad PSF InterpolatedImage.  Default: 1.0")
    parser.add_argument("--oversampling", type=float, default=1.0,
                        help="Factor by which to oversample the PSF InterpolatedImage." +
                             "Default: 1.0")
    parser.add_argument("--psf_vmax", type=float, default=0.0001,
                        help="Matplotlib imshow vmax kwarg for PSF image.  Sets data value that "
                             "maxes out the colorbar range.  Default: 0.001")
    parser.add_argument("--wf_vmax", type=float, default=50.0,
                        help="Matplotlib imshow vmax kwarg for wavefront image.  Sets data value "
                             "that maxes out the colorbar range.  Default: 50.0")
    parser.add_argument("--outfile", type=str, default="psf_wf_movie.mp4",
                        help="Output filename.  Default: psf_wf_movie.mp4")

    args = parser.parse_args()
    make_movie(args)
