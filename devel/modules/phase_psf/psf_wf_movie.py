import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.patches import Wedge
import numpy as np
import galsim
from astropy.utils.console import ProgressBar


def make_movie(args):
    rng = galsim.BaseDeviate(args.seed)
    # GalSim Atmospheric PSF code
    spd = []
    dirn = []
    for i in xrange(args.nlayers):
        spd.append(galsim.UniformDeviate(rng)()*5.0)
        dirn.append(galsim.UniformDeviate(rng)()*360*galsim.degrees)

    atm = galsim.Atmosphere(r0_500=args.r0_500, speed=spd, direction=dirn, rng=rng,
                            time_step=args.time_step, screen_size=args.screen_size,
                            screen_scale=args.screen_scale)

    # Matplotlib animation code
    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title='Wavefront Movie', artist='Matplotlib')
    writer = FFMpegWriter(fps=15, bitrate=5000, metadata=metadata)

    fig = plt.figure(facecolor='k', figsize=(11, 6))

    psf_ax = fig.add_axes([0.08, 0.15, 0.35, 0.7])
    psf_im = psf_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                           vmin=0.0, vmax=args.psf_vmax, cmap='hot',
                           extent=np.r_[-1, 1, -1, 1]*0.5*args.psf_nx*args.psf_scale)

    wf_ax = fig.add_axes([0.51, 0.15, 0.35, 0.7])
    wf_im = wf_ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True,
                         vmin=-args.wf_vmax, vmax=args.wf_vmax, cmap='YlGnBu',
                         extent=np.r_[-1, 1, -1, 1]*0.5*args.wf_nx*args.wf_scale)

    wf_ax.add_patch(Wedge((0.0, 0.0), args.obscuration*args.diam/2.0, 0, 360,
                          alpha=0.4, color='k', edgecolor=None))
    wf_ax.add_patch(Wedge((0.0, 0.0), args.screen_size, 0, 360,
                          width=args.screen_size-args.diam/2.,
                          alpha=0.4, color='k', edgecolor=None))

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

    nstep = int(args.exptime / args.time_step)
    psf_img_sum = np.zeros((args.psf_nx, args.psf_nx), dtype=np.float64)
    wf_img_sum = np.zeros((args.wf_nx, args.wf_nx), dtype=np.float64)

    psf_aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                               screen_list=atm, pad_factor=args.pad_factor,
                               oversampling=args.oversampling)

    wf_aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                              pad_factor=args.pad_factor, oversampling=args.oversampling,
                              _pupil_plane_size=args.wf_scale*args.wf_nx,
                              _pupil_plane_scale=args.wf_scale)

    # psf_aper = galsim.Aperture.fromPhaseScreenList(
    #     atm, lam=args.lam,
    #     diam=args.diam, obscuration=args.obscuration,
    #     pad_factor=args.pad_factor,
    #     oversampling=args.oversampling)
    #
    # wf_aper = galsim.Aperture.fromPhaseScreenList(
    #     atm, lam=args.lam,
    #     diam=args.diam, obscuration=args.obscuration,
    #     pupil_scale=args.wf_scale,
    #     pupil_plane_size=args.wf_nx * args.wf_scale)

    with ProgressBar(nstep) as bar:
        with writer.saving(fig, args.outfile, 100):
            for i in xrange(nstep):
                # GalSim wavefront code
                wf = atm.wavefront(wf_aper) * 2*np.pi / args.lam  # radians
                wf = galsim.InterpolatedImage(galsim.Image(wf, scale=wf_aper.pupil_plane_scale),
                                              normalization='sb')
                wf_img = wf.drawImage(nx=args.wf_nx, ny=args.wf_nx, scale=args.wf_scale,
                                      method='sb')
                # GalSim PSF code
                psf = atm.makePSF(lam=args.lam, aper=psf_aper, exptime=args.time_step)
                psf_img = psf.drawImage(nx=args.psf_nx, ny=args.psf_nx, scale=args.psf_scale)

                wf_img = wf_img.array
                if args.accumulate:
                    psf_img_sum += psf_img.array
                    psf_img = psf_img_sum/(i+1)
                else:
                    psf_img = psf_img.array

                # Matplotlib code
                wf_im.set_array(wf_img)
                wf_ax.set_title("t={:5.2f} s".format(i*args.time_step))
                psf_im.set_array(psf_img)
                import warnings
                with warnings.catch_warnings(FutureWarning):
                    warnings.simplefilter("ignore")
                    writer.grab_frame(facecolor=fig.get_facecolor())
                bar.update()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="")
    parser.add_argument("--r0_500", type=float, default=0.2,
                        help="Fried parameter at 500 nm in meters.  Default: 0.2")
    parser.add_argument("--nlayers", type=int, default=3,
                        help="Number of atmospheric layers.  Default: 3")
    parser.add_argument("--lam", type=float, default=500.0,
                        help="Wavelength in nanometers.  Default: 500.0")
    parser.add_argument("--time_step", type=float, default=0.03,
                        help="Incremental time step for advancing phase screens and accumulating "
                             " instantaneous PSFs in seconds.  Default: 0.03")
    parser.add_argument("--exptime", type=float, default=3.0,
                        help="Total amount of time to integrate in seconds.  Default: 3.0")
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
    parser.add_argument("--screen_size", type=float, default=30.0,
                        help="Size of atmospheric screen in meters.  Note that the screen wraps "
                             "with periodic boundary conditions.  Default: 30")
    parser.add_argument("--screen_scale", type=float, default=0.1,
                        help="Resolution of atmospheric screen in meters.  Default: 0.1")
    parser.add_argument("--pad_factor", type=float, default=1.5,
                        help="Factor by which to pad PSF InterpolatedImage.  Default: 1.5")
    parser.add_argument("--oversampling", type=float, default=1.5,
                        help="Factor by which to oversample the PSF InterpolatedImage." +
                             "Default: 1.5")
    parser.add_argument("--psf_vmax", type=float, default=0.00025,
                        help="Matplotlib imshow vmax kwarg for PSF image.  Sets data value that "
                             "maxes out the colorbar range.  Default: 0.001")
    parser.add_argument("--wf_vmax", type=float, default=50.0,
                        help="Matplotlib imshow vmax kwarg for wavefront image.  Sets data value "
                             "that maxes out the colorbar range.  Default: 50.0")
    parser.add_argument("--outfile", type=str, default="psf_wf_movie.mp4",
                        help="Output filename.  Default: psf_wf_movie.mp4")

    args = parser.parse_args()
    make_movie(args)
