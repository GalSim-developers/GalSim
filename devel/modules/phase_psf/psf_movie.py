import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import galsim
from astropy.utils.console import ProgressBar


def make_movie(args):
    rng = galsim.BaseDeviate(args.seed)
    # GalSim Atmospheric PSF code
    vel = []
    dirn = []
    for i in xrange(args.nlayers):
        vel.append(galsim.UniformDeviate(rng)()*5.0)
        dirn.append(galsim.UniformDeviate(rng)()*360*galsim.degrees)

    atm = galsim.Atmosphere(r0_500=args.r0_500, velocity=vel, direction=dirn, rng=rng,
                            time_step=args.time_step, screen_size=args.screen_size,
                            screen_scale=args.screen_scale)

    # Matplotlib animation code
    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title='PSF Movie', artist='Matplotlib')
    writer = FFMpegWriter(fps=15, bitrate=1000, metadata=metadata)

    fig = plt.figure(facecolor='k')
    ax = fig.add_subplot(111, axisbg='k')
    im = ax.imshow(np.ones((128, 128), dtype=np.float64), animated=True, vmin=0, vmax=args.vmax,
                   extent=np.r_[-1, 1, -1, 1]*0.5*args.nx*args.scale)
    for _, spine in ax.spines.iteritems():
        spine.set_color('w')
    ax.title.set_color('w')
    ax.set_xlabel("Arcsec")
    ax.set_ylabel("Arcsec")
    ax.yaxis.label.set_color('w')
    ax.xaxis.label.set_color('w')
    ax.tick_params(axis='both', colors='w')

    nstep = int(args.exptime / args.time_step)
    img = np.zeros((args.nx, args.nx), dtype=np.float64)
    imgsum = np.zeros_like(img)
    with ProgressBar(nstep) as bar:
        with writer.saving(fig, args.outfile, 100):
            for i in xrange(nstep):
                # GalSim PSF code
                psf = atm.getPSF(exptime=args.time_step, diam=args.diam,
                                 obscuration=args.obscuration, pad_factor=args.pad_factor,
                                 oversampling=args.oversampling)
                psfim = psf.drawImage(nx=args.nx, ny=args.nx, scale=args.scale)
                if args.accumulate:
                    imgsum += psfim.array
                    img = imgsum/(i+1)
                else:
                    img = psfim.array

                # Matplotlib code
                im.set_array(img)
                ax.set_title("t={:5.2f} s".format(i*args.time_step))
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
                        help="Total amount of time to integrate in seconds.  Default: 15.0")
    parser.add_argument("--nx", type=int, default=128,
                        help="Output image dimensions in pixels.  Default: 128")
    parser.add_argument("--scale", type=float, default=0.02,
                        help="Scale of output pixels in arcseconds.  Default: 0.02")
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
                        help="Factor by which to pad phase screen before Fourier transforming into"
                             "a PSF.  Default: 1.5")
    parser.add_argument("--oversampling", type=float, default=1.5,
                        help="Factor by which to oversample phase screen before Fourier "
                             "transforming into a PSF.  Default: 1.5")
    parser.add_argument("--vmax", type=float, default=0.003,
                        help="Matplotlib imshow vmax kwarg.  Sets data value that maxes out the"
                             "colorbar range.  Default: 0.003")
    parser.add_argument("--outfile", type=str, default="psf_movie.mp4",
                        help="Output filename.  Default: psf_movie.mp4")

    args = parser.parse_args()
    make_movie(args)
