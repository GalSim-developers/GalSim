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

import numpy as np
import galsim

try:
    import matplotlib
    import matplotlib.pyplot as plt
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

def simple_moments(img):
    """Compute unweighted 0th, 1st, and 2nd moments of image.  Return result as a dictionary.
    """
    array = img.array
    scale = img.scale
    N = array.shape[0]
    # x = y = np.arange(array.shape[0])*scale
    x = y = np.arange(array.shape[0])-(N/2)
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
    rng = galsim.BaseDeviate(args.seed)
    t = np.arange(-args.n/2, args.n/2)
    corr = np.exp(-0.5*t**2/args.ell**2)
    pk = np.fft.fft(np.fft.fftshift(corr))
    ak = np.sqrt(2*pk)
    phi = np.random.uniform(size=(args.n, args.jmax))
    zk = ak[:, None]*np.exp(2j*np.pi*phi)
    aberrations = args.n/2*np.fft.ifft(zk, axis=0).real

    measured_std = np.mean(np.std(aberrations, axis=0))
    aberrations *= args.sigma/measured_std
    # Make the mean of each aberration zero.
    aberrations -= np.mean(aberrations, axis=0)

    lam = args.lam
    diam = args.diam
    obscuration = args.obscuration
    fft_aper = galsim.Aperture(args.diam, args.lam, obscuration=args.obscuration,
                               pad_factor=args.pad_factor, oversampling=args.oversampling,
                               nstruts=args.nstruts, strut_thick=args.strut_thick,
                               strut_angle=args.strut_angle*galsim.degrees)
    geom_aper = galsim.Aperture(args.diam, args.lam, obscuration=args.obscuration,
                                pad_factor=args.geom_oversampling, oversampling=0.5,
                                nstruts=args.nstruts, strut_thick=args.strut_thick,
                                strut_angle=args.strut_angle*galsim.degrees)

    scale = args.size/args.nx
    extent = np.r_[-1,1,-1,1]*args.size/2

    metadata = dict(title="Optical PSF movie", artist='Matplotlib')
    writer = anim.FFMpegWriter(fps=15, bitrate=10000, metadata=metadata)

    fig = plt.figure(facecolor='k', figsize=(16, 9))

    fft_ax = fig.add_axes([0.07, 0.08, 0.36, 0.9])
    fft_ax.set_xlabel("Arcsec")
    fft_ax.set_ylabel("Arcsec")
    fft_ax.set_title("Fourier Optics")
    fft_im = fft_ax.imshow(np.ones((args.nx, args.nx), dtype=float), animated=True, extent=extent,
                           vmin=0.0, vmax=1e-3)

    # Axis for the wavefront image on the right.
    geom_ax = fig.add_axes([0.50, 0.08, 0.36, 0.9])
    geom_ax.set_xlabel("Arcsec")
    geom_ax.set_ylabel("Arcsec")
    geom_ax.set_title("Geometric Optics")
    geom_im = geom_ax.imshow(np.ones((args.nx, args.nx), dtype=float), animated=True, extent=extent,
                             vmin=0.0, vmax=1e-3)

    # Color items white to show up on black background
    for ax in [fft_ax, geom_ax]:
        for _, spine in ax.spines.items():
            spine.set_color('w')
        ax.title.set_color('w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='both', colors='w')

    ztext = []
    for i in range(2, args.jmax+1):
        x = 0.88
        y = 0.1 + (args.jmax-i)/args.jmax*0.8
        ztext.append(fig.text(x, y, "Z{:d} = {:5.3f}".format(i, 0.0)))
        ztext[-1].set_color('w')


    I_fft = fft_ax.text(0.05, 0.955, '', transform=fft_ax.transAxes)
    I_fft.set_color('w')
    I_phot = geom_ax.text(0.05, 0.955, '', transform=geom_ax.transAxes)
    I_phot.set_color('w')

    etext_fft = fft_ax.text(0.05, 0.91, '', transform=fft_ax.transAxes)
    etext_fft.set_color('w')
    etext_phot = geom_ax.text(0.05, 0.91, '', transform=geom_ax.transAxes)
    etext_phot.set_color('w')

    # plt.show()

    fft_mom = np.empty((args.n, 8), dtype=float)
    geom_mom = np.empty((args.n, 8), dtype=float)

    with ProgressBar(args.n) as bar:
        with writer.saving(fig, args.out+"movie.mp4", 100):
            for i, aberration in enumerate(aberrations):
                fft_psf = galsim.OpticalPSF(lam=lam, diam=diam, obscuration=obscuration,
                                            aper=fft_aper, aberrations=[0]+aberration.tolist())
                fft_img = fft_psf.drawImage(nx=args.nx, ny=args.nx, scale=scale)

                geom_psf = galsim.OpticalPSF(lam=lam, diam=diam, obscuration=obscuration,
                                             aper=geom_aper, aberrations=[0]+aberration.tolist())
                geom_img = geom_psf.drawImage(nx=args.nx, ny=args.nx, scale=scale,
                                              method='phot', n_photons=100000)

                fft_im.set_array(fft_img.array)
                geom_im.set_array(geom_img.array)

                for j, ab in enumerate(aberration):
                    if j == 0:
                        continue
                    ztext[j-1].set_text("Z{:d} = {:5.3f}".format(j+1, ab))

                # Calculate simple estimate of ellipticity
                mom_fft = simple_moments(fft_img)
                mom_phot = simple_moments(geom_img)
                e_fft = ellip(mom_fft)
                e_phot = ellip(mom_phot)

                Is = ("$I_x$={:6.3f}, $I_y$={:6.3f}, $I_{{xx}}$={:6.3f},"
                      " $I_{{yy}}$={:6.3f}, $I_{{xy}}$={:6.3f}")
                I_fft.set_text(Is.format(mom_fft['Ix'], mom_fft['Iy'],
                                         mom_fft['Ixx'], mom_fft['Iyy'], mom_fft['Ixy']))
                I_phot.set_text(Is.format(mom_phot['Ix'], mom_phot['Iy'],
                                          mom_phot['Ixx'], mom_phot['Iyy'], mom_phot['Ixy']))
                etext_fft.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                                   e_fft['e1'], e_fft['e2'], e_fft['rsqr']))
                etext_phot.set_text("$e_1$={:6.3f}, $e_2$={:6.3f}, $r^2$={:6.3f}".format(
                                    e_phot['e1'], e_phot['e2'], e_phot['rsqr']))


                fft_mom[i] = (mom_fft['Ix'], mom_fft['Iy'],
                              mom_fft['Ixx'], mom_fft['Iyy'], mom_fft['Ixy'],
                              e_fft['e1'], e_fft['e2'], e_fft['rsqr'])

                geom_mom[i] = (mom_phot['Ix'], mom_phot['Iy'],
                              mom_phot['Ixx'], mom_phot['Iyy'], mom_phot['Ixy'],
                              e_phot['e1'], e_phot['e2'], e_phot['rsqr'])

                writer.grab_frame(facecolor=fig.get_facecolor())

                bar.update()

    def symmetrize_axis(ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.plot(lim, lim)

    # Centroid plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].scatter(fft_mom[:, 0], geom_mom[:, 0])
    axes[1].scatter(fft_mom[:, 0], geom_mom[:, 0])
    axes[0].set_title("Ix")
    axes[1].set_title("Iy")
    for ax in axes:
        ax.set_xlabel("Fourier Optics")
        ax.set_ylabel("Geometric Optics")
        symmetrize_axis(ax)
    fig.tight_layout()
    fig.savefig(args.out+"centroid.png", dpi=300)

    # Second moment plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].scatter(fft_mom[:, 2], geom_mom[:, 2])
    axes[1].scatter(fft_mom[:, 3], geom_mom[:, 3])
    axes[2].scatter(fft_mom[:, 4], geom_mom[:, 4])
    axes[0].set_title("Ixx")
    axes[1].set_title("Iyy")
    axes[2].set_title("Ixy")
    for ax in axes:
        ax.set_xlabel("Fourier Optics")
        ax.set_ylabel("Geometric Optics")
        symmetrize_axis(ax)
    fig.tight_layout()
    fig.savefig(args.out+"2ndMoment.png", dpi=300)

    # Ellipticity plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].scatter(fft_mom[:, 5], geom_mom[:, 5])
    axes[1].scatter(fft_mom[:, 6], geom_mom[:, 6])
    axes[2].scatter(fft_mom[:, 7], geom_mom[:, 7])
    axes[0].set_title("e1")
    axes[1].set_title("e2")
    axes[2].set_title("rsqr")
    for ax in axes:
        ax.set_xlabel("Fourier Optics")
        ax.set_ylabel("Geometric Optics")
        symmetrize_axis(ax)
    fig.tight_layout()
    fig.savefig(args.out+"ellipticity.png", dpi=300)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=1,
                        help="Random number seed.  Default: 1")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of frames to generate.  Default: 100")
    parser.add_argument("--jmax", type=int, default=15,
                        help="Maximum Zernike to include.  Default: 15")
    parser.add_argument("--ell", type=float, default=4.0,
                        help="Correlation length of Zernike coefficients in frames.  Default: 4.0")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Amplitude of Zernike coefficient fluctuations.  Default: 0.05")

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

    parser.add_argument("--nx", type=int, default=256,
                        help="Output PSF image dimensions in pixels.  Default: 256")
    parser.add_argument("--size", type=float, default=0.6,
                        help="Size of PSF image in arcseconds.  Default: 0.6")

    parser.add_argument("--pad_factor", type=float, default=1.0,
                        help="Factor by which to pad Fourier PSF InterpolatedImage to avoid "
                             "aliasing.  Default: 1.0")
    parser.add_argument("--oversampling", type=float, default=1.0,
                        help="Factor by which to oversample the Fourier PSF InterpolatedImage. "
                             "Default: 1.0")
    parser.add_argument("--geom_oversampling", type=float, default=1.0,
                        help="Factor by which to oversample geometric *pupil plane*.  Default: 1.0")

    parser.add_argument("--out", type=str, default="output/fft_vs_geom_",
                        help="Prefix for output files.  Default['output/fft_vs_geom_']")
    args = parser.parse_args()
    make_movie(args)
