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
"""
This script demonstrates the FourierSqrt operation by doing some toy-model simulations
to compare different image coaddition algorithms.  Using observational parameters roughly
appropriate for LSST, it computes the PSF size and 5-sigma point source magnitude limit
for three different coaddition algorithms, using a log-normal distribution for seeing
and a normal distribution for the depth of the input exposures.  The three algorithms
are:

 - Direct: just adding the images with some weight, then adding their
   PSF models with the same weights.

 - PSF-Matched: all input images are matched to a common PSF and then added
   with some weight.

 - Kaiser: uses a Fourier-space approach to combine images optimally.  The original
   paper that developed the algorithm by Nick Kaiser has not been published (even
   to the arxiv), but the same algorithm can be found in Zackay & Ofek 2015
   (http://adsabs.harvard.edu/abs/2015arXiv151206879Z).

The first two approaches are lossy, and hence imposing a seeing percentile cut
on the input images can improve the results; this can be configured with the
"-i" command-line option.  We weight the exposures by point source SNR.
"""

import argparse
import sys
import os
import numpy
import scipy.stats
import matplotlib.pyplot
import galsim

numpy.random.seed(500)

N_SIGMA_DEPTH = 5    # use 5-sigma (point source) as the measure of depth

# Kwargs to pass to drawImage when creating PSF images.
PSF_DRAW_KWARGS = dict(nx=20, ny=20, scale=0.2, method='no_pixel')


class CoaddMocker(object):
    """Interface specification for classes that know how to mock up an effective
    PSF and variance for a particular coaddition algorithm.
    """

    def mockCoadd(self, variances, fwhms, psfs):
        """Compute the effective PSF and per-pixel variance of a coadd.

        This method must be implemented by subclasses to define the coadd
        algorithm.
        """
        raise NotImplementedError()

    def selectInputs(self, variances, fwhms):
        """Return a mask object (boolean array or slice) that selects images
        that should go into a coadd, given the FWHMs and variances of the input
        exposures.

        Default implementation returns a slice that selects all images.
        """
        slice(None)


class DirectCoaddMocker(object):
    """Coadd mocker for direct coaddition: simply adding images with some weight.
    """

    def __init__(self, included_fraction=1.0):
        self.included_fraction = included_fraction

    def mockCoadd(self, variances, fwhms, psfs):
        weights = 1.0 / (fwhms**2 * variances)
        weights /= weights.sum()
        coadd_psf = galsim.Sum([psf*weight for psf, weight in zip(psfs, weights)])
        coadd_variance = (variances*weights*weights).sum()
        return coadd_variance, coadd_psf

    def selectInputs(self, variances, fwhms):
        cutoff = numpy.percentile(fwhms, 100*self.included_fraction)
        return fwhms < cutoff


class PSFMatchedCoaddMocker(object):
    """Coadd mocker for PSF-matched coaddition, in which each exposure's
    PSF is convolved with a kernel that matches it to the worst input PSF.
    """


    def __init__(self, included_fraction=1.0):
        self.included_fraction = included_fraction

    def mockCoadd(self, variances, fwhms, psfs):
        weights = 1.0 / (fwhms**2 * variances)
        weights /= weights.sum()
        coadd_psf = psfs[fwhms.argmax()]
        # We ignore transfer from variance to covariance from convolution
        # with the matching kernel, because this toy model of coaddition
        # doesn't track covariance at all.  Treating it all as variance
        # should be a better approximation than calculating the transfer
        # and then throwing away the variance as long as the matching
        # kernels are typically smaller than the PSFs.
        coadd_variance = (variances*weights*weights).sum()
        return coadd_variance, coadd_psf

    def selectInputs(self, variances, fwhms):
        cutoff = numpy.percentile(fwhms, 100*self.included_fraction)
        return fwhms < cutoff


class KaiserCoaddMocker(object):
    """Coadd mocker for optimal coaddition in Fourier space;
    see http://adsabs.harvard.edu/abs/2015arXiv151206879Z.
    """

    def mockCoadd(self, variances, fwhms, psfs):
        weights = 1.0 / variances
        weights /= weights.sum()
        coadd_variance = 1.0 / (1.0 / variances).sum()
        coadd_psf = galsim.FourierSqrt(
            galsim.Sum([
                galsim.AutoCorrelate(psf)*weight
                for psf, weight in zip(psfs, weights)
            ])
        )
        return coadd_variance, coadd_psf

    def selectInputs(self, variances, fwhms):
        return slice(None, None, None)


class CoaddMetricCalculator(object):
    """Object that runs several coadd mockers on a set of input PSFs and variances
    representing a single point on the sky, computing the effective FWHM of the
    coadd PSF (from its effective area) and the variance in the coadd pixels.
    """

    def __init__(self, mockers=None, included_fraction=1.0):
        if mockers is None:
            mockers = {
                "direct": DirectCoaddMocker(included_fraction),
                "psf-matched": PSFMatchedCoaddMocker(included_fraction),
                "kaiser": KaiserCoaddMocker(),
            }
        self.mockers = dict(mockers)

    def makePSF(self, fwhm):
        """Create a galsim.GSObject that represents a PSF with the given FWHM.

        For simplicity and speed we just use Gaussian.
        """
        return galsim.Gaussian(fwhm=fwhm)

    def buildInputs(self, fwhms, depths):
        """Build input PSFs and variances for a set of input images from their
        FWHMs and 5-sigma magnitude limits.
        """
        psfs = numpy.zeros(len(fwhms), dtype=object)
        n_effs = numpy.zeros(len(fwhms), dtype=float)
        for n, (fwhm, depth) in enumerate(zip(fwhms, depths)):
            psfs[n] = self.makePSF(fwhm)
            psf_image = psfs[n].drawImage(**PSF_DRAW_KWARGS)
            n_effs[n] = psf_image.array.sum()**2 / (psf_image.array**2).sum()
        flux_limit = 10**(-0.4*depths)
        variances = (flux_limit/N_SIGMA_DEPTH)**2 / n_effs
        fwhm_factor = numpy.median(fwhms / n_effs**0.5)
        return psfs, variances, fwhm_factor

    def computeMetrics(self, coadd_psf, coadd_variance, fwhm_factor):
        """Given a coadd PSF (GSObject), per-pixel variance, compute the
        effective FWHM and 5-sigma magnitude limit of the coadd.

        Effective FWHM is computed as a simple scaling of the square root of
        the PSF effective area; effective area is a more meaningful measure of
        PSF size, but FWHM is more readily understood by humans.  The scaling
        factor for this conversion is given by the fwhm_factor argument.
        """
        coadd_psf_image = coadd_psf.drawImage(**PSF_DRAW_KWARGS)
        coadd_n_eff = coadd_psf_image.array.sum()**2 / (coadd_psf_image.array**2).sum()
        coadd_depth = -2.5*numpy.log10(
            N_SIGMA_DEPTH * (coadd_variance*coadd_n_eff)**0.5
        )
        return fwhm_factor*coadd_n_eff**0.5, coadd_depth

    def __call__(self, fwhms, depths):
        """Compute coadd PSF metrics for input exposures defined by the given
        PSF FWHMs and 5-sigma magnitude limits.

        Returns a dict of metrics with keys "<name>.fwhm" and "<name>.depth",
        along with the fwhm_factor used to compute effective FWHM from PSF
        effective area.
        """
        psfs, variances, fwhm_factor = self.buildInputs(fwhms, depths)
        result = {}
        for name, mocker in self.mockers.items():
            mask = mocker.selectInputs(variances, fwhms)
            coadd_variance, coadd_psf = mocker.mockCoadd(variances[mask], fwhms[mask], psfs[mask])
            coadd_fwhm, coadd_depth = self.computeMetrics(coadd_psf, coadd_variance,
                                                          fwhm_factor)
            result["{0}.fwhm".format(name)] = coadd_fwhm
            result["{0}.depth".format(name)] = coadd_depth
        return result, fwhm_factor


def compareCoadds(depth=24.7, depth_scatter=0.2, fwhm=0.7, fwhm_scatter=0.2,
                  n_exposures=200, n_realizations=100, included_fraction=1.0, file=None):
    """Generate several realizations of mock inputs and plot histograms of coadd quality.
    """
    depths = scipy.stats.norm(depth, depth_scatter).rvs(size=(n_realizations, n_exposures))
    fwhms = scipy.stats.lognorm(s=fwhm_scatter, scale=fwhm).rvs(size=(n_realizations, n_exposures))

    calc = CoaddMetricCalculator(included_fraction=included_fraction)
    result = {}
    for name in calc.mockers:
        result["{0}.fwhm".format(name)] = numpy.zeros(n_realizations, dtype=float)
        result["{0}.depth".format(name)] = numpy.zeros(n_realizations, dtype=float)
    for n in range(n_realizations):
        local, fwhm_factor = calc(fwhms[n], depths[n])
        for k, v in local.items():
            result[k][n] = local[k]

    plot_kwds = dict(bins=150, normed=True, linewidth=0, alpha=0.75)

    fig = matplotlib.pyplot.figure(figsize=(8,10))
    ax1 = fig.add_subplot(2, 1, 1)
    for name in calc.mockers:
        ax1.hist(result["{0}.depth".format(name)], label=name, range=(24.0, 28.0), **plot_kwds)
    ax1.hist(depths.ravel(), label="input exposures", range=(24.0, 28.0), **plot_kwds)
    ax1.set_xlabel("5-sigma magnitude limit")
    ax1.set_xlim(24.0, 28.0)

    ax2 = fig.add_subplot(2, 1, 2)
    for name in calc.mockers:
        ax2.hist(result["{0}.fwhm".format(name)], label=name, range=(0.5, 1.2), **plot_kwds)
    ax2.hist(fwhms.ravel(), label="input exposures", range=(0.5, 1.2), **plot_kwds)
    ax2.set_xlabel("PSF Effective FWHM")
    ax2.set_xlim(0.5, 1.2)
    ax2.legend()

    if file is None:
        matplotlib.pyplot.show()
    else:
        if not os.path.isdir(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        fig.savefig(file)

    return result, fig


def main(argv):
    parser = argparse.ArgumentParser(description="Compare coadd algorithms with toy-level simulations")
    parser.add_argument("-i", "--include", metavar="FRACTION", type=float,
                        help="Fraction of exposures (ordered by seeing) to include for lossy algorithms",
                        default=1.0)
    parser.add_argument("-r", "--realizations", metavar="N", type=int,
                        help="Number of realizations (sets of exposures)",
                        default=100)
    parser.add_argument("-e", "--exposures", metavar="N", type=int,
                        help="Number of exposures at each point",
                        default=200)
    parser.add_argument("--depth", metavar="MAG", type=int,
                        help="5-sigma point source depth of each input exposure, on average",
                        default=24.7)
    parser.add_argument("--depth-scatter", metavar="MAG", type=int,
                        help="magnitude scatter in the per-exposure depth (normal distribution)",
                        default=0.2)
    parser.add_argument("--fwhm", metavar="ARCSEC", type=int,
                        help="Median seeing of input exposures",
                        default=0.7)
    parser.add_argument("--fwhm-scatter", metavar="ARCSEC", type=int,
                        help="RMS scatter in the FWHM of input exposure PSFs (log-normal distribution)",
                        default=0.2)
    parser.add_argument("--file", metavar="FILE", type=str,
                        help="Filename for the output plot; extension sets the format.",
                        default=os.path.join(os.path.dirname(__file__), "output", "demo14.png"))
    parser.add_argument("--display", action="store_const", dest="file", const=None,
                        help="Display the plots in a window instead of writing a file.")
    args = parser.parse_args(argv)
    compareCoadds(
        depth=args.depth, depth_scatter=args.depth_scatter,
        fwhm=args.fwhm, fwhm_scatter=args.fwhm_scatter,
        n_exposures=args.exposures, n_realizations=args.realizations,
        included_fraction=args.include,
        file=args.file
    )


if __name__ == "__main__":
    main(sys.argv[1:])
