# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
Module containing code for estimating shear power spectra from shears at gridded positions.

The code below was developed largely by Joe Zuntz and tweaked by assorted GalSim
developers.  This development and testing took place in a separate (private) repository before the
code was moved into the GalSim repository, but there are some demonstrations of the performance of
this code in devel/modules/lensing_engine.pdf
"""
import numpy as np
import os
import sys

from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError


class PowerSpectrumEstimator(object):
    """Class for estimating the shear power spectrum from gridded shears.

    This class stores all the data used in power spectrum estimation that is fixed with the geometry
    of the problem - the binning and spin weighting factors.

    The only public method is estimate(), which is called with 2D ``g1`` and ``g2`` arrays on a
    square grid.  It assumes the flat sky approximation (where ``ell`` and ``k`` are
    interchangeable), and rebins the observed ell modes into a user-defined number of logarithimic
    bins in ell.  Given that the grid parameters are precomputed and stored when the
    `PowerSpectrumEstimator` is initialized, computation of the PS for multiple sets of shears
    corresponding to the same grid setup can proceed more rapidly than if everything had to be
    recomputed each time.

    Below is an example of how to use this code (relying on GalSim to provide the arrays of g1 and
    g2, though that is by no means required, and assuming that the user is sitting in the examples/
    directory)::

        >>> grid_size = 10.  # Define the total grid extent, in degrees
        >>> ngrid = 100      # Define the number of grid points in each dimension: (ngrid x ngrid)
        >>> n_ell = 15       # Choose the number of logarithmic bins in ell or k for outputs
        >>>
        >>> # Define a lookup-table for the power spectrum as a function of k based on the outputs
        >>> # of iCosmo (see demo11.py for more description of how this was generated).
        >>> my_tab = galsim.LookupTable(file='data/cosmo-fid.zmed1.00.out')
        >>>
        >>> # Generate a galsim.PowerSpectrum with this P(k), noting the units.
        >>> my_ps = galsim.PowerSpectrum(my_tab, units=galsim.radians)
        >>>
        >>> # Build a grid of shear values with the desired parameters.
        >>> g1, g2 = my_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid,
        ...                          units=galsim.degrees)
        >>>
        >>> # Initialize a PowerSpectrumEstimator with the chosen grid geometry and number of ell
        >>> # bins. Note that these values are actually the default, so we didn't technically have
        >>> # to specifythem.
        >>> my_pse = galsim.pse.PowerSpectrumEstimator(ngrid, grid_size, n_ell)
        >>>
        >>> # Estimate the power based on this set of g1, g2.  If we get another set of shears for
        >>> # the same grid geometry, we can reuse the same PowerSpectrumEstimator object.
        >>> ell, P_e, P_b, P_eb = my_pse.estimate(g1, g2)

    The output NumPy arrays ``ell``, ``P_e``, ``P_b``, and ``P_eb`` contain the effective ell
    value, the E-mode auto-power spectrum, the B-mode auto-power spectrum, and the EB cross-power
    spectrum.  The units are inverse radians for ell, and radians^2 for the output power spectra.

    Some important notes:

    1) Power spectrum estimation requires a weight function which decides how the averaging is done
       across ell within each bin.  By default that weighting is flat in ell using an analytic
       calculation of the area in ell space, but this is easy to change with the ``_bin_power``
       function.  (Note this area averaged bin weighting is only approximate for the higher
       frequency bins in which the lower ``ell`` edge is greater than ``pi * ngrid / grid_size``,
       due to the annular ``ell`` region being cut off by the square grid edges beyond this value.)
       A keyword allows for weighting by the power itself.
    2) This is the power spectrum of the gridded *data*, not the underlying field - we do not
       account for the effects of the finite grid (basically, ignoring all the reasons why power
       spectrum estimation is hard - see devel/modules/lensing_engine.pdf in the GalSim repository).
       Users must account for the contribution of noise in ``g1``, ``g2`` and any masking.
    3) The binning is currently fixed as uniform in log(ell).
    4) The code for this class uses the notation of the GREAT10 handbook (Kitching et al. 2011,
       http://dx.doi.org/10.1214/11-AOAS484), equations 17-21.
    """
    def __init__(self, N=100, sky_size_deg=10., nbin=15):
        """Create a PowerSpectrumEstimator object given some grid parameters.

        This constructor precomputes some numbers related to the grid geometry, so the same
        PowerSpectrumEstimator can be used to estimate the power spectrum quickly for many sets of
        shears at gridded positions.

        Parameters:
            N:              The number of pixels along each side of the grid. [default: 100]
            sky_size_deg:   The total grid width (in one dimension) in degrees. [default: 10]
            nbin:           The number of evenly-spaced logarithmic ``ell`` bins to use for
                            estimating the power spectrum. [default: 15]
        """
        # Set up the scales of the sky and pixels
        self.N = N
        self.sky_size_deg = sky_size_deg
        self.nbin = nbin
        self.sky_size = np.radians(sky_size_deg)
        self.dx = self.sky_size / N

        # Define the possible ell range, the bin edges and effective ell values.
        # This is necessary for binning the power spectrum in ell.
        lmin = 2*np.pi / self.sky_size
        lmax = np.sqrt(2.)*np.pi / self.dx # in 2 dimensions
        self.bin_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin+1)
        # By default, report an area-averaged value of ell, which should be fine if there is
        # no weighting (in which case it's recomputed) and if there are many ell modes in
        # each bin.  The latter assumption is most likely to break down at low ell.  Note also that
        # at high ell when the lower ell edge is greater than pi * ngrid / grid_size, due to the
        # annular ell region being cut off by the square grid edges beyond this value, this annular
        # average is only approximate.
        self.ell = (2./3.)*(self.bin_edges[1:]**3-self.bin_edges[:-1]**3) \
                                   / (self.bin_edges[1:]**2-self.bin_edges[:-1]**2)

        # Precompute and store two useful factors, both in the form of 2D grids in Fourier space.
        # These are the lengths of the wavevector |ell| for each point in the space, and the complex
        # valued spin-weighting that takes the complex shear fields -> E,B
        self.l_abs, self.eb_rot = self._generate_eb_rotation()

    def __repr__(self):
        return "galsim.pse.PowerSpectrumEstimator(N=%r, sky_size_deg=%r, nbin=%r)"%(
                self.N, self.sky_size_deg, self.nbin)
    def __eq__(self, other): return self is other or repr(self) == repr(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(repr(self))

    def _generate_eb_rotation(self):
        # Set up the Fourier space grid lx, ly.
        ell = 2*np.pi*np.fft.fftfreq(self.N, self.dx)
        lx, ly = np.meshgrid(ell,ell)

        # Now compute the lengths and angles of the ell vectors.
        l_sq = lx**2 + ly**2

        # Compute exp(-2i psi) where psi = atan2(ly,lx)
        l_sq[0,0] = 1  # Avoid division by 0
        expm2ipsi = (lx - 1j * ly)**2 / l_sq
        l_abs = np.sqrt(l_sq)
        l_abs[0,0] = 0  # Go back to correct value at 0,0.

        self.lx = lx
        self.ly = ly

        return l_abs, expm2ipsi

    def _bin_power(self, C, ell_weight=None):
        # This little utility function bins a 2D C^{E/B, E/B}_{ell} based on |ell|.  The use of
        # histogram is a little hack, but is quite convenient since it means everything is done in C
        # so it is very fast. The first call to `histogram` just returns an array over the
        # logarithmic ell bins of
        # sum_{|ell| in bin} weight(|ell|)*C_{ell_x,ell_y}
        # and the second call returns
        # sum_{|ell| in bin} weight(|ell|).
        # Thus, the ratio is just the mean power in the bin.  If `ell_weight` is None, then weight=1
        # for all ell, corresponding to a simple averaging process.  If `ell_weight` is not None,
        # then some non-flat weighting scheme is used for averaging over the ell values within a
        # bin.
        if ell_weight is not None:
            ell_weight = np.abs(ell_weight)
            P,_ = np.histogram(self.l_abs, self.bin_edges, weights=C*ell_weight)
            count,_ = np.histogram(self.l_abs, self.bin_edges, weights=ell_weight)
        else:
            P,_ = np.histogram(self.l_abs, self.bin_edges, weights=C)
            count,_ = np.histogram(self.l_abs, self.bin_edges)
        if (count == 0).any():
            raise GalSimError("Logarithmic bin definition resulted in >=1 empty bin!")
        return P/count

    def estimate(self, g1, g2, weight_EE=False, weight_BB=False, theory_func=None):
        """Compute the EE, BB, and EB power spectra of two 2D arrays ``g1`` and ``g2``.

        For example usage, see the docstring for the `PowerSpectrumEstimator` class.

        Parameters:
            g1:             The shear component g1 as a square 2D NumPy array.
            g2:             The shear component g2 as a square 2D NumPy array.
            weight_EE:      If True, then the E auto-power spectrum is re-computed weighting by
                            the power within each logarithmically-spaced ell bin. [default: False]
            weight_BB:      If True, then the B auto-power spectrum is re-computed weighting by
                            the power within each logarithmically-spaced ell bin. [default: False]
            theory_func:    An optional callable function that can be used to get an idealized
                            value of power at each point on the grid, and then see what results
                            it gives for our chosen ell binning. [default: None]
        """
        from .table import LookupTable
        # Check for the expected square geometry consistent with the previously-defined grid size.
        if g1.shape != g2.shape:
            raise GalSimIncompatibleValuesError(
                "g1 and g2 grids do not have the same shape.", g1=g1, g2=g2)
        if g1.shape[0] != g1.shape[1]:
            raise GalSimValueError("Input shear arrays must be square.", g1.shape)
        if g1.shape[0] != self.N:
            raise GalSimValueError("Input shear array size is not correct!", g1.shape)

        if not isinstance(weight_EE, bool) or not isinstance(weight_BB, bool):
            raise TypeError("Input weight flags must be bools!")

        # Transform g1+j*g2 into Fourier space and rotate into E-B, then separate into E and B.
        EB = np.fft.ifft2(self.eb_rot * np.fft.fft2(g1 + 1j*g2))
        E = np.fft.fft2(EB.real)
        B = np.fft.fft2(EB.imag)

        # Use the internal function above to bin, and account for the normalization of the FFT.
        # Recall that power has units of angle^2, which is the reason why we need a self.dx^2 in the
        # equations below in addition to the standard 1/N^2 coming from the FFTs.
        C_EE = self._bin_power(E*np.conjugate(E))*(self.dx/self.N)**2
        C_BB = self._bin_power(B*np.conjugate(B))*(self.dx/self.N)**2
        C_EB = self._bin_power(E*np.conjugate(B))*(self.dx/self.N)**2

        if theory_func is not None:
            # theory_func needs to be a callable function
            C_theory_ell = np.zeros_like(self.l_abs)
            C_theory_ell[self.l_abs>0] = theory_func(self.l_abs[self.l_abs>0])
            C_theory = self._bin_power(C_theory_ell)

        if weight_EE or weight_BB:
            # Need to interpolate C_EE to values of self.l_abs.  A bit of kludginess as we go off
            # the end of our final ell grid...
            new_ell = np.zeros(len(self.ell)+2)
            new_ell[1:len(self.ell)+1] = self.ell
            new_ell[len(self.ell)+1] = 10.*max(self.ell)
            if theory_func is not None:
                C_theory = self._bin_power(C_theory_ell, ell_weight=C_theory_ell)

        if weight_EE:
            new_CEE = np.zeros_like(new_ell)
            new_CEE[1:len(self.ell)+1] = np.real(C_EE)
            new_CEE[len(self.ell)+1] = new_CEE[len(self.ell)]
            EE_table = LookupTable(new_ell, new_CEE)
            ell_weight = EE_table(self.l_abs)
            C_EE = self._bin_power(E*np.conjugate(E), ell_weight=ell_weight)*(self.dx/self.N)**2

        if weight_BB:
            new_CBB = np.zeros_like(new_ell)
            new_CBB[1:len(self.ell)+1] = np.real(C_BB)
            new_CBB[len(self.ell)+1] = new_CBB[len(self.ell)]
            BB_table = LookupTable(new_ell, new_CBB)
            ell_weight = BB_table(self.l_abs)
            C_BB = self._bin_power(B*np.conjugate(B), ell_weight=ell_weight)*(self.dx/self.N)**2

        # For convenience, return ell (copied in case the user messes with it) and the three power
        # spectra. If the user requested a binned theoretical spectrum, return that as well.
        if theory_func is None:
            return self.ell.copy(), np.real(C_EE), np.real(C_BB), np.real(C_EB)
        else:
            return self.ell.copy(), np.real(C_EE), np.real(C_BB), np.real(C_EB), np.real(C_theory)
