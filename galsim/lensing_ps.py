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

import numpy as np

from .angle import arcsec, AngleUnit
from .position import PositionD, PositionI
from .bounds import BoundsD, BoundsI
from .interpolant import Quintic, Lanczos
from .image import Image, ImageD
from .random import GaussianDeviate
from .table import LookupTable, LookupTable2D
from . import utilities
from . import integ
from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError
from .errors import GalSimNotImplementedError, galsim_warn

def theoryToObserved(gamma1, gamma2, kappa):
    """Helper function to convert theoretical lensing quantities to observed ones.

    This helper function is used internally by the methods `PowerSpectrum.getShear`,
    `PowerSpectrum.getMagnification`, and `PowerSpectrum.getLensing` to convert from theoretical
    quantities (shear and convergence) to observable ones (reduced shear and magnification).
    Users of `PowerSpectrum.buildGrid` outputs can also apply this method directly to the outputs
    in order to get the values of reduced shear and magnification on the output grid.

    Parameters:
        gamma1:     The first shear component, which must be the NON-reduced shear.  This and
                    all other inputs may be supplied either as individual floating point
                    numbers or lists/arrays of floats.
        gamma2:     The second (x) shear component, which must be the NON-reduced shear.
        kappa:      The convergence.

    Returns:
        the reduced shear and magnification as a tuple (g1, g2, mu)
    """
    gamma1 = np.array(gamma1, copy=False, dtype=float)
    gamma2 = np.array(gamma2, copy=False, dtype=float)
    kappa = np.array(kappa, copy=False, dtype=float)

    g1 = gamma1/(1.-kappa)
    g2 = gamma2/(1.-kappa)
    mu = 1./((1.-kappa)**2 - (gamma1**2 + gamma2**2))
    return g1, g2, mu

class PowerSpectrum(object):
    r"""Class to represent a lensing shear field according to some power spectrum :math:`P(k)`.

    **General considerations**:

    A PowerSpectrum represents some (flat-sky) shear power spectrum, either for gridded points or at
    arbitary positions.  This class is originally initialized with a power spectrum from which we
    would like to generate g1 and g2 (and, optionally, convergence kappa) values.  It generates
    shears on a grid, and if necessary, when `getShear` (or another "get" method) is called, it
    will interpolate to the requested positions.  For detail on how these processes are carried
    out, please see the document in the GalSim repository, ``devel/modules/lensing_engine.pdf``.

    This class generates the shears according to the input power spectrum using a DFT approach,
    which means that we implicitly assume our discrete representation of :math:`P(k)` on a grid is
    one complete cell in an infinite periodic series.  We are making assumptions about what
    :math:`P(k)` is doing outside of our minimum and maximum k range, and those must be kept in
    mind when comparing with theoretical expectations.  Specifically, since the power spectrum is
    realized on only a finite grid it has been been effectively bandpass filtered between a
    minimum and maximum k value in each of the k1, k2 directions.  See the `buildGrid` method for
    more information.

    As a result, the shear generation currently does not include sample variance due to coverage of
    a finite patch.  We explicitly enforce :math:`P(0)=0`, which is true for the full sky in a
    reasonable cosmological model, but it ignores the fact that our little patch of sky might
    reasonably live in some special region with respect to shear correlations.  Our :math:`P(0)=0`
    is essentially setting the integrated power below our minimum k value to zero.  The
    implications of the discrete representation, and the :math:`P(0)=0` choice, are discussed in
    more detail in ``devel/modules/lensing_engine.pdf``.

    The effective shear correlation function for the gridded points will be modified both because of
    the DFT approach to representing shears according to a power spectrum, and because of the power
    cutoff below and above the minimum k values.  The latter effect can be particularly important on
    large scales, so the `buildGrid` method has some keywords that can be used to reduce the
    impact of the minimum k set by the grid extent.  The calculateXi() method can be used to
    calculate the expected shear correlation functions given the minimum and maximum k for some grid
    (but ignoring the discrete vs. continuous Fourier transform effects), for comparison with some
    ideal theoretical correlation function given an infinite k range.

    When interpolating the shears to non-gridded points, the shear correlation function and power
    spectrum are modified; see the `getShear` and other "get" method docstrings for more details.

    **The power spectra to be used**:

    When creating a PowerSpectrum instance, you must specify at least one of the E or B mode power
    spectra, which is normally given as a function :math:`P(k)`.  The typical thing is to just use a lambda
    function in Python (i.e., a function that is not associated with a name); for example, to define
    :math:`P(k)=k^2`, one would use ``lambda k : k**2``.  But the power spectra can also be more complicated
    user-defined functions that take a single argument ``k`` and return the power at that ``k``
    value, or they can be instances of the `LookupTable` class for power spectra that are known at
    particular ``k`` values but for which there is not a simple analytic form.

    Cosmologists often express the power spectra in terms of an expansion in spherical harmonics
    (ell), i.e., the :math:`C_\ell` values.  In the flat-sky limit, we can replace :math:`\ell`
    with :math:`k` and :math:`C_\ell` with :math:`P(k)`.  Thus, :math:`k` and :math:`P(k)` have
    dimensions of inverse angle and angle^2, respectively.  It is quite common for people to plot
    :math:`\ell(\ell+1) C_\ell/2\pi`, a dimensionless quantity; the analogous flat-sky
    quantity is :math:`\Delta^2 = k^2 P(k)/2\pi`.

    By default, the PowerSpectrum object assumes it is getting :math:`P(k)`, but it is possible to
    instead give it :math:`\Delta^2` by setting the optional keyword ``delta2 = True`` in the
    constructor.

    The power functions must return a list/array that is the same size as what they are given, e.g.,
    in the case of no power or constant power, a function that just returns a float would not be
    permitted; it would have to return an array of floats all with the same value.

    It is important to note that the power spectra used to initialize the PowerSpectrum object
    should use the same units for k and :math:`P(k)`, i.e., if k is in inverse radians then
    :math:`P(k)` should be in radians^2 (as is natural for outputs from a cosmological shear power
    spectrum calculator).  However, when we actually draw images, there is a natural scale that
    defines the pitch of the image, which is typically taken to be arcsec.  This definition of a
    specific length scale means that by default we assume all quantities to the PowerSpectrum are
    in arcsec, and those are the units used for internal calculations, but the ``units`` keyword
    can be used to specify different input units for :math:`P(k)` (again, within the constraint
    that k and :math:`P(k)` must be consistent).  If the ``delta2`` keyword is set to specify that
    the input is actually the dimensionless power :math:`\Delta^2`, then the input ``units`` are
    taken to apply only to the k values.

    Parameters:
        e_power_function:   A function or other callable that accepts a NumPy array of abs(k)
                            values, and returns the E-mode power spectrum P_E(abs(k)) in an array of
                            the same shape.  The function should return the power spectrum desired
                            in the E (gradient) mode of the image.
                            It may also be a string that can be converted to a function using
                            ``eval('lambda k : '+e_power_function)``, a `LookupTable`, or
                            ``file_name`` from which to read in a `LookupTable`.  If a ``file_name``
                            is given, the resulting `LookupTable` uses the defaults for the
                            `LookupTable` class, namely spline interpolation in :math:`P(k)`.
                            Users who wish to deviate from those defaults (for example, to
                            interpolate in log(P) and log(k), as might be more natural for
                            power-law functions) should instead read in the file to create a
                            `LookupTable` using the necessary non-default settings. [default: None,
                            which means no E-mode power.]
        b_power_function:   A function or other callable that accepts a NumPy array of abs(k)
                            values, and returns the B-mode power spectrum P_B(abs(k)) in an array of
                            the same shape.  The function should return the power spectrum desired
                            in the B (curl) mode of the image.  See description of
                            ``e_power_function`` for input format options.  [default: None, which
                            means no B-mode power.]
        delta2:             Is the power actually given as dimensionless :math:`\Delta^2`, which
                            requires us to multiply by :math:`2\pi / k^2` to get the shear power
                            :math:`P(k)` in units of angle^2?  [default: False]
        units:              The angular units used for the power spectrum (i.e. the units of
                            k^-1 and sqrt(P)). This should be either an `AngleUnit` instance
                            (e.g. galsim.radians) or a string (e.g. 'radians'). [default: arcsec]
    """
    _opt_params = { 'e_power_function' : str, 'b_power_function' : str,
                    'delta2' : bool, 'units' : str }

    def __init__(self, e_power_function=None, b_power_function=None, delta2=False, units=arcsec):
        # Check that at least one power function is not None
        if e_power_function is None and b_power_function is None:
            raise GalSimIncompatibleValuesError(
                "At least one of e_power_function or b_power_function must be provided.",
                e_power_function=e_power_function, b_power_function=b_power_function)

        self.e_power_function = e_power_function
        self.b_power_function = b_power_function
        self.delta2 = delta2
        self.units = units

        # Try these conversions, but we don't actually keep the output.  This just
        # provides a way to test if the arguments are sane.
        # Note: we redo this in buildGrid for real rather than keeping the outputs
        # (e.g. in self.e_power_function, self.b_power_function) so that PowerSpectrum is
        # picklable.  It turns out lambda functions are not picklable.
        self._convert_power_function(self.e_power_function,'e_power_function')
        self._convert_power_function(self.b_power_function,'b_power_function')

        # Check validity of units
        if isinstance(units, str):
            # if the string is invalid, this raises a reasonable error message.
            units = AngleUnit.from_name(units)
        if not isinstance(units, AngleUnit):
            raise GalSimValueError("units must be either an AngleUnit or a string", units,
                                   ('arcsec', 'arcmin', 'degree', 'hour', 'radian'))

        if units == arcsec:
            self.scale = 1
        else:
            self.scale = units / arcsec

    def __repr__(self):
        s = 'galsim.PowerSpectrum(e_power_function=%r'%self.e_power_function
        if self.b_power_function is not None:
            s += ', b_power_function=%r'%self.b_power_function
        if self.delta2:
            s += ', delta2=%r'%self.delta2
        if self.units != arcsec:
            s += ', units=%r'%self.units
        s += ')'
        return s

    def __str__(self):
        s = 'galsim.PowerSpectrum(e_power_function=%s'%self.e_power_function
        if self.b_power_function is not None:
            s += ', b_power_function=%s'%self.b_power_function
        s += ')'
        return s

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, PowerSpectrum) and
                 self.e_power_function == other.e_power_function and
                 self.b_power_function == other.b_power_function and
                 self.delta2 == other.delta2 and
                 self.scale == other.scale))
    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self): return hash(repr(self))

    def _get_scale_fac(self, units):
        if isinstance(units, str):
            # if the string is invalid, this raises a reasonable error message.
            units = AngleUnit.from_name(units)
        if not isinstance(units, AngleUnit):
            raise GalSimValueError("units must be either an AngleUnit or a string", units,
                                   ('arcsec', 'arcmin', 'degree', 'hour', 'radian'))
        return units / arcsec

    def _get_bandlimit_func(self, bandlimit):
        if bandlimit == 'hard':
            return self._hard_cutoff
        elif bandlimit == 'soft':
            return self._softening_function
        elif bandlimit is None:
            return lambda k, kmax: 1.0
        else:
            raise GalSimValueError("Unrecognized option for band limit!", bandlimit,
                                   (None, 'soft', 'hard'))

    def _get_pk(self, power_function, k_max, bandlimit_func):
        if power_function is None:
            return None
        elif self.delta2:
            # Here we have to go from Delta^2 (dimensionless) to P = 2pi Delta^2 / k^2.  We want to
            # have P and therefore 1/k^2 in units of arcsec, so we won't rescale the k that goes in
            # the denominator.  This naturally gives P(k) in arcsec^2.
            return lambda k : (2.*np.pi) * power_function(self.scale*k)/(k**2) * \
                bandlimit_func(self.scale*k, self.scale*k_max)
        elif self.scale != 1:
            # Here, the scale comes in two places:
            # The units of k have to be converted from 1/arcsec, which GalSim wants to use, into
            # whatever the power spectrum function was defined to use.
            # The units of power have to be converted from (input units)^2 as returned by the power
            # function, to Galsim's units of arcsec^2.
            # Recall that scale is (input units)/arcsec.
            return lambda k : power_function(self.scale*k)*(self.scale**2) * \
                bandlimit_func(self.scale*k, self.scale*k_max)
        else:
            return lambda k : power_function(k) * bandlimit_func(k, k_max)

    def buildGrid(self, grid_spacing, ngrid, rng=None, interpolant=None,
                  center=PositionD(0,0), units=arcsec, get_convergence=False,
                  kmax_factor=1, kmin_factor=1, bandlimit="hard", variance=None):
        """Generate a realization of the current power spectrum on the specified grid.

        **Basic functionality**:

        This function will generate a Gaussian random realization of the specified E and B mode
        shear power spectra at a grid of positions, specified by the input parameters
        ``grid_spacing`` (distance between grid points) and ``ngrid`` (number of grid points in
        each direction.)  Units for ``grid_spacing`` and ``center`` can be specified using the
        ``units`` keyword; the default is arcsec, which is how all values are stored internally.
        It automatically computes and stores grids for the shears and convergence.  However, since
        many users are primarily concerned with shape distortion due to shear, the default is to
        return only the shear components; the ``get_convergence`` keyword can be used to also
        return the convergence.

        The quantities that are returned are the theoretical shears and convergences, usually
        denoted gamma and kappa, respectively.  Users who wish to obtain the more
        observationally-relevant reduced shear and magnification (that describe real lensing
        distortions) can either use the `getShear`, `getMagnification`, or `getLensing` methods
        after `buildGrid`, or can use the convenience function `galsim.lensing_ps.theoryToObserved`
        to convert from theoretical to observed quantities.

        **Caveats of the DFT approach**:

        Note that the shears generated using this method correspond to the PowerSpectrum multiplied
        by a sharp bandpass filter, set by the dimensions of the grid.

        The filter sets :math:`P(k) = 0` for::

            abs(k1), abs(k2) < kmin / 2

        and::

            abs(k1), abs(k2) > kmax + kmin / 2

        where::

            kmin = 2. * pi / (ngrid * grid_spacing)
            kmax = pi / grid_spacing

        and where we have adopted the convention that grid points at a given ``k`` represent the
        interval between (k - dk/2) and (k + dk/2) (noting that the grid spacing dk in k space
        is equivalent to ``kmin``).

        It is worth remembering that this bandpass filter will *not* look like a circular annulus
        in 2D ``k`` space, but is rather more like a thick-sided picture frame, having a small
        square central cutout of dimensions ``kmin`` by ``kmin``.  These properties are visible in
        the shears generated by this method.

        If you care about these effects and want to ameliorate their effect, there are two
        optional kwargs you can provide: ``kmin_factor`` and ``kmax_factor``, both of which are 1
        by default.  These should be integers >= 1 that specify some factor smaller or larger
        (for kmin and kmax respectively) you want the code to use for the underlying grid in
        fourier space.  The final shear grid is returned using the specified ``ngrid`` and
        ``grid_spacing`` parameters.  But the intermediate grid in Fourier space will be larger
        by the specified factors.

        Note: These are really just for convenience, since you could easily get the same effect
        by providing different values of ngrid and grid_spacing and then take a subset of them.
        The ``kmin_factor`` and ``kmax_factor`` just handle the scalings appropriately for you.

        Use of ``kmin_factor`` and ``kmax_factor`` should depend on the desired application.  For
        accurate representation of power spectra, one should not change these values from their
        defaults of 1.  Changing them from one means the E- and B-mode power spectra that are input
        will be valid for the larger intermediate grids that get generated in Fourier space, but not
        necessarily for the smaller ones that get returned to the user.  However, for accurate
        representation of cosmological shear correlation functions, use of ``kmin_factor`` larger
        than one can be helpful in getting the shear correlations closer to the ideal theoretical
        ones (see ``devel/module/lensing_engine.pdf`` for details).

        **Aliasing**:

        If the user provides a power spectrum that does not include a cutoff at kmax, then our
        method of generating shears will result in aliasing that will show up in both E- and
        B-modes.  Thus the `buildGrid` method accepts an optional keyword argument called
        ``bandlimit`` that can tell the PowerSpectrum object to cut off power above kmax
        automatically, where the relevant kmax is larger than the grid Nyquist frequency by a factor
        of ``kmax_factor``.  The allowed values for ``bandlimit`` are None (i.e., do nothing),
        ``hard`` (set power to zero above the band limit), or ``soft`` (use an arctan-based
        softening function to make the power go gradually to zero above the band limit).  By
        default, ``bandlimit=hard``.  Use of this keyword does nothing to the internal
        representation of the power spectrum, so if the user calls the `buildGrid` method again,
        they will need to set ``bandlimit`` again (and if their grid setup is different in a way
        that changes ``kmax``, then that's fine).

        **Interpolation**:

        If the grid is being created for the purpose of later interpolating to random positions, the
        following findings should be kept in mind: since the interpolant modifies the effective
        shear correlation function on scales comparable to <~3x the grid spacing, the grid spacing
        should be chosen to be at least 3 times smaller than the minimum scales on which the user
        wishes to reproduce the shear correlation function accurately.  Ideally, the grid should be
        somewhat larger than the region in which shears at random points are needed, so that edge
        effects in the interpolation will not be important.  For this purpose, there should be >~5
        grid points outside of the region in which interpolation will take place.  Ignoring this
        edge effect and using the grid for interpolation out to its edges can suppress shear
        correlations on all scales by an amount that depends on the grid size; for a 100x100 grid,
        the suppression is ~2-3%.  Note that the above numbers came from tests that use a
        cosmological shear power spectrum; precise figures for this suppression can also depend on
        the shear correlation function itself.

        **Sign conventions and other info**:

        Note also that the convention for axis orientation differs from that for the GREAT10
        challenge, so when using codes that deal with GREAT10 challenge outputs, the sign of our g2
        shear component must be flipped.

        For more information on the effects of finite grid representation of the power spectrum
        see ``devel/modules/lensing_engine.pdf``.

        **Examples**:

        1. Get shears on a grid of points separated by 1 arcsec::

                >>> my_ps = galsim.PowerSpectrum(lambda k : k**2)
                >>> g1, g2 = my_ps.buildGrid(grid_spacing = 1., ngrid = 100)

           The returned g1, g2 are 2-d NumPy arrays of values, corresponding to the values of
           g1 and g2 at the locations of the grid points.

           For a given value of ``grid_spacing`` and ``ngrid``, we could get the x and y values on
           the grid using::

                >>> import numpy as np
                >>> min = (-ngrid/2 + 0.5) * grid_spacing
                >>> max = (ngrid/2 - 0.5) * grid_spacing
                >>> x, y = np.meshgrid(np.arange(min,max+grid_spacing,grid_spacing),
                ...                    np.arange(min,max+grid_spacing,grid_spacing))

           where the center of the grid is taken to be (0,0).

        2. Rebuild the grid using a particular rng and set the location of the center of the grid
           to be something other than the default (0,0)::

                >>> g1, g2 = my_ps.buildGrid(grid_spacing = 8., ngrid = 65,
                ...                          rng = galsim.BaseDeviate(1413231),
                ...                          center = galsim.PositionD(256.5, 256.5) )

        3. Make a `PowerSpectrum` from a tabulated :math:`P(k)` that gets interpolated to find the
           power at all necessary values of k, then generate shears and convergences on a grid, and
           convert to reduced shear and magnification so they can be used to transform galaxy
           images.  E.g., assuming that k and P_k are NumPy arrays containing k and :math:`P(k)`::

                >>> tab_pk = galsim.LookupTable(k, P_k)
                >>> my_ps = galsim.PowerSpectrum(tab_pk)
                >>> g1, g2, kappa = my_ps.buildGrid(grid_spacing = 1., ngrid = 100,
                ...                                 get_convergence = True)
                >>> g1_r, g2_r, mu = galsim.lensing_ps.theoryToObserved(g1, g2, kappa)

        Parameters:
            grid_spacing:       Spacing for an evenly spaced grid of points, by default in arcsec
                                for consistency with the natural length scale of images created
                                using the `GSObject.drawImage` method.  Other units can be
                                specified using the ``units`` keyword.
            ngrid:              Number of grid points in each dimension.  [Must be an integer]
            rng:                A `BaseDeviate` object for drawing the random numbers.
                                [default: None]
            interpolant:        `Interpolant` that will be used for interpolating the gridded shears
                                by methods like `getShear`, `getConvergence`, etc. if they are
                                later called. [default: galsim.Lanczos(5)]
            center:             If setting up a new grid, define what position you want to consider
                                the center of that grid.  Units must be consistent with those for
                                ``grid_spacing``.  [default: galsim.PositionD(0,0)]
            units:              The angular units used for the positions.  [default: arcsec]
            get_convergence:    Return the convergence in addition to the shear?  Regardless of the
                                value of ``get_convergence``, the convergence will still be computed
                                and stored for future use. [default: False]
            kmin_factor:        Factor by which the grid spacing in fourier space is smaller than
                                the default.  i.e.::

                                    kmin = 2. * pi / (ngrid * grid_spacing) / kmin_factor

                                [default: 1; must be an integer]
            kmax_factor:        Factor by which the overall grid in fourier space is larger than
                                the default.  i.e.::

                                    kmax = pi / grid_spacing * kmax_factor

                                [default: 1; must be an integer]
            bandlimit:          Keyword determining how to handle power :math:`P(k)` above the
                                limiting k value, kmax.  The options None, 'hard', and 'soft'
                                correspond to doing nothing (i.e., allow P(>kmax) to be aliased to
                                lower k values), cutting off all power above kmax, and applying a
                                softening filter to gradually cut off power above kmax.  Use of
                                this keyword does not modify the internally-stored power spectrum,
                                just the shears generated for this particular call to `buildGrid`.
                                [default: "hard"]
            variance:           Optionally renormalize the variance of the output shears to a
                                given value.  This is useful if you know the functional form of
                                the power spectrum you want, but not the normalization.  This lets
                                you set the normalization separately.  The resulting shears should
                                have var(g1) + var(g2) ~= variance.  If only ``e_power_function`` is
                                given, then this is also the variance of kappa.  Otherwise, the
                                variance of kappa may be smaller than the specified variance.
                                [default: None]

        Returns:
            the tuple (g1,g2[,kappa]), where each is a 2-d NumPy array and kappa is included
            iff ``get_convergence`` is set to True.
        """
        # Check for validity of integer values
        if not isinstance(ngrid, int):
            if ngrid != int(ngrid):
                raise GalSimValueError("ngrid must be an integer", ngrid)
            ngrid = int(ngrid)
        if not isinstance(kmin_factor, int):
            if kmin_factor != int(kmin_factor):
                raise GalSimValueError("kmin_factor must be an integer", kmin_factor)
            kmin_factor = int(kmin_factor)
        if not isinstance(kmax_factor, int):
            if kmax_factor != int(kmax_factor):
                raise GalSimValueError("kmax_factor must be an integer", kmax_factor)
            kmax_factor = int(kmax_factor)

        # Check if center is a PositionD
        if not isinstance(center, PositionD):
            raise GalSimValueError("center argument for buildGrid must be a PositionD instance",
                                   center)

        # Automatically convert units to arcsec at the outset, then forget about it.  This is
        # because PowerSpectrum by default wants to work in arsec, and all power functions are
        # automatically converted to do so, so we'll also do that here.
        scale_fac = self._get_scale_fac(units)
        center *= scale_fac
        grid_spacing *= scale_fac

        # The final grid spacing that will be in the computed images is grid_spacing/kmax_factor.
        self.grid_spacing = grid_spacing / kmax_factor
        self.center = center

        # It is also convenient to store the bounds within which an input position is allowed.
        self.bounds = BoundsD( center.x - (ngrid-1) * grid_spacing / 2. ,
                               center.x + (ngrid-1) * grid_spacing / 2. ,
                               center.y - (ngrid-1) * grid_spacing / 2. ,
                               center.y + (ngrid-1) * grid_spacing / 2. )
        # Expand the bounds slightly to make sure rounding errors don't lead to points on the
        # edge being considered off the edge.
        self.bounds = self.bounds.expand( 1. + 1.e-15 )

        self.x_grid = np.linspace(self.bounds.xmin, self.bounds.xmax, ngrid)
        self.y_grid = np.linspace(self.bounds.ymin, self.bounds.ymax, ngrid)

        gd = GaussianDeviate(rng)

        # Check that the interpolant is valid.
        if interpolant is None:
            self.interpolant = Lanczos(5)
        else:
            self.interpolant = utilities.convert_interpolant(interpolant)

        # Convert power_functions into callables:
        e_power_function = self._convert_power_function(self.e_power_function,'e_power_function')
        b_power_function = self._convert_power_function(self.b_power_function,'b_power_function')

        # Figure out how to apply band limit if requested.
        # Start by calculating kmax in the appropriate units:
        # Generally, it should be kmax_factor*pi/(input grid spacing).  We have already converted
        # the user-input grid spacing to arcsec, the units that the PowerSpectrum class uses
        # internally, and divided it by kmax_factor to get self.grid_spacing, so here we just use
        # pi/self.grid_spacing.
        k_max = np.pi / self.grid_spacing
        bandlimit_func = self._get_bandlimit_func(bandlimit)

        # If we actually have dimensionless Delta^2, then we must convert to power
        # P(k) = 2pi Delta^2 / k^2,
        # which has dimensions of angle^2.
        # Also apply the bandlimit and/or scale as appropriate.
        p_E = self._get_pk(e_power_function, k_max, bandlimit_func)
        p_B = self._get_pk(b_power_function, k_max, bandlimit_func)

        # Build the grid
        self.ngrid_tot = ngrid * kmin_factor * kmax_factor
        self.pixel_size = grid_spacing/kmax_factor
        psr = PowerSpectrumRealizer(self.ngrid_tot, self.pixel_size, p_E, p_B)
        self.grid_g1, self.grid_g2, self.grid_kappa = psr(gd, variance)
        if kmin_factor != 1 or kmax_factor != 1:
            # Need to make sure the rows are contiguous so we can use it in the constructor
            # of the ImageD objects below.  This requires a copy.
            s = slice(0,ngrid*kmax_factor,kmax_factor)
            self.grid_g1 = np.array(self.grid_g1[s,s], copy=True, order='C')
            self.grid_g2 = np.array(self.grid_g2[s,s], copy=True, order='C')
            self.grid_kappa = np.array(self.grid_kappa[s,s], copy=True, order='C')

        # Set up the images to be interpolated.
        # Note: We don't make the LookupTable2D's yet, since we don't know if
        #       the user wants periodic wrapping or not.
        #       So we wait to create them when we are actually going to use them.
        self.im_g1 = ImageD(self.grid_g1, scale=self.grid_spacing)
        self.im_g2 = ImageD(self.grid_g2, scale=self.grid_spacing)
        self.im_kappa = ImageD(self.grid_kappa, scale=self.grid_spacing)

        if get_convergence:
            return self.grid_g1, self.grid_g2, self.grid_kappa
        else:
            return self.grid_g1, self.grid_g2

    def nRandCallsForBuildGrid(self):
        """Return the number of times the rng() was called the last time `buildGrid` was called.

        This can be useful for keeping rngs in sync if the connection between them is broken
        (e.g. when calling the function through a Proxy object).
        """
        if not hasattr(self,'ngrid_tot'):
            raise GalSimError("BuildGrid has not been called yet.")
        ntot = 0
        # cf. PowerSpectrumRealizer._generate_power_array
        temp = 2 * np.product( (self.ngrid_tot, self.ngrid_tot//2 +1 ) )
        if self.e_power_function is not None:
            ntot += temp
        if self.b_power_function is not None:
            ntot += temp
        return int(ntot)

    def _convert_power_function(self, pf, pf_str):
        if pf is None: return None

        # Convert string inputs to either a lambda function or LookupTable
        if isinstance(pf,str):
            origpf = pf
            import os
            if os.path.isfile(pf):
                pf = LookupTable.from_file(pf)
            else:
                # Detect at least _some_ forms of malformed string input.  Note that this
                # test assumes that the eval string completion is defined for k=1.0.
                try:
                    pf = utilities.math_eval('lambda k : ' + pf)
                    pf(1.0)
                except Exception as e:
                    raise GalSimValueError(
                        "String {0} must either be a valid filename or something that "
                        "can eval to a function of k.\n"
                        "Caught error: {1}".format(pf_str, e), origpf)

        # Check that the function is sane.
        # Note: Only try tests below if it's not a LookupTable.
        #       (If it's a LookupTable, then it could be a valid function that isn't
        #        defined at k=1.)
        if not isinstance(pf, LookupTable):
            pf(np.array((0.1,1.)))
        return pf

    def calculateXi(self, grid_spacing, ngrid, kmax_factor=1, kmin_factor=1, n_theta=100,
                    units=arcsec, bandlimit="hard"):
        r"""Calculate shear correlation functions for the current power spectrum on the specified
        grid.

        This function will calculate the theoretical shear correlation functions, :math:`\xi_+`
        and :math:`\xi_-`, for this power spectrum and the grid configuration specified using
        keyword arguments, taking into account the minimum and maximum k range implied by the grid
        parameters, ``kmin_factor`` and ``kmax_factor``.  Most theoretical correlation function
        calculators assume an infinite k range, so this utility can be used to check how close the
        chosen grid parameters (and the implied minimum and maximum k) come to the "ideal" result.
        This is particularly useful on large scales, since in practice the finite grid extent
        limits the minimum k value and therefore can suppress shear correlations on large scales.
        Note that the actual shear correlation function in the generated shears will still differ
        from the one calculated here due to differences between the discrete and continuous Fourier
        transform.

        The quantities that are returned are three NumPy arrays: separation theta (in the adopted
        units), :math:`\xi_+`, and :math:`\xi_-`.  These are defined in terms of the E- and B-mode
        shear power spectrum as in the document ``devel/modules/lensing_engine.pdf``, equations 2
        and 3.  The values that are returned are for a particular theta value, not an average over
        a range of theta values in some bin of finite width.

        This method has been tested with cosmological shear power spectra; users should check for
        sanity of outputs if attempting to use power spectra that have very different scalings with
        k.

        Parameters:
            grid_spacing:   Spacing for an evenly spaced grid of points, by default in arcsec
                            for consistency with the natural length scale of images created
                            using the `GSObject.drawImage` method.  Other units can be specified
                            using the ``units`` keyword.
            ngrid:          Number of grid points in each dimension.  [Must be an integer]
            units:          The angular units used for the positions.  [default = arcsec]
            kmin_factor:    (Optional) Factor by which the grid spacing in fourier space is
                            smaller than the default.  i.e.::

                                kmin = 2. * pi / (ngrid * grid_spacing) / kmin_factor

                            [default ``kmin_factor = 1``; must be an integer]
            kmax_factor:    (Optional) Factor by which the overall grid in fourier space is
                            larger than the default.  i.e.::

                                kmax = pi / grid_spacing * kmax_factor

                            [default ``kmax_factor = 1``; must be an integer]
            n_theta:        (Optional) Number of logarithmically spaced bins in angular
                            separation. [default ``n_theta=100``]
            bandlimit:      (Optional) Keyword determining how to handle power :math:`P(k)` above
                            the limiting k value, kmax.  The options None, 'hard', and 'soft'
                            correspond to doing nothing (i.e., allow P(>kmax) to be aliased to
                            lower k values), cutting off all power above kmax, and applying a
                            softening filter to gradually cut off power above kmax.  Use of this
                            keyword does not modify the internally-stored power spectrum, just
                            the result generated by this particular call to `calculateXi`.
                            [default ``bandlimit="hard"``]

        Returns:
            the tuple (theta, xi_p, xi_m), 1-d NumPy arrays for the angular separation theta
            and the two shear correlation functions.
        """
        # Normalize inputs
        grid_spacing = float(grid_spacing)
        ngrid = int(ngrid)
        kmin_factor = int(kmin_factor)
        kmax_factor = int(kmax_factor)
        n_theta = int(n_theta)

        # Automatically convert units to arcsec at the outset, then forget about it.  This is
        # because PowerSpectrum by default wants to work in arsec, and all power functions are
        # automatically converted to do so, so we'll also do that here.
        scale_fac = self._get_scale_fac(units)
        grid_spacing *= scale_fac

        # Decide on a grid of separation values.  Do this in arcsec, for consistency with the
        # internals of the PowerSpectrum class.
        min_sep = grid_spacing
        max_sep = ngrid*grid_spacing
        theta = np.logspace(np.log10(min_sep), np.log10(max_sep), n_theta)

        # Set up the power spectrum to use for the calculations, just as in buildGrid.
        # Convert power_functions into callables:
        e_power_function = self._convert_power_function(self.e_power_function,'e_power_function')
        b_power_function = self._convert_power_function(self.b_power_function,'b_power_function')

        # Apply band limit if requested; see comments in 'buildGrid()' for more details.
        k_max = kmax_factor * np.pi / grid_spacing
        bandlimit_func = self._get_bandlimit_func(bandlimit)

        # If we actually have dimensionless Delta^2, then we must convert to power
        # P(k) = 2pi Delta^2 / k^2,
        # which has dimensions of angle^2.
        # Also apply the bandlimit and/or scale as appropriate.
        p_E = self._get_pk(e_power_function, k_max, bandlimit_func)
        p_B = self._get_pk(b_power_function, k_max, bandlimit_func)

        # Get k_min value in arcsec:
        k_min = 2.*np.pi / (ngrid * grid_spacing * kmin_factor)

        # Do the actual integration for each of the separation values, now that we have power
        # spectrum functions p_E and p_B.
        xi_p = np.zeros(n_theta)
        xi_m = np.zeros(n_theta)
        for i_theta in range(n_theta):
            # Usually theory calculations use radians.  However, our k and P are already set up to
            # use arcsec, so we need theta to be in arcsec (which it already is) in order for the
            # units to work out right.
            # xi_p = (1/2pi) \int (P_E + P_B) J_0(k theta) k dk
            # xi_m = (1/2pi) \int (P_E - P_B) J_4(k theta) k dk
            if p_E is not None and p_B is not None:
                integrand_p = xip_integrand(lambda k: p_E(k) + p_B(k), theta[i_theta])
                integrand_m = xim_integrand(lambda k: p_E(k) - p_B(k), theta[i_theta])
            elif p_E is not None:
                integrand_p = xip_integrand(p_E, theta[i_theta])
                integrand_m = xim_integrand(p_E, theta[i_theta])
            else:
                integrand_p = xip_integrand(p_B, theta[i_theta])
                integrand_m = xim_integrand(lambda k: -p_B(k), theta[i_theta])
            xi_p[i_theta] = integ.int1d(integrand_p, k_min, k_max, rel_err=1.e-6,
                                        abs_err=1.e-12)
            xi_m[i_theta] = integ.int1d(integrand_m, k_min, k_max, rel_err=1.e-6,
                                        abs_err=1.e-12)
        xi_p /= (2.*np.pi)
        xi_m /= (2.*np.pi)

        # Now convert the array of separation values back to whatever units were used as inputs to
        # this function.
        theta /= scale_fac

        # Return arrays with results.
        return theta, xi_p, xi_m

    @staticmethod
    def _softening_function(k, k_max):
        # Softening function for the power spectrum band-limiting step, instead of a hard cut in k.
        # We use an arctan function to go smoothly from 1 to 0 above k_max.  The input k values
        # can be in any units, as long as the choice of units for k and k_max is the same.

        # The magic numbers in the code below come from the following:
        # We define the function as
        #     (arctan[A log(k/k_max) + B] + pi/2)/pi
        # For our current purposes, we will define A and B by requiring that this function go to
        # 0.95 (0.05) for k/k_max = 0.95 (1).  This gives two equations:
        #     0.95 = (arctan[log(0.95) A + B] + pi/2)/pi
        #     0.05 = (arctan[B] + pi/2)/pi.
        # We will solve the second equation:
        #     -0.45 pi = arctan(B), or
        #     B = tan(-0.45 pi).
        b = np.tan(-0.45*np.pi)

        # Then, we get A from the first equation:
        #     0.45 pi = arctan[log(0.95) A + B]
        #     tan(0.45 pi) = log(0.95) A  + B
        a = (np.tan(0.45*np.pi)-b) / np.log(0.95)
        return (np.arctan(a*np.log(k/k_max)+b) + np.pi/2.)/np.pi

    @staticmethod
    def _hard_cutoff(k, k_max):
        if isinstance(k, float):
            return float(k < k_max)
        else:
            return (k < k_max).astype(float)

    def getShear(self, pos, units=arcsec, reduced=True, periodic=False):
        """
        This function can interpolate between grid positions to find the shear values for a given
        list of input positions (or just a single position).  Before calling this function, you must
        call `buildGrid` first to define the grid of shears and convergences on which to
        interpolate.  The docstring for `buildGrid` provides some guidance on appropriate grid
        configurations to use when building a grid that is to be later interpolated to random
        positions.

        By default, this method returns the reduced shear, which is defined in terms of shear and
        convergence as reduced shear ``g=gamma/(1-kappa)``; the ``reduced`` keyword can be set to
        False in order to return the non-reduced shear.

        Note that the interpolation (specified when calling `buildGrid`) modifies the effective
        shear power spectrum and correlation function somewhat, though the effects can be limited
        by careful choice of grid parameters (see buildGrid() docstring for details).  Assuming
        those guidelines are followed, then the shear correlation function modifications due to use
        of the quintic, Lanczos-3, and Lanczos-5 interpolants are below 5% on all scales from the
        grid spacing to the total grid extent, typically below 2%.  The linear, cubic, and nearest
        interpolants perform significantly more poorly, with modifications of the correlation
        functions that can reach tens of percent on the scales where the recommended interpolants
        perform well.  Thus, the default interpolant is Lanczos-5, and users should think carefully
        about the acceptability of significant modification of the shear correlation function before
        changing to use linear, cubic, or nearest.

        Users who wish to ensure that the shear power spectrum is preserved post-interpolation
        should consider using the ``periodic`` interpolation option, which assumes the shear field
        is periodic (i.e., the sky is tiled with many copies of the given shear field).  Those who
        care about the correlation function should not use this option, and for this reason it's
        not the default.

        **Examples**:

        1. Get the shear for a particular point::

                >>> g1, g2 = my_ps.getShear(pos = galsim.PositionD(12, 412))

           This time the returned values are just floats and correspond to the shear for the
           provided position.

        2. You can also provide a position as a tuple to save the explicit PositionD construction::

                >>> g1, g2 = my_ps.getShear(pos = (12, 412))

        3. Get the shears for a bunch of points at once::

                >>> xlist = [ 141, 313,  12, 241, 342 ]
                >>> ylist = [  75, 199, 306, 225, 489 ]
                >>> poslist = [ galsim.PositionD(xlist[i],ylist[i]) for i in range(len(xlist)) ]
                >>> g1, g2 = my_ps.getShear( poslist )
                >>> g1, g2 = my_ps.getShear( (xlist, ylist) )

           Both calls do the same thing.  The returned g1, g2 this time are numpy arrays of g1, g2
           values.  The arrays are the same length as the number of input positions.

        Parameters:
            pos:            Position(s) of the source(s), assumed to be post-lensing!
                            Valid ways to input this:

                            - single `Position` instance
                            - tuple of floats: (x,y)
                            - list/array of `Position` instances
                            - tuple of lists/arrays: ( xlist, ylist )

            units:          The angular units used for the positions.  [default: arcsec]
            reduced:        Whether returned shear(s) should be reduced shears. [default: True]
            periodi:        Whether the interpolation should treat the positions as being defined
                            with respect to a periodic grid, which will wrap them around if they
                            are outside the bounds of the original grid on which shears were
                            defined.  If not, then shears are set to zero for positions outside the
                            original grid. [default: False]

        Returns:
            the shear as a tuple, (g1,g2)

        If the input ``pos`` is given a single position, (g1,g2) are the two shear components.
        If the input ``pos`` is given a list/array of positions, they are NumPy arrays.
        """
        if not hasattr(self, 'im_g1'):
            raise GalSimError("PowerSpectrum.buildGrid must be called before getShear")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = utilities._convertPositions(pos, units, 'getShear')
        return self._getShear(pos_x, pos_y, reduced, periodic)

    def _getShear(self, pos_x, pos_y, reduced=True, periodic=False):
        """Equivalent to `getShear`, but without some sanity checks and the positions must be
        given as ``pos_x``, ``pos_y`` in arcsec.

        Parameters:
            pos_x:      x position in arcsec (either a scalar or a numpy array)
            pos_y:      y position in arcsec (either a scalar or a numpy array)
            reduced:    Whether returned shear(s) should be reduced shears. [default: True]
            periodic:   Whether the interpolation should treat the positions as being defined
                        with respect to a periodic grid. [default: False]

        Returns:
            the (possibly reduced) shears as a tuple (g1,g2) (either scalars or numpy arrays)
        """
        g1_grid = self.im_g1.array
        g2_grid = self.im_g2.array

        if reduced:
            # get reduced shear (just discard magnification)
            g1_grid, g2_grid, _ = theoryToObserved(g1_grid, g2_grid, self.im_kappa.array)

        lut_g1 = LookupTable2D(self.x_grid, self.y_grid, g1_grid,
                               edge_mode='wrap' if periodic else 'warn',
                               interpolant=self.interpolant)
        lut_g2 = LookupTable2D(self.x_grid, self.y_grid, g2_grid,
                               edge_mode='wrap' if periodic else 'warn',
                               interpolant=self.interpolant)

        ret = lut_g1(pos_x, pos_y), lut_g2(pos_x, pos_y)
        return ret

    def getConvergence(self, pos, units=arcsec, periodic=False):
        """
        This function can interpolate between grid positions to find the convergence values for a
        given list of input positions (or just a single position).  Before calling this function,
        you must call `buildGrid` first to define the grid of convergences on which to interpolate.
        The docstring for `buildGrid` provides some guidance on appropriate grid configurations to
        use when building a grid that is to be later interpolated to random positions.

        Note that the interpolation (specified when calling `buildGrid`) modifies the effective
        2-point functions of these quantities.  See docstring for `getShear` docstring for caveats
        about interpolation.  The user is advised to be very careful about deviating from the
        default Lanczos-5 interpolant.

        The usage of getConvergence() is the same as for `getShear`, except that it returns only a
        single quantity (convergence value or array of convergence values) rather than two
        quantities.  See documentation for `getShear` for some examples.

        Parameters:
            pos:        Position(s) of the source(s), assumed to be post-lensing!
                        Valid ways to input this:

                        - single `Position` instance
                        - tuple of floats: (x,y)
                        - list or array of `Position` instances
                        - tuple of lists/arrays: ( xlist, ylist )

            units:      The angular units used for the positions.  [default: arcsec]
            periodic:   Whether the interpolation should treat the positions as being defined
                        with respect to a periodic grid, which will wrap them around if they
                        are outside the bounds of the original grid on which shears and
                        convergences were defined.  If not, then convergences are set to zero
                        for positions outside the original grid.  [default: False]

        Returns:
            the convergence, kappa (either a scalar or a numpy array)

        If the input ``pos`` is given a single position, kappa is the convergence value.
        If the input ``pos`` is given a list/array of positions, kappa is a NumPy array.
        """
        if not hasattr(self, 'im_kappa'):
            raise GalSimError("PowerSpectrum.buildGrid must be called before getConvergence")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = utilities._convertPositions(pos, units, 'getConvergence')
        return self._getConvergence(pos_x, pos_y, periodic)

    def _getConvergence(self, pos_x, pos_y, periodic=False):
        """Equivalent to `getConvergence`, but without some sanity checks and the positions must be
        given as ``pos_x``, ``pos_y`` in arcsec.

        Parameters:
            pos_x:      x position in arcsec (either a scalar or a numpy array)
            pos_y:      y position in arcsec (either a scalar or a numpy array)
            periodic:   Whether the interpolation should treat the positions as being defined
                        with respect to a periodic grid. [default: False]

        Returns:
            the convergence, kappa (either a scalar or a numpy array)
        """
        kappa_grid = self.im_kappa.array

        lut_kappa = LookupTable2D(self.x_grid, self.y_grid, kappa_grid,
                                  edge_mode='wrap' if periodic else 'warn',
                                  interpolant=self.interpolant)

        return lut_kappa(pos_x, pos_y)

    def getMagnification(self, pos, units=arcsec, periodic=False):
        """
        This function can interpolate between grid positions to find the lensing magnification (mu)
        values for a given list of input positions (or just a single position).  Before calling this
        function, you must call `buildGrid` first to define the grid of shears and convergences on
        which to interpolate.  The docstring for `buildGrid` provides some guidance on appropriate
        grid configurations to use when building a grid that is to be later interpolated to random
        positions.

        Note that the interpolation (specified when calling `buildGrid`) modifies the effective
        2-point functions of these quantities.  See docstring for `getShear` docstring for caveats
        about interpolation.  The user is advised to be very careful about deviating from the
        default Lanczos-5 interpolant.

        The usage of `getMagnification` is the same as for `getShear`, except that it returns only
        a single quantity (a magnification value or array of magnification values) rather than a
        pair of quantities.  See documentation for `getShear` for some examples.

        Parameters:
            pos:        Position(s) of the source(s), assumed to be post-lensing!
                        Valid ways to input this:

                        - single `Position` instance
                        - tuple of floats: (x,y)
                        - list/array of `Position` instances
                        - tuple of lists/arrays: ( xlist, ylist )

            units:      The angular units used for the positions.  [default: arcsec]
            periodic:   Whether the interpolation should treat the positions as being
                        defined with respect to a periodic grid, which will wrap them around
                        if they are outside the bounds of the original grid on which shears
                        and convergences were defined.  If not, then magnification is set to
                        1 for positions outside the original grid.  [default: False]

        Returns:
            the magnification, mu (either a scalar or a numpy array)

        If the input ``pos`` is given a single position, mu is the magnification value.
        If the input ``pos`` is given a list/array of positions, mu is a NumPy array.
        """
        if not hasattr(self, 'im_kappa'):
            raise GalSimError("PowerSpectrum.buildGrid must be called before getMagnification")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = utilities._convertPositions(pos, units, 'getMagnification')
        return self._getMagnification(pos_x, pos_y, periodic)

    def _getMagnification(self, pos_x, pos_y, periodic=False):
        """Equivalent to `getMagnification`, but without some sanity checks and the positions must
        be given as ``pos_x``, ``pos_y`` in arcsec.

        Parameters:
            pos_x:      x position in arcsec (either a scalar or a numpy array)
            pos_y:      y position in arcsec (either a scalar or a numpy array)
            periodic:   Whether the interpolation should treat the positions as being defined
                        with respect to a periodic grid. [default: False]

        Returns:
            the magnification, mu (either a scalar or a numpy array)
        """
        _, _, mu_grid = theoryToObserved(self.im_g1.array, self.im_g2.array, self.im_kappa.array)
        lut_mu = LookupTable2D(self.x_grid, self.y_grid, mu_grid - 1,
                               edge_mode='wrap' if periodic else 'warn',
                               interpolant=self.interpolant)

        return lut_mu(pos_x, pos_y) + 1

    def getLensing(self, pos, units=arcsec, periodic=False):
        """
        This function can interpolate between grid positions to find the lensing observable
        quantities (reduced shears g1 and g2, and magnification mu) for a given list of input
        positions (or just a single position).  Before calling this function, you must call
        `buildGrid` first to define the grid of shears and convergences on which to interpolate.
        The docstring for `buildGrid` provides some guidance on appropriate grid configurations to
        use when building a grid that is to be later interpolated to random positions.

        Note that the interpolation (specified when calling `buildGrid`) modifies the effective
        2-point functions of these quantities.  See docstring for `getShear` docstring for caveats
        about interpolation.  The user is advised to be very careful about deviating from the
        default Lanczos-5 interpolant.

        The usage of `getLensing` is the same as for `getShear`, except that it returns three
        quantities (two reduced shear components and magnification) rather than two.  See
        documentation for `getShear` for some examples.

        Parameters:
            pos:        Position(s) of the source(s), assumed to be post-lensing!
                        Valid ways to input this:

                        - single `Position` instance
                        - tuple of floats: (x,y)
                        - list/array of `Position` instances
                        - tuple of lists/arrays: ( xlist, ylist )

            units:      The angular units used for the positions.  [default: arcsec]
            periodic:   Whether the interpolation should treat the positions as being
                        defined with respect to a periodic grid, which will wrap them around
                        if they are outside the bounds of the original grid on which shears
                        and convergences were defined.  If not, then shear is set to zero
                        and magnification is set to 1 for positions outside the original
                        grid.  [default: False]

        Returns:
            shear and magnification as a tuple (g1,g2,mu).

        If the input ``pos`` is given a single position, the return values are the shear and
        magnification values at that position.
        If the input ``pos`` is given a list/array of positions, they are NumPy arrays.
        """
        if not hasattr(self, 'im_kappa'):
            raise GalSimError("PowerSpectrum.buildGrid must be called before getLensing")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = utilities._convertPositions(pos, units, 'getLensing')
        return self._getLensing(pos_x, pos_y, periodic)

    def _getLensing(self, pos_x, pos_y, periodic=False):
        """Equivalent to `getLensing`, but without some sanity checks and the positions must
        be given as ``pos_x``, ``pos_y`` in arcsec.

        Parameters:
            pos_x:      x position in arcsec (either a scalar or a numpy array)
            pos_y:      y position in arcsec (either a scalar or a numpy array)
            periodic:   Whether the interpolation should treat the positions as being defined
                        with respect to a periodic grid. [default: False]

        Returns:
            the reduced shear and magnification as a tuple (g1,g2,mu) (either scalars or
            numpy arrays)
        """
        g1_grid, g2_grid, mu_grid = theoryToObserved(
            self.im_g1.array, self.im_g2.array, self.im_kappa.array)

        lut_g1 = LookupTable2D(self.x_grid, self.y_grid, g1_grid,
                               edge_mode='wrap' if periodic else 'warn',
                               interpolant=self.interpolant)
        lut_g2 = LookupTable2D(self.x_grid, self.y_grid, g2_grid,
                               edge_mode='wrap' if periodic else 'warn',
                               interpolant=self.interpolant)
        lut_mu = LookupTable2D(self.x_grid, self.y_grid, mu_grid-1,
                               edge_mode='wrap' if periodic else 'warn',
                               interpolant=self.interpolant)

        return lut_g1(pos_x, pos_y), lut_g2(pos_x, pos_y), lut_mu(pos_x, pos_y)+1

class PowerSpectrumRealizer(object):
    """Class for generating realizations of power spectra with any area and pixel size.

    This class is not one that end-users should expect to interact with.  It is designed to quickly
    generate many realizations of the same shear power spectra on a square grid.  The initializer
    sets up the grids in k-space and computes the power on them.  It also computes spin weighting
    terms.  You can alter any of the setup properties later.  It currently only works for square
    grids (at least, much of the internals would be incorrect for non-square grids), so while it
    nominally contains arrays that could be allowed to be non-square, the constructor itself
    enforces squareness.

    Parameters:
        ngrid:          The size of the grid in one dimension.
        pixel_size:     The size of the pixel sides, in units consistent with the units expected
                        by the power spectrum functions.
        p_E:            Equivalent to ``e_power_function`` in the documentation for the
                        `PowerSpectrum` class.
        p_B:            Equivalent to ``b_power_function`` in the documentation for the
                        `PowerSpectrum` class.
    """
    def __init__(self, ngrid, pixel_size, p_E, p_B):
        # Set up the k grids in x and y, and the instance variables
        self.set_size(ngrid, pixel_size)
        self.set_power(p_E, p_B)

    def __repr__(self):
        return "galsim.lensing_ps.PowerSpectrumRealizer(ngrid=%r, pixel_size=%r, p_E=%r, p_B=%r)"%(
                self.nx, self.pixel_size, self.p_E, self.p_B)
    def __str__(self):
        return "galsim.lensing_ps.PowerSpectrumRealizer(ngrid=%r, pixel_size=%r, p_E=%s, p_B=%s)"%(
                self.nx, self.pixel_size, self.p_E, self.p_B)
    def __eq__(self, other): return self is other or repr(self) == repr(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(repr(self))

    def set_size(self, ngrid, pixel_size):
        self.nx = ngrid
        self.ny = ngrid
        self.pixel_size = float(pixel_size)

        # Setup some handy slices for indexing different parts of k space
        self.ikx = slice(0,self.nx//2+1)       # positive kx values, including 0, nx/2
        self.ikxp = slice(1,(self.nx+1)//2)    # limit to only values with a negative value
        self.ikxn = slice(-1,self.nx//2,-1)    # negative kx values

        # We always call this with nx=ny, so behavior with nx != ny is not tested.
        # However, we make a basic attempt to enable such behavior in the future if needed.
        self.iky = slice(0,self.ny//2+1)
        self.ikyp = slice(1,(self.ny+1)//2)
        self.ikyn = slice(-1,self.ny//2,-1)

        # Set up the scalar k grid. Generally, for a box size of L (in one dimension), the grid
        # spacing in k_x or k_y is Delta k=2pi/L
        self.kx, self.ky = utilities.kxky((self.ny,self.nx))
        self.kx /= self.pixel_size
        self.ky /= self.pixel_size

        # Compute the spin weightings
        self._generate_exp2ipsi()

    def set_power(self, p_E, p_B):
        self.p_E = p_E
        self.p_B = p_B
        if p_E is None:  self.amplitude_E = None
        else:            self.amplitude_E = np.sqrt(self._generate_power_array(p_E))/self.pixel_size
        if p_B is None:  self.amplitude_B = None
        else:            self.amplitude_B = np.sqrt(self._generate_power_array(p_B))/self.pixel_size

    def __call__(self, gd, variance=None):
        """Generate a realization of the current power spectrum.

        Parameters:
            gd:         A Gaussian deviate to use when generating the shear fields.
            variance:   Optionally renormalize the variance of the output shears to a
                        given value.  This is useful if you know the functional form of
                        the power spectrum you want, but not the normalization.  This lets
                        you set the normalization separately.  The resulting shears should
                        have var(g1) + var(g2) ~= variance.  If only ``e_power_function`` is
                        given, then this is also the variance of kappa.  Otherwise, the
                        variance of kappa may be smaller than the specified variance.
                        [default: None]

        @return a tuple of NumPy arrays (g1,g2,kappa) for the shear and convergence.
        """
        ISQRT2 = np.sqrt(1.0/2.0)

        if not isinstance(gd, GaussianDeviate):
            raise TypeError(
                "The gd provided to the PowerSpectrumRealizer is not a GaussianDeviate!")

        # Generate a random complex realization for the E-mode, if there is one
        if self.amplitude_E is not None:
            r1 = utilities.rand_arr(self.amplitude_E.shape, gd)
            r2 = utilities.rand_arr(self.amplitude_E.shape, gd)
            E_k = np.empty((self.ny,self.nx), dtype=complex)
            E_k[:,self.ikx] = self.amplitude_E * (r1 + 1j*r2) * ISQRT2
            # E_k corresponds to real kappa, so E_k[-k] = conj(E_k[k])
            self._make_hermitian(E_k)
        else: E_k = 0

        # Generate a random complex realization for the B-mode, if there is one
        if self.amplitude_B is not None:
            r1 = utilities.rand_arr(self.amplitude_B.shape, gd)
            r2 = utilities.rand_arr(self.amplitude_B.shape, gd)
            B_k = np.empty((self.ny,self.nx), dtype=complex)
            B_k[:,self.ikx] = self.amplitude_B * (r1 + 1j*r2) * ISQRT2
            # B_k corresponds to imag kappa, so B_k[-k] = -conj(B_k[k])
            # However, we later multiply this by i, so that means here B_k[-k] = conj(B_k[k])
            self._make_hermitian(B_k)
        else:
            B_k = 0

        # In terms of kappa, the E mode is the real kappa, and the B mode is imaginary kappa:
        # In fourier space, both E_k and B_k are complex, but the same E + i B relation holds.
        kappa_k = E_k + 1j * B_k

        # Renormalize the variance if desired
        if variance is not None:
            current_var = np.sum(np.abs(kappa_k)**2) / (self.nx * self.ny)
            factor = np.sqrt(variance / current_var)
            kappa_k *= factor
            E_k *= factor  # Need this for the k return value below.

        # Compute gamma_k as exp(2i psi) kappa_k
        # Equation 2.1.12 of Kaiser & Squires (1993, ApJ, 404, 441) is equivalent to:
        #   gamma_k = -self.exp2ipsi * kappa_k
        # But of course, they only considered real (E-mode) kappa.
        # However, this equation has a sign error.  There should not be a minus in front.
        # If you follow their subsequent deviation, you will see that they drop the minus sign
        # when they get to 2.1.15 (another - appears from the derivative).  2.1.15 is correct.
        # e.g. it correctly produces a positive point mass for tangential shear ~ 1/r^2.
        # So this implies that the minus sign in 2.1.12 should not be there.
        gamma_k = self.exp2ipsi * kappa_k

        # And go to real space to get the real-space shear and convergence fields.
        # Note the multiplication by N is needed because the np.fft.ifft2 implicitly includes a
        # 1/N^2, and for proper normalization we need a factor of 1/N.
        gamma = self.nx * np.fft.ifft2(gamma_k)
        # Make them contiguous, since we need to use them in an Image, which requires it.
        g1 = np.ascontiguousarray(np.real(gamma))
        g2 = np.ascontiguousarray(np.imag(gamma))

        # Could do the same thing with kappa..
        #kappa = self.nx * np.fft.ifft2(kappa_k)
        #k = np.ascontiguousarray(np.real(kappa))

        # But, since we don't care about imag(kappa), this is a bit faster:
        if np.all(E_k == 0):
            k = np.zeros((self.ny,self.nx))
        else:
            k = self.nx * np.fft.irfft2(E_k[:,self.ikx], s=(self.ny,self.nx))

        return g1, g2, k

    def _make_hermitian(self, P_k):
        # Make P_k[-k] = conj(P_k[k])
        # First update the kx=0 values to be consistent with this.
        P_k[self.ikyn,0] = np.conj(P_k[self.ikyp,0])
        P_k[0,0] = np.real(P_k[0,0])  # Not reall necessary, since P_k[0,0] = 0, but
                                      # I do it anyway for the sake of pedantry...
        # Then fill the kx<0 values appropriately
        P_k[self.ikyp,self.ikxn] = np.conj(P_k[self.ikyn,self.ikxp])
        P_k[self.ikyn,self.ikxn] = np.conj(P_k[self.ikyp,self.ikxp])
        P_k[0,self.ikxn] = np.conj(P_k[0,self.ikxp])
        # For even nx,ny, there are a few more changes needed.
        if self.ny % 2 == 0:
            # Note: this is a bit more complicated if you have to separately check whether
            # nx and/or ny are even.  I ignore this subtlety until we decide it is needed.
            P_k[self.ikyn,self.nx//2] = np.conj(P_k[self.ikyp,self.nx//2])
            P_k[self.ny//2,self.ikxn] = np.conj(P_k[self.ny//2,self.ikxp])
            P_k[self.ny//2,0] = np.real(P_k[self.ny//2,0])
            P_k[0,self.nx//2] = np.real(P_k[0,self.nx//2])
            P_k[self.ny//2,self.nx//2] = np.real(P_k[self.ny//2,self.nx//2])

    def _generate_power_array(self, power_function):
        from .table import LookupTable
        # Internal function to generate the result of a power function evaluated on a grid,
        # taking into account the symmetries.
        power_array = np.empty((self.ny, self.nx//2+1))

        # Set up the scalar |k| grid using just the positive kx,ky
        k = np.sqrt(self.kx[self.iky,self.ikx]**2 + self.ky[self.iky,self.ikx]**2)

        # Fudge the value at k=0, so we don't have to evaluate power there
        k[0,0] = k[1,0]
        P_k = np.empty_like(k)
        P_k[:,:] = power_function(k)

        # Now fix the k=0 value of power to zero
        P_k[0,0] = type(P_k[0,1])(0.)
        if np.any(P_k < 0):
            raise GalSimError("Negative power found for some values of k!")

        power_array[self.iky, self.ikx] = P_k
        power_array[self.ikyn, self.ikx] = P_k[self.ikyp, self.ikx]
        return power_array

    def _generate_exp2ipsi(self):
        # exp2ipsi = (kx + iky)^2 / |kx + iky|^2 is the phase of the k vector.
        kz = self.kx + self.ky*1j
        # exp(2i psi) = kz^2 / |kz|^2
        ksq = kz*np.conj(kz)
        # Need to adjust denominator for kz=0 to avoid division by 0.
        ksq[0,0] = 1.
        self.exp2ipsi = kz*kz/ksq
        # Note: this leaves exp2ipsi[0,0] = 0, but it turns out that's ok, since we only
        # ever multiply it by something that is 0 anyway (amplitude[0,0] = 0).

def kappaKaiserSquires(g1, g2):
    """Perform a Kaiser & Squires (1993) inversion to get a convergence map from gridded shears.

    This function takes gridded shears and constructs a convergence map from them.  While this is
    complicated in reality by the non-gridded galaxy positions, it is a straightforward
    implementation using Fourier transforms for the case of gridded galaxy positions.  Note that
    there are additional complications when dealing with real observational issues like shape noise
    that are not handled by this function, and likewise there are known edge effects.

    Note that, like any process that attempts to recover information from discretely sampled data,
    the ``kappa_E`` and ``kappa_B`` maps returned by this function are subject to aliasing.
    There will be distortions if there are non-zero frequency modes in the lensing field represented
    by ``g1`` and ``g2`` at more than half the frequency represented by the ``g1``, ``g2`` grid
    spacing.  To avoid this issue in practice you can smooth the input ``g1``, ``g2`` to effectively
    bandlimit them (the same smoothing kernel will be present in the output ``kappa_E``,
    ``kappa_B``).  If applying this function to shears drawn randomly according to some power
    spectrum, the power spectrum that is used should be modified to go to zero above the relevant
    maximum k value for the grid being used.

    Parameters:
        g1:     Square NumPy array containing the first component of shear.
        g2:     Square NumPy array containing the second component of shear.

    Returns:
        the tuple (kappa_E, kappa_B), as NumPy arrays.

    The returned kappa_E represents the convergence field underlying the input shears.
    The returned kappa_B is the convergence field generated were all shears rotated by 45 degrees
    prior to input.
    """
    # Checks on inputs
    if not (isinstance(g1, np.ndarray) and isinstance(g2, np.ndarray)):
        raise TypeError("Input g1 and g2 must be NumPy arrays.")
    if g1.shape != g2.shape:
        raise GalSimIncompatibleValuesError("Input g1 and g2 must be the same shape.", g1=g1, g2=g2)
    if g1.shape[0] != g1.shape[1]:
        raise GalSimNotImplementedError("Non-square input shear grids not supported.")

    # Then setup the kx, ky grids
    kx, ky = utilities.kxky(g1.shape)
    kz = kx + ky*1j

    # exp(2i psi) = kz^2 / |kz|^2
    ksq = kz*np.conj(kz)
    # Need to adjust denominator for kz=0 to avoid division by 0.
    ksq[0,0] = 1.
    exp2ipsi = kz*kz/ksq

    # Build complex g = g1 + i g2
    gz = g1 + g2*1j

    # Go to fourier space
    gz_k = np.fft.fft2(gz)

    # Equation 2.1.12 of Kaiser & Squires (1993) is equivalent to:
    #   kz_k = -np.conj(exp2ipsi)*gz_k
    # However, this equation has a sign error.  There should not be a minus in front.
    # If you follow their subsequent deviation, you will see that they drop the minus sign
    # when they get to 2.1.15 (another - appears from the derivative).  2.1.15 is correct.
    # e.g. it correctly produces a positive point mass for tangential shear ~ 1/r^2.
    # So this implies that the minus sign in 2.1.12 should not be there.
    kz_k = np.conj(exp2ipsi)*gz_k

    # Come back to real space
    kz = np.fft.ifft2(kz_k)

    # kz = kappa_E + i kappa_B
    kappaE = np.real(kz)
    kappaB = np.imag(kz)
    return kappaE, kappaB

class xip_integrand:
    """Utility class to assist in calculating the xi_+ shear correlation function from power
    spectra."""
    def __init__(self, pk, r):
        self.pk = pk
        self.r = r
    def __call__(self, k):
        from .bessel import j0
        return k * self.pk(k) * j0(self.r*k)

class xim_integrand:
    """Utility class to assist in calculating the xi_- shear correlation function from power
    spectra."""
    def __init__(self, pk, r):
        self.pk = pk
        self.r = r
    def __call__(self, k):
        from .bessel import jn
        return k * self.pk(k) * jn(4,self.r*k)
