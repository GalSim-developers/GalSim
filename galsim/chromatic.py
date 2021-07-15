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

from __future__ import division
import numpy as np

from .gsobject import GSObject
from .sed import SED
from .bandpass import Bandpass
from .position import Position, PositionD, _PositionD
from .utilities import lazy_property, doc_inherit
from .gsparams import GSParams
from .phase_psf import OpticalPSF
from .table import _LookupTable
from . import utilities
from . import integ
from .errors import GalSimError, GalSimRangeError, GalSimSEDError, GalSimValueError
from .errors import GalSimIncompatibleValuesError, GalSimNotImplementedError, galsim_warn

class ChromaticObject(object):
    """Base class for defining wavelength-dependent objects.

    This class primarily serves as the base class for chromatic subclasses.  See the docstrings for
    subclasses for more details.

    A ChromaticObject can be instantiated directly from an existing `GSObject`.  In this case, the
    newly created ChromaticObject will act in nearly the same way as the original `GSObject` works,
    except that it has access to the ChromaticObject transformation methods described below (e.g.,
    expand(), dilate(), shift(), withFlux(), ...)  These can all take functions as arguments to
    describe wavelength-dependent transformations.  E.g.,::

        >>> gsobj = galsim.Gaussian(fwhm=1)
        >>> chrom_obj = galsim.ChromaticObject(gsobj).dilate(lambda wave: (wave/500.)**(-0.2))

    In this and similar cases, the argument to the transformation method should be a python callable
    that accepts wavelength in nanometers and returns whatever type the transformation method
    normally accepts (so an int or float above).

    One caveat to creating a ChromaticObject directly from a `GSObject` like this is that even
    though the source `GSObject` instance has flux units in photons/s/cm^2, the newly formed
    ChromaticObject will be interpreted as dimensionless, i.e., it will have a dimensionless `SED`
    (and have its .dimensionless attribute set to True).  See below for more discussion about the
    dimensions of ChromaticObjects.

    Another way to instantiate a ChromaticObject from a `GSObject` is to multiply by an `SED`.
    This can be useful to consistently generate the same galaxy observed through different filters,
    or, with `ChromaticSum`, to construct multi-component galaxies, each component with a different
    `SED`.  For example, a bulge+disk galaxy could be constructed::

        >>> bulge_SED = user_function_to_get_bulge_spectrum()
        >>> disk_SED = user_function_to_get_disk_spectrum()
        >>> bulge_mono = galsim.DeVaucouleurs(half_light_radius=1.0)
        >>> disk_mono = galsim.Exponential(half_light_radius=2.0)
        >>> bulge = bulge_mono * bulge_SED
        >>> disk = disk_mono * disk_SED
        >>> gal = bulge + disk

    The `SED` instances above describe the flux density in photons/nm/cm^2/s of an object, possibly
    normalized with either the `SED.withFlux` or `SED.withMagnitude` methods (see their docstrings
    for details about these and other normalization options).  Note that for dimensional
    consistency, in this case, the ``flux`` attribute of the multiplied `GSObject` is interpreted
    as being dimensionless instead of in its normal units of [photons/s/cm^2].  The photons/s/cm^2
    units are (optionally) carried by the `SED` instead, or even left out entirely if the `SED` is
    dimensionless itself (see discussion on ChromaticObject dimensions below).  The `GSObject`
    ``flux`` attribute *does* still contribute to the ChromaticObject normalization, though.

    For example, the following are equivalent::

        >>> chrom_obj = (sed * 3.0) * gsobj
        >>> chrom_obj2 = sed * (gsobj * 3.0)

    Subclasses that instantiate a ChromaticObject directly, such as `ChromaticAtmosphere`, also
    exist.  Even in this case, however, the underlying implementation always eventually wraps one
    or more `GSObject` instances.

    **Dimensions**:

    ChromaticObjects can generally be sorted into two distinct types: those that represent galaxies
    or stars and have dimensions of [photons/wavelength-interval/area/time/solid-angle], and those
    that represent other types of wavelength dependence besides flux, like chromatic PSFs (these
    have dimensions of [1/solid-angle]).  The former category of ChromaticObjects will have their
    ``.spectral`` attribute set to True, while the latter category of ChromaticObjects will have
    their ``.dimensionless`` attribute set to True.  These two classes of ChromaticObjects have
    different restrictions associated with them.  For example, only spectral ChromaticObjects can
    be drawn using `drawImage`, only ChromaticObjects of the same type can be added together, and
    at most one spectral ChromaticObject can be part of a `ChromaticConvolution`.

    Multiplying a dimensionless ChromaticObject a spectral `SED` produces a spectral ChromaticObject
    (though note that the new object's `SED` may not be equal to the SED being multiplied by since
    the original ChromaticObject may not have had unit normalization.)

    **Methods**:

    `evaluateAtWavelength` returns the monochromatic surface brightness profile (as a `GSObject`)
    at a given wavelength (in nanometers).

    `interpolate` can be used for non-separable ChromaticObjects to expedite the image rendering
    process.  See the docstring of that method for more details and discussion of when this is a
    useful tool (and the interplay between interpolation, object transformations, and convolutions).

    Also, ChromaticObject has most of the same methods as `GSObject` with the following exceptions:

    The `GSObject` access methods (e.g. `GSObject.xValue`, `GSObject.maxk`, etc.) are not available.
    Instead, you would need to evaluate the profile at a particular wavelength and access what you
    want from that.

    The `withFlux`, `withFluxDensity`, and `withMagnitude` methods will return a new chromatic
    object with the appropriate spatially integrated flux, flux density, or magnitude.

    The `drawImage` method draws the object as observed through a particular bandpass, so the
    arguments are somewhat different.  See the docstring of `drawImage` for more details.
    """

    # ChromaticObjects should adhere to the following invariants:
    # - Objects should define the attributes/properties:
    #   * .SED, .separable, .wave_list, .interpolated, .deinterpolated, .spectral, .dimensionless
    # - obj.evaluateAtWavelength(lam).drawImage().array.sum() == obj.SED(lam)
    #   == obj.evaluateAtWavelength(lam).flux
    # - if obj.spectral:
    #       obj.SED.calculateFlux(bandpass) == obj.calculateFlux(bandpass)
    #       == obj.drawImage(bandpass).array.sum()
    # - .separable is a boolean indicating whether or not the profile can be factored into a
    #   spatial part and a spectral part.
    # - .wave_list is a numpy array indicating wavelengths of particular interest, for instance, the
    #   wavelengths at which the SED is explicitly defined via a LookupTable.  These are the
    #   wavelengths that will be used (in addition to those in bandpass.wave_list) when drawing an
    #   image of the chromatic profile.
    # - .interpolated is a boolean indicating whether any part of the object hierarchy includes an
    #   InterpolatedChromaticObject.
    # - .spectral indicates obj.SED.spectral
    # - .dimensionless indicates obj.SED.dimensionless

    def __init__(self, obj):
        self._obj = obj
        if isinstance(obj, GSObject):
            self.SED = SED(obj.flux, 'nm', '1')
        elif isinstance(obj, ChromaticObject):
            self.SED = obj.SED
        else:
            raise TypeError("Can only directly instantiate ChromaticObject with a GSObject "
                            "or ChromaticObject argument.")
        self.separable = obj.separable
        self.interpolated = obj.interpolated
        self.wave_list = obj.wave_list
        self.deinterpolated = obj.deinterpolated

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._obj.gsparams

    @property
    def redshift(self):
        """The redshift of the object.
        """
        return self.SED.redshift

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given gsparams

        Note: if this object wraps other objects (e.g. Convolution, Sum, Transformation, etc.)
        those component objects will also have their gsparams updated to the new value.
        """
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._obj = self._obj.withGSParams(gsparams, **kwargs)
        return ret

    @staticmethod
    def _get_multiplier(sed, bandpass, wave_list):
        """Cached integral of product of sed and bandpass."""
        wave_list = np.array(wave_list)
        if len(wave_list) > 0:
            bp = _LookupTable(wave_list, bandpass(wave_list), 'linear')
            multiplier = bp.integrate_product(sed)
        else:
            multiplier = integ.int1d(lambda w: sed(w) * bandpass(w),
                                     bandpass.blue_limit, bandpass.red_limit)
        return multiplier

    @staticmethod
    def resize_multiplier_cache(maxsize):
        """Resize the cache (default size=10) containing the integral over the product of an `SED`
        and a `Bandpass`, which is used by `ChromaticObject.drawImage`.

        Parameters:
            maxsize:    The new number of products to cache.
        """
        ChromaticObject._multiplier_cache.resize(maxsize)

    def _fiducial_profile(self, bandpass):
        """
        Return a fiducial achromatic profile of a chromatic object that can be used to estimate
        default output image characteristics, or in the case of separable profiles, can be scaled to
        give the monochromatic profile at any wavelength or the wavelength-integrated profile.
        """
        bpwave = bandpass.effective_wavelength
        bpwave, prof0 = self._approxWavelength(bpwave)
        if prof0.flux != 0:
            return bpwave, prof0

        candidate_waves = np.concatenate(
            [np.array([0.5 * (bandpass.blue_limit + bandpass.red_limit)]),
             bandpass.wave_list,
             self.wave_list])
        # Prioritize wavelengths near the bandpass effective wavelength.
        candidate_waves = candidate_waves[np.argsort(np.abs(candidate_waves - bpwave))]
        for w in candidate_waves:
            if bandpass.blue_limit <= w <= bandpass.red_limit:
                prof0 = self.evaluateAtWavelength(w)
                if prof0.flux != 0:
                    return w, prof0

        raise GalSimError("Could not locate fiducial wavelength where SED * Bandpass is nonzero.")

    def _approxWavelength(self, wave):
        # If a class doesn't have any more appropriate choice, just use evaluateAtWavelength
        # InterpolatedChromaticObject has a better choice when phot=True, so overrides this.
        return wave, self.evaluateAtWavelength(wave)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticObject) and
                 hasattr(other, '_obj') and  # not all ChromaticObjects have an _obj attribute.
                 self._obj == other._obj))

    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self): return hash(("galsim.ChromaticObject", self._obj))

    def __repr__(self):
        return 'galsim.ChromaticObject(%r)'%self._obj

    def __str__(self):
        return 'galsim.ChromaticObject(%s)'%self._obj

    def interpolate(self, waves, **kwargs):
        """Build interpolation images to (possibly) speed up subsequent `drawImage` calls.

        This method is used as a pre-processing step that can expedite image rendering using objects
        that have to be built up as sums of `GSObject` instances with different parameters at each
        wavelength, by interpolating between images at each wavelength instead of making a more
        costly instantiation of the relevant `GSObject` at each value of wavelength at which the
        bandpass is defined.

        This routine does a costly initialization process to build up a grid of `Image` instances to
        be used for the interpolation later on.  However, the object can get reused with different
        bandpasses, so there should not be any need to make many versions of this object, and there
        is a significant savings each time it is drawn into an image.

        As a general rule of thumb, chromatic objects that are separable do not benefit from this
        particular optimization, whereas those that involve making `GSObject` instances with
        wavelength-dependent keywords or transformations do benefit from it.

        Note that the interpolation scheme is simple linear interpolation in wavelength, and no
        extrapolation beyond the originally-provided range of wavelengths is permitted.  However,
        the overall flux at each wavelength will use the exact `SED` at that wavelength to give
        more accurate final flux values.  You can disable this feature by setting
        ``use_exact_SED = False``.

        The speedup involved in using interpolation depends in part on the bandpass used for
        rendering (since that determines how many full profile evaluations are involved in rendering
        the image).  However, for `ChromaticAtmosphere` with simple profiles like `Kolmogorov`, the
        speedup in some simple example cases is roughly a factor of three, whereas for more
        expensive to render profiles like the `ChromaticOpticalPSF`, the speedup is more typically a
        factor of 10-50.

        Achromatic transformations can be applied either before or after setting up interpolation,
        with the best option depending on the application.  For example, when rendering many times
        with the same achromatic transformation applied, it is typically advantageous to apply the
        transformation before setting up the interpolation.  But there is no value in this when
        applying different achromatic transformation to each object.  Chromatic transformations
        should be applied before setting up interpolation; attempts to render images of
        `ChromaticObject` instances with interpolation followed by a chromatic transformation will
        result in the interpolation being unset and the full calculation being done.

        Because of the clever way that the `ChromaticConvolution` routine works, convolutions of
        separable chromatic objects with non-separable ones that use interpolation will still
        benefit from these optimizations.  For example, a non-separable chromatic PSF that uses
        interpolation, when convolved with a sum of two separable galaxy components each with their
        own `SED`, will be able to take advantage of this optimization.  In contrast, when
        convolving two non-separable profiles that already have interpolation set up, there is no
        way to take advantage of that interpolation optimization, so it will be ignored and the
        full calculation will be done.  However, interpolation can be set up for the convolution of
        two non-separable profiles, after the convolution step.  This could be beneficial for
        example when convolving a chromatic optical PSF and chromatic atmosphere, before convolving
        with multiple galaxy profiles.

        For use cases requiring a high level of precision, we recommend a comparison between the
        interpolated and the more accurate calculation for at least one case, to ensure that the
        required precision has been reached.

        The input parameter ``waves`` determines the input grid on which images are precomputed.  It
        is difficult to give completely general guidance as to how many wavelengths to choose or how
        they should be spaced; some experimentation compared with the exact calculation is warranted
        for each particular application.  The best choice of settings might depend on how strongly
        the parameters of the object depend on wavelength.

        Parameters:
            waves:              The list, tuple, or NumPy array of wavelengths to be used when
                                building up the grid of images for interpolation.  The wavelengths
                                should be given in nanometers, and they should span the full range
                                of wavelengths covered by any bandpass to be used for drawing an
                                `Image` (i.e., this class will not extrapolate beyond the given
                                range of wavelengths).  They can be spaced any way the user likes,
                                not necessarily linearly, though interpolation will be linear in
                                wavelength between the specified wavelengths.
            oversample_fac:     Factor by which to oversample the stored profiles compared to the
                                default, which is to sample them at the Nyquist frequency for
                                whichever wavelength has the highest Nyquist frequency.
                                ``oversample_fac``>1 results in higher accuracy but costlier
                                pre-computations (more memory and time). [default: 1]
            use_exact_SED:      If true, then rescale the interpolated image for a given wavelength
                                by the ratio of the exact `SED` at that wavelength to the linearly
                                interpolated `SED` at that wavelength.  Thus, the flux of the
                                interpolated object should be correct, at the possible expense of
                                other features. [default: True]

        Returns:
            the version of the Chromatic object that uses interpolation
            (This will be an `InterpolatedChromaticObject` instance.)
        """
        return InterpolatedChromaticObject(self, waves, **kwargs)

    @property
    def spectral(self):
        """Boolean indicating if the `ChromaticObject` has units compatible with a spectral density.
        """
        return self.SED.spectral

    @property
    def dimensionless(self):
        """Boolean indicating if the `ChromaticObject` is dimensionless.
        """
        return self.SED.dimensionless

    @staticmethod
    def _get_integrator(integrator, wave_list):
        # Decide on integrator.  If the user passed one of the integrators from galsim.integ, that's
        # fine.  Otherwise we decide based on the adopted integration rule and the presence/absence
        # of `wave_list`.
        if isinstance(integrator, str):
            if integrator == 'quadratic':
                rule = integ.quadRule
            elif integrator == 'trapezoidal':
                rule = integ.trapzRule
            elif integrator == 'midpoint':
                rule = integ.midptRule
            else:
                raise GalSimValueError("Unrecognized integration rule", integrator,
                                       ('trapezoidal', 'midpoint', 'quadratic'))
            if len(wave_list) > 0:
                integrator = integ.SampleIntegrator(rule)
            else:
                integrator = integ.ContinuousIntegrator(rule)
        if not isinstance(integrator, integ.ImageIntegrator):
            raise TypeError("Invalid type passed in for integrator!")
        return integrator

    def drawImage(self, bandpass, image=None, integrator='quadratic', **kwargs):
        """Base implementation for drawing an image of a `ChromaticObject`.

        Some subclasses may choose to override this for specific efficiency gains.  For instance,
        most GalSim use cases will probably finish with a convolution, in which case
        `ChromaticConvolution.drawImage` will be used.

        The task of drawImage() in a chromatic context is to integrate a chromatic surface
        brightness profile multiplied by the throughput of ``bandpass``, over the wavelength
        interval indicated by ``bandpass``.

        Several integrators are available in galsim.integ to do this integration when using the
        first method (non-interpolated integration).  By default, `galsim.integ.SampleIntegrator`
        will be used if either ``bandpass.wave_list`` or ``self.wave_list`` have len() > 0.

        If lengths of both are zero, which may happen if both the bandpass throughput and the `SED`
        associated with ``self`` are analytic python functions for example, then
        `galsim.integ.ContinuousIntegrator` will be used instead.  This latter case by default will
        evaluate the integrand at 250 equally-spaced wavelengths between ``bandpass.blue_limit``
        and ``bandpass.red_limit``.

        By default, the above two integrators will use the ``rule`` `galsim.integ.quadRule`
        for integration.  The midpoint rule for integration can be specified instead by passing an
        integrator that has been initialized with the ``rule`` set to `galsim.integ.midptRule`.
        When creating a `ContinuousIntegrator`, the number of samples ``N`` is also an argument.
        For example::

            >>> integrator = galsim.integ.ContinuousIntegrator(rule=galsim.integ.midptRule, N=100)
            >>> image = chromatic_obj.drawImage(bandpass, integrator=integrator)

        Finally, this method uses a cache to avoid recomputing the integral over the product of
        the bandpass and object `SED` when possible (i.e., for separable profiles).  Because the
        cache size is finite, users may find that it is more efficient when drawing many images
        to group images using the same SEDs and bandpasses together in order to hit the cache more
        often.  The default cache size is 10, but may be resized using the
        `ChromaticObject.resize_multiplier_cache` method.

        Parameters:
            bandpass:       A `Bandpass` object representing the filter against which to
                            integrate.
            image:          Optionally, the Image to draw onto.  (See `GSObject.drawImage`
                            for details.)  [default: None]
            integrator:     When doing the exact evaluation of the profile, this argument should
                            be one of the image integrators from galsim.integ, or a string
                            'trapezoidal', 'midpoint', or 'quadratic', in which case the routine
                            will use a `SampleIntegrator` or `ContinuousIntegrator` depending on
                            whether or not the object has a ``wave_list``.  [default: 'quadratic',
                            which will try to select an appropriate integrator using the
                            quadratic integration rule automatically.]
            **kwargs:       For all other kwarg options, see `GSObject.drawImage`

        Returns:
            the drawn `Image`.
        """
        from .photon_array import WavelengthSampler

        # Store the last bandpass used and any extra kwargs.
        self._last_bp = bandpass
        if self.SED.dimensionless:
            raise GalSimSEDError("Can only draw ChromaticObjects with spectral SEDs.", self.SED)

        # setup output image using fiducial profile
        wave0, prof0 = self._fiducial_profile(bandpass)
        image = prof0.drawImage(image=image, setup_only=True, **kwargs)
        _remove_setup_kwargs(kwargs)

        # determine combined self.wave_list and bandpass.wave_list
        wave_list, _, _ = utilities.combine_wave_list(self, bandpass)

        # If there are photon ops, they'll probably need valid wavelengths, so add
        # WavelengthSampler as the first op in the list.
        if 'photon_ops' in kwargs:
            wave_sampler = WavelengthSampler(self.SED, bandpass)
            kwargs['photon_ops'] = [wave_sampler] + kwargs['photon_ops']

        if self.separable:
            multiplier = ChromaticObject._multiplier_cache(self.SED, bandpass, tuple(wave_list))
            prof0 *= multiplier/self.SED(wave0)
            image = prof0.drawImage(image=image, **kwargs)
            return image

        integrator = self._get_integrator(integrator, wave_list)

        # merge self.wave_list into bandpass.wave_list if using a sampling integrator
        if isinstance(integrator, integ.SampleIntegrator):
            if len(wave_list) < 2:
                raise GalSimIncompatibleValuesError(
                    "Cannot use SampleIntegrator when Bandpass and SED are both analytic.",
                    integrator=integrator, bandpass=bandpass, sed=self.SED)
            bandpass = Bandpass(_LookupTable(wave_list, bandpass(wave_list), 'linear'), 'nm')

        add_to_image = kwargs.pop('add_to_image', False)
        integral = integrator(self.evaluateAtWavelength, bandpass, image, kwargs)

        # For performance profiling, store the number of evaluations used for the last integration
        # performed.  Note that this might not be very useful for ChromaticSum instances, which are
        # drawn one profile at a time, and hence _last_n_eval will only represent the final
        # component drawn.
        self._last_n_eval = integrator.last_n_eval

        # Apply integral to the initial image appropriately.
        # Note: Don't do image = integral and return that for add_to_image==False.
        #       Remember that python doesn't actually do assignments, so this won't update the
        #       original image if the user provided one.  The following procedure does work.
        if not add_to_image:
            image.setZero()
        image += integral
        self._last_wcs = image.wcs
        return image

    def drawKImage(self, bandpass, image=None, integrator='quadratic', **kwargs):
        """Base implementation for drawing the Fourier transform of a `ChromaticObject`.

        The task of drawKImage() in a chromatic context is exactly analogous to the task of
        `drawImage` in a chromatic context: to integrate the ``sed * bandpass`` weighted Fourier
        profiles over wavelength.

        See `drawImage` for details on integration options.

        Parameters:
            bandpass:   A `Bandpass` object representing the filter against which to integrate.
            image:      If provided, this will be the complex `Image` onto which to draw the
                        k-space image.  If ``image`` is None, then an automatically-sized image
                        will be created.  If ``image`` is given, but its bounds are undefined,
                        then it will be resized appropriately based on the profile's size.
                        [default: None]
            integrator: When doing the exact evaluation of the profile, this argument should be
                        one of the image integrators from galsim.integ, or a string
                        'trapezoidal', 'midpoint', or 'quadratic', in which case the routine will
                        use a `SampleIntegrator` or `ContinuousIntegrator` depending on whether or
                        not the object has a ``wave_list``.  [default: 'quadratic', which will try
                        to select an appropriate integrator using the quadratic integration rule
                        automatically.]
            **kwargs:   For all other kwarg options, see `GSObject.drawKImage`.

        Returns:
            a complex `Image` instance (created if necessary)
        """
        if self.SED.dimensionless:
            raise GalSimSEDError("Can only drawK ChromaticObjects with spectral SEDs.", self.SED)

        # setup output image (semi-arbitrarily using the bandpass effective wavelength)
        prof0 = self.evaluateAtWavelength(bandpass.effective_wavelength)
        image = prof0.drawKImage(image=image, setup_only=True, **kwargs)
        _remove_setup_kwargs(kwargs)

        # determine combined self.wave_list and bandpass.wave_list
        wave_list, _ , _ = utilities.combine_wave_list(self, bandpass)

        if self.separable:
            multiplier = ChromaticObject._multiplier_cache(self.SED, bandpass, tuple(wave_list))
            prof0 *= multiplier/self.SED(bandpass.effective_wavelength)
            image = prof0.drawKImage(image=image, **kwargs)
            return image

        integrator = self._get_integrator(integrator, wave_list)

        # merge self.wave_list into bandpass.wave_list if using a sampling integrator
        if isinstance(integrator, integ.SampleIntegrator):
            bandpass = Bandpass(_LookupTable(wave_list, bandpass(wave_list), 'linear'), 'nm')

        add_to_image = kwargs.pop('add_to_image', False)
        image_int = integrator(self.evaluateAtWavelength, bandpass, image, kwargs, doK=True)

        # For performance profiling, store the number of evaluations used for the last integration
        # performed.  Note that this might not be very useful for ChromaticSum instances, which are
        # drawn one profile at a time, and hence _last_n_eval will only represent the final
        # component drawn.
        self._last_n_eval = integrator.last_n_eval

        # Apply integral to the initial image appropriately.
        # Note: Don't do image = integral and return that for add_to_image==False.
        #       Remember that python doesn't actually do assignments, so this won't update the
        #       original image if the user provided one.  The following procedure does work.
        if add_to_image:
            image += image_int
        else:
            image.copyFrom(image_int)
        return image

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        # Subclasses all override this.
        return self._obj.evaluateAtWavelength(wave)

    def _shoot(self, photons, rng):
        self._obj._shoot(photons, rng)

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the chromatic profile as a convolution to an existing photon array.

        This method allows instances of this class to duck type as a PhotonOp, so one can apply it
        in a photon_ops list.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use to effect the convolution.
                            [default: None]
        """
        from .photon_array import PhotonArray
        from .random import BaseDeviate
        if not photon_array.hasAllocatedWavelengths():
            raise GalSimError("Using ChromaticObject as a PhotonOp requires wavelengths be set")
        p1 = PhotonArray(len(photon_array))
        p1.wavelength = photon_array.wavelength
        obj = local_wcs.toImage(self) if local_wcs is not None else self
        rng = BaseDeviate(rng)
        obj._shoot(p1, rng)
        photon_array.convolve(p1, rng)

    # Make op* and op*= work to adjust the flux of the object
    def __mul__(self, flux_ratio):
        """Scale the flux of the object by the given flux ratio, which may be an `SED`, a float, or
        a univariate callable function (of wavelength in nanometers) that returns a float.

        The normalization of a `ChromaticObject` is tracked through its ``.SED`` attribute, which
        may have dimensions of either [photons/wavelength-interval/area/time/solid-angle] or
        [1/solid-angle].

        If ``flux_ratio`` is a spectral `SED` (i.e., ``flux_ratio.spectral==True``), then self.SED
        must be dimensionless for dimensional consistency.  The returned object will have a
        spectral SED attribute.  On the other hand, if ``flux_ratio`` is a dimensionless `SED`,
        float, or univariate callable function, then the returned object will have ``.spectral``
        and ``.dimensionless`` matching ``self.spectral`` and ``self.dimensionless``.

        Parameters:
            flux_ratio:     The factor by which to scale the normalization of the object.
                            ``flux_ratio`` may be a float, univariate callable function, in which
                            case the argument should be wavelength in nanometers and return value
                            the flux ratio for that wavelength, or an `SED`.

        Returns:
            a new object with scaled flux.
        """
        return self.withScaledFlux(flux_ratio)

    __rmul__ = __mul__

    # Likewise for op/ and op/=
    def __div__(self, other):
        return self.__mul__(1./other)

    __truediv__ = __div__

    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by ``flux_ratio``

        Parameters:
            flux_ratio:     The factor by which to scale the normalization of the object.
                            ``flux_ratio`` may be a float, univariate callable function, in which
                            case the argument should be wavelength in nanometers and return value
                            the flux ratio for that wavelength, or an `SED`.

        Returns:
            a new object with scaled flux.
        """
        from .transform import Transform
        return Transform(self, flux_ratio=flux_ratio)

    def withFlux(self, target_flux, bandpass):
        """Return a new `ChromaticObject` with flux through the `Bandpass` ``bandpass`` set to
        ``target_flux``.

        Parameters:
            target_flux:    The desired flux normalization of the `ChromaticObject`.
            bandpass:       A `Bandpass` object defining a filter bandpass.

        Returns:
            the new normalized `ChromaticObject`.
        """
        current_flux = self.calculateFlux(bandpass)
        norm = target_flux/current_flux
        return self * norm

    def withMagnitude(self, target_magnitude, bandpass):
        """Return a new `ChromaticObject` with magnitude through ``bandpass`` set to
        ``target_magnitude``.  Note that this requires ``bandpass`` to have been assigned a
        zeropoint using `Bandpass.withZeropoint`.

        Parameters:
            target_magnitude:   The desired magnitude of the `ChromaticObject`.
            bandpass:           A `Bandpass` object defining a filter bandpass.

        Returns:
            the new normalized `ChromaticObject`.
        """
        if bandpass.zeropoint is None:
            raise GalSimError("Cannot call ChromaticObject.withMagnitude on this bandpass, because"
                              " it does not have a zeropoint.  See `Bandpass.withZeropoint`")
        current_magnitude = self.calculateMagnitude(bandpass)
        norm = 10**(-0.4*(target_magnitude - current_magnitude))
        return self * norm

    def withFluxDensity(self, target_flux_density, wavelength):
        """Return a new `ChromaticObject` with flux density set to ``target_flux_density`` at
        wavelength ``wavelength``.

        Parameters:
            target_flux_density:    The target normalization in photons/nm/cm^2/s.
            wavelength:             The wavelength, in nm, at which the flux density will be set.

        Returns:
            the new normalized `SED`.
        """
        from astropy import units
        _photons = units.astrophys.photon/(units.s * units.cm**2 * units.nm)

        if self.dimensionless:
            raise GalSimSEDError("Cannot set flux density of dimensionless ChromaticObject.", self)
        if isinstance(wavelength, units.Quantity):
            wavelength_nm = wavelength.to(units.nm, units.spectral())
            current_flux_density = self.SED(wavelength_nm.value)
        else:
            wavelength_nm = wavelength * units.nm
            current_flux_density = self.SED(wavelength)
        if isinstance(target_flux_density, units.Quantity):
            target_flux_density = target_flux_density.to(
                    _photons, units.spectral_density(wavelength_nm)).value
        factor = target_flux_density / current_flux_density
        return self * factor

    def atRedshift(self, redshift):
        """Create a version of the current object with a different redshift.

        This will both adjust the SED to have the given redshift and set a ``redshift`` attribute
        with the given value.

        Returns:
            the object with the new redshift
        """
        return ChromaticTransformation(self, redshift=redshift)

    def calculateCentroid(self, bandpass):
        """Determine the centroid of the wavelength-integrated surface brightness profile.

        Parameters:
            bandpass:   The bandpass through which the observation is made.

        Returns:
            the centroid of the integrated surface brightness profile, as a PositionD.
        """
        # if either the Bandpass or self maintain a wave_list, evaluate integrand only at
        # those wavelengths.
        if len(bandpass.wave_list) > 0 or len(self.wave_list) > 0:
            w, _, _ = utilities.combine_wave_list(self, bandpass)
            objs = [self.evaluateAtWavelength(ww) for ww in w]
            fluxes = np.array([o.flux for o in objs])
            centroids = [o.centroid for o in objs]
            xcentroids = np.array([c.x for c in centroids])
            ycentroids = np.array([c.y for c in centroids])
            bp = _LookupTable(w, bandpass(w), 'linear')
            flux = bp.integrate_product(_LookupTable(w, fluxes, 'linear'))
            xcentroid = bp.integrate_product(_LookupTable(w, fluxes * xcentroids, 'linear')) / flux
            ycentroid = bp.integrate_product(_LookupTable(w, fluxes * ycentroids, 'linear')) / flux
            return _PositionD(xcentroid, ycentroid)
        else:
            flux_integrand = lambda w: self.evaluateAtWavelength(w).flux * bandpass(w)
            def xcentroid_integrand(w):
                mono = self.evaluateAtWavelength(w)
                return mono.centroid.x * mono.flux * bandpass(w)
            def ycentroid_integrand(w):
                mono = self.evaluateAtWavelength(w)
                return mono.centroid.y * mono.flux * bandpass(w)
            flux = integ.int1d(flux_integrand, bandpass.blue_limit, bandpass.red_limit)
            xcentroid = 1./flux * integ.int1d(xcentroid_integrand,
                                              bandpass.blue_limit,
                                              bandpass.red_limit)
            ycentroid = 1./flux * integ.int1d(ycentroid_integrand,
                                              bandpass.blue_limit,
                                              bandpass.red_limit)
            return _PositionD(xcentroid, ycentroid)

    def calculateFlux(self, bandpass):
        """Return the flux (photons/cm^2/s) of the `ChromaticObject` through a `Bandpass` bandpass.

        Parameters:
            bandpass:   A `Bandpass` object representing a filter, or None to compute the bolometric
                        flux.  For the bolometric flux the integration limits will be set to
                        (0, infinity) unless overridden by non-None `SED` attributes ``blue_limit``
                        or ``red_limit``.  Note that an `SED` defined using a `LookupTable`
                        automatically has ``blue_limit`` and ``red_limit`` set.

        Returns:
            the flux through the bandpass.
        """
        if self.SED.dimensionless:
            raise GalSimSEDError("Cannot calculate flux of dimensionless ChromaticObject.",
                                 self.SED)
        return self.SED.calculateFlux(bandpass)

    def calculateMagnitude(self, bandpass):
        """Return the `ChromaticObject` magnitude through a `Bandpass` ``bandpass``.

        Note that this requires ``bandpass`` to have been assigned a zeropoint using
        `Bandpass.withZeropoint`.

        Parameters:
            bandpass:   A `Bandpass` object representing a filter, or None to compute the
                        bolometric magnitude.  For the bolometric magnitude the integration
                        limits will be set to (0, infinity) unless overridden by non-None `SED`
                        attributes ``blue_limit`` or ``red_limit``.  Note that an `SED` defined
                        using a `LookupTable` automatically has ``blue_limit`` and ``red_limit``
                        set.

        Returns:
            the bandpass magnitude.
        """
        if self.SED.dimensionless:
            raise GalSimSEDError("Cannot calculate magnitude of dimensionless ChromaticObject.",
                                 self.SED)
        return self.SED.calculateMagnitude(bandpass)

    # Add together ChromaticObjects and/or GSObjects
    def __add__(self, other):
        return ChromaticSum(self, other)

    # Subtract ChromaticObjects and/or GSObjects
    def __sub__(self, other):
        return ChromaticSum(self, -other)

    def __neg__(self):
        return -1. * self

    # Following functions work to apply affine transformations to a ChromaticObject.
    #
    def expand(self, scale):
        """Expand the linear size of the profile by the given (possibly wavelength-dependent)
        scale factor ``scale``, while preserving surface brightness.

        This doesn't correspond to either of the normal operations one would typically want to
        do to a galaxy.  The functions dilate() and magnify() are the more typical usage.  But this
        function is conceptually simple.  It rescales the linear dimension of the profile, while
        preserving surface brightness.  As a result, the flux will necessarily change as well.

        See dilate() for a version that applies a linear scale factor while preserving flux.

        See magnify() for a version that applies a scale factor to the area while preserving surface
        brightness.

        Parameters:
            scale:      The factor by which to scale the linear dimension of the object.  In
                        addition, ``scale`` may be a callable function, in which case the argument
                        should be wavelength in nanometers and the return value the scale for that
                        wavelength.

        Returns:
            the expanded object
        """
        from .transform import Transform
        if hasattr(scale, '__call__'):
            def buildScaleJac(w):
                s = scale(w)
                return np.diag([s,s])
            jac = buildScaleJac
        else:
            jac = None if scale == 1 else np.diag([scale, scale])
        return Transform(self, jac=jac)

    def dilate(self, scale):
        """Dilate the linear size of the profile by the given (possibly wavelength-dependent)
        ``scale``, while preserving flux.

        e.g. ``half_light_radius`` <-- ``half_light_radius * scale``

        See expand() and magnify() for versions that preserve surface brightness, and thus
        change the flux.

        Parameters:
            scale:      The linear rescaling factor to apply.  In addition, ``scale`` may be a
                        callable function, in which case the argument should be wavelength in
                        nanometers and the return value the scale for that wavelength.

        Returns:
            the dilated object.
        """
        if hasattr(scale, '__call__'):
            return self.expand(scale).withScaledFlux(lambda w: 1./scale(w)**2)
        else:
            return self.expand(scale).withScaledFlux(1./scale**2)

    def magnify(self, mu):
        """Apply a lensing magnification, scaling the area and flux by ``mu`` at fixed surface
        brightness.

        This process applies a lensing magnification ``mu``, which scales the linear dimensions of the
        image by the factor sqrt(mu), i.e., ``half_light_radius`` <-- ``half_light_radius * sqrt(mu)``
        while increasing the flux by a factor of ``mu``.  Thus, magnify() preserves surface
        brightness.

        See dilate() for a version that applies a linear scale factor while preserving flux.

        Parameters:
            mu:         The lensing magnification to apply.  In addition, ``mu`` may be a callable
                        function, in which case the argument should be wavelength in nanometers
                        and the return value the magnification for that wavelength.

        Returns:
            the magnified object.
        """
        import math
        if hasattr(mu, '__call__'):
            return self.expand(lambda w: math.sqrt(mu(w)))
        else:
            return self.expand(math.sqrt(mu))

    def shear(self, *args, **kwargs):
        """Apply an area-preserving shear to this object, where arguments are either a `Shear`,
        or arguments that will be used to initialize one.

        For more details about the allowed keyword arguments, see the `Shear` docstring.

        The shear() method precisely preserves the area.  To include a lensing distortion with
        the appropriate change in area, either use shear() with magnify(), or use lens(), which
        combines both operations.

        Note that, while gravitational shear is monochromatic, the shear method may be used for
        many other use cases including some which may be wavelength-dependent, such as
        intrinsic galaxy shape, telescope dilation, atmospheric PSF shape, etc.  Thus, the
        shear argument is allowed to be a function of wavelength like other transformations.

        Parameters:
            shear:      The shear to be applied. Or, as described above, you may instead supply
                        parameters to construct a `Shear` directly.  eg. ``obj.shear(g1=g1,g2=g2)``.
                        In addition, the ``shear`` parameter may be a callable function, in which
                        case the argument should be wavelength in nanometers and the return value
                        the shear for that wavelength, returned as a `galsim.Shear` instance.

        Returns:
            the sheared object.
        """
        from .transform import Transform
        from .shear import Shear
        if len(args) == 1:
            if kwargs:
                raise TypeError("Gave both unnamed and named arguments!")
            if not hasattr(args[0], '__call__') and not isinstance(args[0], Shear):
                raise TypeError("Unnamed argument is not a Shear or function returning Shear!")
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Too many unnamed arguments!")
        elif 'shear' in kwargs:
            # Need to break this out specially in case it is a function of wavelength
            shear = kwargs.pop('shear')
            if kwargs:
                raise TypeError("Too many kwargs provided!")
        else:
            shear = Shear(**kwargs)
        if hasattr(shear, '__call__'):
            jac = lambda w: shear(w).getMatrix()
        else:
            jac = shear.getMatrix()
        return Transform(self, jac=jac)

    def _shear(self, shear):
        """Equivalent to `ChromaticObject.shear`, but only valid for a galsim.Shear object,
        not any of the possible wavelength-dependent options.

        Parameters:
            shear:      The `Shear` to be applied.

        Returns:
            the sheared object.
        """
        from .transform import Transform
        return Transform(self, shear.getMatrix())

    def lens(self, g1, g2, mu):
        """Apply a lensing shear and magnification to this object.

        This `ChromaticObject` method applies a lensing (reduced) shear and magnification.
        The shear must be specified using the g1, g2 definition of shear (see `Shear` for details).
        This is the same definition as the outputs of the `PowerSpectrum` and `NFWHalo` classes,
        which compute shears according to some lensing power spectrum or lensing by an NFW dark
        matter halo.  The magnification determines the rescaling factor for the object area and
        flux, preserving surface brightness.

        While gravitational lensing is achromatic, we do allow the parameters ``g1``, ``g2``, and
        ``mu`` to be callable functions to be parallel to all the other transformations of
        chromatic objects.  In this case, the functions should take the wavelength in nanometers as
        the argument, and the return values are the corresponding value at that wavelength.

        Parameters:
            g1:         First component of lensing (reduced) shear to apply to the object.
            g2:         Second component of lensing (reduced) shear to apply to the object.
            mu:         Lensing magnification to apply to the object.  This is the factor by which
                        the solid angle subtended by the object is magnified, preserving surface
                        brightness.

        Returns:
            the lensed object.
        """
        from .shear import Shear
        if any(hasattr(g, '__call__') for g in (g1,g2)):
            _g1 = g1
            _g2 = g2
            if not hasattr(g1, '__call__'): _g1 = lambda w: g1
            if not hasattr(g2, '__call__'): _g2 = lambda w: g2
            S = lambda w: Shear(g1=_g1(w), g2=_g2(w))
            sheared = self.shear(S)
        else:
            sheared = self.shear(g1=g1,g2=g2)
        return sheared.magnify(mu)

    def _lens(self, g1, g2, mu):
        """Equivalent to `ChromaticObject.lens`, but without the overhead of some of the sanity
        checks or any of the possible wavelength-dependent options.

        Parameters:
            g1:         First component of lensing (reduced) shear to apply to the object.
            g2:         Second component of lensing (reduced) shear to apply to the object.
            mu:         Lensing magnification to apply to the object.  This is the factor by which
                        the solid angle subtended by the object is magnified, preserving surface
                        brightness.

        Returns:
            the lensed object.
        """
        from .shear import _Shear
        from .transform import Transform
        import math
        shear = _Shear(g1 + 1j*g2)
        return Transform(self, shear.getMatrix() * math.sqrt(mu))

    def rotate(self, theta):
        """Rotate this object by an `Angle` ``theta``.

        Parameters:
            theta:      Rotation angle (`Angle` object, +ve anticlockwise). In addition, ``theta``
                        may be a callable function, in which case the argument should be wavelength
                        in nanometers and the return value the rotation angle for that wavelength,
                        returned as a `galsim.Angle` instance.

        Returns:
            the rotated object.
        """
        from .transform import Transform
        if hasattr(theta, '__call__'):
            def buildRMatrix(w):
                sth, cth = theta(w).sincos()
                R = np.array([[cth, -sth],
                              [sth,  cth]], dtype=float)
                return R
            jac = buildRMatrix
        else:
            sth, cth = theta.sincos()
            jac = np.array([[cth, -sth],
                            [sth,  cth]], dtype=float)
        return Transform(self, jac=jac)

    def transform(self, dudx, dudy, dvdx, dvdy):
        """Apply a transformation to this object defined by an arbitrary Jacobian matrix.

        This works the same as `GSObject.transform`, so see that method's docstring for more
        details.

        As with the other more specific chromatic trasnformations, dudx, dudy, dvdx, and dvdy
        may be callable functions, in which case the argument should be wavelength in nanometers
        and the return value the appropriate value for that wavelength.

        Parameters:
            dudx:       du/dx, where (x,y) are the current coords, and (u,v) are the new coords.
            dudy:       du/dy, where (x,y) are the current coords, and (u,v) are the new coords.
            dvdx:       dv/dx, where (x,y) are the current coords, and (u,v) are the new coords.
            dvdy:       dv/dy, where (x,y) are the current coords, and (u,v) are the new coords.

        Returns:
            the transformed object.
        """
        from .transform import Transform
        if any(hasattr(dd, '__call__') for dd in (dudx, dudy, dvdx, dvdy)):
            _dudx = dudx
            _dudy = dudy
            _dvdx = dvdx
            _dvdy = dvdy
            if not hasattr(dudx, '__call__'): _dudx = lambda w: dudx
            if not hasattr(dudy, '__call__'): _dudy = lambda w: dudy
            if not hasattr(dvdx, '__call__'): _dvdx = lambda w: dvdx
            if not hasattr(dvdy, '__call__'): _dvdy = lambda w: dvdy
            jac = lambda w: np.array([[_dudx(w), _dudy(w)],
                                      [_dvdx(w), _dvdy(w)]], dtype=float)
        else:
            jac = np.array([[dudx, dudy],
                            [dvdx, dvdy]], dtype=float)
        return Transform(self, jac=jac)

    def shift(self, *args, **kwargs):
        """Apply a (possibly wavelength-dependent) (dx, dy) shift to this chromatic object.

        For a wavelength-independent shift, you may supply ``dx,dy`` as either two arguments, as a
        tuple, or as a PositionD or PositionI object.

        For a wavelength-dependent shift, you may supply two functions of wavelength in nanometers
        which will be interpreted as ``dx(wave)`` and ``dy(wave)``, or a single function of
        wavelength in nanometers that returns either a 2-tuple, PositionD, or PositionI.

        Parameters:
            dx:     Horizontal shift to apply (float or function).
            dy:     Vertical shift to apply (float or function).

        Returns:
            the shifted object.
        """
        from .transform import Transform
        # This follows along the galsim.utilities.pos_args function, but has some
        # extra bits to account for the possibility of dx,dy being functions.
        # First unpack args/kwargs
        if len(args) == 0:
            # Then dx,dy need to be kwargs
            # If not, then python will raise an appropriate error.
            try:
                dx = kwargs.pop('dx')
                dy = kwargs.pop('dy')
            except KeyError:
                raise TypeError('shift() requires exactly 2 arguments (dx, dy)')
            offset = None
        elif len(args) == 1:
            if hasattr(args[0], '__call__'):
                try:
                    args[0](700.).x
                    # If the function returns a Position, recast it as a function returning
                    # a numpy array.
                    def offset_func(w):
                        d = args[0](w)
                        return np.asarray( (d.x, d.y) )
                    offset = offset_func
                except AttributeError:
                    # Then it's a function returning a tuple or list or array.
                    # Just make sure it is actually an array to make our life easier later.
                    offset = lambda w: np.asarray(args[0](w))
            elif isinstance(args[0], Position):
                offset = np.asarray( (args[0].x, args[0].y) )
            else:
                # Let python raise the appropriate exception if this isn't valid.
                dx, dy = args[0]
                offset = np.asarray( (dx, dy) )
        elif len(args) == 2:
            dx = args[0]
            dy = args[1]
            offset = None
        else:
            raise TypeError("Too many arguments supplied!")
        if kwargs:
            raise TypeError("Got unexpected keyword arguments: %s",kwargs.keys())

        if offset is None:
            offset = utilities.functionize(lambda x,y:(x,y))(dx, dy)

        return Transform(self, offset=offset)

    def _shift(self, dx, dy):
        """Equivalent to `ChromaticObject.shift`, but only valid for a scalar shift (dx, dy)
        not any of the possible wavelength-dependent options.

        Parameters:
            dx:         The x-component of the shift to apply
            dy:         The y-component of the shift to apply

        Returns:
            the shifted object.
        """
        from .transform import Transform
        return Transform(self, offset=_PositionD(dx,dy))

ChromaticObject._multiplier_cache = utilities.LRU_Cache(
    ChromaticObject._get_multiplier, maxsize=10)


class InterpolatedChromaticObject(ChromaticObject):
    """A `ChromaticObject` that uses interpolation of predrawn images to speed up subsequent
    rendering.

    This class wraps another `ChromaticObject`, which is stored in the attribute ``deinterpolated``.
    Any `ChromaticObject` can be used, although the interpolation procedure is most effective
    for non-separable objects, which can sometimes be very slow to render.

    Normally, you would not create an InterpolatedChromaticObject directly.  It is the
    return type from `ChromaticObject.interpolate`.  See the description of that function
    for more details.

    Parameters:
        original:       The `ChromaticObject` to be interpolated.
        waves:          The list, tuple, or NumPy array of wavelengths to be used when
                        building up the grid of images for interpolation.  The wavelengths
                        should be given in nanometers, and they should span the full range
                        of wavelengths covered by any bandpass to be used for drawing an `Image`
                        (i.e., this class will not extrapolate beyond the given range of
                        wavelengths).  They can be spaced any way the user likes, not
                        necessarily linearly, though interpolation will be linear in
                        wavelength between the specified wavelengths.
        oversample_fac: Factor by which to oversample the stored profiles compared to the
                        default, which is to sample them at the Nyquist frequency for
                        whichever wavelength has the highest Nyquist frequency.
                        ``oversample_fac``>1 results in higher accuracy but costlier
                        pre-computations (more memory and time). [default: 1]
        use_exact_SED:  If true, then rescale the interpolated image for a given wavelength by
                        the ratio of the exact `SED` at that wavelength to the linearly
                        interpolated `SED` at that wavelength.  Thus, the flux of the interpolated
                        object should be correct, at the possible expense of other features.
                        [default: True]
    """
    def __init__(self, original, waves, oversample_fac=1.0, use_exact_SED=True):

        self.waves = np.sort(np.array(waves))
        self.oversample = oversample_fac
        self.use_exact_SED = use_exact_SED

        self.separable = original.separable
        self.interpolated = True
        self.SED = original.SED
        self.wave_list = original.wave_list

        # Don't interpolate an interpolation.  Go back to the original.
        self.deinterpolated = original.deinterpolated
        self._build_objs()

    def _build_objs(self):
        # Make the objects between which we are going to interpolate.  Note that these do not have
        # to be saved for later, unlike the images.
        self._objs = [ self.deinterpolated.evaluateAtWavelength(wave) for wave in self.waves ]

        # Find the Nyquist scale for each, and to be safe, choose the minimum value to use for the
        # array of images that is being stored.
        nyquist_scale_vals = [ obj.nyquist_scale for obj in self._objs ]
        scale = np.min(nyquist_scale_vals) / self.oversample

        # Find the suggested image size for each object given the choice of scale, and use the
        # maximum just to be safe.
        possible_im_sizes = [ obj.getGoodImageSize(scale) for obj in self._objs ]
        im_size = np.max(possible_im_sizes)

        # Find the stepk and maxk values for each object.  These will be used later on, so that we
        # can force these values when instantiating InterpolatedImages before drawing.
        self.stepk_vals = np.array([ obj.stepk for obj in self._objs ])
        self.maxk_vals = np.array([ obj.maxk for obj in self._objs ])

        # Finally, now that we have an image scale and size, draw all the images.  Note that
        # `no_pixel` is used (we want the object on its own, without a pixel response).
        self.ims = [ obj.drawImage(scale=scale, nx=im_size, ny=im_size, method='no_pixel')
                     for obj in self._objs ]
        self.fluxes = [ obj.flux for obj in self._objs ]

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self.deinterpolated.gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret.deinterpolated = self.deinterpolated.withGSParams(gsparams, **kwargs)
        ret._build_objs()
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, InterpolatedChromaticObject) and
                 self.deinterpolated == other.deinterpolated and
                 np.array_equal(self.waves, other.waves) and
                 self.oversample == other.oversample and
                 self.use_exact_SED == other.use_exact_SED))

    def __hash__(self):
        return hash(("galsim.InterpolatedChromaticObject", self.deinterpolated, tuple(self.waves),
                     self.oversample, self.use_exact_SED))

    def __repr__(self):
        s = 'galsim.InterpolatedChromaticObject(%r,%r'%(self.deinterpolated, self.waves)
        if self.oversample != 1.0:
            s += ', oversample_fac=%r'%self.oversample
        if not self.use_exact_SED:
            s += ', use_exact_SED=False'
        s += ')'
        return s

    def __str__(self):
        return 'galsim.InterpolatedChromaticObject(%s,%s)'%(self.deinterpolated, self.waves)

    def _imageAtWavelength(self, wave):
        """
        Get an image of the object at a particular wavelength, using linear interpolation between
        the originally-stored images.  Also returns values for step_k and max_k, to be used to
        expedite the instantation of `InterpolatedImage`.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            an `Image` of the object at the given wavelength.
        """
        # First, some wavelength-related sanity checks.
        if wave < self.waves[0] or wave > self.waves[-1]:
            raise GalSimRangeError("Requested wavelength is outside the allowed range.",
                                   wave, self.waves[0], self.waves[-1])

        # Figure out where the supplied wavelength is compared to the list of wavelengths on which
        # images were originally tabulated.
        lower_idx, frac = _findWave(self.waves, wave)

        # Actually do the interpolation for the image, stepk, and maxk.
        im = _linearInterp(self.ims, frac, lower_idx)
        stepk = _linearInterp(self.stepk_vals, frac, lower_idx)
        maxk = _linearInterp(self.maxk_vals, frac, lower_idx)

        # Rescale to use the exact flux or normalization if requested.
        if self.use_exact_SED:
            interp_norm = _linearInterp(self.fluxes, frac, lower_idx)
            exact_norm = self.SED(wave)
            im *= exact_norm/interp_norm

        return im, stepk, maxk

    def _approxWavelength(self, wave):
        # More efficient to use one of the original objects, not a new InterpolatedImage.
        k = np.searchsorted(self.waves, wave)
        if k >= len(self.waves) or (k > 0 and wave-self.waves[k-1] < self.waves[k]-wave):
            k = k - 1
        return self.waves[k], self._objs[k]

    def evaluateAtWavelength(self, wave):
        """
        Evaluate this `ChromaticObject` at a particular wavelength using interpolation.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength, as a `GSObject`.
        """
        from .interpolatedimage import InterpolatedImage
        im, stepk, maxk = self._imageAtWavelength(wave)
        return InterpolatedImage(im, _force_stepk=stepk, _force_maxk=maxk)

    def _shoot(self, photons, rng):
        from .photon_array import PhotonArray
        from .random import UniformDeviate

        w = photons.wavelength
        if np.any((w < self.waves[0]) | (w > self.waves[-1])):
            bad_waves = [w for w in photons.wavelength if w < self.waves[0] or w > self.waves[-1]]
            raise GalSimRangeError("Shooting photons outside the interpolated wave_list",
                                   bad_waves, self.waves[0], self.waves[-1])

        k = np.searchsorted(self.waves, w)
        k[k==0] = 1  # if k == 0, then w == min(waves). Using k=1 instead is fine for this.
        #assert np.all(k > 0)
        #assert np.all(k < len(self.waves))

        # For each w, these are the wavelengthat that bracket w:
        w0 = self.waves[k-1]
        w1 = self.waves[k]
        #assert np.all(w0 <= w)
        #assert np.all(w <= w1)

        # If we could get away with averaging photons shot at each wavelength,
        # these would be relative fractions.  So e.g. x = x0 f0 + x1 f1 would be the
        # right weighted average to use.
        f0 = (w1-w) / (w1-w0)
        #f1 = (w-w0) / (w1-w0)  (We don't need this quantity below.)

        # Instead of averaging these, we can do the averaging probabilistically by selecting
        # each photon with a probability equal to the relative weight we would have used
        # in the average.
        u = np.empty(len(photons))
        UniformDeviate(rng).generate(u)
        use_k = k - (u<f0).astype(int)  # The second term is either 0 or 1.

        # Draw photons from the saved profiles according to when we have selected to use each one.
        for kk, obj in enumerate(self._objs):
            use = (use_k == kk)  # True for each photon where this is the object to shoot from
            temp = PhotonArray(np.sum(use))
            obj._shoot(temp, rng)
            photons.x[use] = temp.x
            photons.y[use] = temp.y
            # It will have tried to shoot the right total flux.  But that's not correct.
            # Rescale it down by the fraction of the total flux we actually want in this set.
            photons.flux[use] = temp.flux * (len(temp)/len(photons))

    def _get_interp_image(self, bandpass, image=None, integrator='quadratic',
                          _flux_ratio=None, **kwargs):
        from .interpolatedimage import InterpolatedImage
        if integrator == 'quadratic':
            rule = integ.quadRule
        elif integrator == 'trapezoidal':
            rule = integ.trapzRule
        elif integrator == 'midpoint':
            rule = integ.midptRule
        else:
            raise GalSimValueError("Unrecognized integration rule", integrator,
                                    ('trapezoidal', 'midpoint', 'quadratic'))

        if _flux_ratio is None:
            _flux_ratio = lambda w: np.ones_like(w)
        # Constant flux_ratio is already an SED at this point, so can treat as function.
        #assert hasattr(_flux_ratio, '__call__')

        # setup output image (semi-arbitrarily using the bandpass effective wavelength).
        # Note: we cannot just use self._imageAtWavelength, because that routine returns an image
        # with whatever pixel scale was required to sample all the images properly.  We want to set
        # up an output image that has the requested pixel scale, which might change the image size
        # and so on.
        _, prof0 = self._fiducial_profile(bandpass)
        image = prof0.drawImage(image=image, setup_only=True, **kwargs)
        _remove_setup_kwargs(kwargs)

        # determine combination of self.wave_list and bandpass.wave_list
        wave_objs = [self, bandpass]
        if isinstance(_flux_ratio, SED):
            wave_objs += [_flux_ratio]
        wave_list, _, _ = utilities.combine_wave_list(wave_objs)

        if np.any((wave_list < self.waves[0]) | (wave_list > self.waves[-1])):  # pragma: no cover
            # MJ: I'm pretty sure it's impossible to hit this.
            #     But just in case I'm wrong, I'm leaving it here but with pragma: no cover.
            bad_waves = [w for w in wave_list if w < self.waves[0] or w > self.waves[-1]]
            raise GalSimRangeError("Requested wavelength is outside the allowed range.",
                                   bad_waves, self.waves[0], self.waves[-1])

        # weights are the weights to use at each of the given wavelengths for the integration.
        weights = rule.calculateWeights(wave_list, bandpass)
        # im_weights are the weights for the stored images.
        im_weights = np.zeros(len(self.waves))
        for w, wt in zip(wave_list, weights):
            # Find where this is with respect to the wavelengths on which images are stored.
            lower_idx, frac = _findWave(self.waves, w)
            assert 0 <= lower_idx < len(self.waves)-1

            # Rescale to use the exact flux or normalization if requested.
            if self.use_exact_SED:
                interp_norm = _linearInterp(self.fluxes, frac, lower_idx)
                exact_norm = self.SED(w)
                wt *= exact_norm/interp_norm

            im_weights[lower_idx] += (1.-frac) * wt * _flux_ratio(w)
            im_weights[lower_idx+1] += frac * wt * _flux_ratio(w)

        # Do the integral as a weighted sum.
        integral = sum(wt*im for wt,im in zip(im_weights, self.ims) if wt!=0)

        # Get the stepk, maxk using the same weights
        stepk = np.average(self.stepk_vals, weights=im_weights)
        maxk = np.average(self.maxk_vals, weights=im_weights)

        # Instantiate the InterpolatedImage, using these conservative stepk and maxk choices.
        return InterpolatedImage(integral, _force_stepk=stepk, _force_maxk=maxk)

    def drawImage(self, bandpass, image=None, integrator='quadratic', **kwargs):
        """Draw an image as seen through a particular bandpass using the stored interpolated
        images at the specified wavelengths.

        This integration will take place using interpolation between stored images that were
        setup when the object was constructed.  (See interpolate() for more details.)

        Parameters:
            bandpass:       A `Bandpass` object representing the filter against which to
                            integrate.
            image:          Optionally, the `Image` to draw onto.  (See `GSObject.drawImage`
                            for details.)  [default: None]
            integrator:     The integration algorithm to use, given as a string.  Either
                            'midpoint', 'trapezoidal', or 'quadratic' is allowed.
                            [default: 'quadratic']
            **kwargs:       For all other kwarg options, see `GSObject.drawImage`.

        Returns:
            the drawn `Image`.
        """
        # Store the last bandpass used.
        self._last_bp = bandpass
        if self.SED.dimensionless:
            raise GalSimSEDError("Can only draw ChromaticObjects with spectral SEDs.", self.SED)

        int_im = self._get_interp_image(bandpass, image=image, integrator=integrator, **kwargs)
        image = int_im.drawImage(image=image, **kwargs)
        self._last_wcs = image.wcs
        return image


class ChromaticAtmosphere(ChromaticObject):
    """A `ChromaticObject` implementing two atmospheric chromatic effects: differential
    chromatic refraction (DCR) and wavelength-dependent seeing.

    Due to DCR, blue photons land closer to the zenith than red photons.  Kolmogorov turbulence
    also predicts that blue photons get spread out more by the atmosphere than red photons,
    specifically FWHM is proportional to wavelength^(-0.2).  Both of these effects can be
    implemented by wavelength-dependent shifts and dilations.

    Since DCR depends on the zenith angle and the parallactic angle (which is the position angle of
    the zenith measured from North through East) of the object being drawn, these must be specified
    via keywords.  There are four ways to specify these values:

      1) explicitly provide ``zenith_angle`` as a keyword of type `Angle`, and
         ``parallactic_angle`` will be assumed to be 0 by default.
      2) explicitly provide both ``zenith_angle`` and ``parallactic_angle`` as keywords of type
         `Angle`.
      3) provide the coordinates of the object ``obj_coord`` and the coordinates of the zenith
         ``zenith_coord`` as keywords of type `CelestialCoord`.
      4) provide the coordinates of the object ``obj_coord`` as a `CelestialCoord`, the hour angle
         of the object ``HA`` as an `Angle`, and the latitude of the observer ``latitude`` as an
         `Angle`.

    DCR also depends on temperature, pressure and water vapor pressure of the atmosphere.  The
    default values for these are expected to be appropriate for LSST at Cerro Pachon, Chile, but
    they are broadly reasonable for most observatories.

    Note that a ChromaticAtmosphere by itself is NOT the correct thing to use to draw an image of a
    star. Stars (and galaxies too, of course) have an `SED` that is not flat. To draw a real star,
    you should either multiply the ChromaticAtmosphere object by an `SED`, or convolve it with a
    point source multiplied by an `SED`::

        >>> psf = galsim.ChromaticAtmosphere(...)
        >>> star = galsim.DeltaFunction() * psf_sed
        >>> final_star = galsim.Convolve( [psf, star] )
        >>> final_star.drawImage(bandpass = bp, ...)

    Parameters:
        base_obj:           Fiducial PSF, equal to the monochromatic PSF at ``base_wavelength``
        base_wavelength:    Wavelength represented by the fiducial PSF, in nanometers.
        scale_unit:         Units used by base_obj for its linear dimensions.
                            [default: galsim.arcsec]
        alpha:              Power law index for wavelength-dependent seeing.  [default: -0.2,
                            the prediction for Kolmogorov turbulence]
        zenith_angle:       `Angle` from object to zenith [default: 0]
        parallactic_angle:  Parallactic angle, i.e. the position angle of the zenith, measured
                            from North through East.  [default: 0]
        obj_coord:          Celestial coordinates of the object being drawn as a
                            `CelestialCoord`. [default: None]
        zenith_coord:       Celestial coordinates of the zenith as a `CelestialCoord`.
                            [default: None]
        HA:                 Hour angle of the object as an `Angle`. [default: None]
        latitude:           Latitude of the observer as an `Angle`. [default: None]
        pressure:           Air pressure in kiloPascals.  [default: 69.328 kPa]
        temperature:        Temperature in Kelvins.  [default: 293.15 K]
        H2O_pressure:       Water vapor pressure in kiloPascals.  [default: 1.067 kPa]
    """
    def __init__(self, base_obj, base_wavelength, scale_unit=None, **kwargs):
        from .angle import Angle, _Angle, AngleUnit, arcsec
        from . import dcr

        self.separable = False
        self.interpolated = False
        self.deinterpolated = self
        self.SED = SED(base_obj.flux, 'nm', '1')
        self.wave_list = np.array([], dtype=float)

        self.base_obj = base_obj
        self.base_wavelength = base_wavelength
        self._gsparams = base_obj.gsparams

        if scale_unit is None:
            scale_unit = arcsec
        elif isinstance(scale_unit, str):
            scale_unit = AngleUnit.from_name(scale_unit)
        self.scale_unit = scale_unit
        self.alpha = kwargs.pop('alpha', -0.2)
        self.zenith_angle, self.parallactic_angle, self.kw = dcr.parse_dcr_angles(**kwargs)

        # Any remaining kwargs will get forwarded to galsim.dcr.get_refraction
        # Check that they're valid
        for kw in self.kw:
            if kw not in ('temperature', 'pressure', 'H2O_pressure'):
                raise TypeError("Got unexpected keyword: {0}".format(kw))

        self.base_refraction = dcr.get_refraction(self.base_wavelength, self.zenith_angle,
                                                  **self.kw)

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self.base_obj.gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret.base_obj = self.base_obj.withGSParams(gsparams, **kwargs)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticAtmosphere) and
                 self.base_obj == other.base_obj and
                 self.base_wavelength == other.base_wavelength and
                 self.alpha == other.alpha and
                 self.zenith_angle == other.zenith_angle and
                 self.parallactic_angle == other.parallactic_angle and
                 self.scale_unit == other.scale_unit and
                 self.kw == other.kw))

    def __hash__(self):
        return hash(("galsim.ChromaticAtmosphere", self.base_obj, self.base_wavelength,
                     self.alpha, self.zenith_angle, self.parallactic_angle, self.scale_unit,
                     frozenset(self.kw.items())))

    def __repr__(self):
        s = 'galsim.ChromaticAtmosphere(%r, base_wavelength=%r, alpha=%r'%(
                self.base_obj, self.base_wavelength, self.alpha)
        s += ', zenith_angle=%r, parallactic_angle=%r'%(self.zenith_angle, self.parallactic_angle)
        s += ', scale_unit=%r'%(self.scale_unit)
        for k,v in self.kw.items():
            s += ', %s=%r'%(k,v)
        s += ')'
        return s

    def __str__(self):
        return 'galsim.ChromaticAtmosphere(%s, base_wavelength=%s, alpha=%s)'%(
                self.base_obj, self.base_wavelength, self.alpha)

    def build_obj(self):
        """Build a `ChromaticTransformation` object for this `ChromaticAtmosphere`.

        We don't do this right away to help make `ChromaticAtmosphere` objects be picklable.
        Building this is quite fast, so we do it on the fly in `evaluateAtWavelength` and
        `ChromaticObject.drawImage`.
        """
        from . import dcr
        from .angle import radians
        def shift_fn(w):
            shift_magnitude = dcr.get_refraction(w, self.zenith_angle, **self.kw)
            shift_magnitude -= self.base_refraction
            shift_magnitude = shift_magnitude * radians / self.scale_unit
            sinp, cosp = self.parallactic_angle.sincos()
            shift = (-shift_magnitude * sinp, shift_magnitude * cosp)
            return shift

        def jac_fn(w):
            scale = (w/self.base_wavelength)**self.alpha
            return np.diag([scale, scale])

        flux_ratio = lambda w: (w/self.base_wavelength)**(-2.*self.alpha)

        return ChromaticTransformation(self.base_obj, jac=jac_fn, offset=shift_fn,
                                       flux_ratio=flux_ratio)

    def _shoot(self, photons, rng):
        from . import dcr
        from .angle import radians

        # Start with the base PSF
        self.base_obj._shoot(photons, rng)

        w = photons.wavelength

        # Apply the wavelength-dependent scaling
        if self.alpha != 0.:
            scale = (w/self.base_wavelength)**self.alpha
            photons.x *= scale
            photons.y *= scale

        # Apply DCR
        shift_magnitude = dcr.get_refraction(w, self.zenith_angle, **self.kw)
        shift_magnitude -= self.base_refraction
        shift_magnitude *= radians / self.scale_unit
        sinp, cosp = self.parallactic_angle.sincos()
        photons.x += -shift_magnitude * sinp
        photons.y += shift_magnitude * cosp

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        return self.build_obj().evaluateAtWavelength(wave)


class ChromaticTransformation(ChromaticObject):
    """A class for modeling a wavelength-dependent affine transformation of a `ChromaticObject`
    instance.

    Typically, you do not need to construct a ChromaticTransformation object explicitly.
    This is the type returned by the various transformation methods of `ChromaticObject` such as
    `ChromaticObject.shear`, `ChromaticObject.rotate`, `ChromaticObject.shift`, etc.

    All the various transformations can be described as a combination of a jacobian matrix
    (i.e. `ChromaticObject.transform`) and a translation (`ChromaticObject.shift`), which are
    described by (dudx,dudy,dvdx,dvdy) and (dx,dy) respectively.

    Parameters:
        obj:                The object to be transformed.
        jac:                A list or tuple (dudx, dudy, dvdx, dvdy), or a numpy.array object
                            [[dudx, dudy], [dvdx, dvdy]] describing the Jacobian to apply.  May
                            also be a function of wavelength returning a numpy array.
                            Use None to indicate that the Jacobian is the 2x2 unit matrix.
                            [default: None]
        offset:             A galsim.PositionD or list or tuple or numpy array giving the offset
                            (dx,dy) by which to shift the profile.  May also be a function of
                            wavelength returning a numpy array.  [default: None]
        flux_ratio:         A factor by which to multiply the flux of the object. [default: 1]
        redshift:           A redshift to apply to the wavelength when evaluating. [default: None]
        gsparams:           An optional `GSParams` argument.  See the docstring for `GSParams` for
                            details. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]
    """
    def __init__(self, obj, jac=None, offset=(0.,0.), flux_ratio=1., redshift=None,
                 gsparams=None, propagate_gsparams=True):
        if isinstance(offset, Position):
            offset = (offset.x, offset.y)
        if not hasattr(jac,'__call__') and jac is not None:
            jac = np.asarray(jac).reshape(2,2)
        if not hasattr(offset,'__call__'):
            offset = np.asarray(offset)

        self.chromatic = hasattr(jac,'__call__') or hasattr(offset,'__call__')
        # Technically, if the only chromatic transformation is a flux_ratio, and the original object
        # is separable, then the transformation is still separable (for instance, galsim.Chromatic),
        # but we'll ignore that here.
        self.separable = obj.separable and not self.chromatic

        self.SED = obj.SED * flux_ratio
        self.wave_list, _, _ = utilities.combine_wave_list(obj, self.SED)

        self._redshift = redshift
        if redshift is not None:
            self.SED = self.SED.atRedshift(redshift)
            self.wave_list *= (1.+redshift)

        # Need to account for non-unit determinant jacobian in normalization.
        if hasattr(jac, '__call__'):
            @np.vectorize
            def detjac(w):
                return np.linalg.det(np.asarray(jac(w)).reshape(2,2))
            self.SED *= detjac
        elif jac is not None:
            self.SED *= np.linalg.det(np.asarray(jac).reshape(2,2))

        if obj.interpolated and self.chromatic:
            galsim_warn("Cannot render image with chromatic transformation applied to it "
                        "using interpolation between stored images.  Reverting to "
                        "non-interpolated version.")
            obj = obj.deinterpolated
        self.interpolated = obj.interpolated

        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams

        if (isinstance(obj, ChromaticTransformation) and not self.chromatic
                and not obj.chromatic and self._redshift is None and obj._redshift is None):
            # If both transformations are not chromatic, then it is useful to combine them.
            # Especially if the original object is interpolated, since we have special handling
            # for that case in drawImage.
            # However, if either one is chromatic, then it's hard to combine them in a way that
            # preserves the ability to call the functions on either numpy arrays or scalars,
            # so just leave this as two consecutive transformations in that case.
            # Similarly, if there are redshifts involved, then the flux_ratio function needs to
            # be called on two different wavelengths, so again just leave it separate.
            # (I think this last case could be remedied, so if there is a use case where it
            # is important, we could try to implement it.)
            self._original = obj.original
            if jac is None:
                self._jac = obj._jac
                self._offset = obj._offset + offset
            else:
                self._jac = jac if obj._jac is None else jac.dot(obj._jac)
                self._offset = jac.dot(obj._offset) + offset
            if hasattr(flux_ratio, '__call__') or hasattr(obj._flux_ratio, '__call__'):
                self._flux_ratio = SED(flux_ratio, 'nm', '1') * obj._flux_ratio
            else:
                self._flux_ratio = obj._flux_ratio * flux_ratio
        else:
            self._original = obj
            self._jac = jac
            self._offset = offset
            self._flux_ratio = flux_ratio

        if self._propagate_gsparams:
            self._original = self._original.withGSParams(self._gsparams)

        if self.interpolated:
            self.deinterpolated = ChromaticTransformation(
                    self._original.deinterpolated,
                    jac = self._jac,
                    offset = self._offset,
                    flux_ratio = self._flux_ratio,
                    redshift = self._redshift,
                    gsparams = self._gsparams,
                    propagate_gsparams = self._propagate_gsparams)
        else:
            self.deinterpolated = self

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._original = self._original.withGSParams(ret._gsparams)
        if self.interpolated:
            ret.deinterpolated = self.deinterpolated.withGSParams(ret._gsparams)
        else:
            ret.deinterpolated = ret
        return ret

    @property
    def original(self):
        """The original object that was transformed.
        """
        return self._original

    def __eq__(self, other):
        if self is other: return True
        if not (isinstance(other, ChromaticTransformation) and
                self.original == other.original and
                self._gsparams == other._gsparams and
                self._propagate_gsparams == other._propagate_gsparams):
            return False
        # There's really no good way to check that two callables are equal, except if they literally
        # point to the same object.  So we'll just check for that for _jac, _offset, _flux_ratio.
        for attr in ('_jac', '_offset', '_flux_ratio'):
            selfattr = getattr(self, attr)
            otherattr = getattr(other, attr)
            # For this attr, either both need to be chromatic or neither.
            if ((hasattr(selfattr, '__call__') and not hasattr(otherattr, '__call__')) or
                (hasattr(otherattr, '__call__') and not hasattr(selfattr, '__call__'))):
                return False
            # If chromatic, then check that attrs compare equal
            if hasattr(selfattr, '__call__'):
                if selfattr != otherattr:
                    return False
            else: # Otherwise, check that attr arrays are equal.
                if not np.array_equal(selfattr, otherattr):
                    return False
        return True

    def __hash__(self):
        # This one's a bit complicated, so we'll go ahead and cache the hash.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.ChromaticTransformation", self.original, self._gsparams,
                               self._propagate_gsparams))
            # achromatic _jac and _offset are ndarrays, so need to be handled separately.
            for attr in ('_jac', '_offset'):
                selfattr = getattr(self, attr)
                if hasattr(selfattr, '__call__'):
                    self._hash ^= hash(selfattr)
                elif selfattr is not None:
                    self._hash ^= hash(tuple(selfattr.ravel().tolist()))
            self._hash ^= hash(self._flux_ratio)
        return self._hash

    def __repr__(self):
        if hasattr(self._jac, '__call__'):
            jac = self._jac
        elif self._jac is None:
            jac = None
        else:
            jac = self._jac.ravel().tolist()
        if hasattr(self._offset, '__call__'):
            offset = self._offset
        else:
            offset = _PositionD(*(self._offset.tolist()))
        return ('galsim.ChromaticTransformation(%r, jac=%r, offset=%r, flux_ratio=%r, '
                'redshift=%r, gsparams=%r, propagate_gsparams=%r)')%(
            self.original, jac, offset, self._flux_ratio, self._redshift,
            self._gsparams, self._propagate_gsparams)

    def __str__(self):
        from .transform import Transformation
        s = str(self.original)
        if hasattr(self._jac, '__call__'):
            s += '.transform(%s)'%self._jac
        elif self._jac is not None:
            s += Transformation._str_from_jac(self._jac)
        if hasattr(self._offset, '__call__'):
            s += '.shift(%s)'%self._offset
        elif not np.array_equal(self._offset,(0,0)):
            s += '.shift(%s,%s)'%(self._offset[0],self._offset[1])
        if self._flux_ratio != 1.:
            s += '.withScaledFlux(%s)'%self._flux_ratio
        if self._redshift is not None:
            s += '.atRedshift(%s)'%(self._redshift)
        return s

    def _getTransformations(self, wave):
        if hasattr(self._jac, '__call__'):
            jac = self._jac(wave)
        else:
            jac = self._jac
        if hasattr(self._offset, '__call__'):
            offset = self._offset(wave)
        else:
            offset = self._offset
        if hasattr(self._flux_ratio, '__call__'):
            flux_ratio = self._flux_ratio(wave)
        else:
            flux_ratio = self._flux_ratio
        return jac, offset, flux_ratio

    def _approxWavelength(self, wave):
        # Same as evaluateAtWavelength, except the starting point is also _approxWavelength
        from .transform import Transformation
        wave1 = wave / (1.+self._redshift) if self._redshift is not None else wave
        wave2, ret = self.original._approxWavelength(wave1)
        wave = wave2 * (1.+self._redshift) if self._redshift is not None else wave2
        jac, offset, flux_ratio = self._getTransformations(wave)
        offset = _PositionD(*offset)
        return wave, Transformation(ret, jac=jac, offset=offset, flux_ratio=flux_ratio,
                                    gsparams=self._gsparams,
                                    propagate_gsparams=self._propagate_gsparams)

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        from .transform import Transformation
        if self._redshift is not None:
            wave1 = wave / (1.+self._redshift)
        else:
            wave1 = wave
        ret = self.original.evaluateAtWavelength(wave1)
        jac, offset, flux_ratio = self._getTransformations(wave)
        offset = _PositionD(*offset)
        return Transformation(ret, jac=jac, offset=offset, flux_ratio=flux_ratio,
                              gsparams=self._gsparams, propagate_gsparams=self._propagate_gsparams)

    def _shoot(self, photons, rng):
        self._original._shoot(photons, rng)
        wave = photons.wavelength
        jac, offset, flux_ratio = self._getTransformations(wave)

        # cf. Transformation._fwd_normal
        if jac is not None:
            temp = jac[0,1] * photons.y
            photons.y *= jac[1,1]
            photons.y += jac[1,0] * photons.x
            photons.x *= jac[0,0]
            photons.x += temp

            det = jac[0,0] * jac[1,1] - jac[0,1] * jac[1,0]
            flux_ratio *= np.abs(det)

        photons.x += offset[0]
        photons.y += offset[1]

        photons.flux *= flux_ratio

    def drawImage(self, bandpass, image=None, integrator='quadratic', **kwargs):
        """
        See `ChromaticObject.drawImage` for a full description.

        This version usually just calls that one, but if the transformed object (self.original) is
        an `InterpolatedChromaticObject`, and the transformation is achromatic, then it will still
        be able to use the interpolation.

        Parameters:
            bandpass:       A `Bandpass` object representing the filter against which to
                            integrate.
            image:          Optionally, the `Image` to draw onto.  (See `GSObject.drawImage`
                            for details.)  [default: None]
            integrator:     When doing the exact evaluation of the profile, this argument should
                            be one of the image integrators from galsim.integ, or a string
                            'trapezoidal', 'midpoint', 'quadratic', in which case the routine will
                            use a `SampleIntegrator` or `ContinuousIntegrator` depending on whether
                            or not the object has a ``wave_list``.  [default: 'quadratic',
                            which will try to select an appropriate integrator using the
                            quadratic integration rule automatically.]
                            If the object being transformed is an `InterpolatedChromaticObject`,
                            then ``integrator`` can only be a string, either 'midpoint',
                            'trapezoidal', or 'quadratic'.
            **kwargs:       For all other kwarg options, see `GSObject.drawImage`.

        Returns:
            the drawn `Image`.
        """
        from .transform import Transform
        # Store the last bandpass used.
        self._last_bp = bandpass
        if self.SED.dimensionless:
            raise GalSimSEDError("Can only draw ChromaticObjects with spectral SEDs.", self.SED)
        if isinstance(self.original, InterpolatedChromaticObject):
            # Pass self._flux_ratio, which *could* depend on wavelength, to _get_interp_image,
            # where it will be used to reweight the stored images.
            int_im = self.original._get_interp_image(bandpass, image=image, integrator=integrator,
                                                     _flux_ratio=self._flux_ratio, **kwargs)
            # Get shape transformations at bandpass.red_limit (they are achromatic so it doesn't
            # matter where you get them).
            jac, offset, _ = self._getTransformations(bandpass.red_limit)
            offset = _PositionD(*offset)
            int_im = Transform(int_im, jac=jac, offset=offset, gsparams=self._gsparams,
                               propagate_gsparams=self._propagate_gsparams)
            image = int_im.drawImage(image=image, **kwargs)
            self._last_wcs = image.wcs
            return image
        else:
            image = ChromaticObject.drawImage(self, bandpass, image, integrator, **kwargs)
            self._last_wcs = image.wcs
            return image

    @lazy_property
    def noise(self):
        """An estimate of the noise already in the profile.
        """
        from .transform import _Transform
        from .correlatednoise import BaseCorrelatedNoise
        # Condition for being able to propagate noise:
        # 1) All transformations are achromatic.
        # 2) This ChromaticTransformation wraps a ChromaticConvolution with a valid noise property.
        if (hasattr(self._jac, '__call__') or
            hasattr(self._offset, '__call__') or
            hasattr(self._flux_ratio, '__call__')):
            raise GalSimError("Cannot propagate noise through chromatic transformation")
        noise = self.original.noise
        jac = self._jac
        flux_ratio = self._flux_ratio
        return BaseCorrelatedNoise(noise.rng,
                                   _Transform(noise._profile, jac, flux_ratio=flux_ratio**2),
                                   noise.wcs)


class ChromaticSum(ChromaticObject):
    """A sum of several `ChromaticObject` and/or `GSObject` instances.

    Any `GSObject` in the sum is assumed to have a flat `SED` with spectral density of 1
    photon/s/cm**2/nm.

    This is the type returned from `galsim.Add` if any of the objects are a `ChromaticObject`.

    Typically, you do not need to construct a ChromaticSum object explicitly.  Normally, you
    would just use the + operator, which returns a ChromaticSum when used with chromatic objects::

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8) * bulge_sed
        >>> disk = galsim.Exponential(half_light_radius=1.4) * disk_sed
        >>> gal = bulge + disk

    You can also use the `Add` factory function, which returns a ChromaticSum object if any of
    the individual objects are chromatic::

        >>> gal = galsim.Add([bulge,disk])

    Parameters:
        args:               Unnamed args should be a list of objects to add.
        gsparams:           An optional `GSParams` argument.  See the docstring for `GSParams` for
                            details. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]
    """
    def __init__(self, *args, **kwargs):
        # Check kwargs first:
        gsparams = kwargs.pop("gsparams", None)
        self._propagate_gsparams = kwargs.pop("propagate_gsparams", True)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError("Got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise TypeError("Must provide at least one GSObject or ChromaticObject.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject, ChromaticObject or a list of these.
            if isinstance(args[0], (GSObject, ChromaticObject)):
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject, a ChromaticObject,"
                                " or list of them.")
        # else args is already the list of objects

        # Figure out what gsparams to use
        if gsparams is None:
            # If none is given, take the most restrictive combination from the obj_list.
            self._gsparams = GSParams.combine([obj.gsparams for obj in args])
        else:
            # If something explicitly given, then use that.
            self._gsparams = GSParams.check(gsparams)

        self.interpolated = any(arg.interpolated for arg in args)
        if self.interpolated:
            self.deinterpolated = ChromaticSum([arg.deinterpolated for arg in args],
                                               gsparams=self._gsparams)
        else:
            self.deinterpolated = self

        # We can only add ChromaticObjects together if they're either all SED'd or all non-SED'd
        dimensionless = all(a.dimensionless for a in args)
        spectral = all(a.spectral for a in args)
        if not (dimensionless or spectral):
            raise GalSimIncompatibleValuesError(
                "Cannot add dimensionless and spectral ChromaticObjects.", args=args)

        # Sort arguments into inseparable objects and groups of separable objects.  Note that
        # separable groups are only identified if the constituent objects have the *same* SED even
        # though a proportional SED is mathematically sufficient for separability.  It's basically
        # impossible to identify if two SEDs are proportional (or even equal) unless they point to
        # the same memory, so we just accept this limitation.

        # Each input summand will either end up in SED_dict if it's separable, or in self._obj_list
        # if it's inseparable.  Use an OrderedDict to ensure deterministic results.
        from collections import OrderedDict
        SED_dict = OrderedDict()
        self._obj_list = []
        for obj in args:
            if self._propagate_gsparams:
                obj = obj.withGSParams(self._gsparams)
            if obj.separable:
                if obj.SED not in SED_dict:
                    SED_dict[obj.SED] = []
                SED_dict[obj.SED].append(obj)
            else:
                self._obj_list.append(obj)

        # If everything ended up in a single SED_dict entry (and self._obj_list is empty) then this
        # ChromaticSum is separable.
        self.separable = (len(self._obj_list) == 0 and len(SED_dict) == 1)
        if self.separable:
            the_one_SED = list(SED_dict)[0]
            self._obj_list = SED_dict[the_one_SED]
            # Since we know that the chromatic objects' SEDs already include all relevant
            # normalizations, we can just multiply the_one_SED by the number of objects.
            self.SED = the_one_SED * len(SED_dict[the_one_SED])
        else:
            # Sum is not separable, put partial sums might be.  Search for them.
            for v in SED_dict.values():
                if len(v) == 1:
                    self._obj_list.append(v[0])
                else:
                    self._obj_list.append(ChromaticSum(v))
            # and assemble self normalization:
            self.SED = self._obj_list[0].SED
            for obj in self._obj_list[1:]:
                self.SED += obj.SED

        self.wave_list, _, _ = utilities.combine_wave_list(self._obj_list)

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams
    @property
    def obj_list(self):
        """The list of objects being added.
        """
        return self._obj_list

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._obj_list = [ obj.withGSParams(ret._gsparams) for obj in self.obj_list ]
        return ret

    @doc_inherit
    def atRedshift(self, redshift):
        from copy import copy
        ret = copy(self)
        ret._obj_list = [ obj.atRedshift(redshift) for obj in self.obj_list ]
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticSum) and
                 self.obj_list == other.obj_list and
                 self._gsparams == other._gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.ChromaticSum", tuple(self.obj_list), self._gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return 'galsim.ChromaticSum(%r, gsparams=%r, propagate_gsparams=%r)'%(
                self.obj_list, self._gsparams, self._propagate_gsparams)

    def __str__(self):
        str_list = [ str(obj) for obj in self.obj_list ]
        return 'galsim.ChromaticSum([%s])'%', '.join(str_list)

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength ``wave``.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        from .sum import Add
        return Add([obj.evaluateAtWavelength(wave) for obj in self.obj_list],
                   gsparams=self._gsparams, propagate_gsparams=self._propagate_gsparams)

    def _shoot(self, photons, rng):
        raise GalSimNotImplementedError("ChromaticSum cannot be used as a PhotonOp")

    def drawImage(self, bandpass, image=None, integrator='quadratic', **kwargs):
        """Slightly optimized draw method for `ChromaticSum` instances.

        Draws each summand individually and add resulting images together.  This might waste time if
        two or more summands are separable and have the same `SED`, and another summand with a
        different `SED` is also added, in which case the summands should be added together first and
        the resulting `Sum` object can then be chromaticized.  In general, however, drawing
        individual sums independently can help with speed by identifying chromatic profiles that
        are separable into spectral and spatial factors.

        Parameters:
            bandpass:       A `Bandpass` object representing the filter against which to
                            integrate.
            image:          Optionally, the `Image` to draw onto.  (See `GSObject.drawImage`
                            for details.)  [default: None]
            integrator:     When doing the exact evaluation of the profile, this argument should
                            be one of the image integrators from galsim.integ, or a string
                            'trapezoidal', 'midpoint', 'quadratic', in which case the routine will
                            use a `SampleIntegrator` or `ContinuousIntegrator` depending on whether
                            or not the object has a ``wave_list``.  [default: 'quadratic',
                            which will try to select an appropriate integrator using the
                            quadratic integration rule automatically.]
            **kwargs:       For all other kwarg options, see `GSObject.drawImage`.

        Returns:
            the drawn `Image`.
        """
        # Store the last bandpass used.
        self._last_bp = bandpass
        if self.SED.dimensionless:
            raise GalSimSEDError("Can only draw ChromaticObjects with spectral SEDs.", self.SED)
        add_to_image = kwargs.pop('add_to_image', False)
        # Use given add_to_image for the first one, then add_to_image=True for the rest.
        image = self.obj_list[0].drawImage(
                bandpass, image=image, add_to_image=add_to_image, **kwargs)
        _remove_setup_kwargs(kwargs)
        for obj in self.obj_list[1:]:
            image = obj.drawImage(bandpass, image=image, add_to_image=True, **kwargs)
        self._last_wcs = image.wcs
        return image

    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by ``flux_ratio``

        Parameters:
            flux_ratio:     The factor by which to scale the flux.

        Returns:
            the object with the new flux.
        """
        new_obj = ChromaticSum([ obj.withScaledFlux(flux_ratio) for obj in self.obj_list ])
        if hasattr(self, 'covspec'):
            new_covspec = self.covspec * flux_ratio**2
            new_obj.covspec = new_covspec
        return new_obj


class ChromaticConvolution(ChromaticObject):
    """A convolution of several `ChromaticObject` and/or `GSObject` instances.

    Any `GSObject` in the convolution is assumed to have a flat `SED` with spectral density of 1
    photon/s/cm**2/nm.

    This is the type returned from `galsim.Convolve` if any of the objects is a `ChromaticObject`.

    The normal way to use this class is to use the `Convolve` factory function::

        >>> gal = galsim.Sersic(n, half_light_radius) * galsim.SED(sed_file, 'nm', 'flambda')
        >>> psf = galsim.ChromaticAtmosphere(...)
        >>> final = galsim.Convolve([gal, psf])

    The objects to be convolved may be provided either as multiple unnamed arguments (e.g.
    ``Convolve(psf, gal, pix)``) or as a list (e.g. ``Convolve([psf, gal, pix])``).  Any number of
    objects may be provided using either syntax.  (Well, the list has to include at least 1 item.)

    Parameters:
        args:               Unnamed args should be a list of objects to convolve.
        real_space:         Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
        gsparams:           An optional `GSParams` argument.  See the docstring for `GSParams` for
                            details. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]
    """
    def __init__(self, *args, **kwargs):
        # First check for number of arguments != 0
        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise TypeError("Must provide at least one GSObject or ChromaticObject")
        elif len(args) == 1:
            if isinstance(args[0], (GSObject, ChromaticObject)):
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject, or a ChromaticObject,"
                                " or list of them.")
        # else args is already the list of objects

        # Check kwargs
        # real space convolution is not implemented for chromatic objects.
        real_space = kwargs.pop("real_space", None)
        if real_space:
            raise GalSimNotImplementedError(
                "Real space convolution of chromatic objects not implemented.")
        gsparams = kwargs.pop("gsparams", None)
        self._propagate_gsparams = kwargs.pop("propagate_gsparams", True)

        # Figure out what gsparams to use
        if gsparams is None:
            # If none is given, take the most restrictive combination from the obj_list.
            self._gsparams = GSParams.combine([obj.gsparams for obj in args])
        else:
            # If something explicitly given, then use that.
            self._gsparams = GSParams.check(gsparams)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError("Got unexpected keyword argument(s): %s"%kwargs.keys())

        # Accumulate convolutant .SEDs.  Check if more than one is spectral.
        nspectral = sum(arg.spectral for arg in args)
        if nspectral > 1:
            raise GalSimIncompatibleValuesError(
                "Cannot convolve more than one spectral ChromaticObject.", args=args)
        self.SED = args[0].SED
        for obj in args[1:]:
            self.SED *= obj.SED

        self._obj_list = []
        # Unfold convolution of convolution.
        for obj in args:
            if self._propagate_gsparams:
                obj = obj.withGSParams(self._gsparams)
            if isinstance(obj, ChromaticConvolution):
                self._obj_list.extend(obj.obj_list)
            else:
                self._obj_list.append(obj)

        self.separable = all(obj.separable for obj in self._obj_list)
        self.interpolated = any(obj.interpolated for obj in self._obj_list)
        if self.interpolated:
            self.deinterpolated = ChromaticConvolution(
                    [obj.deinterpolated for obj in self._obj_list],
                    gsparams=self._gsparams, propagate_gsparams=self._propagate_gsparams)
        else:
            self.deinterpolated = self

        # Check quickly whether we are convolving two non-separable things that aren't
        # ChromaticSums, >1 of which uses interpolation.  If so, emit a warning that the
        # interpolation optimization is being ignored and full evaluation is necessary.
        # For the case of ChromaticSums, as long as each object in the sum is separable (even if the
        # entire object is not) then interpolation can still be used.  So we do not warn about this
        # here.
        n_nonsep = 0
        n_interp = 0
        for obj in self._obj_list:
            if not obj.separable and not isinstance(obj, ChromaticSum): n_nonsep += 1
            if obj.interpolated: n_interp += 1
        if n_nonsep>1 and n_interp>0:
            galsim_warn(
                "Image rendering for this convolution cannot take advantage of "
                "interpolation-related optimization.  Will use full profile evaluation.")

        # Assemble wave_lists
        self.wave_list, _, _ = utilities.combine_wave_list(self._obj_list)

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams
    @property
    def obj_list(self):
        """The list of objects being convolved.
        """
        return self._obj_list

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._obj_list = [ obj.withGSParams(ret._gsparams) for obj in self.obj_list ]
        return ret

    @staticmethod
    def _get_effective_prof(insep_obj, bandpass, iimult, integrator, gsparams):
        from .interpolatedimage import InterpolatedImage
        # Find scale at which to draw effective profile
        # Use smallest nyquist scale among the fiducial profile and at the two limits of the bp.
        _, prof0 = insep_obj._fiducial_profile(bandpass)
        prof1 = insep_obj.evaluateAtWavelength(bandpass.red_limit)
        prof2 = insep_obj.evaluateAtWavelength(bandpass.blue_limit)
        iiscale = min(prof0.nyquist_scale, prof1.nyquist_scale, prof2.nyquist_scale)
        iiscale /= 2  # This seems to be required to make test_monochromatic_sed to pass.
                      # Not sure why, since I thought straight nyquist should be good enough.
                      # But if it's needed there, it's probably worth always doing, rather than
                      # having that test use iimult=2.  And definitions of Nyquist are somewhat
                      # confusing, so it's possible that we should expect to need a factor of
                      # 2 smaller than nyquist for the pixel scale.  :-S
        if iimult is not None:
            iiscale /= iimult

        # Prevent infinite recursive loop by using ChromaticObject.drawImage() on a
        # ChromaticConvolution.
        if isinstance(insep_obj, ChromaticConvolution):
            effective_prof_image = ChromaticObject.drawImage(
                    insep_obj, bandpass, scale=iiscale,
                    integrator=integrator, method='no_pixel')
        else:
            effective_prof_image = insep_obj.drawImage(
                    bandpass, scale=iiscale, integrator=integrator,
                    method='no_pixel')

        return InterpolatedImage(effective_prof_image, gsparams=gsparams)

    @staticmethod
    def resize_effective_prof_cache(maxsize):
        """Resize the cache containing effective profiles.

        These are wavelength-integrated products of separable profile SEDs, inseparable profiles,
        and Bandpasses) used by `ChromaticConvolution.drawImage`.

        Parameters:
            maxsize:    The new number of effective profiles to cache.
        """
        ChromaticConvolution._effective_prof_cache.resize(maxsize)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticConvolution) and
                 self.obj_list == other.obj_list and
                 self._gsparams == other._gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.ChromaticConvolution", tuple(self.obj_list), self._gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return 'galsim.ChromaticConvolution(%r, gsparams=%r, propagate_gsparams=%r)'%(
                self.obj_list, self._gsparams, self._propagate_gsparams)

    def __str__(self):
        str_list = [ str(obj) for obj in self.obj_list ]
        return 'galsim.ChromaticConvolution([%s])'%', '.join(str_list)

    def _approxWavelength(self, wave):
        from .convolve import Convolve
        # If any of the components prefer a different wavelength, use that for all.
        achrom_objs = []
        for k, obj in enumerate(self.obj_list):
            new_wave, aobj = obj._approxWavelength(wave)
            if new_wave != wave:
                # Break the loop and use evaluateAtWavelength for everything else.
                achrom_objs = ([o.evaluateAtWavelength(new_wave) for o in self.obj_list[:k]] +
                               [aobj] +
                               [o.evaluateAtWavelength(new_wave) for o in self.obj_list[k+1:]])
                break
            else:
                achrom_objs.append(aobj)

        return new_wave, Convolve(achrom_objs, gsparams=self._gsparams,
                                  propagate_gsparams=self._propagate_gsparams)

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength ``wave``.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        from .convolve import Convolve
        return Convolve([obj.evaluateAtWavelength(wave) for obj in self.obj_list],
                        gsparams=self._gsparams, propagate_gsparams=self._propagate_gsparams)

    def _shoot(self, photons, rng):
        raise GalSimNotImplementedError("ChromaticConvolution cannot be used as a PhotonOp")

    def drawImage(self, bandpass, image=None, integrator='quadratic', iimult=None, **kwargs):
        """Optimized draw method for the `ChromaticConvolution` class.

        Works by finding sums of profiles which include separable portions, which can then be
        integrated before doing any convolutions, which are pushed to the end.

        This method uses a cache to avoid recomputing 'effective' profiles, which are the
        wavelength-integrated products of inseparable profiles, the spectral components of
        separable profiles, and the bandpass.  Because the cache size is finite, users may find
        that it is more efficient when drawing many images to group images using the same
        SEDs, bandpasses, and inseparable profiles (generally PSFs) together in order to hit the
        cache more often.  The default cache size is 10, but may be resized using the
        `ChromaticConvolution.resize_effective_prof_cache` method.

        Parameters:
            bandpass:       A `Bandpass` object representing the filter against which to
                            integrate.
            image:          Optionally, the `Image` to draw onto.  (See `GSObject.drawImage`
                            for details.)  [default: None]
            integrator:     When doing the exact evaluation of the profile, this argument should
                            be one of the image integrators from galsim.integ, or a string
                            'trapezoidal', 'midpoint', or 'quadratic', in which case the routine
                            will use a `SampleIntegrator` or `ContinuousIntegrator` depending on
                            whether or not the object has a ``wave_list``.  [default: 'quadratic',
                            which will try to select an appropriate integrator using the
                            quadratic integration rule automatically.]
            iimult:         Oversample any intermediate `InterpolatedImage` created to hold
                            effective profiles by this amount. [default: None]
            **kwargs:       For all other kwarg options, see `GSObject.drawImage`.

        Returns:
            the drawn `Image`.
        """
        from .convolve import Convolve
        from .random import BaseDeviate
        # Store the last bandpass used.
        self._last_bp = bandpass
        if self.SED.dimensionless:
            raise GalSimSEDError("Can only draw ChromaticObjects with spectral SEDs.", self.SED)
        # `ChromaticObject.drawImage()` can just as efficiently handle separable cases.
        if self.separable:
            image = ChromaticObject.drawImage(self, bandpass, image=image, **kwargs)
            self._last_wcs = image.wcs
            return image

        # Now split up any `ChromaticSum`s:
        # This is the tricky part.  Some notation first:
        #     int(f(x,y,lambda)) denotes the integral over wavelength of chromatic surface
        #         brightness profile f(x,y,lambda).
        #     (f1 * f2) denotes the convolution of surface brightness profiles f1 & f2.
        #     (f1 + f2) denotes the addition of surface brightness profiles f1 & f2.
        #
        # In general, chromatic s.b. profiles can be classified as either separable or inseparable,
        # depending on whether they can be factored into spatial and spectral components or not.
        # Write separable profiles as g(x,y) * h(lambda), and leave inseparable profiles as
        # f(x,y,lambda).
        # We will suppress the arguments `x`, `y`, `lambda`, hereforward, but generally an `f`
        # refers to an inseparable profile, a `g` refers to the spatial part of a separable
        # profile, and an `h` refers to the spectral part of a separable profile.
        #
        # Now, analyze a typical scenario, a bulge+disk galaxy model (each of which is separable,
        # e.g., an SED times an exponential profile for the disk, and a different SED times a DeV
        # profile for the bulge).  Suppose the PSF is inseparable.  (Chromatic PSF's will generally
        # be inseparable since we usually think of the spatial part of the PSF being normalized to
        # unit integral for any fixed wavelength.)  Say there's also an achromatic pixel to
        # convolve with.
        # The formula for this might look like:
        #
        # img = int((bulge + disk) * PSF * pix)
        #     = int((g1 h1 + g2 h2) * f3 * g4)               # note pix is lambda-independent
        #     = int(g1 h1 * f3 * g4 + g2 h2 * f3 * g4)       # distribute the + over the *
        #     = int(g1 h1 * f3 * g4) + int(g2 h2 * f3 * g4)  # distribute the + over the int
        #     = g1 * g4 * int(h1 f3) + g2 * g4 * int(h2 f3)  # move lambda-indep terms out of int
        #
        # The result is that the integral is now inside the convolution, meaning we only have to
        # compute two convolutions instead of a convolution for each wavelength at which we evaluate
        # the integrand.  This technique, making an "effective" PSF profile for each of the bulge
        # and disk, is a significant time savings in most cases.
        #
        # In general, we make effective profiles by splitting up ChromaticSum items and collecting
        # the inseparable terms on which to do integration first, and then finish with convolution
        # last.

        phot = kwargs.get('method', 'auto') == 'phot'

        # This optimization is not actually helpful when photon shooting.
        if not phot:
            # Here is the logic to turn
            #   int((g1 h1 + g2 h2) * f3)
            #   -> g1 * int(h1 f3) + g2 * int(h2 f3)
            for i, obj in enumerate(self.obj_list):
                if isinstance(obj, ChromaticSum):
                    # say obj.obj_list = [A,B,C], where obj is a ChromaticSum object
                    # Assemble temporary list of convolutants excluding the ChromaticSum in question.
                    tmplist = list(self.obj_list)
                    del tmplist[i]  # remove ChromaticSum object from obj_list
                    tmplist.append(obj.obj_list[0])  # Append first summand, i.e., A, to convolutants
                    # now draw this image
                    tmpobj = ChromaticConvolution(tmplist)
                    add_to_image = kwargs.pop('add_to_image', False)
                    image = tmpobj.drawImage(bandpass, image=image, integrator=integrator,
                                            iimult=iimult, add_to_image=add_to_image, **kwargs)
                    # Now add in the rest of the summands in turn, i.e., B and C
                    for summand in obj.obj_list[1:]:
                        tmplist = list(self.obj_list)
                        del tmplist[i]
                        tmplist.append(summand)
                        tmpobj = ChromaticConvolution(tmplist)
                        # add to previously started image
                        _remove_setup_kwargs(kwargs)
                        image = tmpobj.drawImage(bandpass, image=image, integrator=integrator,
                                                iimult=iimult, add_to_image=True, **kwargs)
                    # Return the image here, breaking the loop early.  If there are two ChromaticSum
                    # instances in obj_list, then the above procedure will repeat in the recursion,
                    # effectively distributing the multiplication over both sums.
                    self._last_wcs = image.wcs
                    return image

        # If program gets this far, the objects in obj_list should be atomic (non-ChromaticSum
        # and non-ChromaticConvolution).  (The latter case was dealt with in the constructor.)

        # setup output image (semi-arbitrarily using the bandpass effective wavelength)
        wave0, prof0 = self._fiducial_profile(bandpass)
        image = prof0.drawImage(image=image, setup_only=True, **kwargs)
        _remove_setup_kwargs(kwargs)

        # If we are photon shooting, then we can move all non-spectral objects to the photon_ops
        # list and deal with them that way.  This both more accurate and more efficient for most
        # chromatic PSFs.
        if phot:
            psfs = [obj for obj in self.obj_list if obj.dimensionless]
            gals = [obj for obj in self.obj_list if obj.spectral]
            assert len(gals) == 1  # Should have been checked by constructor.
            gal = gals[0]
            kwargs['photon_ops'] = psfs + kwargs.get('photon_ops', [])

            # Need to calculate n_photons now using the fiducial profile, not gal, in case the
            # PSF has an interpolated image (e.g. OpticalPSF) which needs more photons.
            flux = self.calculateFlux(bandpass)
            prof1 = prof0.withFlux(flux)
            n_photons = kwargs.pop('n_photons', 0)
            poisson_flux = kwargs.pop('poisson_flux', n_photons == 0.)
            max_extra_noise = kwargs.pop('max_extra_noise', 0.)
            rng = BaseDeviate(kwargs.get('rng', None))
            n_photons, _ = prof1._calculate_nphotons(n_photons, poisson_flux, max_extra_noise, rng)
            return gal.drawImage(bandpass, image=image, integrator=integrator,
                                 n_photons=n_photons, **kwargs)

        # Separate convolutants into a Convolution of inseparable profiles multiplied by the
        # wavelength-dependent normalization of separable profiles, and the achromatic part of
        # separable profiles.
        insep_obj = [obj for obj in self.obj_list if not obj.separable]

        # Note that len(insep_obj) > 0, since purely separable ChromaticConvolutions were
        # already handled above.
        # Don't wrap in Convolution if not needed.  Single item can draw itself better than
        # Convolution can.
        if len(insep_obj) == 1:
            insep_obj = insep_obj[0]
        else:
            insep_obj = Convolve(insep_obj, gsparams=self._gsparams,
                                 propagate_gsparams=self._propagate_gsparams)

        sep_profs = []
        for obj in self.obj_list:
            if not obj.separable:
                continue
            wave0, prof0 = obj._fiducial_profile(bandpass)
            sep_profs.append(prof0 / obj.SED(wave0))
            insep_obj *= obj.SED

        # Collapse inseparable profiles and chromatic normalizations into one effective profile
        # Note that at this point, insep_obj.SED should *not* be None.
        effective_prof = ChromaticConvolution._effective_prof_cache(
                insep_obj, bandpass, iimult, integrator, self._gsparams)

        # append effective profile to separable profiles (which should all be GSObjects)
        sep_profs.append(effective_prof)
        # finally, convolve and draw.
        final_prof = Convolve(sep_profs, gsparams=self._gsparams,
                              propagate_gsparams=self._propagate_gsparams)
        image = final_prof.drawImage(image=image, **kwargs)
        self._last_wcs = image.wcs
        return image

    @lazy_property
    def noise(self):
        """An estimate of the noise already in the profile.
        """
        from .convolve import Convolve
        # Condition for being able to propagate noise:
        # Exactly one of the convolutants has a .covspec attribute.
        covspecs = [ obj.covspec for obj in self.obj_list if hasattr(obj, 'covspec') ]
        if len(covspecs) != 1:
            raise GalSimError("Cannot compute noise for ChromaticConvolution for which number "
                              "of convolutants with covspec attribute is not 1.")
        if not hasattr(self, '_last_bp'):
            raise GalSimError("Cannot compute noise for ChromaticConvolution until after drawImage "
                              "has been called.")
        covspec = covspecs[0]
        other = Convolve([obj for obj in self.obj_list if not hasattr(obj, 'covspec')])
        return covspec.toNoise(self._last_bp, other, self._last_wcs)  # rng=?


ChromaticConvolution._effective_prof_cache = utilities.LRU_Cache(
    ChromaticConvolution._get_effective_prof, maxsize=10)


class ChromaticDeconvolution(ChromaticObject):
    """A class for deconvolving a `ChromaticObject`.

    The ChromaticDeconvolution class represents a wavelength-dependent deconvolution kernel.

    You may also specify a gsparams argument.  See the docstring for `GSParams` for more
    information about this option.  Note: if ``gsparams`` is unspecified (or None), then the
    ChromaticDeconvolution instance inherits the same `GSParams` as the object being deconvolved.

    This is the type returned from `galsim.Deconvolve` if the argument is a `ChromaticObject`.
    This is the normal way to construct this class.

    Parameters:
        obj:                The object to deconvolve.
        gsparams:           An optional `GSParams` argument.  See the docstring for `GSParams` for
                            details. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]
    """
    def __init__(self, obj, gsparams=None, propagate_gsparams=True):
        if not obj.SED.dimensionless:
            raise GalSimSEDError("Cannot deconvolve by spectral ChromaticObject.", obj.SED)
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams
        if self._propagate_gsparams:
            self._obj = obj.withGSParams(self._gsparams)
        else:
            self._obj = obj
        self.separable = obj.separable
        self.interpolated = obj.interpolated
        if self.interpolated:
            self.deinterpolated = ChromaticDeconvolution(self._obj.deinterpolated, self._gsparams,
                                                         self._propagate_gsparams)
        else:
            self.deinterpolated = self
        self.SED = SED(lambda w: 1./obj.SED(w), 'nm', '1')
        self.wave_list = obj.wave_list

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._obj = self._obj.withGSParams(ret._gsparams)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticDeconvolution) and
                 self._obj == other._obj and
                 self.gsparams == other.gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.ChromaticDeconvolution", self._obj, self.gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return 'galsim.ChromaticDeconvolution(%r, gsparams=%r, propagate_gsparams=%r)'%(
                self._obj, self.gsparams, self._propagate_gsparams)

    def __str__(self):
        return 'galsim.ChromaticDeconvolution(%s)'%self._obj

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength ``wave``.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        from .convolve import Deconvolve
        return Deconvolve(self._obj.evaluateAtWavelength(wave), gsparams=self.gsparams,
                          propagate_gsparams=self._propagate_gsparams)

    def _shoot(self, photons, rng):
        raise GalSimNotImplementedError("ChromaticDeconvolution cannot use method='phot'")


class ChromaticAutoConvolution(ChromaticObject):
    """A special class for convolving a `ChromaticObject` with itself.

    It is equivalent in functionality to ``galsim.Convolve([obj,obj])``, but takes advantage of
    the fact that the two profiles are the same for some efficiency gains.

    This is the type returned from `galsim.AutoConvolve` if the argument is a `ChromaticObject`.
    This is the normal way to construct this class.

    Parameters:
        obj:                The object to be convolved with itself.
        real_space:         Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
        gsparams:           An optional `GSParams` argument.  See the docstring for `GSParams` for
                            details. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]
    """
    def __init__(self, obj, real_space=None, gsparams=None, propagate_gsparams=True):
        if not obj.SED.dimensionless:
            raise GalSimSEDError("Cannot autoconvolve spectral ChromaticObject.", obj.SED)
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams
        if self._propagate_gsparams:
            self._obj = obj.withGSParams(self._gsparams)
        else:
            self._obj = obj
        self._real_space = real_space
        self.separable = obj.separable
        self.interpolated = obj.interpolated
        if self.interpolated:
            self.deinterpolated = ChromaticAutoConvolution(self._obj.deinterpolated, real_space,
                                                           self._gsparams, self._propagate_gsparams)
        else:
            self.deinterpolated = self
        self.SED = obj.SED * obj.SED
        self.wave_list = obj.wave_list

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._obj = self._obj.withGSParams(ret._gsparams)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticAutoConvolution) and
                 self._obj == other._obj and
                 self._real_space == other._real_space and
                 self.gsparams == other.gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.ChromaticAutoConvolution", self._obj, self._real_space, self.gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return ('galsim.ChromaticAutoConvolution(%r, real_space=%r, gsparams=%r, '
                'propagate_gsparams=%r)')%(
                self._obj, self._real_space, self.gsparams, self._propagate_gsparams)

    def __str__(self):
        return 'galsim.ChromaticAutoConvolution(%s)'%self._obj

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength ``wave``.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        from .convolve import AutoConvolve
        return AutoConvolve(self._obj.evaluateAtWavelength(wave), self._real_space, self._gsparams,
                            self._propagate_gsparams)

    def _shoot(self, photons, rng):
        raise GalSimNotImplementedError("ChromaticAutoConvolution cannot be used as a PhotonOp")


class ChromaticAutoCorrelation(ChromaticObject):
    """A special class for correlating a `ChromaticObject` with itself.

    It is equivalent in functionality to::

        galsim.Convolve([obj,obj.rotate(180.*galsim.degrees)])

    but takes advantage of the fact that the two profiles are the same for some efficiency gains.

    This is the type returned from `galsim.AutoCorrelate` if the argument is a `ChromaticObject`.
    This is the normal way to construct this class.

    Parameters:
        obj:                The object to be convolved with itself.
        real_space:         Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
        gsparams:           An optional `GSParams` argument.  See the docstring for `GSParams` for
                            details. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]
    """
    def __init__(self, obj, real_space=None, gsparams=None, propagate_gsparams=True):
        if not obj.SED.dimensionless:
            raise GalSimSEDError("Cannot autocorrelate spectral ChromaticObject.", obj.SED)
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams
        if self._propagate_gsparams:
            self._obj = obj.withGSParams(self._gsparams)
        else:
            self._obj = obj
        self._real_space = real_space
        self.separable = obj.separable
        self.interpolated = obj.interpolated
        if self.interpolated:
            self.deinterpolated = ChromaticAutoCorrelation(self._obj.deinterpolated,
                                                           self._real_space, self._gsparams,
                                                           self._propagate_gsparams)
        else:
            self.deinterpolated = self
        self.SED = obj.SED * obj.SED
        self.wave_list = obj.wave_list

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._obj = self._obj.withGSParams(ret._gsparams)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticAutoCorrelation) and
                 self._obj == other._obj and
                 self._real_space == other._real_space and
                 self.gsparams == other.gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.ChromaticAutoCorrelation", self._obj, self._real_space, self.gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return ('galsim.ChromaticAutoCorrelation(%r, real_space=%r, gsparams=%r, '
                'propagate_gsparams=%r)')%(
                self._obj, self._real_space, self.gsparams, self._propagate_gsparams)

    def __str__(self):
        return 'galsim.ChromaticAutoCorrelation(%s)'%self._obj

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength ``wave``.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        from .convolve import AutoCorrelate
        return AutoCorrelate(self._obj.evaluateAtWavelength(wave), self._real_space, self.gsparams,
                             self._propagate_gsparams)

    def _shoot(self, photons, rng):
        raise GalSimNotImplementedError("ChromaticAutoCorrelation cannot be used as a PhotonOp")


class ChromaticFourierSqrtProfile(ChromaticObject):
    """A class for computing the Fourier-space square root of a `ChromaticObject`.

    The ChromaticFourierSqrtProfile class represents a wavelength-dependent Fourier-space square
    root of a profile.

    You may also specify a gsparams argument.  See the docstring for `GSParams` for more
    information about this option.  Note: if ``gsparams`` is unspecified (or None), then the
    ChromaticFourierSqrtProfile inherits the same `GSParams` as the object being operated on.

    The normal way to use this class is to use the `FourierSqrt` factory function::

        >>> fourier_sqrt = galsim.FourierSqrt(chromatic_obj)

    If ``chromatic_obj`` is indeed a `ChromaticObject`, then that function will create a
    ChromaticFourierSqrtProfile object.

    Parameters:
        obj:                The object to compute the Fourier-space square root of.
        gsparams:           An optional `GSParams` argument.  See the docstring for `GSParams` for
                            details. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]
    """
    def __init__(self, obj, gsparams=None, propagate_gsparams=True):
        import math
        if not obj.SED.dimensionless:
            raise GalSimSEDError("Cannot take Fourier sqrt of spectral ChromaticObject.", obj.SED)
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams
        if self._propagate_gsparams:
            self._obj = obj.withGSParams(self._gsparams)
        else:
            self._obj = obj
        self.separable = obj.separable
        self.interpolated = obj.interpolated
        if self.interpolated:
            self.deinterpolated = ChromaticFourierSqrtProfile(
                    self._obj.deinterpolated, self._gsparams, self._propagate_gsparams)
        else:
            self.deinterpolated = self
        self.SED = SED(lambda w:math.sqrt(obj.SED(w)), 'nm', '1')
        self.wave_list = obj.wave_list

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._obj = self._obj.withGSParams(ret._gsparams)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticFourierSqrtProfile) and
                 self._obj == other._obj and
                 self.gsparams == other.gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.ChromaticFourierSqrtProfile", self._obj, self.gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return 'galsim.ChromaticFourierSqrtProfile(%r, gsparams=%r, propagate_gsparams=%r)'%(
                self._obj, self.gsparams, self._propagate_gsparams)

    def __str__(self):
        return 'galsim.ChromaticFourierSqrtProfile(%s)'%self._obj

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength ``wave``.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            the monochromatic object at the given wavelength.
        """
        from .fouriersqrt import FourierSqrt
        return FourierSqrt(self._obj.evaluateAtWavelength(wave), self.gsparams,
                           self._propagate_gsparams)

    def _shoot(self, photons, rng):
        raise GalSimNotImplementedError("ChromaticFourierSqrtProfile cannot use method='phot'")


class ChromaticOpticalPSF(ChromaticObject):
    """A subclass of ChromaticObject meant to represent chromatic optical PSFs.

    Chromaticity plays two roles in optical PSFs. First, it determines the diffraction limit, via
    the wavelength/diameter factor.  Second, aberrations such as defocus, coma, etc. are typically
    defined in physical distances, but their impact on the PSF depends on their size in units of
    wavelength.  Other aspects of the optical PSF do not require explicit specification of their
    chromaticity, e.g., once the obscuration and struts are specified in units of the aperture
    diameter, their chromatic dependence gets taken care of automatically.  Note that the
    ChromaticOpticalPSF implicitly defines diffraction limits in units of ``scale_units``, which by
    default are arcsec, but can in principle be set to any of our GalSim angle units.

    When using interpolation to speed up image rendering (see the `ChromaticObject.interpolate`
    method for details), the ideal number of wavelengths to use across a given bandpass depends on
    the application and accuracy requirements.  In general it will be necessary to do a test in
    comparison with a more exact calculation to ensure convergence.  However, a typical calculation
    might use ~10-15 samples across a typical optical bandpass, with ``oversample_fac`` in the range
    1.5-2; for moderate accuracy, ~5 samples across the bandpass and ``oversample_fac=1`` may
    suffice. All of these statements assume that aberrations are not very large (typically <~0.25
    waves, which is commonly satisfied by space telescopes); if they are larger than that, then more
    stringent settings are required.

    Note that a ChromaticOpticalPSF by itself is NOT the correct thing to use to draw an image of a
    star. Stars (and galaxies too, of course) have an `SED` that is not flat. To draw a real star,
    you should either multiply the ChromaticOpticalPSF object by an `SED`, or convolve it with a
    point source multiplied by an `SED`::

        >>> psf = galsim.ChromaticOpticalPSF(...)
        >>> star = galsim.DeltaFunction() * psf_sed
        >>> final_star = galsim.Convolve( [psf, star] )
        >>> final_star.drawImage(bandpass = bp, ...)

    .. note::

        When geometric_shooting is False (the default), the photon shooting implementation is
        only approximately correct with respect to the wavelength dependence.  It is also
        not particularly fast, since it generates three optical screens to span the wavelength
        range and shoots from these (with a subsequent adjustment to improve the accuracy
        of this approximation).  We expect that most users who want to use photon shooting in
        conjunction with this class will prefer to make an InterpolatedChromaticObject
        (by calling ``psf.interpolate(...)``), especially if it is a good approximation to
        use the same optical PSF for a whole exposure or CCD image, so the setup time for
        the interpolation is able to be amortized for many objects.

    Parameters:
        lam:            Fiducial wavelength for which diffraction limit and aberrations are
                        initially defined, in nanometers.
        diam:           Telescope diameter in meters.  Either ``diam`` or ``lam_over_diam`` must be
                        specified.
        lam_over_diam:  Ratio of (fiducial wavelength) / telescope diameter in units of
                        ``scale_unit``.  Either ``diam`` or ``lam_over_diam`` must be specified.
        aberrations:    An array of aberrations, in units of fiducial wavelength ``lam``.  The
                        size and format of this array is described in the OpticalPSF docstring.
        scale_unit:     Units used to define the diffraction limit and draw images.
                        [default: galsim.arcsec]
        gsparams:       An optional `GSParams` argument.  See the docstring for `GSParams` for
                        details. [default: None]
        geometric_shooting: If True, then when drawing using photon shooting, use geometric
                        optics approximation where the photon angles are derived from the
                        phase screen gradient.  If False, then first draw using Fourier
                        optics and then shoot from the derived InterpolatedImage. [default: False]
        **kwargs:       Any other keyword arguments to be passed to OpticalPSF, for example,
                        related to struts, obscuration, oversampling, etc.  See OpticalPSF
                        docstring for a complete list of options.
    """
    _req_params = { 'lam' : float }
    _opt_params = { k:v for k,v in OpticalPSF._opt_params.items() if k != 'diam' }
    _single_params = [ {'diam' : float, 'lam_over_diam' : float} ]

    def __init__(self, lam, diam=None, lam_over_diam=None, aberrations=None,
                 scale_unit=None, gsparams=None, **kwargs):
        from .angle import AngleUnit, arcsec, radians
        # First, take the basic info.
        if scale_unit is None:
            scale_unit = arcsec
        elif isinstance(scale_unit, str):
            scale_unit = AngleUnit.from_name(scale_unit)
        self.scale_unit = scale_unit
        self._gsparams = GSParams.check(gsparams)

        # We have to require either diam OR lam_over_diam:
        if ( (diam is None and lam_over_diam is None) or
             (diam is not None and lam_over_diam is not None) ):
            raise GalSimIncompatibleValuesError(
                "Need to specify telescope diameter OR wavelength/diam ratio",
                diam=diam, lam_over_diam=lam_over_diam)
        if diam is not None:
            self.lam_over_diam = (1.e-9*lam/diam)*radians/self.scale_unit
            self.diam = diam
        else:
            self.lam_over_diam = lam_over_diam
            self.diam = (lam*1e-9/lam_over_diam)*radians/self.scale_unit
        self.lam = lam

        if aberrations is not None:
            self.aberrations = np.asarray(aberrations)
            if len(self.aberrations) < 12:
                self.aberrations = np.append(self.aberrations, [0] * (12-len(self.aberrations)))
        else:
            self.aberrations = np.zeros(12)

        # Pop named aberrations from kwargs so aberrations=[0,0,0,0,1] means the same as
        # defocus=1 (w/ all other named aberrations 0).
        for i, ab in enumerate(['defocus', 'astig1', 'astig2', 'coma1', 'coma2', 'trefoil1',
                                'trefoil2', 'spher']):
            if ab in kwargs:
                self.aberrations[i+4] = kwargs.pop(ab)
        if 'fft_sign' in kwargs:
            fft_sign = kwargs['fft_sign']
            if fft_sign not in ['+', '-']:
                raise GalSimValueError("Invalid fft_sign", fft_sign, allowed_values=['+','-'])

        self.kwargs = kwargs

        # Define the necessary attributes for this ChromaticObject.
        self.separable = False
        self.interpolated = False
        self.deinterpolated = self
        self.SED = SED(1, 'nm', '1')
        self.wave_list = np.array([], dtype=float)

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticOpticalPSF) and
                 self.lam == other.lam and
                 self.lam_over_diam == other.lam_over_diam and
                 np.array_equal(self.aberrations, other.aberrations) and
                 self.scale_unit == other.scale_unit and
                 self.gsparams == other.gsparams and
                 self.kwargs == other.kwargs))

    def __hash__(self):
        return hash(("galsim.ChromaticOpticalPSF", self.lam, self.lam_over_diam,
                     tuple(self.aberrations), self.scale_unit, self.gsparams,
                     frozenset(self.kwargs.items())))

    def __repr__(self):
        from .angle import arcsec
        s = 'galsim.ChromaticOpticalPSF(lam=%r, lam_over_diam=%r, aberrations=%r'%(
                self.lam, self.lam_over_diam, self.aberrations.tolist())
        if self.scale_unit != arcsec:
            s += ', scale_unit=%r'%self.scale_unit
        for k,v in self.kwargs.items():
            s += ', %s=%r'%(k,v)
        s += ', gsparams=%r'%self.gsparams
        s += ')'
        return s

    def __str__(self):
        return 'galsim.ChromaticOpticalPSF(lam=%s, lam_over_diam=%s, aberrations=%s)'%(
                self.lam, self.lam_over_diam, self.aberrations.tolist())

    def evaluateAtWavelength(self, wave):
        """
        Method to directly instantiate a monochromatic instance of this object.

        Parameters:
             wave:  Wavelength in nanometers.
        """
        # The aberrations were in units of wavelength for the fiducial wavelength, so we have to
        # convert to units of waves for *this* wavelength.
        wave_factor = self.lam / wave

        # stepk and maxk also scale basically with this ratio, and they are fairly slow to
        # calculate, so once we've done this once, store the results and just rescale all future
        # versions with this factor.
        if hasattr(self, '_stepk'):
            return OpticalPSF(
                    lam=wave, diam=self.diam,
                    aberrations=self.aberrations*wave_factor, scale_unit=self.scale_unit,
                    _force_stepk=self._stepk*wave_factor, _force_maxk=self._maxk*wave_factor,
                    gsparams=self.gsparams, **self.kwargs)
        else:
            ret = OpticalPSF(
                    lam=wave, diam=self.diam,
                    aberrations=self.aberrations*wave_factor, scale_unit=self.scale_unit,
                    gsparams=self.gsparams, **self.kwargs)
            self._stepk = ret.stepk / wave_factor
            self._maxk = ret.maxk / wave_factor
            return ret

    def _shoot(self, photons, rng):
        from .photon_array import PhotonArray
        from .random import UniformDeviate

        if self.kwargs.get('geometric_shooting',False):
            # In the geometric shooting approximation, the lambda factors out, and this
            # becomes the same kind of calculation we did for ChromaticAiry.
            # Use the mean wavelength for the base profile.
            mean_wave = np.mean(photons.wavelength)
            obj = self.evaluateAtWavelength(mean_wave)
            obj._shoot(photons, rng)
            factor = photons.wavelength / mean_wave
            photons.scaleXY(factor)
        else:
            # When not using geometric shooting, the following isn't exact.
            # The exact method would involve doing the fourier transform for each wavelength
            # in the photon list.  Obviously, that's not tenable.
            # So instead, we shoot with the same random seed for 3 different profiles:
            # The minimum wavelength, the mean, and the maximum.
            # Then interpolate between the results for each photon.
            # This should (hopefully!) be good enough for most use cases if the bandpass
            # isn't extremely wide and the wavelength dependence is modest over the range.

            wave1 = np.min(photons.wavelength)
            wave2 = np.mean(photons.wavelength)
            wave3 = np.max(photons.wavelength)

            prof1 = self.evaluateAtWavelength(wave1)
            if wave1 == wave3:
                # Interjection at this point -- if min=mean=max, then this is easy.
                return prof1._shoot(photons, rng)
            else:
                # Otherwise we're ok dividing by wave2-wave1 and wave3-wave2 below.
                assert wave2 != wave1
                assert wave3 != wave2
            prof2 = self.evaluateAtWavelength(wave2)
            prof3 = self.evaluateAtWavelength(wave3)
            # For each photon, shoot using one of these profiles according to the given
            # wavelength.
            # For wavelenghts with w1 < w < w2, select from prof1 or prof2 with probabilities
            #     P(use prof1) = (w2-w)/(w2-w1)
            #     P(use prof2) = (w-w1)/(w2-w1)
            # Likewise when w2 < w < w3:
            #     P(use prof2) = (w3-w)/(w3-w2)
            #     P(use prof3) = (w-w2)/(w3-w2)
            u = np.empty(len(photons))
            UniformDeviate(rng).generate(u)
            w = photons.wavelength
            use_p1 = (wave1 <= w) & (w < wave2) & (u <= (wave2-w)/(wave2-wave1))
            use_p2 = (wave1 <= w) & (w < wave2) & (u > (wave2-w)/(wave2-wave1))
            use_p2 |= (wave2 <= w) & (w <= wave3) & (u <= (wave3-w)/(wave3-wave2))
            use_p3 = (wave2 <= w) & (w <= wave3) & (u > (wave3-w)/(wave3-wave2))
            assert np.all(use_p1 | use_p2 | use_p3)
            assert not np.any(use_p1 & use_p2)
            assert not np.any(use_p2 & use_p3)
            assert not np.any(use_p1 & use_p3)

            temp1 = PhotonArray(np.sum(use_p1))
            temp2 = PhotonArray(np.sum(use_p2))
            temp3 = PhotonArray(np.sum(use_p3))
            prof1._shoot(temp1, rng)
            prof2._shoot(temp2, rng)
            prof3._shoot(temp3, rng)
            photons.x[use_p1] = temp1.x * (w[use_p1] / wave1)
            photons.y[use_p1] = temp1.y * (w[use_p1] / wave1)
            photons.flux[use_p1] = temp1.flux * (len(temp1)/len(photons))
            photons.x[use_p2] = temp2.x * (w[use_p2] / wave2)
            photons.y[use_p2] = temp2.y * (w[use_p2] / wave2)
            photons.flux[use_p2] = temp2.flux * (len(temp2)/len(photons))
            photons.x[use_p3] = temp3.x * (w[use_p3] / wave3)
            photons.y[use_p3] = temp3.y * (w[use_p3] / wave3)
            photons.flux[use_p3] = temp3.flux * (len(temp3)/len(photons))


class ChromaticAiry(ChromaticObject):
    """A subclass of `ChromaticObject` meant to represent chromatic Airy profiles.

    For more information about the basics of Airy profiles, please see `galsim.Airy`.

    This class is a chromatic representation of Airy profiles, including the wavelength-dependent
    diffraction limit.  One can also get this functionality using the `ChromaticOpticalPSF` class,
    but that class includes additional complications beyond a simple Airy profile, and thus has a
    more complicated internal representation.  For users who only want a (possibly obscured) Airy
    profile, the ChromaticAiry class is likely to be a less computationally expensive and more
    accurate option.

    Parameters:
        lam:            Fiducial wavelength for which diffraction limit is initially defined, in
                        nanometers.
        diam:           Telescope diameter in meters.  Either ``diam`` or ``lam_over_diam`` must be
                        specified.
        lam_over_diam:  Ratio of (fiducial wavelength) / telescope diameter in units of
                        ``scale_unit``.  Either ``diam`` or ``lam_over_diam`` must be specified.
        scale_unit:     Units used to define the diffraction limit and draw images.
                        [default: galsim.arcsec]
        gsparams:       An optional `GSParams` argument.  See the docstring for `GSParams` for
                        details. [default: None]
        **kwargs:       Any other keyword arguments to be passed to `Airy`: either flux, or
                        gsparams.  See `galsim.Airy` docstring for a complete description of these
                        options.
    """
    _req_params = { 'lam' : float }
    _opt_params = { 'scale_unit' : str }
    _single_params = [ {'diam' : float, 'lam_over_diam' : float} ]

    def __init__(self, lam, diam=None, lam_over_diam=None, scale_unit=None, gsparams=None,
                 **kwargs):
        from .angle import AngleUnit, arcsec, radians
        # First, take the basic info.
        # We have to require either diam OR lam_over_diam:
        if scale_unit is None:
            scale_unit = arcsec
        elif isinstance(scale_unit, str):
            scale_unit = AngleUnit.from_name(scale_unit)
        self.scale_unit = scale_unit
        self._gsparams = GSParams.check(gsparams)

        if ( (diam is None and lam_over_diam is None) or
             (diam is not None and lam_over_diam is not None) ):
            raise GalSimIncompatibleValuesError(
                "Need to specify telescope diameter OR wavelength/diam ratio",
                diam=diam, lam_over_diam=lam_over_diam)
        if diam is not None:
            self.lam_over_diam = (1.e-9*lam/diam)*radians/self.scale_unit
        else:
            self.lam_over_diam = float(lam_over_diam)
        self.lam = float(lam)

        self.kwargs = kwargs

        # Define the necessary attributes for this ChromaticObject.
        self.separable = False
        self.interpolated = False
        self.deinterpolated = self
        self.SED = SED(1, 'nm', '1')
        self.wave_list = np.array([], dtype=float)

    @property
    def gsparams(self):
        """The `GSParams` for this object.
        """
        return self._gsparams

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, ChromaticAiry) and
                 self.lam == other.lam and
                 self.lam_over_diam == other.lam_over_diam and
                 self.scale_unit == other.scale_unit and
                 self.gsparams == other.gsparams and
                 self.kwargs == other.kwargs))

    def __hash__(self):
        return hash(("galsim.ChromaticAiry", self.lam, self.lam_over_diam, self.scale_unit,
                     self.gsparams, frozenset(self.kwargs.items())))

    def __repr__(self):
        from .angle import arcsec
        s = 'galsim.ChromaticAiry(lam=%r, lam_over_diam=%r'%(self.lam, self.lam_over_diam)
        if self.scale_unit != arcsec:
            s += ', scale_unit=%r'%self.scale_unit
        for k,v in self.kwargs.items():
            s += ', %s=%r'%(k,v)
        s += ', gsparams=%r'%self.gsparams
        s += ')'
        return s

    def __str__(self):
        return 'galsim.ChromaticAiry(lam=%s, lam_over_diam=%s)'%(self.lam, self.lam_over_diam)

    def evaluateAtWavelength(self, wave):
        """
        Method to directly instantiate a monochromatic instance of this object.

        Parameters:
             wave:  Wavelength in nanometers.
        """
        from .airy import Airy
        # We need to rescale the stored lam/diam by the ratio of input wavelength to stored fiducial
        # wavelength.
        ret = Airy(
            lam_over_diam=self.lam_over_diam*(wave/self.lam), scale_unit=self.scale_unit,
            gsparams=self.gsparams, **self.kwargs)
        return ret

    def _shoot(self, photons, rng):
        from .airy import Airy
        # Start with the convolution at the reference wavelength
        obj = Airy(lam_over_diam=self.lam_over_diam, scale_unit=self.scale_unit,
                   gsparams=self.gsparams, **self.kwargs)
        obj._shoot(photons, rng)

        # Now adjust the positions according to the wavelengths
        factor = photons.wavelength / self.lam
        photons.scaleXY(factor)


def _findWave(wave_list, wave):
    # Helper routine to search a sorted NumPy array of wavelengths (not necessarily evenly spaced)
    # to find where a particular wavelength ``wave`` would fit in, and return the index below along
    # with the fraction of the way to the next entry in the array.
    lower_idx = np.searchsorted(wave_list, wave)-1
    # There can be edge issues, so watch out for that:
    if lower_idx < 0: lower_idx = 0
    if lower_idx > len(wave_list)-1: lower_idx = len(wave_list)-1

    frac = (wave-wave_list[lower_idx]) / (wave_list[lower_idx+1]-wave_list[lower_idx])
    return lower_idx, frac

def _linearInterp(list, frac, lower_idx):
    # Helper routine for linear interpolation between values in lists (which could be lists of
    # images, just not numbers, hence the need to avoid a LookupTable).  Not really worth
    # splitting out on its own now, but could be useful to have separate routines for the
    # interpolation later on if we want to enable something other than linear interpolation.
    return frac*list[lower_idx+1] + (1.-frac)*list[lower_idx]

def _remove_setup_kwargs(kwargs):
    # Helper function to remove from kwargs anything that is only used for setting up image and that
    # might otherwise interfere with drawImage.
    kwargs.pop('dtype', None)
    kwargs.pop('scale', None)
    kwargs.pop('wcs', None)
    kwargs.pop('nx', None)
    kwargs.pop('ny', None)
    kwargs.pop('bounds', None)
