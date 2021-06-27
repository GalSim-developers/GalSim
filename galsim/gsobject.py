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
import math

from . import _galsim
from .gsparams import GSParams
from .position import _PositionD, _PositionI, Position
from .utilities import lazy_property, parse_pos_args
from .errors import GalSimError, GalSimRangeError, GalSimValueError, GalSimIncompatibleValuesError
from .errors import GalSimFFTSizeError, GalSimNotImplementedError, convert_cpp_errors, galsim_warn


class GSObject(object):
    """Base class for all GalSim classes that represent some kind of surface brightness profile.

    A GSObject is not intended to be constructed directly.  Normally, you would use whatever
    derived class is appropriate for the surface brightness profile you want::

        >>> gal = galsim.Sersic(n=4, half_light_radius=4.3)
        >>> psf = galsim.Moffat(beta=3, fwhm=2.85)
        >>> conv = galsim.Convolve([gal,psf])

    All of these classes are subclasses of GSObject, so you should see those docstrings for
    more details about how to construct the various profiles.  Here we discuss attributes and
    methods that are common to all GSObjects.

    GSObjects are always defined in sky coordinates.  So all sizes and other linear dimensions
    should be in terms of some kind of units on the sky, arcsec for instance.  Only later (when
    they are drawn) is the connection to pixel coordinates established via a pixel scale or WCS.
    (See the documentation for galsim.BaseWCS for more details about how to specify various kinds
    of world coordinate systems more complicated than a simple pixel scale.)

    For instance, if you eventually draw onto an image that has a pixel scale of 0.2 arcsec/pixel,
    then the normal thing to do would be to define your surface brightness profiles in terms of
    arcsec and then draw with ``pixel_scale=0.2``.  However, while arcsec are the usual choice of
    units for the sky coordinates, if you wanted, you could instead define the sizes of all your
    galaxies and PSFs in terms of radians and then use ``pixel_scale=0.2/206265`` when you draw
    them.

    **Transforming methods**:

    The GSObject class uses an "immutable" design[1], so all methods that would potentially modify
    the object actually return a new object instead.  This uses pointers and such behind the
    scenes, so it all happens efficiently, but it makes using the objects a bit simpler, since
    you don't need to worry about some function changing your object behind your back.

    In all cases below, we just give an example usage.  See the docstrings for the methods for
    more details about how to use them.::

        >>> obj = obj.shear(shear)      # Apply a shear to the object.
        >>> obj = obj.dilate(scale)     # Apply a flux-preserving dilation.
        >>> obj = obj.magnify(mu)       # Apply a surface-brightness-preserving magnification.
        >>> obj = obj.rotate(theta)     # Apply a rotation.
        >>> obj = obj.shift(dx,dy)      # Shft the object in real space.
        >>> obj = obj.transform(dudx,dudy,dvdx,dvdy)    # Apply a general jacobian transformation.
        >>> obj = obj.lens(g1,g2,mu)    # Apply both a lensing shear and magnification.
        >>> obj = obj.withFlux(flux)    # Set a new flux value.
        >>> obj = obj * ratio           # Scale the surface brightness profile by some factor.

    **Access Methods**:

    There are some access methods and properties that are available for all GSObjects.
    Again, see the docstrings for each method for more details.::

        >>> obj.flux
        >>> obj.centroid
        >>> obj.nyquist_scale
        >>> obj.stepk
        >>> obj.maxk
        >>> obj.has_hard_edges
        >>> obj.is_axisymmetric
        >>> obj.is_analytic_x
        >>> obj.is_analytic_k
        >>> obj.xValue(x,y) or obj.xValue(pos)
        >>> obj.kValue(kx,ky) os obj.kValue(kpos)

    Most subclasses have additional methods that are available for values that are particular to
    that specific surface brightness profile.  e.g. ``sigma = gauss.sigma``.  However, note
    that class-specific methods are not available after performing one of the above transforming
    operations.::

        >>> gal = galsim.Gaussian(sigma=5)
        >>> gal = gal.shear(g1=0.2, g2=0.05)
        >>> sigma = gal.sigma               # This will raise an exception.

    It is however possible to access the original object that was transformed via the
    ``original`` attribute.::

        >>> sigma = gal.original.sigma      # This works.

    No matter how many transformations are performed, the ``original`` attribute will contain the
    _original_ object (not necessarily the most recent ancestor).

    **Drawing Methods**:

    The main thing to do with a GSObject once you have built it is to draw it onto an image.
    There are two methods that do this.  In both cases, there are lots of optional parameters.
    See the docstrings for these methods for more details.::

        >>> image = obj.drawImage(...)
        >>> kimage = obj.drawKImage(...)

    There two attributes that may be available for a GSObject.

    Attributes:
        original:   This was mentioned above as a way to access the original object that has
                    been transformed by one of the transforming methods.
        noise:      Some types, like `RealGalaxy`, set this attribute to be the intrinsic noise that
                    is already inherent in the profile and will thus be present when you draw the
                    object.  The noise is propagated correctly through the various transforming
                    methods, as well as convolutions and flux rescalings.  Note that the ``noise``
                    attribute can be set directly by users even for GSObjects that do not naturally
                    have one. The typical use for this attribute is to use it to whiten the noise in
                    the image after drawing.  See `BaseCorrelatedNoise` for more details.

    **GSParams**:

    All GSObject classes take an optional ``gsparams`` argument, so we document that feature here.
    For all documentation about the specific derived classes, please see the docstring for each
    one individually.

    The ``gsparams`` argument can be used to specify various numbers that govern the tradeoff
    between accuracy and speed for the calculations made in drawing a GSObject.  The numbers are
    encapsulated in a class called `GSParams`, and the user should make careful choices whenever
    they opt to deviate from the defaults.  For more details about the parameters and their default
    values, please see the docstring of the `GSParams` class.

    For example, let's say you want to do something that requires an FFT larger than 4096 x 4096
    (and you have enough memory to handle it!).  Then you can create a new `GSParams` object with a
    larger ``maximum_fft_size`` and pass that to your GSObject on construction::

        >>> gal = galsim.Sersic(n=4, half_light_radius=4.3)
        >>> psf = galsim.Moffat(beta=3, fwhm=2.85)
        >>> conv = galsim.Convolve([gal,psf])
        >>> im = galsim.Image(1000,1000, scale=0.02)        # Note the very small pixel scale!
        >>> im = conv.drawImage(image=im)                   # This uses the default GSParams.
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "galsim/gsobject.py", line 1666, in drawImage
            added_photons = prof.drawFFT(draw_image, add)
          File "galsim/gsobject.py", line 1877, in drawFFT
            kimage, wrap_size = self.drawFFT_makeKImage(image)
          File "galsim/gsobject.py", line 1802, in drawFFT_makeKImage
            raise GalSimFFTSizeError("drawFFT requires an FFT that is too large.", Nk)
        galsim.errors.GalSimFFTSizeError: drawFFT requires an FFT that is too large.
        The required FFT size would be 12288 x 12288, which requires 3.38 GB of memory.
        If you can handle the large FFT, you may update gsparams.maximum_fft_size.
        >>> big_fft_params = galsim.GSParams(maximum_fft_size=12300)
        >>> conv = galsim.Convolve([gal,psf],gsparams=big_fft_params)
        >>> im = conv.drawImage(image=im)                   # Now it works (but is slow!)
        >>> im.write('high_res_sersic.fits')

    Note that for compound objects such as `Convolution` or `Sum`, not all `GSParams` can be
    changed when the compound object is created.  In the example given here, it is possible to
    change parameters related to the drawing, but not the Fourier space parameters for the
    components that go into the `Convolution`.  To get better sampling in Fourier space,
    for example, the ``gal`` and/or ``psf`` should be created with ``gsparams`` that have a
    non-default value of ``folding_threshold``.  This statement applies to the threshold and
    accuracy parameters.
    """
    _gsparams_opt = { 'minimum_fft_size' : int,
                      'maximum_fft_size' : int,
                      'folding_threshold' : float,
                      'stepk_minimum_hlr' : float,
                      'maxk_threshold' : float,
                      'kvalue_accuracy' : float,
                      'xvalue_accuracy' : float,
                      'realspace_relerr' : float,
                      'realspace_abserr' : float,
                      'integration_relerr' : float,
                      'integration_abserr' : float,
                      'shoot_accuracy' : float,
                      'allowed_flux_variation' : float,
                      'range_division_for_extrema' : int,
                      'small_fraction_of_flux' : float
                    }
    def __init__(self):
        raise NotImplementedError("The GSObject base class should not be instantiated directly.")

    # Note: subclasses are expected to define the following attributes or properties:
    #
    # Required for all profiles:
    #
    #     _flux (the object's flux, natch)
    #     _gsparams (use GSParams.check(None) if you just want the default)
    #     _stepk (the sampling in k space necessary to avoid folding of image in x space)
    #     _maxk (the value of k beyond which aliasing can be neglected)
    #     _has_hard_edges (true if should use real-space convolution with another hard edge profile)
    #     _is_axisymmetric (true if f(x,y) = f(r))
    #     _is_analytic_x (true if _xValue and _drawReal are implemented)
    #     _is_analytic_k (true if _kValue and _drawKImage are implemented)
    #
    # Required and typically class attributes, but there are defaults in each case:
    #
    #     _req_params (dict of required config parameters: name : type, default: {})
    #     _opt_params (dict of optional config parameters: name : type, default: {})
    #     _single_params (list of dicts for parameters where exactly one of several is required,
    #                    default: [])
    #     _takes_rng (bool specifying whether rng is an input parameter, default: False)
    #
    # Optional
    #
    #     _centroid (default = PositionD(0,0), which is often the right value)
    #     _positive_flux (default = _flux + _negative_flux)
    #     _negative_flux (default = 0; note: this should be absolute value of the negative flux)
    #     _max_sb (default 1.e500, which in this context is equivalent to "unknown")
    #     _noise (default None)
    #
    # In addition, subclasses should typically define most of the following methods.
    # The default in each case is to raise a NotImplementedError, so if you cannot implement one,
    # you may simply not define it.
    #
    #     _xValue(self, pos)
    #     _kValue(self, kpos)
    #     _drawReal(self, image)
    #     _shoot(self, photons, rng):
    #     _drawKImage(self, image)
    #
    # Required for real-space convolution
    #
    #     _sbp which must be an attribute or property providing a C++-layer SBProfile instance.
    #
    # Note that most objects don't need to implement real-space convolution, so use of a C++-layer
    # SBProfile sub-class is usually only an implementation detail to improve efficiency.
    #
    # TODO: For now, _sbp is also required for transformations, but this is expected to be
    #       addressed in a future PR.

    @property
    def flux(self):
        """The flux of the profile.
        """
        return self._flux

    @property
    def gsparams(self):
        """A `GSParams` object that sets various parameters relevant for speed/accuracy trade-offs.
        """
        return self._gsparams

    @property
    def maxk(self):
        """The value of k beyond which aliasing can be neglected.
        """
        return self._maxk

    @property
    def stepk(self):
        """The sampling in k space necessary to avoid folding of image in x space.
        """
        return self._stepk

    @property
    def nyquist_scale(self):
        """The pixel spacing that does not alias maxk.
        """
        return math.pi / self.maxk

    @property
    def has_hard_edges(self):
        """Whether there are any hard edges in the profile, which would require very small k
        spacing when working in the Fourier domain.
        """
        return self._has_hard_edges

    @property
    def is_axisymmetric(self):
        """Whether the profile is axially symmetric; affects efficiency of evaluation.
        """
        return self._is_axisymmetric

    @property
    def is_analytic_x(self):
        """Whether the real-space values can be determined immediately at any position without
        requiring a Discrete Fourier Transform.
        """
        return self._is_analytic_x

    @property
    def is_analytic_k(self):
        """Whether the k-space values can be determined immediately at any position without
        requiring a Discrete Fourier Transform.
        """
        return self._is_analytic_k

    @property
    def centroid(self):
        """The (x, y) centroid of an object as a `PositionD`.
        """
        return self._centroid

    @lazy_property
    def _centroid(self):
        # Most profiles are centered at 0,0, so make this the default.
        return _PositionD(0,0)

    @property
    def positive_flux(self):
        """The expectation value of flux in positive photons.

        Some profiles, when rendered with photon shooting, need to shoot both positive- and
        negative-flux photons.  For such profiles, this method returns the total flux
        of the positive-valued photons.

        For profiles that don't have this complication, this is equivalent to getFlux().

        It should be generally true that ``obj.positive_flux - obj.negative_flux`` returns the same
        thing as ``obj.flux``.  Small difference may accrue from finite numerical accuracy in
        cases involving lookup tables, etc.
        """
        return self._positive_flux

    @property
    def negative_flux(self):
        """Returns the expectation value of flux in negative photons.

        Some profiles, when rendered with photon shooting, need to shoot both positive- and
        negative-flux photons.  For such profiles, this method returns the total absolute flux
        of the negative-valued photons (i.e. as a positive value).

        For profiles that don't have this complication, this returns 0.

        It should be generally true that ``obj.positive_flux - obj.negative_flux`` returns the same
        thing as ``obj.flux``.  Small difference may accrue from finite numerical accuracy in
        cases involving lookup tables, etc.
        """
        return self._negative_flux

    @lazy_property
    def _positive_flux(self):
        # The usual case.
        return self.flux + self._negative_flux

    @lazy_property
    def _negative_flux(self):
        # The usual case.
        return 0.

    @lazy_property
    def _flux_per_photon(self):
        # The usual case.
        return 1.

    def _calculate_flux_per_photon(self):
        # If negative_flux is overriden, then _flux_per_photon should be overridden as well
        # to return this calculation.
        posflux = self.positive_flux
        negflux = self.negative_flux
        eta = negflux / (posflux + negflux)
        return 1.-2.*eta

    @property
    def max_sb(self):
        """An estimate of the maximum surface brightness of the object.

        Some profiles will return the exact peak SB, typically equal to the value of
        obj.xValue(obj.centroid).  However, not all profiles (e.g. Convolution) know how to
        calculate this value without just drawing the image and checking what the maximum value is.
        Clearly, this would be inefficient, so in these cases, some kind of estimate is returned,
        which will generally be conservative on the high side.

        This routine is mainly used by the photon shooting process, where an overestimate of
        the maximum surface brightness is acceptable.

        Note, for negative-flux profiles, this will return the absolute value of the most negative
        surface brightness.  Technically, it is an estimate of the maximum deviation from zero,
        rather than the maximum value.  For most profiles, these are the same thing.
        """
        return self._max_sb

    @lazy_property
    def _max_sb(self):
        # The way this is used, overestimates are conservative.
        # So the default value of 1.e500 will skip the optimization involving the maximum sb.
        return 1.e500

    @property
    def noise(self):
        """An estimate of the noise already in the profile.

        Some profiles have some noise already in their definition.  E.g. those that come from
        observations of galaxies in real data.  In GalSim, `RealGalaxy` objects are an example of
        this.  In these cases, the noise attribute gives an estimate of the Noise object that
        would generate noise consistent with that already in the profile.

        It is permissible to attach a noise estimate to an existing object with::

            >>> obj.noise = noise    # Some BaseNoise instance
        """
        return self._noise

    @noise.setter
    def noise(self, n):
        # We allow the user to set the noise with obj.noise = n
        self._noise = n

    @lazy_property
    def _noise(self):
        # Most profiles don't have any noise.
        return None

    # a few definitions for using GSObjects as duck-typed ChromaticObjects
    @property
    def separable(self): return True
    @property
    def interpolated(self): return False
    @property
    def deinterpolated(self): return self
    @property
    def SED(self):
        from .sed import SED
        return SED(self.flux, 'nm', '1')
    @property
    def spectral(self): return False
    @property
    def dimensionless(self): return True
    @property
    def wave_list(self): return np.array([], dtype=float)
    @property
    def redshift(self): return getattr(self, '_redshift', 0.)

    # Also need these methods to duck-type as a ChromaticObject
    def evaluateAtWavelength(self, wave):
        return self
    def _approxWavelength(self, wave):
        return wave, self

    # Make op+ of two GSObjects work to return an Add object
    # Note: we don't define __iadd__ and similar.  Let python handle this automatically
    # to make obj += obj2 be equivalent to obj = obj + obj2.
    def __add__(self, other):
        """Add two GSObjects.

        Equivalent to Add(self, other)
        """
        from .sum import Add
        return Add([self, other])

    # op- is unusual, but allowed.  It subtracts off one profile from another.
    def __sub__(self, other):
        """Subtract two GSObjects.

        Equivalent to Add(self, -1 * other)
        """
        from .sum import Add
        return Add([self, (-1. * other)])

    # Make op* work to adjust the flux of an object
    def __mul__(self, other):
        """Scale the flux of the object by the given factor.

        obj * flux_ratio is equivalent to obj.withScaledFlux(flux_ratio)

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location scaled by the given amount.

        You can also multiply by an `SED`, which will create a `ChromaticObject` where the `SED`
        acts like a wavelength-dependent ``flux_ratio``.
        """
        return self.withScaledFlux(other)

    def __rmul__(self, other):
        """Equivalent to obj * other.  See `__mul__` for details."""
        return self.__mul__(other)

    # Likewise for op/
    def __div__(self, other):
        """Equivalent to obj * (1/other).  See `__mul__` for details."""
        return self * (1. / other)

    __truediv__ = __div__

    def __neg__(self):
        return -1. * self

    # Some calculations that can be done for all GSObjects.
    def calculateHLR(self, size=None, scale=None, centroid=None, flux_frac=0.5):
        """Returns the half-light radius of the object.

        If the profile has a half_light_radius attribute, it will just return that, but in the
        general case, we draw the profile and estimate the half-light radius directly.

        This function (by default at least) is only accurate to a few percent, typically.
        Possibly worse depending on the profile being measured.  If you care about a high
        precision estimate of the half-light radius, the accuracy can be improved using the
        optional parameter scale to change the pixel scale used to draw the profile.

        The default scale is half the Nyquist scale, which were found to produce results accurate
        to a few percent on our internal tests.  Using a smaller scale will be more accurate at
        the expense of speed.

        In addition, you can optionally specify the size of the image to draw. The default size is
        None, which means `drawImage` will choose a size designed to contain around 99.5% of the
        flux.  This is overkill for this calculation, so choosing a smaller size than this may
        speed up this calculation somewhat.

        Also, while the name of this function refers to the half-light radius, in fact it can also
        calculate radii that enclose other fractions of the light, according to the parameter
        ``flux_frac``.  E.g. for r90, you would set flux_frac=0.90.

        The default scale should usually be acceptable for things like testing that a galaxy
        has a reasonable resolution, but they should not be trusted for very fine grain
        discriminations.

        Parameters:
            size:           If given, the stamp size to use for the drawn image. [default: None,
                            which will let `drawImage` choose the size automatically]
            scale:          If given, the pixel scale to use for the drawn image. [default:
                            0.5 * self.nyquist_scale]
            centroid:       The position to use for the centroid. [default: self.centroid]
            flux_frac:      The fraction of light to be enclosed by the returned radius.
                            [default: 0.5]

        Returns:
            an estimate of the half-light radius in physical units
        """
        try:
            # It there is a half_light_radius attribute, use that.
            return self.half_light_radius
        except (AttributeError, GalSimError):
            # Otherwise, or (e.g. with Airy where it is only implemented for obscuration=0)
            # if there is an error trying to use it, then keep going with this calculation.
            pass

        if scale is None:
            scale = self.nyquist_scale * 0.5

        if centroid is None:
            centroid = self.centroid

        # Draw the image.  Note: need a method that integrates over pixels to get flux right.
        # The offset is to make all the rsq values different to help the precision a bit.
        offset = _PositionD(0.2, 0.33)
        im = self.drawImage(nx=size, ny=size, scale=scale, offset=offset, dtype=float)

        center = im.true_center + offset + centroid/scale
        return im.calculateHLR(center=center, flux=self.flux, flux_frac=flux_frac)

    def calculateMomentRadius(self, size=None, scale=None, centroid=None, rtype='det'):
        """Returns an estimate of the radius based on unweighted second moments.

        The second moments are defined as:

        Q_ij = int( I(x,y) i j dx dy ) / int( I(x,y) dx dy )
        where i,j may be either x or y.

        If I(x,y) is a Gaussian, then T = Tr(Q) = Qxx + Qyy = 2 sigma^2.  Thus, one reasonable
        choice for a "radius" for an arbitrary profile is sqrt(T/2).

        In addition, det(Q) = sigma^4.  So another choice for an arbitrary profile is det(Q)^1/4.

        This routine can return either of these measures according to the value of the ``rtype``
        parameter.  ``rtype='trace'`` will cause it to return sqrt(T/2).  ``rtype='det'`` will cause
        it to return det(Q)^1/4.  And ``rtype='both'`` will return a tuple with both values.

        Note that for the special case of a Gaussian profile, no calculation is necessary, and
        the ``sigma`` attribute will be used in both cases.  In the limit as scale->0, this
        function will return the same value, but because finite pixels are drawn, the results
        will not be precisely equal for real use cases.  The approximation being made is that
        the integral of I(x,y) i j dx dy over each pixel can be approximated as
        int(I(x,y) dx dy) * i_center * j_center.

        This function (by default at least) is only accurate to a few percent, typically.
        Possibly worse depending on the profile being measured.  If you care about a high
        precision estimate of the radius, the accuracy can be improved using the optional
        parameters size and scale to change the size and pixel scale used to draw the profile.

        The default is to use the the Nyquist scale for the pixel scale and let `drawImage`
        choose a size for the stamp that will enclose at least 99.5% of the flux.  These
        were found to produce results accurate to a few percent on our internal tests.
        Using a smaller scale and larger size will be more accurate at the expense of speed.

        The default parameters should usually be acceptable for things like testing that a galaxy
        has a reasonable resolution, but they should not be trusted for very fine grain
        discriminations.  For a more accurate estimate, see galsim.hsm.FindAdaptiveMom.

        Parameters:
            size:           If given, the stamp size to use for the drawn image. [default: None,
                            which will let `drawImage` choose the size automatically]
            scale:          If given, the pixel scale to use for the drawn image. [default:
                            self.nyquist_scale]
            centroid:       The position to use for the centroid. [default: self.centroid]
            rtype:          There are three options for this parameter:
                            - 'trace' means return sqrt(T/2)
                            - 'det' means return det(Q)^1/4
                            - 'both' means return both: (sqrt(T/2), det(Q)^1/4)
                            [default: 'det']

        Returns:
            an estimate of the radius in physical units (or both estimates if rtype == 'both')
        """
        if rtype not in ('trace', 'det', 'both'):
            raise GalSimValueError("Invalid rtype.", rtype, ('trace', 'det', 'both'))

        if hasattr(self, 'sigma'):
            if rtype == 'both':
                return self.sigma, self.sigma
            else:
                return self.sigma

        if scale is None:
            scale = self.nyquist_scale

        if centroid is None:
            centroid = self.centroid

        # Draw the image.  Note: need a method that integrates over pixels to get flux right.
        im = self.drawImage(nx=size, ny=size, scale=scale, dtype=float)

        center = im.true_center + centroid/scale
        return im.calculateMomentRadius(center=center, flux=self.flux, rtype=rtype)

    def calculateFWHM(self, size=None, scale=None, centroid=None):
        """Returns the full-width half-maximum (FWHM) of the object.

        If the profile has a fwhm attribute, it will just return that, but in the general case,
        we draw the profile and estimate the FWHM directly.

        As with `calculateHLR` and `calculateMomentRadius`, this function optionally takes size and
        scale values to use for the image drawing.  The default is to use the the Nyquist scale
        for the pixel scale and let `drawImage` choose a size for the stamp that will enclose at
        least 99.5% of the flux.  These were found to produce results accurate to well below
        one percent on our internal tests, so it is unlikely that you will want to adjust
        them for accuracy.  However, using a smaller size than default could help speed up
        the calculation, since the default is usually much larger than is needed.

        Parameters:
            size:           If given, the stamp size to use for the drawn image. [default: None,
                            which will let `drawImage` choose the size automatically]
            scale:          If given, the pixel scale to use for the drawn image. [default:
                            self.nyquist_scale]
            centroid:       The position to use for the centroid. [default: self.centroid]

        Returns:
            an estimate of the full-width half-maximum in physical units
        """
        if hasattr(self, 'fwhm'):
            return self.fwhm

        if scale is None:
            scale = self.nyquist_scale

        if centroid is None:
            centroid = self.centroid

        # Draw the image.  Note: draw with method='sb' here, since the fwhm is a property of the
        # raw surface brightness profile, not integrated over pixels.
        # The offset is to make all the rsq values different to help the precision a bit.
        offset = _PositionD(0.2, 0.33)

        im = self.drawImage(nx=size, ny=size, scale=scale, offset=offset, method='sb', dtype=float)

        # Get the maximum value, assuming the maximum is at the centroid.
        if self.is_analytic_x:
            Imax = self.xValue(centroid)
        else:
            im1 = self.drawImage(nx=1, ny=1, scale=scale, method='sb', offset=-centroid/scale)
            Imax = im1(1,1)

        center = im.true_center + offset + centroid/scale
        return im.calculateFWHM(center=center, Imax=Imax)

    def xValue(self, *args, **kwargs):
        """Returns the value of the object at a chosen 2D position in real space.

        This function returns the surface brightness of the object at a particular position
        in real space.  The position argument may be provided as a `PositionD` or `PositionI`
        argument, or it may be given as x,y (either as a tuple or as two arguments).

        The object surface brightness profiles are typically defined in world coordinates, so
        the position here should be in world coordinates as well.

        Not all `GSObject` classes can use this method.  Classes like Convolution that require a
        Discrete Fourier Transform to determine the real space values will not do so for a single
        position.  Instead a GalSimError will be raised.  The xValue() method is available if and
        only if ``obj.is_analytic_x == True``.

        Users who wish to use the xValue() method for an object that is the convolution of other
        profiles can do so by drawing the convolved profile into an image, using the image to
        initialize a new `InterpolatedImage`, and then using the xValue() method for that new
        object.

        Parameters:
            position:    The position at which you want the surface brightness of the object.

        Returns:
            the surface brightness at that position.
        """
        pos = parse_pos_args(args,kwargs,'x','y')
        return self._xValue(pos)

    def _xValue(self, pos):
        """Equivalent to `xValue`, but ``pos`` must be a `galsim.PositionD` instance

        Parameters:
            pos:        The position at which you want the surface brightness of the object.

        Returns:
            the surface brightness at that position.
        """
        raise NotImplementedError("%s does not implement xValue"%self.__class__.__name__)

    def kValue(self, *args, **kwargs):
        """Returns the value of the object at a chosen 2D position in k space.

        This function returns the amplitude of the fourier transform of the surface brightness
        profile at a given position in k space.  The position argument may be provided as a
        `Position` argument, or it may be given as kx,ky (either as a tuple or as two arguments).

        Technically, kValue() is available if and only if the given obj has ``obj.is_analytic_k
        == True``, but this is the case for all `GSObject` classes currently, so that should never
        be an issue (unlike for `xValue`).

        Parameters:
            position:    The position in k space at which you want the fourier amplitude.

        Returns:
            the amplitude of the fourier transform at that position.
        """
        kpos = parse_pos_args(args,kwargs,'kx','ky')
        return self._kValue(kpos)

    def _kValue(self, kpos):  # pragma: no cover  (all our classes override this)
        """Equivalent to `kValue`, but ``kpos`` must be a `galsim.PositionD` instance.
        """
        raise NotImplementedError("%s does not implement kValue"%self.__class__.__name__)

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given `GSParams`.
        """
        # Note to developers: objects that wrap other objects should override this in order
        # to apply the new gsparams to the components.
        # This implementation relies on getstate/setstate clearing out any _sbp or similar
        # attribute that depends on the details of gsparams.  If there are stored calculations
        # aside from these, you should also clear them as well, or update them.
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        return ret

    def withFlux(self, flux):
        """Create a version of the current object with a different flux.

        This function is equivalent to ``obj.withScaledFlux(flux / obj.flux)``.

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location rescaled such that the total flux will be
        the given value.  Note that if ``flux`` is an `SED`, the return value will be a
        `ChromaticObject` with specified `SED`.

        Parameters:
            flux:       The new flux for the object.

        Returns:
            the object with the new flux
        """
        return self.withScaledFlux(flux / self.flux)

    def withScaledFlux(self, flux_ratio):
        """Create a version of the current object with the flux scaled by the given ``flux_ratio``.

        This function is equivalent to ``obj.withFlux(flux_ratio * obj.flux)``.  Indeed, withFlux()
        is implemented in terms of this one.

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location scaled by the given amount.  If ``flux_ratio`` is an
        `SED`, then the returned object is a `ChromaticObject` with the `SED` multiplied by
        its current ``flux``.

        Note that in this case the ``flux`` attribute of the `GSObject` being scaled gets
        interpreted as being dimensionless, instead of having its normal units of [photons/s/cm^2].
        The photons/s/cm^2 units are (optionally) carried by the `SED` instead, or even left out
        entirely if the `SED` is dimensionless itself (see discussion in the `ChromaticObject`
        docstring).  The `GSObject` ``flux`` attribute *does* still contribute to the
        `ChromaticObject` normalization, though.  For example, the following are equivalent::

            >>> chrom_obj = gsobj.withScaledFlux(sed * 3.0)
            >>> chrom_obj2 = (gsobj * 3.0).withScaledFlux(sed)

        An equivalent, and usually simpler, way to effect this scaling is::

            >>> obj = obj * flux_ratio

        Parameters:
            flux_ratio:     The ratio by which to rescale the flux of the object when creating a new
                            one.

        Returns:
            the object with the new flux.
        """
        from .sed import SED
        from .transform import Transform
        # Prohibit non-SED callable flux_ratio here as most likely an error.
        if hasattr(flux_ratio, '__call__') and not isinstance(flux_ratio, SED):
            raise TypeError('callable flux_ratio must be an SED.')

        if flux_ratio == 1:
            return self
        else:
            return Transform(self, flux_ratio=flux_ratio)

    def expand(self, scale):
        """Expand the linear size of the profile by the given ``scale`` factor, while preserving
        surface brightness.

        e.g. ``half_light_radius`` <-- ``half_light_radius * scale``

        This doesn't correspond to either of the normal operations one would typically want to do to
        a galaxy.  The functions dilate() and magnify() are the more typical usage.  But this
        function is conceptually simple.  It rescales the linear dimension of the profile, while
        preserving surface brightness.  As a result, the flux will necessarily change as well.

        See dilate() for a version that applies a linear scale factor while preserving flux.

        See magnify() for a version that applies a scale factor to the area while preserving surface
        brightness.

        Parameters:
            scale:      The factor by which to scale the linear dimension of the object.

        Returns:
            the expanded object.
        """
        from .transform import Transform
        return Transform(self, jac=[scale, 0., 0., scale])

    def dilate(self, scale):
        """Dilate the linear size of the profile by the given ``scale`` factor, while preserving
        flux.

        e.g. ``half_light_radius`` <-- ``half_light_radius * scale``

        See expand() and magnify() for versions that preserve surface brightness, and thus
        changes the flux.

        Parameters:
            scale:      The linear rescaling factor to apply.

        Returns:
            the dilated object.
        """
        from .transform import Transform
        # equivalent to self.expand(scale) * (1./scale**2)
        return Transform(self, jac=[scale, 0., 0., scale], flux_ratio=scale**-2)

    def magnify(self, mu):
        """Create a version of the current object with a lensing magnification applied to it,
        scaling the area and flux by ``mu`` at fixed surface brightness.

        This process applies a lensing magnification mu, which scales the linear dimensions of the
        image by the factor sqrt(mu), i.e., ``half_light_radius`` <--
        ``half_light_radius * sqrt(mu)`` while increasing the flux by a factor of mu.  Thus,
        magnify() preserves surface brightness.

        See dilate() for a version that applies a linear scale factor while preserving flux.

        See expand() for a version that applies a linear scale factor while preserving surface
        brightness.

        Parameters:
            mu:     The lensing magnification to apply.

        Returns:
            the magnified object.
        """
        return self.expand(math.sqrt(mu))

    def shear(self, *args, **kwargs):
        """Create a version of the current object with an area-preserving shear applied to it.

        The arguments may be either a `Shear` instance or arguments to be used to initialize one.

        For more details about the allowed keyword arguments, see the `Shear` docstring.

        The shear() method precisely preserves the area.  To include a lensing distortion with
        the appropriate change in area, either use shear() with magnify(), or use lens(), which
        combines both operations.

        Parameters:
            shear:      The `Shear` to be applied. Or, as described above, you may instead supply
                        parameters do construct a shear directly.  eg. ``obj.shear(g1=g1,g2=g2)``.

        Returns:
            the sheared object.
        """
        from .transform import Transform
        from .shear import Shear
        if len(args) == 1:
            if kwargs:
                raise TypeError("Error, gave both unnamed and named arguments to GSObject.shear!")
            if not isinstance(args[0], Shear):
                raise TypeError("Error, unnamed argument to GSObject.shear is not a Shear!")
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Error, too many unnamed arguments to GSObject.shear!")
        elif len(kwargs) == 0:
            raise TypeError("Error, shear argument is required")
        else:
            shear = Shear(**kwargs)
        return Transform(self, shear.getMatrix())

    def _shear(self, shear):
        """Equivalent to `GSObject.shear`, but without the overhead of sanity checks or other
        ways to input the ``shear`` value.

        Also, it won't propagate any noise attribute.

        Parameters:
            shear:      The `Shear` to be applied.

        Returns:
            the sheared object.
        """
        from .transform import _Transform
        return _Transform(self, shear.getMatrix())

    def lens(self, g1, g2, mu):
        """Create a version of the current object with both a lensing shear and magnification
        applied to it.

        This `GSObject` method applies a lensing (reduced) shear and magnification.  The shear must
        be specified using the g1, g2 definition of shear (see `Shear` for more details).
        This is the same definition as the outputs of the PowerSpectrum and NFWHalo classes, which
        compute shears according to some lensing power spectrum or lensing by an NFW dark matter
        halo.  The magnification determines the rescaling factor for the object area and flux,
        preserving surface brightness.

        Parameters:
            g1:         First component of lensing (reduced) shear to apply to the object.
            g2:         Second component of lensing (reduced) shear to apply to the object.
            mu:         Lensing magnification to apply to the object.  This is the factor by which
                        the solid angle subtended by the object is magnified, preserving surface
                        brightness.

        Returns:
            the lensed object.
        """
        from .transform import Transform
        from .shear import Shear
        shear = Shear(g1=g1, g2=g2)
        return Transform(self, shear.getMatrix() * math.sqrt(mu))

    def _lens(self, g1, g2, mu):
        """Equivalent to `GSObject.lens`, but without the overhead of some of the sanity checks.

        Also, it won't propagate any noise attribute.

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
        from .transform import _Transform
        shear = _Shear(g1 + 1j*g2)
        return _Transform(self, shear.getMatrix() * math.sqrt(mu))

    def rotate(self, theta):
        """Rotate this object by an `Angle` ``theta``.

        Parameters:
            theta:      Rotation angle (`Angle` object, positive means anticlockwise).

        Returns:
            the rotated object.
        """
        from .angle import Angle
        from .transform import Transform
        if not isinstance(theta, Angle):
            raise TypeError("Input theta should be an Angle")
        s, c = theta.sincos()
        return Transform(self, jac=[c, -s, s, c])

    def transform(self, dudx, dudy, dvdx, dvdy):
        """Create a version of the current object with an arbitrary Jacobian matrix transformation
        applied to it.

        This applies a Jacobian matrix to the coordinate system in which this object
        is defined.  It changes a profile defined in terms of (x,y) to one defined in
        terms of (u,v) where:

            u = dudx x + dudy y
            v = dvdx x + dvdy y

        That is, an arbitrary affine transform, but without the translation (which is
        easily effected via the `shift` method).

        Note that this function is similar to expand in that it preserves surface brightness,
        not flux.  If you want to preserve flux, you should also do::

            >>> prof *= 1./abs(dudx*dvdy - dudy*dvdx)

        Parameters:
            dudx:       du/dx, where (x,y) are the current coords, and (u,v) are the new coords.
            dudy:       du/dy, where (x,y) are the current coords, and (u,v) are the new coords.
            dvdx:       dv/dx, where (x,y) are the current coords, and (u,v) are the new coords.
            dvdy:       dv/dy, where (x,y) are the current coords, and (u,v) are the new coords.

        Returns:
            the transformed object
        """
        from .transform import Transform
        return Transform(self, jac=[dudx, dudy, dvdx, dvdy])

    def shift(self, *args, **kwargs):
        """Create a version of the current object shifted by some amount in real space.

        After this call, the caller's type will be a `GSObject`.
        This means that if the caller was a derived type that had extra methods or properties
        beyond those defined in `GSObject` (e.g. `Gaussian.sigma`), then these methods are no
        longer available.

        Note: in addition to the dx,dy parameter names, you may also supply dx,dy as a tuple,
        or as a `Position` object.

        The shift coordinates here are sky coordinates.  `GSObject` profiles are always defined in
        sky coordinates and only later (when they are drawn) is the connection to pixel coordinates
        established (via a pixel_scale or WCS).  So a shift of dx moves the object horizontally
        in the sky (e.g. west in the local tangent plane of the observation), and dy moves the
        object vertically (north in the local tangent plane).

        The units are typically arcsec, but we don't enforce that anywhere.  The units here just
        need to be consistent with the units used for any size values used by the `GSObject`.
        The connection of these units to the eventual image pixels is defined by either the
        ``pixel_scale`` or the ``wcs`` parameter of `GSObject.drawImage`.

        Note: if you want to shift the object by a set number (or fraction) of pixels in the
        drawn image, you probably want to use the ``offset`` parameter of `GSObject.drawImage`
        rather than this method.

        Parameters:
            dx:         Horizontal shift to apply.
            dy:         Vertical shift to apply.

        Alternatively, you may supply a single parameter as a `Position` instance, rather than
        the two components separately if that is more convenient.

        Parameter:
            offset:     The shift to apply, given as PositionD(dx,dy) or PositionI(dx,dy)

        Returns:
            the shifted object.
        """
        from .transform import Transform
        offset = parse_pos_args(args, kwargs, 'dx', 'dy')
        return Transform(self, offset=offset)

    def _shift(self, dx, dy):
        """Equivalent to `shift`, but without the overhead of sanity checks or option
        to give the shift as a PositionD.

        Also, it won't propagate any noise attribute.

        Parameters:
            dx:         The x-component of the shift to apply
            dy:         The y-component of the shift to apply

        Returns:
            the shifted object.
        """
        from .transform import _Transform
        new_obj = _Transform(self, offset=(dx,dy))
        return new_obj

    def atRedshift(self, redshift):
        """Create a version of the current object with a different redshift.

        For regular GSObjects, this method doesn't do anything aside from setting a ``redshift``
        attribute with the given value.  But this allows duck typing with ChromaticObjects
        where this function will adjust the SED appropriately.

        Returns:
            the object with the new redshift
        """
        from copy import copy
        ret = copy(self)
        ret._redshift = redshift
        return ret

    # Make sure the image is defined with the right size and wcs for drawImage()
    def _setup_image(self, image, nx, ny, bounds, add_to_image, dtype, center, odd=False):
        from .image import Image
        from .bounds import _BoundsI

        # If image is given, check validity of nx,ny,bounds:
        if image is not None:
            if bounds is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide bounds if image is provided", bounds=bounds, image=image)
            if nx is not None or ny is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide nx,ny if image is provided", nx=nx, ny=ny, image=image)
            if dtype is not None and image.array.dtype != dtype:
                raise GalSimIncompatibleValuesError(
                    "Cannot specify dtype != image.array.dtype if image is provided",
                    dtype=dtype, image=image)

            # Resize the given image if necessary
            if not image.bounds.isDefined():
                # Can't add to image if need to resize
                if add_to_image:
                    raise GalSimIncompatibleValuesError(
                        "Cannot add_to_image if image bounds are not defined",
                        add_to_image=add_to_image, image=image)
                N = self.getGoodImageSize(1.0)
                if odd: N += 1
                bounds = _BoundsI(1,N,1,N)
                image.resize(bounds)
            # Else use the given image as is

        # Otherwise, make a new image
        else:
            # Can't add to image if none is provided.
            if add_to_image:
                raise GalSimIncompatibleValuesError(
                    "Cannot add_to_image if image is None", add_to_image=add_to_image, image=image)
            # Use bounds or nx,ny if provided
            if bounds is not None:
                if nx is not None or ny is not None:
                    raise GalSimIncompatibleValuesError(
                        "Cannot set both bounds and (nx, ny)", nx=nx, ny=ny, bounds=bounds)
                if not bounds.isDefined():
                    raise GalSimValueError("Cannot use undefined bounds", bounds)
                image = Image(bounds=bounds, dtype=dtype)
            elif nx is not None or ny is not None:
                if nx is None or ny is None:
                    raise GalSimIncompatibleValuesError(
                        "Must set either both or neither of nx, ny", nx=nx, ny=ny)
                image = Image(nx, ny, dtype=dtype)
                if center is not None:
                    image.shift(_PositionI(np.floor(center.x+0.5-image.true_center.x),
                                           np.floor(center.y+0.5-image.true_center.y)))
            else:
                N = self.getGoodImageSize(1.0)
                if odd: N += 1
                image = Image(N, N, dtype=dtype)
                if center is not None:
                    image.setCenter(_PositionI(np.ceil(center.x), np.ceil(center.y)))

        return image

    def _local_wcs(self, wcs, image, offset, center, use_true_center, new_bounds):
        # Get the local WCS at the location of the object.

        if wcs._isUniform:
            return wcs.local()
        elif image is None:
            bounds = new_bounds
        else:
            bounds = image.bounds
        if not bounds.isDefined():
            raise GalSimIncompatibleValuesError(
                "Cannot provide non-local wcs with automatically sized image",
                wcs=wcs, image=image, bounds=new_bounds)
        elif center is not None:
            obj_cen = center
        elif use_true_center:
            obj_cen = bounds.true_center
        else:
            obj_cen = bounds.center
            # Convert from PositionI to PositionD
            obj_cen = _PositionD(obj_cen.x, obj_cen.y)
        # _parse_offset has already turned offset=None into PositionD(0,0), so it is safe to add.
        obj_cen += offset
        return wcs.local(image_pos=obj_cen)

    def _parse_offset(self, offset):
        if offset is None:
            return _PositionD(0,0)
        elif isinstance(offset, Position):
            return _PositionD(offset.x, offset.y)
        else:
            # Let python raise the appropriate exception if this isn't valid.
            return _PositionD(offset[0], offset[1])

    def _parse_center(self, center):
        # Almost the same as _parse_offset, except we leave it as None in that case.
        if center is None:
            return None
        elif isinstance(center, Position):
            return _PositionD(center.x, center.y)
        else:
            # Let python raise the appropriate exception if this isn't valid.
            return _PositionD(center[0], center[1])

    def _get_new_bounds(self, image, nx, ny, bounds, center):
        from .bounds import BoundsI
        if image is not None and image.bounds.isDefined():
            return image.bounds
        elif nx is not None and ny is not None:
            b = BoundsI(1,nx,1,ny)
            if center is not None:
                b = b.shift(_PositionI(np.floor(center.x+0.5)-b.center.x,
                                       np.floor(center.y+0.5)-b.center.y))
            return b
        elif bounds is not None and bounds.isDefined():
            return bounds
        else:
            return BoundsI()

    def _adjust_offset(self, new_bounds, offset, center, use_true_center):
        # Note: this assumes self is in terms of image coordinates.
        if center is not None:
            if new_bounds.isDefined():
                offset += center - new_bounds.center
            else:
                # Then will be created as even sized image.
                offset += _PositionD(center.x-np.ceil(center.x), center.y-np.ceil(center.y))
        elif use_true_center:
            # For even-sized images, the SBProfile draw function centers the result in the
            # pixel just up and right of the real center.  So shift it back to make sure it really
            # draws in the center.
            # Also, remember that numpy's shape is ordered as [y,x]
            dx = offset.x
            dy = offset.y
            shape = new_bounds.numpyShape()
            if shape[1] % 2 == 0: dx -= 0.5
            if shape[0] % 2 == 0: dy -= 0.5
            offset = _PositionD(dx,dy)
        return offset

    def _determine_wcs(self, scale, wcs, image, default_wcs=None):
        from .wcs import BaseWCS, PixelScale
        # Determine the correct wcs given the input scale, wcs and image.
        if wcs is not None:
            if scale is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both wcs and scale", wcs=wcs, scale=scale)
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs must be a BaseWCS instance")
            if image is not None: image.wcs = None
        elif scale is not None:
            wcs = PixelScale(scale)
            if image is not None: image.wcs = None
        elif image is not None and image.wcs is not None:
            wcs = image.wcs

        # If the input scale <= 0, or wcs is still None at this point, then use the Nyquist scale:
        if wcs is None or (wcs._isPixelScale and wcs.scale <= 0):
            if default_wcs is None:
                wcs = PixelScale(self.nyquist_scale)
            else:
                wcs = default_wcs

        return wcs

    def _prepareDraw(self):
        # Do any work that was postponed until drawImage.
        pass

    def drawImage(self, image=None, nx=None, ny=None, bounds=None, scale=None, wcs=None, dtype=None,
                  method='auto', area=1., exptime=1., gain=1., add_to_image=False,
                  center=None, use_true_center=True, offset=None,
                  n_photons=0., rng=None, max_extra_noise=0.,
                  poisson_flux=None, sensor=None, photon_ops=(), n_subsample=3, maxN=None,
                  save_photons=False, bandpass=None, setup_only=False,
                  surface_ops=None):
        """Draws an `Image` of the object.

        The drawImage() method is used to draw an `Image` of the current object using one of several
        possible rendering methods (see below).  It can create a new `Image` or can draw onto an
        existing one if provided by the ``image`` parameter.  If the ``image`` is given, you can
        also optionally add to the given `Image` if ``add_to_image = True``, but the default is to
        replace the current contents with new values.

        **Providing an input image**:

        Note that if you provide an ``image`` parameter, it is the image onto which the profile
        will be drawn.  The provided image *will be modified*.  A reference to the same image
        is also returned to provide a parallel return behavior to when ``image`` is ``None``
        (described above).

        This option is useful in practice because you may want to construct the image first and
        then draw onto it, perhaps multiple times. For example, you might be drawing onto a
        subimage of a larger image. Or you may want to draw different components of a complex
        profile separately.  In this case, the returned value is typically ignored.  For example::

            >>> im1 = bulge.drawImage()
            >>> im2 = disk.drawImage(image=im1, add_to_image=True)
            >>> assert im1 is im2

            >>> full_image = galsim.Image(2048, 2048, scale=pixel_scale)
            >>> b = galsim.BoundsI(x-32, x+32, y-32, y+32)
            >>> stamp = obj.drawImage(image = full_image[b])
            >>> assert (stamp.array == full_image[b].array).all()

        **Letting drawImage create the image for you**:

        If drawImage() will be creating the image from scratch for you, then there are several ways
        to control the size of the new image.  If the ``nx`` and ``ny`` keywords are present, then
        an image with these numbers of pixels on a side will be created.  Similarly, if the ``bounds``
        keyword is present, then an image with the specified bounds will be created.  Note that it
        is an error to provide an existing `Image` when also specifying ``nx``, ``ny``, or
        ``bounds``.  In the absence of ``nx``, ``ny``, and ``bounds``, drawImage will decide a good
        size to use based on the size of the object being drawn.  Basically, it will try to use an
        area large enough to include at least 99.5% of the flux.

        .. note::
            This value 0.995 is really ``1 - folding_threshold``.  You can change the value of
            ``folding_threshold`` for any object via `GSParams`.

        You can set the pixel scale of the constructed image with the ``scale`` parameter, or set
        a WCS function with ``wcs``.  If you do not provide either ``scale`` or ``wcs``, then
        drawImage() will default to using the Nyquist scale for the current object.

        You can also set the data type used in the new `Image` with the ``dtype`` parameter that has
        the same options as for the `Image` constructor.

        **The drawing "method"**:

        There are several different possible methods drawImage() can use for rendering the image.
        This is set by the ``method`` parameter.  The options are:

        auto
                This is the default, which will normally be equivalent to 'fft'.  However,
                if the object being rendered is simple (no convolution) and has hard edges
                (e.g. a Box or a truncated Moffat or Sersic), then it will switch to
                'real_space', since that is often both faster and more accurate in these
                cases (due to ringing in Fourier space).
        fft
                The integration of the light within each pixel is mathematically equivalent
                to convolving by the pixel profile (a `Pixel` object) and sampling the result
                at the centers of the pixels.  This method will do that convolution using
                a discrete Fourier transform.  Furthermore, if the object (or any component
                of it) has been transformed via shear(), dilate(), etc., then these
                transformations are done in Fourier space as well.
        real_space
                This uses direct integrals (using the Gauss-Kronrod-Patterson algorithm)
                in real space for the integration over the pixel response.  It is usually
                slower than the 'fft' method, but if the profile has hard edges that cause
                ringing in Fourier space, it can be faster and/or more accurate.  If you
                use 'real_space' with something that is already a Convolution, then this
                will revert to 'fft', since the double convolution that is required to also
                handle the pixel response is far too slow to be practical using real-space
                integrals.
        phot
                This uses a technique called photon shooting to render the image.
                Essentially, the object profile is taken as a probability distribution
                from which a finite number of photons are "shot" onto the image.  Each
                photon's flux gets added to whichever pixel the photon hits.  This process
                automatically accounts for the integration of the light over the pixel
                area, since all photons that hit any part of the pixel are counted.
                Convolutions and transformations are simple geometric processes in this
                framework.  However, there are two caveats with this method: (1) the
                resulting image will have Poisson noise from the finite number of photons,
                and (2) it is not available for all object types (notably anything that
                includes a Deconvolution).
        no_pixel
                Instead of integrating over the pixels, this method will sample the profile
                at the centers of the pixels and multiply by the pixel area.  If there is
                a convolution involved, the choice of whether this will use an FFT or
                real-space calculation is governed by the ``real_space`` parameter of the
                Convolution class.  This method is the appropriate choice if you are using
                a PSF that already includes a convolution by the pixel response.  For
                example, if you are using a PSF from an observed image of a star, then it
                has already been convolved by the pixel, so you would not want to do so
                again.  Note: The multiplication by the pixel area gets the flux
                normalization right for the above use case.  cf. ``method = 'sb'``.
        sb
                This is a lot like 'no_pixel', except that the image values will simply be
                the sampled object profile's surface brightness, not multiplied by the
                pixel area.  This does not correspond to any real observing scenario, but
                it could be useful if you want to view the surface brightness profile of an
                object directly, without including the pixel integration.

        The 'phot' method has a few extra parameters that adjust how it functions.  The total
        number of photons to shoot is normally calculated from the object's flux.  This flux is
        taken to be given in photons/cm^2/s, so for most simple profiles, this times area * exptime
        will equal the number of photons shot.  (See the discussion in Rowe et al, 2015, for why
        this might be modified for `InterpolatedImage` and related profiles.)  However, you can
        manually set a different number of photons with ``n_photons``.  You can also set
        ``max_extra_noise`` to tell drawImage() to use fewer photons than normal (and so is faster)
        such that no more than that much extra noise is added to any pixel.  This is particularly
        useful if you will be subsequently adding sky noise, and you can thus tolerate more noise
        than the normal number of photons would give you, since using fewer photons is of course
        faster.  Finally, the default behavior is to have the total flux vary as a Poisson random
        variate, which is normally appropriate with photon shooting.  But you can turn this off with
        ``poisson_flux=False``.  It also defaults to False if you set an explicit value for
        ``n_photons``.

        Given the periodicity implicit in the use of FFTs, there can occasionally be artifacts due
        to wrapping at the edges, particularly for objects that are quite extended (e.g., due to
        the nature of the radial profile). See `GSParams` for parameters that you can use to reduce
        the level of these artifacts, in particular ``folding_threshold`` may be helpful if you see
        such artifacts in your images.

        Setting the offset:

        The object will by default be drawn with its nominal center at the center location of the
        image.  There is thus a qualitative difference in the appearance of the rendered profile
        when drawn on even- and odd-sized images.  For a profile with a maximum at (0,0), this
        maximum will fall in the central pixel of an odd-sized image, but in the corner of the four
        central pixels of an even-sized image.  There are three parameters that can affect this
        behavior.  First, you can specify any arbitrary pixel position to center the object using
        the ``center`` parameter.  If this is None, then it will pick one of the two potential
        "centers" of the image, either ``image.true_center`` or ``image.center``.  The latter is
        an integer position, which always corresponds to the center of some pixel, which for even
        sized images won't (cannot) be the actual "true" center of the image.  You can choose which
        of these two centers you want to use with the ``use_true_center`` parameters, which
        defaults to False.  You can also arbitrarily offset the profile from the image center with
        the ``offset`` parameter to handle any aribtrary offset you want from the chosen center.
        (Typically, one would use only one of ``center`` or ``offset`` but it is permissible to use
        both.)

        Setting the overall normalization:

        Normally, the flux of the object should be equal to the sum of all the pixel values in the
        image, less some small amount of flux that may fall off the edge of the image (assuming you
        don't use ``method='sb'``).  However, you may optionally set a ``gain`` value, which
        converts between photons and ADU (so-called analog-to-digital units), the units of the
        pixel values in real images.  Normally, the gain of a CCD is in electrons/ADU, but in
        GalSim, we fold the quantum efficiency into the gain as well, so the units are photons/ADU.

        Another caveat is that, technically, flux is really in units of photons/cm^2/s, not photons.
        So if you want, you can keep track of this properly and provide an ``area`` and ``exptime``
        here. This detail is more important with chromatic objects where the `SED` is typically
        given in erg/cm^2/s/nm, so the exposure time and area are important details. With achromatic
        objects however, it is often more convenient to ignore these details and just consider the
        flux to be the total number of photons for this exposure, in which case, you would leave the
        area and exptime parameters at their default value of 1.

        On return, the image will have an attribute ``added_flux``, which will be set to the total
        flux added to the image.  This may be useful as a sanity check that you have provided a
        large enough image to catch most of the flux.  For example::

            >>> obj.drawImage(image)
            >>> assert image.added_flux > 0.99 * obj.flux

        The appropriate threshold will depend on your particular application, including what kind
        of profile the object has, how big your image is relative to the size of your object,
        whether you are keeping ``poisson_flux=True``, etc.

        The following code snippet illustrates how ``gain``, ``exptime``, ``area``, and ``method``
        can all influence the relationship between the ``flux`` attribute of a `GSObject` and
        both the pixel values and ``.added_flux`` attribute of an `Image` drawn with
        ``drawImage()``::

            >>> obj = galsim.Gaussian(fwhm=1)
            >>> obj.flux
            1.0
            >>> im = obj.drawImage()
            >>> im.added_flux
            0.9999630988657515
            >>> im.array.sum()
            0.99996305
            >>> im = obj.drawImage(exptime=10, area=10)
            >>> im.added_flux
            0.9999630988657525
            >>> im.array.sum()
            99.996315
            >>> im = obj.drawImage(exptime=10, area=10, method='sb', scale=0.5, nx=10, ny=10)
            >>> im.added_flux
            0.9999973790505298
            >>> im.array.sum()
            399.9989
            >>> im = obj.drawImage(exptime=10, area=10, gain=2)
            >>> im.added_flux
            0.9999630988657525
            >>> im.array.sum()
            49.998158

        Using a non-trivial sensor:

        Normally the sensor is modeled as an array of pixels where any photon that hits a given
        pixel is accumulated into that pixel.  The final pixel value then just reflects the total
        number of pixels that hit each sensor.  However, real sensors do not (quite) work this way.

        In real CCDs, the photons travel some distance into the silicon before converting to
        electrons.  Then the electrons diffuse laterally some amount as they are pulled by the
        electric field toward the substrate.  Finally, previous electrons that have already been
        deposited will repel subsequent electrons, both slowing down their descent, leading to
        more diffusion, and pushing them laterally toward neighboring pixels, which is called
        the brighter-fatter effect.

        Users interested in modeling this kind of effect can supply a ``sensor`` object to use
        for the accumulation step.  See `SiliconSensor` for a class that models silicon-based CCD
        sensors.

        Some related effects may need to be done to the photons at the surface layer before being
        passed into the sensor object.  For instance, the photons may need to be given appropriate
        incidence angles according to the optics of the telescope (since this matters for where the
        photons are converted to electrons).  You may also need to give the photons wavelengths
        according to the `SED` of the object.  Such steps are specified in a ``photon_ops``
        parameter, which should be a list of any such operations you wish to perform on the photon
        array before passing them to the sensor.  See `FRatioAngles` and `WavelengthSampler` for
        two examples of such photon operators.

        Since the sensor deals with photons, it is most natural to use this feature in conjunction
        with photon shooting (``method='phot'``).  However, it is allowed with FFT methods too.
        But there is a caveat one should be aware of in this case.  The FFT drawing is used to
        produce an intermediate image, which is then converted to a `PhotonArray` using the
        factory function `PhotonArray.makeFromImage`.  This assigns photon positions randomly
        within each pixel where they were drawn, which isn't always a particularly good
        approximation.

        To improve this behavior, the intermediate image is drawn with smaller pixels than the
        target image, so the photons are given positions closer to their true locations.  The
        amount of subsampling is controlled by the ``n_subsample`` parameter, which defaults to 3.
        Larger values will be more accurate at the expense of larger FFTs (i.e. slower and using
        more memory).

        Parameters:
            image:          If provided, this will be the image on which to draw the profile.
                            If ``image`` is None, then an automatically-sized `Image` will be
                            created.  If ``image`` is given, but its bounds are undefined (e.g. if
                            it was constructed with ``image = galsim.Image()``), then it will be
                            resized appropriately based on the profile's size [default: None].
            nx:             If provided and ``image`` is None, use to set the x-direction size of
                            the image.  Must be accompanied by ``ny``.
            ny:             If provided and ``image`` is None, use to set the y-direction size of
                            the image.  Must be accompanied by ``nx``.
            bounds:         If provided and ``image`` is None, use to set the bounds of the image.
            scale:          If provided, use this as the pixel scale for the image.
                            If ``scale`` is None and ``image`` is given, then take the provided
                            image's pixel scale.
                            If ``scale`` is None and ``image`` is None, then use the Nyquist scale.
                            If ``scale <= 0`` (regardless of ``image``), then use the Nyquist scale.
                            If ``scale > 0`` and ``image`` is given, then override ``image.scale``
                            with the value given as a keyword.  [default: None]
            wcs:            If provided, use this as the wcs for the image (possibly overriding any
                            existing ``image.wcs``).  At most one of ``scale`` or ``wcs`` may be
                            provided.  [default: None]
            dtype:          The data type to use for an automatically constructed image.  Only
                            valid if ``image`` is None. [default: None, which means to use
                            numpy.float32]
            method:         Which method to use for rendering the image.  See discussion above
                            for the various options and what they do. [default: 'auto']
            area:           Collecting area of telescope in cm^2.  [default: 1.]
            exptime:        Exposure time in s.  [default: 1.]
            gain:           The number of photons per ADU ("analog to digital units", the units of
                            the numbers output from a CCD).  [default: 1]
            add_to_image:   Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing.
                            Note: This requires that ``image`` be provided and that it have defined
                            bounds. [default: False]
            center:         The position on the image at which to place the nominal center of the
                            object (usually the centroid, but not necessarily).  [default: None]
            use_true_center: If ``center`` is None, then the object is normally centered at the
                            true center of the image (using the property image.true_center).
                            If you would rather use the integer center (given by image.center),
                            set this to ``False``.  [default: True]
            offset:         The offset in pixel coordinates at which to center the profile being
                            drawn relative to either ``center`` (if given) or the center of the
                            image (either the true center or integer center according to the
                            ``use_true_center`` parameter). [default: None]
            n_photons:      If provided, the number of photons to use for photon shooting.
                            If not provided (i.e. ``n_photons = 0``), use as many photons as
                            necessary to result in an image with the correct Poisson shot
                            noise for the object's flux.  For positive definite profiles, this
                            is equivalent to ``n_photons = flux``.  However, some profiles need
                            more than this because some of the shot photons are negative
                            (usually due to interpolants).
                            [default: 0]
            rng:            If provided, a random number generator to use for photon shooting,
                            which may be any kind of `BaseDeviate` object.  If ``rng`` is None, one
                            will be automatically created.  [default: None]
            max_extra_noise: If provided, the allowed extra noise in each pixel when photon
                            shooting.  This is only relevant if ``n_photons=0``, so the number of
                            photons is being automatically calculated.  In that case, if the image
                            noise is dominated by the sky background, then you can get away with
                            using fewer shot photons than the full ``n_photons = flux``.
                            Essentially each shot photon can have a ``flux > 1``, which increases
                            the noise in each pixel.  The ``max_extra_noise`` parameter specifies
                            how much extra noise per pixel is allowed because of this approximation.
                            A typical value for this might be ``max_extra_noise = sky_level / 100``
                            where ``sky_level`` is the flux per pixel due to the sky.  Note that
                            this uses a "variance" definition of noise, not a "sigma" definition.
                            [default: 0.]
            poisson_flux:   Whether to allow total object flux scaling to vary according to
                            Poisson statistics for ``n_photons`` samples when photon shooting.
                            [default: True, unless ``n_photons`` is given, in which case the default
                            is False]
            sensor:         An optional `Sensor` instance, which will be used to accumulate the
                            photons onto the image. [default: None]
            photon_ops:     A list of operators that can modify the photon array that will be
                            applied in order before accumulating the photons on the sensor.
                            [default: ()]
            n_subsample:    The number of sub-pixels per final pixel to use for fft drawing when
                            using a sensor.  The sensor step needs to know the sub-pixel positions
                            of the photons, which is lost in the fft method.  So using smaller
                            pixels for the fft step keeps some of that information, making the
                            assumption of uniform flux per pixel less bad of an approximation.
                            [default: 3]
            maxN:           Sets the maximum number of photons that will be added to the image
                            at a time.  (Memory requirements are proportional to this number.)
                            [default: None, which means no limit]
            save_photons:   If True, save the `PhotonArray` as ``image.photons``. Only valid if
                            method is 'phot' or sensor is not None.  [default: False]
            bandpass:       This parameter is ignored, but is allowed to enable duck typing
                            eqivalence between this method and the ChromaticObject.drawImage
                            method. [default: None]
            setup_only:     Don't actually draw anything on the image.  Just make sure the image
                            is set up correctly.  This is used internally by GalSim, but there
                            may be cases where the user will want the same functionality.
                            [default: False]

        Returns:
            the drawn `Image`.
        """
        from .image import Image, ImageD
        from .convolve import Convolve, Convolution, Deconvolve
        from .box import Pixel
        from .wcs import PixelScale
        from .photon_array import PhotonArray

        if surface_ops is not None:
            from .deprecated import depr
            depr('surface_ops', 2.3, 'photon_ops')
            photon_ops = surface_ops

        # Check that image is sane
        if image is not None and not isinstance(image, Image):
            raise TypeError("image is not an Image instance", image)

        # Make sure (gain, area, exptime) have valid values:
        if gain <= 0.:
            raise GalSimRangeError("Invalid gain <= 0.", gain, 0., None)
        if area <= 0.:
            raise GalSimRangeError("Invalid area <= 0.", area, 0., None)
        if exptime <= 0.:
            raise GalSimRangeError("Invalid exptime <= 0.", exptime, 0., None)

        if method not in ('auto', 'fft', 'real_space', 'phot', 'no_pixel', 'sb'):
            raise GalSimValueError("Invalid method name", method,
                                   ('auto', 'fft', 'real_space', 'phot', 'no_pixel', 'sb'))

        # Check that the user isn't convolving by a Pixel already.  This is almost always an error.
        if method == 'auto' and isinstance(self, Convolution):
            if any([ isinstance(obj, Pixel) for obj in self.obj_list ]):
                galsim_warn(
                    "You called drawImage with ``method='auto'`` "
                    "for an object that includes convolution by a Pixel.  "
                    "This is probably an error.  Normally, you should let GalSim "
                    "handle the Pixel convolution for you.  If you want to handle the Pixel "
                    "convolution yourself, you can use method=no_pixel.  Or if you really meant "
                    "for your profile to include the Pixel and also have GalSim convolve by "
                    "an _additional_ Pixel, you can suppress this warning by using method=fft.")

        # Some parameters are only relevant for method == 'phot'
        if method != 'phot' and sensor is None:
            if n_photons != 0.:
                raise GalSimIncompatibleValuesError(
                    "n_photons is only relevant for method='phot'",
                    method=method, sensor=sensor, n_photons=n_photons)
            if rng is not None:
                raise GalSimIncompatibleValuesError(
                    "rng is only relevant for method='phot'",
                    method=method, sensor=sensor, rng=rng)
            if max_extra_noise != 0.:
                raise GalSimIncompatibleValuesError(
                    "max_extra_noise is only relevant for method='phot'",
                    method=method, sensor=sensor, max_extra_noise=max_extra_noise)
            if poisson_flux is not None:
                raise GalSimIncompatibleValuesError(
                    "poisson_flux is only relevant for method='phot'",
                    method=method, sensor=sensor, poisson_flux=poisson_flux)
            if photon_ops != ():
                raise GalSimIncompatibleValuesError(
                    "photon_ops are only relevant for method='phot'",
                    method=method, sensor=sensor, photon_ops=photon_ops)
            if maxN != None:
                raise GalSimIncompatibleValuesError(
                    "maxN is only relevant for method='phot'",
                    method=method, sensor=sensor, maxN=maxN)
            if save_photons:
                raise GalSimIncompatibleValuesError(
                    "save_photons is only valid for method='phot'",
                    method=method, sensor=sensor, save_photons=save_photons)
        else:
            # If we want to save photons, it doesn't make sense to limit the number per shoot call.
            if save_photons and maxN is not None:
                raise GalSimIncompatibleValuesError(
                    "Setting maxN is incompatible with save_photons=True")

        # Do any delayed computation needed by fft or real_space drawing.
        if method != 'phot':
            self._prepareDraw()

        # Figure out what wcs we are going to use.
        wcs = self._determine_wcs(scale, wcs, image)

        # Make sure offset and center are PositionD, converting from other formats (tuple, array,..)
        # Note: If None, offset is converted to PositionD(0,0), but center will remain None.
        offset = self._parse_offset(offset)
        center = self._parse_center(center)

        # Determine the bounds of the new image for use below (if it can be known yet)
        new_bounds = self._get_new_bounds(image, nx, ny, bounds, center)

        # Get the local WCS, accounting for the offset correctly.
        local_wcs = self._local_wcs(wcs, image, offset, center, use_true_center, new_bounds)

        # Account for area and exptime.
        flux_scale = area * exptime
        # For surface brightness normalization, also scale by the pixel area.
        if method == 'sb':
            flux_scale /= local_wcs.pixelArea()
        # Only do the gain here if not photon shooting, since need the number of photons to
        # reflect that actual photons, not ADU.
        if gain != 1 and method != 'phot' and sensor is None:
            flux_scale /= gain

        # Determine the offset, and possibly fix the centering for even-sized images
        offset = self._adjust_offset(new_bounds, offset, center, use_true_center)

        # Convert the profile in world coordinates to the profile in image coordinates:
        prof = local_wcs.profileToImage(self, flux_ratio=flux_scale, offset=offset)
        if offset != _PositionD(0,0):
            local_wcs = local_wcs.shiftOrigin(offset)

        # If necessary, convolve by the pixel
        if method in ('auto', 'fft', 'real_space'):
            if method == 'auto':
                real_space = None
            elif method == 'fft':
                real_space = False
            else:
                real_space = True
            prof_no_pixel = prof
            prof = Convolve(prof, Pixel(scale=1.0, gsparams=self.gsparams),
                            real_space=real_space, gsparams=self.gsparams)

        # Make sure image is setup correctly
        image = prof._setup_image(image, nx, ny, bounds, add_to_image, dtype, center)
        image.wcs = wcs

        if setup_only:
            image.added_flux = 0.
            return image

        # Making a view of the image lets us change the center without messing up the original.
        imview = image._view()
        imview._shift(-image.center)  # equiv. to setCenter(0,0), but faster
        imview.wcs = PixelScale(1.0)
        orig_center = image.center  # Save the original center to pass to sensor.accumulate
        if method == 'phot':
            added_photons, photons = prof.drawPhot(imview, gain, add_to_image,
                                                   n_photons, rng, max_extra_noise, poisson_flux,
                                                   sensor, photon_ops, maxN,
                                                   orig_center, local_wcs)
        else:
            # If not using phot, but doing sensor, then make a copy.
            if sensor is not None:
                if imview.dtype in (np.float32, np.float64):
                    dtype = None
                else:
                    dtype = np.float64
                draw_image = imview.real.subsample(n_subsample, n_subsample, dtype=dtype)
                draw_image._shift(-draw_image.center)  # eqiv. to setCenter(0,0)
                if method in ('auto', 'fft', 'real_space'):
                    # Need to reconvolve by the new smaller pixel instead
                    prof = Convolve(
                            prof_no_pixel,
                            Pixel(scale=1.0/n_subsample, gsparams=self.gsparams),
                            real_space=real_space, gsparams=self.gsparams)
                elif n_subsample != 1:
                    # We can't just pull off the pixel-free version, so we need to deconvolve
                    # by the original pixel and reconvolve by the smaller one.
                    prof = Convolve(
                            prof,
                            Deconvolve(Pixel(scale=1.0, gsparams=self.gsparams)),
                            Pixel(scale=1.0/n_subsample, gsparams=self.gsparams),
                            gsparams=self.gsparams)
                add = False
                if not add_to_image: imview.setZero()
            else:
                draw_image = imview
                add = add_to_image

            if prof.is_analytic_x:
                added_photons = prof.drawReal(draw_image, add)
            else:
                added_photons = prof.drawFFT(draw_image, add)

            if sensor is not None:
                photons = PhotonArray.makeFromImage(draw_image, rng=rng)
                for op in photon_ops:
                    op.applyTo(photons, local_wcs, rng)
                if imview.dtype in (np.float32, np.float64):
                    added_photons = sensor.accumulate(photons, imview, orig_center)
                else:
                    # Need a temporary
                    im1 = ImageD(bounds=imview.bounds)
                    added_photons = sensor.accumulate(photons, im1, orig_center)
                    imview.array[:,:] += im1.array.astype(imview.dtype, copy=False)

        image.added_flux = added_photons / flux_scale
        if save_photons:
            image.photons = photons

        return image

    def drawReal(self, image, add_to_image=False):
        """
        Draw this profile into an `Image` by direct evaluation at the location of each pixel.

        This is usually called from the `drawImage` function, rather than called directly by the
        user.  In particular, the input image must be already set up with defined bounds.  The
        profile will be drawn centered on whatever pixel corresponds to (0,0) with the given
        bounds, not the image center (unlike `drawImage`).  The image also must have a `PixelScale`
        wcs.  The profile being drawn should have already been converted to image coordinates via::

            >>> image_profile = original_wcs.toImage(original_profile)

        Note that the image produced by ``drawReal`` represents the profile sampled at the center
        of each pixel and then multiplied by the pixel area.  That is, the profile is NOT
        integrated over the area of the pixel.  This is equivalent to method='no_pixel' in
        `drawImage`.  If you want to render a profile integrated over the pixel, you can convolve
        with a `Pixel` first and draw that.

        Parameters:
            image:          The `Image` onto which to place the flux. [required]
            add_to_image:   Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing. [default: False]

        Returns:
            The total flux drawn inside the image bounds.
        """
        from .image import ImageD, ImageF
        if image.wcs is None or not image.wcs._isPixelScale:
            raise GalSimValueError("drawReal requires an image with a PixelScale wcs", image)

        if image.dtype in (np.float64, np.float32) and not add_to_image and image.iscontiguous:
            self._drawReal(image)
            return image.array.sum(dtype=float)
        else:
            # Need a temporary
            if image.dtype in (np.complex128, np.int32, np.uint32):
                im1 = ImageD(bounds=image.bounds, scale=image.scale)
            else:
                im1 = ImageF(bounds=image.bounds, scale=image.scale)
            self._drawReal(im1)
            if add_to_image:
                image.array[:,:] += im1.array.astype(image.dtype, copy=False)
            else:
                image.array[:,:] = im1.array
            return im1.array.sum(dtype=float)

    def _drawReal(self, image, jac=None, offset=(0.,0.), flux_scaling=1.):
        """A version of `drawReal` without the sanity checks or some options.

        This is nearly equivalent to the regular ``drawReal(image, add_to_image=False)``, but
        the image's dtype must be either float32 or float64, and it must have a c_contiguous array
        (``image.iscontiguous`` must be True).
        """
        raise NotImplementedError("%s does not implement drawReal"%self.__class__.__name__)

    def getGoodImageSize(self, pixel_scale):
        """Return a good size to use for drawing this profile.

        The size will be large enough to cover most of the flux of the object.  Specifically,
        at least (1-gsparams.folding_threshold) (i.e. 99.5% by default) of the flux should fall
        in the image.

        Also, the returned size is always an even number, which is usually desired in practice.
        Of course, if you prefer an odd-sized image, you can add 1 to the result.

        Parameters:
            pixel_scale:    The desired pixel scale of the image to be built.

        Returns:
            N, a good (linear) size of an image on which to draw this object.
        """
        # Start with a good size from stepk and the pixel scale
        Nd = 2. * math.pi / (pixel_scale * self.stepk)

        # Make it an integer
        # (Some slop to keep from getting extra pixels due to roundoff errors in calculations.)
        N = int(math.ceil(Nd*(1.-1.e-12)))

        # Round up to an even value
        N = 2 * ((N+1) // 2)
        return N

    def drawFFT_makeKImage(self, image):
        """
        This is a helper routine for drawFFT that just makes the (blank) k-space image
        onto which the profile will be drawn.  This can be useful if you want to break
        up the calculation into parts for extra efficiency.  E.g. save the k-space image of
        the PSF so drawing many models of the galaxy with the given PSF profile can avoid
        drawing the PSF each time.

        Parameters:
            image:      The `Image` onto which to place the flux.

        Returns:
            (kimage, wrap_size), where wrap_size is either the size of kimage or smaller if
            the result should be wrapped before doing the inverse fft.
        """
        from .bounds import _BoundsI
        from .image import ImageCD, ImageCF
        # Start with what this profile thinks a good size would be given the image's pixel scale.
        N = self.getGoodImageSize(image.scale)

        # We must make something big enough to cover the target image size:
        image_N = max(np.max(np.abs((image.bounds._getinitargs()))) * 2,
                      np.max(image.bounds.numpyShape()))
        N = max(N, image_N)

        # Round up to a good size for making FFTs:
        N = image.good_fft_size(N)

        # Make sure we hit the minimum size specified in the gsparams.
        N = max(N, self.gsparams.minimum_fft_size)

        dk = 2.*np.pi / (N * image.scale)

        maxk = self.maxk
        if N*dk/2 > maxk:
            Nk = N
        else:
            # There will be aliasing.  Make a larger image and then wrap it.
            Nk = int(np.ceil(maxk/dk)) * 2

        if Nk > self.gsparams.maximum_fft_size:
            raise GalSimFFTSizeError("drawFFT requires an FFT that is too large.", Nk)

        bounds = _BoundsI(0,Nk//2,-Nk//2,Nk//2)
        if image.dtype in (np.complex128, np.float64, np.int32, np.uint32):
            kimage = ImageCD(bounds=bounds, scale=dk)
        else:
            kimage = ImageCF(bounds=bounds, scale=dk)
        return kimage, N

    def drawFFT_finish(self, image, kimage, wrap_size, add_to_image):
        """
        This is a helper routine for drawFFT that finishes the calculation, based on the
        drawn k-space image.

        It applies the Fourier transform to ``kimage`` and adds the result to ``image``.

        Parameters:
            image:          The `Image` onto which to place the flux.
            kimage:         The k-space `Image` where the object was drawn.
            wrap_size:      The size of the region to wrap kimage, which must be either the same
                            size as kimage or smaller.
            add_to_image:   Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing.

        Returns:
            The total flux drawn inside the image bounds.
        """
        from .bounds import _BoundsI
        from .image import Image
        # Wrap the full image to the size we want for the FT.
        # Even if N == Nk, this is useful to make this portion properly Hermitian in the
        # N/2 column and N/2 row.
        bwrap = _BoundsI(0, wrap_size//2, -wrap_size//2, wrap_size//2-1)
        kimage_wrap = kimage._wrap(bwrap, True, False)

        # Perform the fourier transform.
        breal = _BoundsI(-wrap_size//2, wrap_size//2+1, -wrap_size//2, wrap_size//2-1)
        real_image = Image(breal, dtype=float)
        with convert_cpp_errors():
            _galsim.irfft(kimage_wrap._image, real_image._image, True, True)

        # Add (a portion of) this to the original image.
        temp = real_image.subImage(image.bounds)
        if add_to_image:
            image += temp
        else:
            image.copyFrom(temp)
        added_photons = temp.array.sum(dtype=float)
        return added_photons

    def drawFFT(self, image, add_to_image=False):
        """
        Draw this profile into an `Image` by computing the k-space image and performing an FFT.

        This is usually called from the `drawImage` function, rather than called directly by the
        user.  In particular, the input image must be already set up with defined bounds.  The
        profile will be drawn centered on whatever pixel corresponds to (0,0) with the given
        bounds, not the image center (unlike `drawImage`).  The image also must have a `PixelScale`
        wcs.  The profile being drawn should have already been converted to image coordinates via::

            >>> image_profile = original_wcs.toImage(original_profile)

        Note that the `Image` produced by drawFFT represents the profile sampled at the center
        of each pixel and then multiplied by the pixel area.  That is, the profile is NOT
        integrated over the area of the pixel.  This is equivalent to method='no_pixel' in
        `drawImage`.  If you want to render a profile integrated over the pixel, you can convolve
        with a `Pixel` first and draw that.

        Parameters:
            image:          The `Image` onto which to place the flux. [required]
            add_to_image:   Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing. [default: False]

        Returns:
            The total flux drawn inside the image bounds.
        """
        if image.wcs is None or not image.wcs._isPixelScale:
            raise GalSimValueError("drawPhot requires an image with a PixelScale wcs", image)

        kimage, wrap_size = self.drawFFT_makeKImage(image)
        self._drawKImage(kimage)
        return self.drawFFT_finish(image, kimage, wrap_size, add_to_image)

    def _calculate_nphotons(self, n_photons, poisson_flux, max_extra_noise, rng):
        """Calculate how many photons to shoot and what flux_ratio (called g) each one should
        have in order to produce an image with the right S/N and total flux.

        This routine is normally called by `drawPhot`.

        Returns:
            n_photons, g
        """
        from .random import PoissonDeviate
        # For profiles that are positive definite, then N = flux. Easy.
        #
        # However, some profiles shoot some of their photons with negative flux. This means that
        # we need a few more photons to get the right S/N = sqrt(flux). Take eta to be the
        # fraction of shot photons that have negative flux.
        #
        # S^2 = (N+ - N-)^2 = (N+ + N- - 2N-)^2 = (Ntot - 2N-)^2 = Ntot^2(1 - 2 eta)^2
        # N^2 = Var(S) = (N+ + N-) = Ntot
        #
        # So flux = (S/N)^2 = Ntot (1-2eta)^2
        # Ntot = flux / (1-2eta)^2
        #
        # However, if each photon has a flux of 1, then S = (1-2eta) Ntot = flux / (1-2eta).
        # So in fact, each photon needs to carry a flux of g = 1-2eta to get the right
        # total flux.
        #
        # That's all the easy case. The trickier case is when we are sky-background dominated.
        # Then we can usually get away with fewer shot photons than the above.  In particular,
        # if the noise from the photon shooting is much less than the sky noise, then we can
        # use fewer shot photons and essentially have each photon have a flux > 1. This is ok
        # as long as the additional noise due to this approximation is "much less than" the
        # noise we'll be adding to the image for the sky noise.
        #
        # Let's still have Ntot photons, but now each with a flux of g. And let's look at the
        # noise we get in the brightest pixel that has a nominal total flux of Imax.
        #
        # The number of photons hitting this pixel will be Imax/flux * Ntot.
        # The variance of this number is the same thing (Poisson counting).
        # So the noise in that pixel is:
        #
        # N^2 = Imax/flux * Ntot * g^2
        #
        # And the signal in that pixel will be:
        #
        # S = Imax/flux * (N+ - N-) * g which has to equal Imax, so
        # g = flux / Ntot(1-2eta)
        # N^2 = Imax/Ntot * flux / (1-2eta)^2
        #
        # As expected, we see that lowering Ntot will increase the noise in that (and every
        # other) pixel.
        # The input max_extra_noise parameter is the maximum value of spurious noise we want
        # to allow.
        #
        # So setting N^2 = Imax + nu, we get
        #
        # Ntot = flux / (1-2eta)^2 / (1 + nu/Imax)
        # g = (1 - 2eta) * (1 + nu/Imax)
        #
        # Returns the total flux placed inside the image bounds by photon shooting.
        #

        flux = self.flux
        if flux == 0.0:
            return 0, 1.0

        # The _flux_per_photon property is (1-2eta)
        # This factor will already be accounted for by the shoot function, so don't include
        # that as part of our scaling here.  There may be other adjustments though, so g=1 here.
        eta_factor = self._flux_per_photon
        mod_flux = flux / (eta_factor * eta_factor)
        g = 1.

        # If requested, let the target flux value vary as a Poisson deviate
        if poisson_flux:
            # If we have both positive and negative photons, then the mix of these
            # already gives us some variation in the flux value from the variance
            # of how many are positive and how many are negative.
            # The number of negative photons varies as a binomial distribution.
            # <F-> = eta * Ntot * g
            # <F+> = (1-eta) * Ntot * g
            # <F+ - F-> = (1-2eta) * Ntot * g = flux
            # Var(F-) = eta * (1-eta) * Ntot * g^2
            # F+ = Ntot * g - F- is not an independent variable, so
            # Var(F+ - F-) = Var(Ntot*g - 2*F-)
            #              = 4 * Var(F-)
            #              = 4 * eta * (1-eta) * Ntot * g^2
            #              = 4 * eta * (1-eta) * flux
            # We want the variance to be equal to flux, so we need an extra:
            # delta Var = (1 - 4*eta + 4*eta^2) * flux
            #           = (1-2eta)^2 * flux
            absflux = abs(flux)
            mean = eta_factor*eta_factor * absflux
            pd = PoissonDeviate(rng, mean)
            pd_val = pd() - mean + absflux
            ratio = pd_val / absflux
            g *= ratio
            mod_flux *= ratio

        if n_photons == 0.:
            n_photons = abs(mod_flux)
            if max_extra_noise > 0.:
                gfactor = 1. + max_extra_noise / abs(self.max_sb)
                n_photons /= gfactor
                g *= gfactor

        # Make n_photons an integer.
        iN = int(n_photons + 0.5)

        return iN, g


    def makePhot(self, n_photons=0, rng=None, max_extra_noise=0., poisson_flux=None,
                 photon_ops=(), local_wcs=None, surface_ops=None):
        """
        Make photons for a profile.

        This is equivalent to drawPhot, except that the photons are not placed onto
        an image.  Instead, it just returns the PhotonArray.

        .. note::

            The (x,y) positions returned are in the same units as the distance units
            of the GSObject being rendered.  If you want (x,y) in pixel coordinates, you
            should call this function for the profile in image coordinates::

                >>> photons = image.wcs.toImage(obj).makePhot()

            Or if you just want a simple pixel scale conversion from sky coordinates to image
            coordinates, you can instead do

                >>> photons = obj.makePhot()
                >>> photons.scaleXY(1./pixel_scale)

        Parameters:
            n_photons:      If provided, the number of photons to use for photon shooting.
                            If not provided (i.e. ``n_photons = 0``), use as many photons as
                            necessary to result in an image with the correct Poisson shot
                            noise for the object's flux.  For positive definite profiles, this
                            is equivalent to ``n_photons = flux``.  However, some profiles need
                            more than this because some of the shot photons are negative
                            (usually due to interpolants).  [default: 0]
            rng:            If provided, a random number generator to use for photon shooting,
                            which may be any kind of `BaseDeviate` object.  If ``rng`` is None, one
                            will be automatically created, using the time as a seed.
                            [default: None]
            max_extra_noise: If provided, the allowed extra noise in each pixel when photon
                            shooting.  This is only relevant if ``n_photons=0``, so the number of
                            photons is being automatically calculated.  In that case, if the image
                            noise is dominated by the sky background, then you can get away with
                            using fewer shot photons than the full ``n_photons = flux``.
                            Essentially each shot photon can have a ``flux > 1``, which increases
                            the noise in each pixel.  The ``max_extra_noise`` parameter specifies
                            how much extra noise per pixel is allowed because of this approximation.
                            A typical value for this might be ``max_extra_noise = sky_level / 100``
                            where ``sky_level`` is the flux per pixel due to the sky.  Note that
                            this uses a "variance" definition of noise, not a "sigma" definition.
                            [default: 0.]
            poisson_flux:   Whether to allow total object flux scaling to vary according to
                            Poisson statistics for ``n_photons`` samples when photon shooting.
                            [default: True, unless ``n_photons`` is given, in which case the default
                            is False]
            photon_ops:     A list of operators that can modify the photon array that will be
                            applied in order before accumulating the photons on the sensor.
                            [default: ()]
            local_wcs:      The local wcs in the original image. [default: None]

        Returns:
            - a `PhotonArray` with the data about the photons.
        """
        if surface_ops is not None:
            from .deprecated import depr
            depr('surface_ops', 2.3, 'photon_ops')
            photon_ops = surface_ops

        # Make sure the type of n_photons is correct and has a valid value:
        if n_photons < 0.:
            raise GalSimRangeError("Invalid n_photons < 0.", n_photons, 0., None)

        if poisson_flux is None:
            # If n_photons is given, poisson_flux = False
            poisson_flux = (n_photons == 0.)

        # Check that either n_photons is set to something or flux is set to something
        if n_photons == 0. and self.flux == 1.:
            galsim_warn(
                    "Warning: drawImage for object with flux == 1, area == 1, and "
                    "exptime == 1, but n_photons == 0.  This will only shoot a single photon.")

        Ntot, g = self._calculate_nphotons(n_photons, poisson_flux, max_extra_noise, rng)

        try:
            photons = self.shoot(Ntot, rng)
        except (GalSimError, NotImplementedError) as e:
            raise GalSimNotImplementedError(
                    "Unable to draw this GSObject with photon shooting.  Perhaps it "
                    "is a Deconvolve or is a compound including one or more "
                    "Deconvolve objects.\nOriginal error: %r"%(e))

        if g != 1.:
            photons.scaleFlux(g)

        for op in photon_ops:
            op.applyTo(photons, local_wcs, rng)

        return photons


    def drawPhot(self, image, gain=1., add_to_image=False,
                 n_photons=0, rng=None, max_extra_noise=0., poisson_flux=None,
                 sensor=None, photon_ops=(), maxN=None, orig_center=_PositionI(0,0),
                 local_wcs=None, surface_ops=None):
        """
        Draw this profile into an `Image` by shooting photons.

        This is usually called from the `drawImage` function, rather than called directly by the
        user.  In particular, the input image must be already set up with defined bounds.  The
        profile will be drawn centered on whatever pixel corresponds to (0,0) with the given
        bounds, not the image center (unlike `drawImage`).  The image also must have a `PixelScale`
        wcs.  The profile being drawn should have already been converted to image coordinates via::

            >>> image_profile = original_wcs.toImage(original_profile)

        Note that the image produced by `drawPhot` represents the profile integrated over the
        area of each pixel.  This is equivalent to convolving the profile by a square `Pixel`
        profile and sampling the value at the center of each pixel, although this happens
        automatically by the shooting algorithm, so you do not need to manually convolve by
        a `Pixel` as you would for `drawReal` or `drawFFT`.

        Parameters:
            image:          The `Image` onto which to place the flux. [required]
            gain:           The number of photons per ADU ("analog to digital units", the units of
                            the numbers output from a CCD). [default: 1.]
            add_to_image:   Whether to add to the existing images rather than clear out
                            anything in the image before drawing.  [default: False]
            n_photons:      If provided, the number of photons to use for photon shooting.
                            If not provided (i.e. ``n_photons = 0``), use as many photons as
                            necessary to result in an image with the correct Poisson shot
                            noise for the object's flux.  For positive definite profiles, this
                            is equivalent to ``n_photons = flux``.  However, some profiles need
                            more than this because some of the shot photons are negative
                            (usually due to interpolants).  [default: 0]
            rng:            If provided, a random number generator to use for photon shooting,
                            which may be any kind of `BaseDeviate` object.  If ``rng`` is None, one
                            will be automatically created, using the time as a seed.
                            [default: None]
            max_extra_noise: If provided, the allowed extra noise in each pixel when photon
                            shooting.  This is only relevant if ``n_photons=0``, so the number of
                            photons is being automatically calculated.  In that case, if the image
                            noise is dominated by the sky background, then you can get away with
                            using fewer shot photons than the full ``n_photons = flux``.
                            Essentially each shot photon can have a ``flux > 1``, which increases
                            the noise in each pixel.  The ``max_extra_noise`` parameter specifies
                            how much extra noise per pixel is allowed because of this approximation.
                            A typical value for this might be ``max_extra_noise = sky_level / 100``
                            where ``sky_level`` is the flux per pixel due to the sky.  Note that
                            this uses a "variance" definition of noise, not a "sigma" definition.
                            [default: 0.]
            poisson_flux:   Whether to allow total object flux scaling to vary according to
                            Poisson statistics for ``n_photons`` samples when photon shooting.
                            [default: True, unless ``n_photons`` is given, in which case the default
                            is False]
            sensor:         An optional `Sensor` instance, which will be used to accumulate the
                            photons onto the image. [default: None]
            photon_ops:     A list of operators that can modify the photon array that will be
                            applied in order before accumulating the photons on the sensor.
                            [default: ()]
            maxN:           Sets the maximum number of photons that will be added to the image
                            at a time.  (Memory requirements are proportional to this number.)
                            [default: None, which means no limit]
            orig_center:    The position of the image center in the original image coordinates.
                            [default: (0,0)]
            local_wcs:      The local wcs in the original image. [default: None]

        Returns:
            (added_flux, photons) where:
            - added_flux is the total flux of photons that landed inside the image bounds, and
            - photons is the `PhotonArray` that was applied to the image.
        """
        from .sensor import Sensor
        from .image import ImageD

        if surface_ops is not None:
            from .deprecated import depr
            depr('surface_ops', 2.3, 'photon_ops')
            photon_ops = surface_ops

        # Make sure the type of n_photons is correct and has a valid value:
        if n_photons < 0.:
            raise GalSimRangeError("Invalid n_photons < 0.", n_photons, 0., None)

        if poisson_flux is None:
            # If n_photons is given, poisson_flux = False
            poisson_flux = (n_photons == 0.)

        # Check that either n_photons is set to something or flux is set to something
        if n_photons == 0. and self.flux == 1.:
            galsim_warn(
                    "Warning: drawImage for object with flux == 1, area == 1, and "
                    "exptime == 1, but n_photons == 0.  This will only shoot a single photon.")

        # Make sure the image is set up to have unit pixel scale and centered at 0,0.
        if image.wcs is None or not image.wcs._isPixelScale:
            raise GalSimValueError("drawPhot requires an image with a PixelScale wcs", image)

        if sensor is None:
            sensor = Sensor()
        elif not isinstance(sensor, Sensor):
            raise TypeError("The sensor provided is not a Sensor instance")

        Ntot, g = self._calculate_nphotons(n_photons, poisson_flux, max_extra_noise, rng)

        if gain != 1.:
            g /= gain

        # total flux falling inside image bounds, this will be returned on exit.
        added_flux = 0.

        if maxN is None:
            maxN = Ntot

        if not add_to_image: image.setZero()

        # Nleft is the number of photons remaining to shoot.
        Nleft = Ntot
        photons = None  # Just in case Nleft is already 0.
        resume = False
        while Nleft > 0:
            # Shoot at most maxN at a time
            thisN = min(maxN, Nleft)

            try:
                photons = self.shoot(thisN, rng)
            except (GalSimError, NotImplementedError) as e:
                raise GalSimNotImplementedError(
                        "Unable to draw this GSObject with photon shooting.  Perhaps it "
                        "is a Deconvolve or is a compound including one or more "
                        "Deconvolve objects.\nOriginal error: %r"%(e))

            if g != 1. or thisN != Ntot:
                photons.scaleFlux(g * thisN / Ntot)

            if image.scale != 1.:
                photons.scaleXY(1./image.scale)  # Convert x,y to image coords if necessary

            for op in photon_ops:
                op.applyTo(photons, local_wcs, rng)

            if image.dtype in (np.float32, np.float64):
                added_flux += sensor.accumulate(photons, image, orig_center, resume=resume)
                resume = True  # Resume from this point if there are any further iterations.
            else:
                # Need a temporary
                im1 = ImageD(bounds=image.bounds)
                added_flux += sensor.accumulate(photons, im1, orig_center)
                image.array[:,:] += im1.array.astype(image.dtype, copy=False)

            Nleft -= thisN

        return added_flux, photons


    def shoot(self, n_photons, rng=None):
        """Shoot photons into a `PhotonArray`.

        Parameters:
            n_photons:  The number of photons to use for photon shooting.
            rng:        If provided, a random number generator to use for photon shooting,
                        which may be any kind of `BaseDeviate` object.  If ``rng`` is None, one
                        will be automatically created, using the time as a seed.
                        [default: None]

        Returns:
            A `PhotonArray`.
        """
        from .random import BaseDeviate
        from .photon_array import PhotonArray

        photons = PhotonArray(n_photons)
        if n_photons == 0:
            # It's ok to shoot 0, but downstream can have problems with it, so just stop now.
            return photons
        if rng is None:
            rng = BaseDeviate()

        self._shoot(photons, rng)
        return photons

    def _shoot(self, photons, rng):
        """Shoot photons into the given `PhotonArray`.

        This is the backend implementation of `shoot` once the `PhotonArray` has been constructed.

        Parameters:
            photons:    A `PhotonArray` instance into which the photons should be placed.
            rng:        A `BaseDeviate` instance to use for the photon shooting,
        """
        raise NotImplementedError("%s does not implement shoot"%self.__class__.__name__)

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply this surface brightness profile as a convolution to an existing photon array.

        This method allows a GSObject to duck type as a PhotonOp, so one can apply a PSF
        in a photon_ops list.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use to effect the convolution.
                            [default: None]
        """
        from .photon_array import PhotonArray
        p1 = PhotonArray(len(photon_array))
        obj = local_wcs.toImage(self) if local_wcs is not None else self
        obj._shoot(p1, rng)
        photon_array.convolve(p1, rng)

    def drawKImage(self, image=None, nx=None, ny=None, bounds=None, scale=None,
                   add_to_image=False, recenter=True, bandpass=None, setup_only=False):
        """Draws the k-space (complex) `Image` of the object, with bounds optionally set by input
        `Image` instance.

        Normalization is always such that image(0,0) = flux.  Unlike the real-space `drawImage`
        function, the (0,0) point will always be one of the actual pixel values.  For even-sized
        images, it will be 1/2 pixel above and to the right of the true center of the image.

        Another difference from  `drawImage` is that a wcs other than a simple pixel scale is not
        allowed.  There is no ``wcs`` parameter here, and if the images have a non-trivial wcs (and
        you don't override it with the ``scale`` parameter), a TypeError will be raised.

        Also, there is no convolution by a pixel.  This is just a direct image of the Fourier
        transform of the surface brightness profile.

        Parameters:
            image:          If provided, this will be the `Image` onto which to draw the k-space
                            image.  If ``image`` is None, then an automatically-sized image will be
                            created.  If ``image`` is given, but its bounds are undefined, then it
                            will be resized appropriately based on the profile's size.
                            [default: None]
            nx:             If provided and ``image`` is None, use to set the x-direction size of
                            the image.  Must be accompanied by ``ny``.
            ny:             If provided and ``image`` is None, use to set the y-direction size of
                            the image.  Must be accompanied by ``nx``.
            bounds:         If provided and ``image`` is None, use to set the bounds of the image.
            scale:          If provided, use this as the pixel scale, dk, for the images.
                            If ``scale`` is None and ``image`` is given, then take the provided
                            images' pixel scale (which must be equal).
                            If ``scale`` is None and ``image`` is None, then use the Nyquist scale.
                            If ``scale <= 0`` (regardless of ``image``), then use the Nyquist scale.
                            [default: None]
            add_to_image:   Whether to add to the existing images rather than clear out
                            anything in the image before drawing.
                            Note: This requires that ``image`` be provided and that it has defined
                            bounds. [default: False]
            recenter:       Whether to recenter the image to put k = 0 at the center (True) or to
                            trust the provided bounds (False).  [default: True]
            bandpass:       This parameter is ignored, but is allowed to enable duck typing
                            eqivalence between this method and the ChromaticObject.drawImage
                            method. [default: None]
            setup_only:     Don't actually draw anything on the image.  Just make sure the image
                            is set up correctly.  This is used internally by GalSim, but there
                            may be cases where the user will want the same functionality.
                            [default: False]

        Returns:
            an `Image` instance (created if necessary)
        """
        from .wcs import PixelScale
        from .image import Image
        # Make sure provided image is complex
        if image is not None:
            if not isinstance(image, Image):
                raise TypeError("Provided image must be galsim.Image", image)

            if not image.iscomplex:
                raise GalSimValueError("Provided image must be complex", image)

        # Possibly get the scale from image.
        if image is not None and scale is None:
            # Grab the scale to use from the image.
            # This will raise a TypeError if image.wcs is not a PixelScale
            scale = image.scale

        # The input scale (via scale or image.scale) is really a dk value, so call it that for
        # clarity here, since we also need the real-space pixel scale, which we will call dx.
        if scale is None or scale <= 0:
            dk = self.stepk
        else:
            dk = float(scale)
        if image is not None and image.bounds.isDefined():
            dx = np.pi/( max(image.array.shape) // 2 * dk )
        elif scale is None or scale <= 0:
            dx = self.nyquist_scale
        else:
            # Then dk = scale, which implies that we need to have dx smaller than nyquist_scale
            # by a factor of (dk/stepk)
            dx = self.nyquist_scale * dk / self.stepk

        # If the profile needs to be constructed from scratch, the _setup_image function will
        # do that, but only if the profile is in image coordinates for the real space image.
        # So make that profile.
        if image is None or not image.bounds.isDefined():
            real_prof = PixelScale(dx).profileToImage(self)
            dtype = np.complex128 if image is None else image.dtype
            image = real_prof._setup_image(image, nx, ny, bounds, add_to_image, dtype,
                                           center=None, odd=True)
        else:
            # Do some checks that setup_image would have done for us
            if bounds is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide bounds if image is provided", bounds=bounds, image=image)
            if nx is not None or ny is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide nx,ny if image is provided", nx=nx, ny=ny, image=image)

        # Can't both recenter a provided image and add to it.
        if recenter and image.center != _PositionI(0,0) and add_to_image:
            raise GalSimIncompatibleValuesError(
                "Cannot use add_to_image=True unless image is centered at (0,0) or recenter=False",
                recenter=recenter, image=image, add_to_image=add_to_image)

        # Set the center to 0,0 if appropriate
        if recenter and image.center != _PositionI(0,0):
            image._shift(-image.center)

        # Set the wcs of the images to use the dk scale size
        image.scale = dk

        if setup_only:
            return image

        if not add_to_image and image.iscontiguous:
            self._drawKImage(image)
        else:
            im2 = Image(bounds=image.bounds, dtype=image.dtype, scale=image.scale)
            self._drawKImage(im2)
            image += im2
        return image

    def _drawKImage(self, image, jac=None):  # pragma: no cover  (all our classes override this)
        """A version of `drawKImage` without the sanity checks or some options.

        Equivalent to ``drawKImage(image, add_to_image=False, recenter=False, add_to_image=False)``,
        but without the option to create the image automatically.

        The input image must be provided as a complex `Image` instance (dtype=complex64 or
        complex128), and the bounds should be set up appropriately (e.g. with 0,0 in the center if
        so desired).  This corresponds to recenter=False for the normal `drawKImage`.  And, it must
        have a c_contiguous array (image.iscontiguous must be True).

        Parameters:
            image:      The `Image` onto which to draw the k-space image. [required]
        """
        raise NotImplementedError("%s does not implement drawKImage"%self.__class__.__name__)

    # Derived classes should define the __eq__ function
    def __ne__(self, other): return not self.__eq__(other)
