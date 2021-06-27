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

from heapq import heappush, heappop
import numpy as np

from .gsobject import GSObject
from .gsparams import GSParams
from .angle import radians, degrees, arcsec, Angle, AngleUnit
from .image import Image, _Image
from .bounds import _BoundsI
from .wcs import PixelScale
from .interpolatedimage import InterpolatedImage
from .utilities import doc_inherit, OrderedWeakRef, rotate_xy, lazy_property, basestring
from .errors import GalSimValueError, GalSimRangeError, GalSimIncompatibleValuesError
from .errors import GalSimFFTSizeError, galsim_warn


class Aperture(object):
    """Class representing a telescope aperture embedded in a larger pupil plane array -- for use
    with the `PhaseScreenPSF` class to create PSFs via Fourier or geometric optics.

    The pupil plane array is completely specified by its size, sampling interval, and pattern of
    illuminated pixels.  Pupil plane arrays can be specified either geometrically or using an image
    to indicate the illuminated pixels.  In both cases, various options exist to control the pupil
    plane size and sampling interval.

    **Geometric pupil specification**:

    The first way to specify the details of the telescope aperture is through a series of keywords
    indicating the diameter, size of the central obscuration, and the nature of the struts
    holding up the secondary mirror (or prime focus cage, etc.).  The struts are assumed to be
    rectangular obscurations extending from the outer edge of the pupil to the outer edge of the
    obscuration disk (or to the pupil center if ``obscuration = 0.``).  You can specify how many
    struts there are (evenly spaced in angle), how thick they are as a fraction of the pupil
    diameter, and what angle they start at relative to the positive y direction.

    The size (in meters) and sampling interval (in meters) of the pupil plane array representing the
    aperture can be set directly using the the ``pupil_plane_size`` and ``pupil_plane_scale``
    keywords.  However, in most situations, it's probably more convenient to let GalSim set these
    automatically based on the pupil geometry and the nature of the (potentially time-varying)
    phase aberrations from which a PSF is being derived.

    The pupil plane array physical size is by default set to twice the pupil diameter producing a
    Nyquist sampled PSF image.  While this would always be sufficient if using sinc interpolation
    over the PSF image for subsequent operations, GalSim by default uses the much faster (though
    approximate) quintic interpolant, which means that in some cases -- in particular, for
    significantly aberrated optical PSFs without atmospheric aberrations -- it may be useful to
    further increase the size of the pupil plane array, thereby increasing the sampling rate of the
    resulting PSF image.  This can be done by increasing the ``oversampling`` keyword.

    A caveat to the above occurs when using ``geometric_shooting=True`` to draw using
    photon-shooting.  In this case, we only need an array just large enough to avoid clipping the
    pupil, which we can get by setting ``oversampling=0.5``.

    The pupil plane array physical sampling interval (which is directly related to the resulting PSF
    image physical size) is set by default to the same interval as would be used to avoid
    significant aliasing (image folding) for an obscured `Airy` profile with matching diameter and
    obscuration and for the value of ``folding_threshold`` in the optionally specified gsparams
    argument.  If the phase aberrations are significant, however, the PSF image size computed this
    way may still not be sufficiently large to avoid aliasing.  To further increase the pupil plane
    sampling rate (and hence the PSF image size), you can increase the value of the ``pad_factor``
    keyword.

    An additional way to set the pupil sampling interval for a particular set of phase screens
    (i.e., for a particular `PhaseScreenList`) is to provide the screens in the ``screen_list``
    argument.  Each screen in the list computes its own preferred sampling rate and the
    `PhaseScreenList` appropriately aggregates these. This last option also requires that a
    wavelength ``lam`` be specified, and is particularly helpful for creating PSFs derived from
    turbulent atmospheric screens.

    Finally, when specifying the pupil geometrically, Aperture may choose to make a small adjustment
    to ``pupil_plane_scale`` in order to produce an array with a good size for FFTs.  If your
    application depends on knowing the size and scale used with the Fourier optics framework, you
    can obtain these from the ``aper.pupil_plane_size`` and ``aper.pupil_plane_scale`` attributes.

    **Pupil image specification**:

    The second way to specify the pupil plane configuration is by passing in an image of it.  This
    can be useful, for example, if the struts are not evenly spaced or are not radially directed, as
    is assumed by the simple model for struts described above.  In this case, an exception is raised
    if keywords related to struts are also given.  On the other hand, the ``obscuration`` keyword is
    still used to ensure that the PSF images are not aliased, though it is ignored during the actual
    construction of the pupil plane illumination pattern.  Note that for complicated pupil
    configurations, it may be desireable to increase ``pad_factor`` for more fidelity at the expense
    of slower running time.  Finally, the ``pupil_plane_im`` that is passed in can be rotated during
    internal calculations by specifying a ``pupil_angle`` keyword.

    If you choose to pass in a pupil plane image, it must be a square array in which the image of
    the pupil is centered.  The areas that are illuminated should have some value >0, and the other
    areas should have a value of precisely zero.  Based on what the Aperture class determines is a
    good PSF sampling interval, the image of the pupil plane that is passed in might be zero-padded
    during internal calculations.  (The pupil plane array size and scale values can be accessed via
    the ``aper.pupil_plane_size`` and ``aper.pupil_plane_scale`` attributes.) The pixel scale of
    the pupil plane can be specified in one of three ways.  In descending order of priority, these
    are:

      1.  The ``pupil_plane_scale`` keyword argument (units are meters).
      2.  The ``pupil_plane_im.scale`` attribute (units are meters).
      3.  If (1) and (2) are both None, then the scale will be inferred by assuming that the
          illuminated pixel farthest from the image center is at a physical distance of self.diam/2.

    The ``pupil_plane_size`` and ``lam`` keywords are both ignored when constructing an Aperture
    from an image.

    Parameters:
        diam:               Aperture diameter in meters.
        lam:                Wavelength in nanometers.  [default: None]
        circular_pupil:     Adopt a circular pupil?  [default: True]
        obscuration:        Linear dimension of central obscuration as fraction of aperture
                            linear dimension. [0., 1.).  [default: 0.0]
        nstruts:            Number of radial support struts to add to the central obscuration.
                            [default: 0]
        strut_thick:        Thickness of support struts as a fraction of aperture diameter.
                            [default: 0.05]
        strut_angle:        `Angle` made between the vertical and the strut starting closest to it,
                            defined to be positive in the counter-clockwise direction; must be an
                            `Angle` instance. [default: 0. * galsim.degrees]
        oversampling:       Optional oversampling factor *in the image plane* for the PSF
                            eventually constructed using this `Aperture`.  Setting
                            ``oversampling < 1`` will produce aliasing in the PSF (not good).
                            [default: 1.0]
        pad_factor:         Additional multiple by which to extend the PSF image to avoid
                            folding.  [default: 1.0]
        screen_list:        An optional `PhaseScreenList` object.  If present, then get a good
                            pupil sampling interval using this object.  [default: None]
        pupil_plane_im:     The GalSim.Image, NumPy array, or name of file containing the pupil
                            plane image, to be used instead of generating one based on the
                            obscuration and strut parameters.  [default: None]
        pupil_angle:        If ``pupil_plane_im`` is not None, rotation angle for the pupil plane
                            (positive in the counter-clockwise direction).  Must be an `Angle`
                            instance. [default: 0. * galsim.degrees]
        pupil_plane_scale:  Sampling interval in meters to use for the pupil plane array.  In
                            most cases, it's a good idea to leave this as None, in which case
                            GalSim will attempt to find a good value automatically.  The
                            exception is when specifying the pupil arrangement via an image, in
                            which case this keyword can be used to indicate the sampling of that
                            image.  See also ``pad_factor`` for adjusting the pupil sampling scale.
                            [default: None]
        pupil_plane_size:   Size in meters to use for the pupil plane array.  In most cases, it's
                            a good idea to leave this as None, in which case GalSim will attempt
                            to find a good value automatically.  See also ``oversampling`` for
                            adjusting the pupil size.  [default: None]
        gsparams:           An optional `GSParams` argument. [default: None]
    """
    def __init__(self, diam, lam=None, circular_pupil=True, obscuration=0.0,
                 nstruts=0, strut_thick=0.05, strut_angle=0.0*radians,
                 oversampling=1.0, pad_factor=1.0, screen_list=None,
                 pupil_plane_im=None, pupil_angle=0.0*radians,
                 pupil_plane_scale=None, pupil_plane_size=None,
                 gsparams=None):

        self._diam = diam  # Always need to explicitly specify an aperture diameter.
        self._lam = lam
        self._circular_pupil = circular_pupil
        self._obscuration = obscuration
        self._nstruts = nstruts
        self._strut_thick = strut_thick
        self._strut_angle = strut_angle
        self._oversampling = oversampling
        self._pad_factor = pad_factor
        self._screen_list = screen_list
        self._pupil_plane_im = pupil_plane_im
        self._pupil_angle = pupil_angle
        self._input_pupil_plane_scale = pupil_plane_scale
        self._input_pupil_plane_size = pupil_plane_size
        self._gsparams = GSParams.check(gsparams)

        if diam <= 0.:
            raise GalSimRangeError("Invalid diam.", diam, 0.)
        if obscuration < 0. or obscuration >= 1.:
            raise GalSimRangeError("Invalid obscuration.", obscuration, 0., 1.)
        if not isinstance(strut_angle, Angle):
            raise TypeError("strut_angle must be a galsim.Angle instance.")
        if not isinstance(pupil_angle, Angle):
            raise TypeError("pupil_angle must be a galsim.Angle instance.")

        # You can either set geometric properties, or use a pupil image, but not both, so check for
        # that here.  One caveat is that we allow sanity checking the sampling of a pupil_image by
        # comparing it to the sampling GalSim would have used for an (obscured) Airy profile.  So
        # it's okay to specify an obscuration and a pupil_plane_im together, for example, but not
        # a pupil_plane_im and struts.
        is_default_geom = (circular_pupil and
                           nstruts == 0 and
                           strut_thick == 0.05 and
                           strut_angle == 0.0*radians)
        if not is_default_geom and pupil_plane_im is not None:
            raise GalSimIncompatibleValuesError(
                "Can't specify both geometric parameters and pupil_plane_im.",
                circular_pupil=circular_pupil, nstruts=nstruts, strut_thick=strut_thick,
                strut_angle=strut_angle, pupil_plane_im=pupil_plane_im)

        if screen_list is not None and lam is None:
            raise GalSimIncompatibleValuesError(
                "Wavelength ``lam`` must be specified with ``screen_list``.",
                screen_list=screen_list, lam=lam)

    # For each of these, the actual value is defined during the construction of the _illuminated
    # array, so access that (lazy) property first.
    @property
    def pupil_plane_scale(self):
        """The scale_size of the pupil-plane image.
        """
        self._illuminated
        return self._pupil_plane_scale
    @property
    def pupil_plane_size(self):
        """The size of the pupil-plane image.
        """
        self._illuminated
        return self._pupil_plane_size
    @property
    def npix(self):
        """The number of pixels in each direction of the pupil-plane image.
        """
        self._illuminated
        return self._npix

    @lazy_property
    def good_pupil_size(self):
        """An estimate of a good pupil-plane image size.
        """
        # Although the user can set the pupil plane size and scale directly if desired, in most
        # cases it's nicer to have GalSim try to pick good values for these.

        # For the pupil plane size, we'll achieve Nyquist sampling in the focal plane if we sample
        # out to twice the diameter of the actual aperture in the pupil plane (completely
        # independent of wavelength, struts, obscurations, GSparams, and so on!).  This corresponds
        # to oversampling=1.0.  In fact, if we were willing to always use sinc interpolation, there
        # would never be any reason to go beyond this.  In practice, we usually use a faster, but
        # less accurate, quintic interpolant, which means we can benefit from improved sampling
        # (oversampling > 1.0) in some cases, especially when we're *not* modeling an atmosphere
        # which would otherwise tend to damp contributions at large k.
        return 2 * self.diam * self._oversampling

    @lazy_property
    def good_pupil_scale(self):
        """An estimate of a good pupil-plane image scale.
        """
        from .airy import Airy
        # For the pupil plane sampling interval, details like the obscuration and GSParams *are*
        # important as they affect the amount of aliasing encountered.  (An Airy profile has an
        # infinite extent in real space, so it *always* aliases at some level, more so with an
        # obscuration than without.  The GSParams settings indicate how much aliasing we're
        # willing to tolerate, so it's required here.)  To pick a good sampling interval, we start
        # with the interval that would be used for an obscured Airy GSObject profile.  If the
        # `screen_list` argument was supplied, then we also check its .stepk propertry, which
        # aggregates a good sampling interval from all of the wrapped PhaseScreens, and keep the
        # smaller stepk.
        if self._lam is None:
            # For Airy, pupil_plane_scale is independent of wavelength.  We could build an Airy with
            # lam_over_diam=1.0 and then alter the `good_pupil_scale = ...` line below
            # appropriately, but it's easier to just arbitrarily set `lam=500` if it wasn't set.
            lam = 500.0
        else:
            lam = self._lam
        airy = Airy(diam=self.diam, lam=lam, obscuration=self.obscuration, gsparams=self.gsparams)
        stepk = airy.stepk
        if self._screen_list is not None:
            screen_list = PhaseScreenList(self._screen_list)
            stepk = min(stepk,
                        screen_list._getStepK(lam=lam, diam=self.diam, obscuration=self.obscuration,
                                              gsparams=self.gsparams))
        return stepk * lam * 1.e-9 * (radians / arcsec) / (2 * np.pi * self._pad_factor)

    @lazy_property
    def _illuminated(self):
        # Now that we have good candidate sizes and scales, we load or generate the pupil plane
        # array.
        if self._pupil_plane_im is not None:  # Use image of pupil plane
            return self._load_pupil_plane()
        else:  # Use geometric parameters.
            if self._input_pupil_plane_scale is not None:
                self._pupil_plane_scale = self._input_pupil_plane_scale
                # Check input scale and warn if looks suspicious.
                if self._pupil_plane_scale > self.good_pupil_scale:
                    ratio = self.good_pupil_scale / self._pupil_plane_scale
                    galsim_warn("Input pupil_plane_scale may be too large for good sampling.\n"
                                "Consider decreasing pupil_plane_scale by a factor %f, and/or "
                                "check PhaseScreenPSF outputs for signs of folding in real "
                                "space."%(1./ratio))
            else:
                self._pupil_plane_scale = self.good_pupil_scale
            if self._input_pupil_plane_size is not None:
                self._pupil_plane_size = self._input_pupil_plane_size
                # Check input size and warn if looks suspicious
                if self._pupil_plane_size < self.good_pupil_size:
                    ratio = self.good_pupil_size / self._pupil_plane_size
                    galsim_warn("Input pupil_plane_size may be too small for good focal-plane"
                                "sampling.\n"
                                "Consider increasing pupil_plane_size by a factor %f, and/or "
                                "check PhaseScreenPSF outputs for signs of undersampling."%ratio)
            else:
                self._pupil_plane_size = self.good_pupil_size
            return self._generate_pupil_plane()

    def _generate_pupil_plane(self):
        """ Create an array of illuminated pixels parameterically.
        """
        ratio = self._pupil_plane_size/self._pupil_plane_scale
        # Fudge a little to prevent good_fft_size() from turning 512.0001 into 768.
        ratio *= (1.0 - 1.0/2**14)
        self._npix = Image.good_fft_size(int(np.ceil(ratio)))

        # Check FFT size
        if self._npix > self.gsparams.maximum_fft_size:
            raise GalSimFFTSizeError("Created pupil plane array that is too large.",self._npix)

        # Shrink scale such that size = scale * npix exactly.
        self._pupil_plane_scale = self._pupil_plane_size / self._npix

        radius = 0.5*self.diam
        if self._circular_pupil:
            illuminated = (self.rsqr < radius**2)
            if self.obscuration > 0.:
                illuminated *= self.rsqr >= (radius*self.obscuration)**2
        else:
            illuminated = (np.abs(self.u) < radius) & (np.abs(self.v) < radius)
            if self.obscuration > 0.:
                illuminated *= ((np.abs(self.u) >= radius*self.obscuration) *
                                      (np.abs(self.v) >= radius*self.obscuration))

        if self._nstruts > 0:
            # Add the initial rotation if requested, converting to radians.
            rot_u, rot_v = self.u, self.v
            if self._strut_angle.rad != 0.:
                rot_u, rot_v = rotate_xy(rot_u, rot_v, -self._strut_angle)
            rotang = 360. * degrees / self._nstruts
            # Then loop through struts setting to zero the regions which lie under the strut
            for istrut in range(self._nstruts):
                rot_u, rot_v = rotate_xy(rot_u, rot_v, -rotang)
                illuminated *= ((np.abs(rot_u) >= radius * self._strut_thick) + (rot_v < 0.0))
        return illuminated

    def _load_pupil_plane(self):
        """ Create an array of illuminated pixels with appropriate size and scale from an input
        image of the pupil.  The basic strategy is:

        1.  Read in array.
        2.  Determine the scale.
        3.  Pad the input array with zeros to meet the requested pupil size.
        4.  Check that the pupil plane sampling interval is at least as small as requested.
        5.  Optionally rotate pupil plane.
        """
        from . import fits
        # Handle multiple types of input: NumPy array, galsim.Image, or string for filename with
        # image.
        if isinstance(self._pupil_plane_im, np.ndarray):
            # Make it into an image.
            self._pupil_plane_im = Image(self._pupil_plane_im)
        elif isinstance(self._pupil_plane_im, Image):
            # Make sure not to overwrite input image.
            self._pupil_plane_im = self._pupil_plane_im.copy()
        else:
            # Read in image of pupil plane from file.
            self._pupil_plane_im = fits.read(self._pupil_plane_im)
        # scale = pupil_plane_im.scale # Interpret as either the pixel scale in meters, or None.
        pp_arr = self._pupil_plane_im.array
        self._npix = pp_arr.shape[0]

        # Check FFT size
        if self._npix > self.gsparams.maximum_fft_size:
            raise GalSimFFTSizeError("Loaded pupil plane array that is too large.", self._npix)

        # Sanity checks
        if self._pupil_plane_im.array.shape[0] != self._pupil_plane_im.array.shape[1]:
            raise GalSimValueError("Input pupil_plane_im must be square.",
                                   self._pupil_plane_im.array.shape)
        if self._pupil_plane_im.array.shape[0] % 2 == 1:
            raise GalSimValueError("Input pupil_plane_im must have even sizes.",
                                   self._pupil_plane_im.array.shape)

        # Set the scale, priority is:
        # 1.  pupil_plane_scale kwarg
        # 2.  image.scale if not None
        # 3.  Use diameter and farthest illuminated pixel.
        if self._input_pupil_plane_scale is not None:
            self._pupil_plane_scale = self._input_pupil_plane_scale
        elif self._pupil_plane_im.scale is not None:
            self._pupil_plane_scale = self._pupil_plane_im.scale
        else:
            # If self._pupil_plane_scale is not set yet, then figure it out from the distance
            # of the farthest illuminated pixel from the image center and the aperture diameter.
            # below is essentially np.linspace(-0.5, 0.5, self._npix)
            u = np.fft.fftshift(np.fft.fftfreq(self._npix))
            u, v = np.meshgrid(u, u)
            r = np.hypot(u, v)
            rmax_illum = np.max(r*(self._pupil_plane_im.array > 0))
            self._pupil_plane_scale = self.diam / (2.0 * rmax_illum * self._npix)
        self._pupil_plane_size = self._pupil_plane_scale * self._npix

        # Check the pupil plane size here and bump it up if necessary.
        if self._pupil_plane_size < self.good_pupil_size:
            new_npix = Image.good_fft_size(int(np.ceil(
                    self.good_pupil_size/self._pupil_plane_scale)))
            pad_width = (new_npix-self._npix)//2
            pp_arr = np.pad(pp_arr, [(pad_width, pad_width)]*2, mode='constant')
            self._npix = new_npix
            self._pupil_plane_size = self._pupil_plane_scale * self._npix

        # Check sampling interval and warn if it's not good enough.
        if self._pupil_plane_scale > self.good_pupil_scale:
            ratio = self._pupil_plane_scale / self.good_pupil_scale
            galsim_warn("Input pupil plane image may not be sampled well enough!\n"
                        "Consider increasing sampling by a factor %f, and/or check "
                        "PhaseScreenPSF outputs for signs of folding in real space."%ratio)

        if self._pupil_angle.rad == 0.:
            return pp_arr.astype(bool)
        else:
            # Rotate the pupil plane image as required based on the `pupil_angle`, being careful to
            # ensure that the image is one of the allowed types.  We ignore the scale.
            b = _BoundsI(1,self._npix,1,self._npix)
            im = _Image(pp_arr, b, PixelScale(1.))
            int_im = InterpolatedImage(im, x_interpolant='linear',
                                       calculate_stepk=False, calculate_maxk=False)
            int_im = int_im.rotate(self._pupil_angle)
            new_im = Image(pp_arr.shape[1], pp_arr.shape[0])
            new_im = int_im.drawImage(image=new_im, scale=1., method='no_pixel')
            pp_arr = new_im.array
            # Restore hard edges that might have been lost during the interpolation.  To do this, we
            # check the maximum value of the entries.  Values after interpolation that are >half
            # that maximum value are kept as nonzero (True), but those that are <half the maximum
            # value are set to zero (False).
            max_pp_val = np.max(pp_arr)
            pp_arr[pp_arr < 0.5*max_pp_val] = 0.
            return pp_arr.astype(bool)

    @property
    def gsparams(self):
        """The `GSParams` of this object.
        """
        return self._gsparams

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current aperture with the given gsparams
        """
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        return ret

    # Used in Aperture.__str__ and OpticalPSF.__str__
    def _geometry_str(self):
        s = ""
        if not self._circular_pupil:
            s += ", circular_pupil=False"
        if self.obscuration != 0.0:
            s += ", obscuration=%s"%self.obscuration
        if self._nstruts != 0:
            s += ", nstruts=%s"%self._nstruts
            if self._strut_thick != 0.05:
                s += ", strut_thick=%s"%self._strut_thick
            if self._strut_angle != 0*radians:
                s += ", strut_angle=%s"%self._strut_angle
        return s

    def __str__(self):
        s = "galsim.Aperture(diam=%r"%self.diam
        if self._pupil_plane_im is None:
            # Pupil was created geometrically, so use that here.
            s += self._geometry_str()
        s += ")"
        return s

    def _geometry_repr(self):
        s = ""
        if not self._circular_pupil:
            s += ", circular_pupil=False"
        if self.obscuration != 0.0:
            s += ", obscuration=%r"%self.obscuration
        if self._nstruts != 0:
            s += ", nstruts=%r"%self._nstruts
            if self._strut_thick != 0.05:
                s += ", strut_thick=%r"%self._strut_thick
            if self._strut_angle != 0*radians:
                s += ", strut_angle=%r"%self._strut_angle
        return s

    def __repr__(self):
        s = "galsim.Aperture(diam=%r"%self.diam
        if self._pupil_plane_im is None:
            # Pupil was created geometrically, so use that here.
            s += self._geometry_repr()
            s += ", pupil_plane_scale=%r"%self._input_pupil_plane_scale
            s += ", pupil_plane_size=%r"%self._input_pupil_plane_size
            s += ", oversampling=%r"%self._oversampling
            s += ", pad_factor=%r"%self._pad_factor
        else:
            # Pupil was created from image, so use that instead.
            # It's slightly less annoying to see an enormous stream of zeros fly by than an enormous
            # stream of Falses, so convert to int16.
            tmp = self.illuminated.astype(np.int16).tolist()
            s += ", pupil_plane_im=array(%r"%tmp+", dtype='int16')"
            s += ", pupil_plane_scale=%r"%self._pupil_plane_scale
        if self.gsparams != GSParams():
            s += ", gsparams=%r"%self.gsparams
        s += ")"
        return s

    def __eq__(self, other):
        if self is other: return True
        if not (isinstance(other, Aperture) and
                self.diam == other.diam and
                self._gsparams == other._gsparams):
            return False
        if self._pupil_plane_im is not None:
            return (self.pupil_plane_scale == other.pupil_plane_scale and
                    np.array_equal(self.illuminated, other.illuminated))
        else:
            return (other._pupil_plane_im is None and
                    self._circular_pupil == other._circular_pupil and
                    self._obscuration == other._obscuration and
                    self._nstruts == other._nstruts and
                    self._strut_thick == other._strut_thick and
                    self._strut_angle == other._strut_angle and
                    self._input_pupil_plane_scale == other._input_pupil_plane_scale and
                    self._input_pupil_plane_size == other._input_pupil_plane_size and
                    self._oversampling == other._oversampling and
                    self._pad_factor == other._pad_factor)

    def __hash__(self):
        # Cache since self.illuminated may be large.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.Aperture", self.diam, self.pupil_plane_scale))
            self._hash ^= hash(tuple(self.illuminated.ravel()))
        return self._hash

    # Properties show up nicely in the interactive terminal for
    #     >>>help(Aperture)
    # So we make a thin wrapper here.
    @property
    def illuminated(self):
        """A boolean array indicating which positions in the pupil plane are exposed to the sky.
        """
        return self._illuminated

    @lazy_property
    def rho(self):
        """Unit-disk normalized pupil plane coordinate as a complex number:
        (x, y) => x + 1j * y.
        """
        self._illuminated
        u = np.fft.fftshift(np.fft.fftfreq(self._npix, self.diam/self._pupil_plane_size/2.0))
        u, v = np.meshgrid(u, u)
        return u + 1j * v

    @lazy_property
    def _uv(self):
        if not hasattr(self, '_npix'):
            # Need this check, since `_uv` is used by `_illuminated`, so need to make sure we
            # don't have an infinite loop.
            self._illuminated
        u = np.fft.fftshift(np.fft.fftfreq(self._npix, 1./self._pupil_plane_size))
        u, v =  np.meshgrid(u, u)
        return u, v

    @property
    def u(self):
        """Pupil horizontal coordinate array in meters."""
        return self._uv[0]

    @property
    def v(self):
        """Pupil vertical coordinate array in meters."""
        return self._uv[1]

    @lazy_property
    def u_illuminated(self):
        """The u values for only the `illuminated` pixels.
        """
        return self.u[self.illuminated]

    @lazy_property
    def v_illuminated(self):
        """The v values for only the `illuminated` pixels.
        """
        return self.v[self.illuminated]

    @lazy_property
    def rsqr(self):
        """Pupil radius squared array in meters squared."""
        return self.u**2 + self.v**2

    @property
    def diam(self):
        """Aperture diameter in meters"""
        return self._diam

    @property
    def obscuration(self):
        """Fraction linear obscuration of pupil."""
        return self._obscuration

    def __getstate__(self):
        # Let unpickled object reconstruct cached values on-the-fly instead of including them in the
        # pickle.
        d = self.__dict__.copy()
        for k in ('rho', '_uv', 'rsqr', 'u_illuminated', 'v_illuminated'):
            d.pop(k, None)
        # Only reconstruct _illuminated if we made it from geometry.  If loaded, it's probably
        # faster to serialize the array.
        if self._pupil_plane_im is None:
            d.pop('_illuminated', None)
        return d

    # Some quick notes for Josh:
    # - Relation between real-space grid with size theta and pitch dtheta (dimensions of angle)
    #   and corresponding (fast) Fourier grid with size 2*maxk and pitch stepk (dimensions of
    #   inverse angle):
    #     stepk = 2*pi/theta
    #     maxk = pi/dtheta
    # - Relation between aperture of size L and pitch dL (dimensions of length, not angle!) and
    #   (fast) Fourier grid:
    #     dL = stepk * lambda / (2 * pi)
    #     L = maxk * lambda / pi
    # - Implies relation between aperture grid and real-space grid:
    #     dL = lambda/theta
    #     L = lambda/dtheta
    #
    # MJ: Of these four, only _sky_scale is still used.  The rest are left here for informational
    # purposes, but nothing actually calls them.
    def _getStepK(self, lam, scale_unit=arcsec):
        """Return the Fourier grid spacing for this aperture at given wavelength.

        Parameters:
            lam:        Wavelength in nanometers.
            scale_unit: Inverse units in which to return result [default: galsim.arcsec]

        Returns:
            Fourier grid spacing.
        """
        return 2*np.pi*self.pupil_plane_scale/(lam*1e-9) * scale_unit/radians

    def _getMaxK(self, lam, scale_unit=arcsec):
        """Return the Fourier grid half-size for this aperture at given wavelength.

        Parameters:
            lam:        Wavelength in nanometers.
            scale_unit: Inverse units in which to return result [default: galsim.arcsec]

        Returns:
            Fourier grid half-size.
        """
        return np.pi*self.pupil_plane_size/(lam*1e-9) * scale_unit/radians

    def _sky_scale(self, lam, scale_unit=arcsec):
        """Return the image scale for this aperture at given wavelength.

        Parameters:
            lam:        Wavelength in nanometers.
            scale_unit: Units in which to return result [default: galsim.arcsec]

        Returns:
            Image scale.
        """
        return (lam*1e-9) / self.pupil_plane_size * radians/scale_unit

    def _sky_size(self, lam, scale_unit=arcsec):
        """Return the image size for this aperture at given wavelength.

        Parameters:
            lam:        Wavelength in nanometers.
            scale_unit: Units in which to return result [default: galsim.arcsec]

        Returns:
            Image size.
        """
        return (lam*1e-9) / self.pupil_plane_scale * radians/scale_unit


class PhaseScreenList(object):
    """List of phase screens that can be turned into a PSF.  Screens can be either atmospheric
    layers or optical phase screens.  Generally, one would assemble a PhaseScreenList object using
    the function `Atmosphere`.  Layers can be added, removed, appended, etc. just like items can
    be manipulated in a python list.  For example::

        # Create an atmosphere with three layers.
        >>> screens = galsim.PhaseScreenList([galsim.AtmosphericScreen(...),
                                              galsim.AtmosphericScreen(...),
                                              galsim.AtmosphericScreen(...)])
        # Add another layer
        >>> screens.append(galsim.AtmosphericScreen(...))
        # Remove the second layer
        >>> del screens[1]
        # Switch the first and second layer.  Silly, but works...
        >>> screens[0], screens[1] = screens[1], screens[0]

    Parameters:
        layers:     Sequence of phase screens.
    """
    def __init__(self, *layers):
        from .phase_screens import AtmosphericScreen, OpticalScreen
        if len(layers) == 1:
            # First check if layers[0] is a PhaseScreenList, so we avoid nesting.
            if isinstance(layers[0], PhaseScreenList):
                self._layers = layers[0]._layers
            else:
                # Next, see if layers[0] is iterable.  E.g., to catch generator expressions.
                try:
                    self._layers = list(layers[0])
                except TypeError:
                    self._layers = list(layers)
        else:
            self._layers = list(layers)
        self._update_attrs()
        self._pending = []  # Pending PSFs to calculate upon first drawImage.

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, index):
        try:
            items = self._layers[index]
        except TypeError:
            msg = "{cls.__name__} indices must be integers or slices"
            raise TypeError(msg.format(cls=self.__class__))
        try:
            index + 1   # Regular in indices are the norm, so try something that works for it,
                        # but not for slices, where we need different handling.
        except TypeError:
            # index is a slice, so items is a list.
            return PhaseScreenList(items)
        else:
            # index is an int, so items is just one screen.
            return items

    def __setitem__(self, index, layer):
        self._layers[index] = layer
        self._update_attrs()

    def __delitem__(self, index):
        del self._layers[index]
        self._update_attrs()

    def append(self, layer):
        self._layers.append(layer)
        self._update_attrs()

    def extend(self, layers):
        self._layers.extend(layers)
        self._update_attrs()

    def __str__(self):
        return "galsim.PhaseScreenList([%s])" % ",".join(str(l) for l in self._layers)

    def __repr__(self):
        return "galsim.PhaseScreenList(%r)" % self._layers

    def __eq__(self, other):
        return (self is other or
                (isinstance(other,PhaseScreenList) and self._layers == other._layers))

    def __ne__(self, other): return not self == other

    __hash__ = None  # Mutable means not hashable.

    def _update_attrs(self):
        # If any of the wrapped PhaseScreens have an rng, then eval(repr(screen_list)) will run, but
        # fail to round-trip to the original object.  So we search for that here and set/delete a
        # dummy rng sentinel attribute so do_pickle() will know to skip the obj == eval(repr(obj))
        # test.
        self.__dict__.pop('rng', None)
        self.dynamic = any(l.dynamic for l in self)
        self.reversible = all(l.reversible for l in self)
        self.__dict__.pop('r0_500_effective', None)

    def _seek(self, t):
        """Set all layers' internal clocks to time t."""
        for layer in self:
            try:
                layer._seek(t)
            except AttributeError:
                # Time indep phase screen
                pass
        self._update_attrs()

    def _reset(self):
        """Reset phase screens back to time=0."""
        for layer in self:
            try:
                layer._reset()
            except AttributeError:
                # Time indep phase screen
                pass
        self._update_attrs()

    def instantiate(self, pool=None, _bar=None, **kwargs):
        """Instantiate the screens in this `PhaseScreenList`.

        Parameters:
            pool:       A multiprocessing.Pool object to use to instantiate screens in parallel.
            **kwargs:   Keyword arguments to forward to screen.instantiate().
        """
        _bar = _bar if _bar else dict()  # with dict() _bar.update() is a trivial no op.
        if pool is not None:
            results = []
            for layer in self:
                try:
                    results.append(pool.apply_async(layer.instantiate, kwds=kwargs))
                except AttributeError:  # OpticalScreen has no instantiate method
                    pass
                _bar.update()
            for r in results:
                r.wait()
        else:
            for layer in self:
                try:
                    layer.instantiate(**kwargs)
                except AttributeError:
                    pass
                _bar.update()

    def _delayCalculation(self, psf):
        """Add psf to delayed calculation list."""
        heappush(self._pending, (psf.t0, OrderedWeakRef(psf)))

    def _prepareDraw(self):
        """Calculate previously delayed PSFs."""
        if not self._pending:
            return
        # See if we have any dynamic screens.  If not, then we can immediately compute each PSF
        # in a simple loop.
        if not self.dynamic:
            for _, psfref in self._pending:
                psf = psfref()
                if psf is not None:
                    psf._step()
                    psf._finalize()
            self._pending = []
            self._update_time_heap = []
            return

        # If we do have time-evolving screens, then iteratively increment the time while being
        # careful to always stop at multiples of each PSF's time_step attribute to update that PSF.
        # Use a heap (in _pending list) to track the next time to stop at.
        while(self._pending):
            # Get and seek to next time that has a PSF update.
            t, psfref = heappop(self._pending)
            # Check if this PSF weakref is still alive
            psf = psfref()
            if psf is not None:
                # If it's alive, update this PSF
                self._seek(t)
                psf._step()
                # If that PSF's next possible update time doesn't extend past its exptime, then
                # push it back on the heap.
                t += psf.time_step
                if t < psf.t0 + psf.exptime:
                    heappush(self._pending, (t, OrderedWeakRef(psf)))
                else:
                    psf._finalize()
        self._pending = []

    def wavefront(self, u, v, t, theta=(0.0*radians, 0.0*radians)):
        """ Compute cumulative wavefront due to all phase screens in `PhaseScreenList`.

        Wavefront here indicates the distance by which the physical wavefront lags or leads the
        ideal plane wave (pre-optics) or spherical wave (post-optics).

        Parameters:
            u:      Horizontal pupil coordinate (in meters) at which to evaluate wavefront.  Can
                    be a scalar or an iterable.  The shapes of u and v must match.
            v:      Vertical pupil coordinate (in meters) at which to evaluate wavefront.  Can
                    be a scalar or an iterable.  The shapes of u and v must match.
            t:      Times (in seconds) at which to evaluate wavefront.  Can be None, a scalar or an
                    iterable.  If None, then the internal time of the phase screens will be used
                    for all u, v.  If scalar, then the size will be broadcast up to match that of
                    u and v.  If iterable, then the shape must match the shapes of u and v.
            theta:  Field angle at which to evaluate wavefront, as a 2-tuple of `galsim.Angle`
                    instances. [default: (0.0*galsim.arcmin, 0.0*galsim.arcmin)]
                    Only a single theta is permitted.

        Returns:
            Array of wavefront lag or lead in nanometers.
        """
        if len(self._layers) > 1:
            return np.sum([layer.wavefront(u, v, t, theta) for layer in self], axis=0)
        else:
            return self._layers[0].wavefront(u, v, t, theta)

    def wavefront_gradient(self, u, v, t, theta=(0.0*radians, 0.0*radians)):
        """ Compute cumulative wavefront gradient due to all phase screens in `PhaseScreenList`.

        Parameters:
            u:      Horizontal pupil coordinate (in meters) at which to evaluate wavefront.  Can
                    be a scalar or an iterable.  The shapes of u and v must match.
            v:      Vertical pupil coordinate (in meters) at which to evaluate wavefront.  Can
                    be a scalar or an iterable.  The shapes of u and v must match.
            t:      Times (in seconds) at which to evaluate wavefront gradient.  Can be None, a
                    scalar or an iterable.  If None, then the internal time of the phase screens
                    will be used for all u, v.  If scalar, then the size will be broadcast up to
                    match that of u and v.  If iterable, then the shape must match the shapes of
                    u and v.
            theta:  Field angle at which to evaluate wavefront, as a 2-tuple of `galsim.Angle`
                    instances. [default: (0.0*galsim.arcmin, 0.0*galsim.arcmin)]
                    Only a single theta is permitted.

        Returns:
            Arrays dWdu and dWdv of wavefront lag or lead gradient in nm/m.
        """
        if len(self._layers) > 1:
            return np.sum([layer.wavefront_gradient(u, v, t, theta) for layer in self], axis=0)
        else:
            return self._layers[0].wavefront_gradient(u, v, t, theta)

    def _wavefront(self, u, v, t, theta):
        if len(self._layers) > 1:
            return np.sum([layer._wavefront(u, v, t, theta) for layer in self], axis=0)
        else:
            return self._layers[0]._wavefront(u, v, t, theta)

    def _wavefront_gradient(self, u, v, t, theta):
        gradx, grady = self._layers[0]._wavefront_gradient(u, v, t, theta)
        for layer in self._layers[1:]:
            gx, gy = layer._wavefront_gradient(u, v, t, theta)
            gradx += gx
            grady += gy
        return gradx, grady

    def makePSF(self, lam, **kwargs):
        """Create a PSF from the current `PhaseScreenList`.

        Parameters:
            lam:                Wavelength in nanometers at which to compute PSF.
            t0:                 Time at which to start exposure in seconds.  [default: 0.0]
            exptime:            Time in seconds over which to accumulate evolving instantaneous
                                PSF.  [default: 0.0]
            time_step:          Time interval in seconds with which to sample phase screens when
                                drawing using real-space or Fourier methods, or when using
                                photon-shooting without the geometric optics approximation.  Note
                                that the default value of 0.025 is fairly arbitrary.  For careful
                                studies, we recommend checking that results are stable when
                                decreasing time_step.  Also note that when drawing using
                                photon-shooting with the geometric optics approximation this
                                keyword is ignored, as the phase screen can be sampled
                                continuously in this case instead of at discrete intervals.
                                [default: 0.025]
            flux:               Flux of output PSF.  [default: 1.0]
            theta:              Field angle of PSF as a 2-tuple of `Angle` instances.
                                [default: (0.0*galsim.arcmin, 0.0*galsim.arcmin)]
            interpolant:        Either an Interpolant instance or a string indicating which
                                interpolant should be used.  Options are 'nearest', 'sinc',
                                'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                                integer order to use. [default: galsim.Quintic()]
            scale_unit:         Units to use for the sky coordinates of the output profile.
                                [default: galsim.arcsec]
            ii_pad_factor:      Zero-padding factor by which to extend the image of the PSF when
                                creating the ``InterpolatedImage``.  See the
                                ``InterpolatedImage`` docstring for more details.  [default: 1.5]
            suppress_warning:   If ``pad_factor`` is too small, the code will emit a warning
                                telling you its best guess about how high you might want to raise
                                it.  However, you can suppress this warning by using
                                ``suppress_warning=True``.  [default: False]
            geometric_shooting: If True, then when drawing using photon shooting, use geometric
                                optics approximation where the photon angles are derived from the
                                phase screen gradient.  If False, then first draw using Fourier
                                optics and then shoot from the derived InterpolatedImage.
                                [default: True]
            aper:               `Aperture` to use to compute PSF(s).  [default: None]
            second_kick:        An optional second kick to also convolve by when using geometric
                                photon-shooting.  (This can technically be any `GSObject`, though
                                usually it should probably be a SecondKick object).  If None, then a
                                good second kick will be chosen automatically based on
                                ``screen_list``.  If False, then a second kick won't be applied.
                                [default: None]
            kcrit:              Critical Fourier scale (in units of 1/r0) at which to separate low-k
                                and high-k turbulence.  The default value was chosen based on
                                comparisons between Fourier optics and geometric optics with a
                                second kick correction.  While most values of kcrit smaller than the
                                default produce similar results, we caution the user to compare the
                                affected geometric PSFs against Fourier optics PSFs carefully before
                                changing this value.  [default: 0.2]
            fft_sign:           The sign (+/-) to use in the exponent of the Fourier kernel when
                                evaluating the Fourier optics PSF.  As of version 2.3, GalSim uses a
                                plus sign by default, which we believe to be consistent with, for
                                example, how Zemax computes a Fourier optics PSF on DECam.  Before
                                version 2.3, the default was a negative sign.  Input should be
                                either the string '+' or the string '-'.  [default: '+']
            gsparams:           An optional `GSParams` argument. [default: None]

        The following are optional keywords to use to setup the aperture if ``aper`` is not
        provided.

        Parameters:
            diam:               Aperture diameter in meters.
            circular_pupil:     Adopt a circular pupil?  [default: True]
            obscuration:        Linear dimension of central obscuration as fraction of aperture
                                linear dimension. [0., 1.).  [default: 0.0]
            nstruts:            Number of radial support struts to add to the central
                                obscuration. [default: 0]
            strut_thick:        Thickness of support struts as a fraction of aperture diameter.
                                [default: 0.05]
            strut_angle:        `Angle` made between the vertical and the strut starting closest to
                                it, defined to be positive in the counter-clockwise direction;
                                must be an `Angle` instance. [default: 0. * galsim.degrees]
            oversampling:       Optional oversampling factor *in the image plane* for the PSF
                                eventually constructed using this `Aperture`.  Setting
                                ``oversampling < 1`` will produce aliasing in the PSF (not good).
                                [default: 1.0]
            pad_factor:         Additional multiple by which to extend the PSF image to avoid
                                folding.  [default: 1.0]
            pupil_plane_im:     The GalSim.Image, NumPy array, or name of file containing the
                                pupil plane image, to be used instead of generating one based on
                                the obscuration and strut parameters.  [default: None]
            pupil_angle:        If ``pupil_plane_im`` is not None, rotation angle for the pupil
                                plane (positive in the counter-clockwise direction).  Must be an
                                `Angle` instance. [default: 0. * galsim.degrees]
            pupil_plane_scale:  Sampling interval in meters to use for the pupil plane array.  In
                                most cases, it's a good idea to leave this as None, in which case
                                GalSim will attempt to find a good value automatically.  The
                                exception is when specifying the pupil arrangement via an image,
                                in which case this keyword can be used to indicate the sampling
                                of that image.  See also ``pad_factor`` for adjusting the pupil
                                sampling scale. [default: None]
            pupil_plane_size:   Size in meters to use for the pupil plane array.  In most cases,
                                it's a good idea to leave this as None, in which case GalSim will
                                attempt to find a good value automatically.  See also
                                ``oversampling`` for adjusting the pupil size.  [default: None]
        """
        return PhaseScreenPSF(self, lam, **kwargs)

    @lazy_property
    def r0_500_effective(self):
        """Effective r0_500 for set of screens in list that define an r0_500 attribute."""
        r0_500s = np.array([l.r0_500 for l in self if hasattr(l, 'r0_500')])
        if len(r0_500s) == 0:
            return None
        else:
            return np.sum(r0_500s**(-5./3))**(-3./5)

    def _getStepK(self, **kwargs):
        """Return an appropriate stepk for this list of phase screens.

        The required set of parameters depends on the types of the individual `PhaseScreen`
        instances in the `PhaseScreenList`.  See the documentation for the individual
        `PhaseScreen.pupil_plane_scale` methods for more details.

        Returns:
            stepk.
        """
        # Generically, GalSim propagates stepk for convolutions using
        #   stepk = sum(s**-2 for s in stepks)**(-0.5)
        # We're not actually doing convolution between screens here, though.  In fact, the right
        # relation for Kolmogorov screens uses exponents -5./3 and -3./5:
        #   stepk = sum(s**(-5./3) for s in stepks)**(-3./5)
        # Since most of the layers in a PhaseScreenList are likely to be (nearly) Kolmogorov
        # screens, we'll use that relation.
        return np.sum([layer._getStepK(**kwargs)**(-5./3) for layer in self])**(-3./5)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_pending'] = []
        return d


class PhaseScreenPSF(GSObject):
    """A PSF surface brightness profile constructed by integrating over time the instantaneous PSF
    derived from a set of phase screens and an aperture.

    There are two equivalent ways to construct a PhaseScreenPSF given a `PhaseScreenList`::

        >>> psf = screen_list.makePSF(...)
        >>> psf = PhaseScreenPSF(screen_list, ...)

    Computing a PSF from a phase screen also requires an `Aperture` be specified.  This can be done
    either directly via the ``aper`` keyword, or by setting a number of keywords that will be passed
    to the `Aperture` constructor.  The ``aper`` keyword always takes precedence.

    There are effectively three ways to draw a PhaseScreenPSF (or `GSObject` that includes a
    PhaseScreenPSF):

    1) Fourier optics

        This is the default, and is performed for all drawImage methods except method='phot'.  This
        is generally the most accurate option.  For a `PhaseScreenList` that includes an
        `AtmosphericScreen`, however, this can be prohibitively slow.  For `OpticalPSF`, though,
        this can sometimes be a good option.

    2) Photon-shooting from an image produced using Fourier optics.

        This is done if geometric_shooting=False when creating the PhaseScreenPSF, and method='phot'
        when calling drawImage.  This actually performs the same calculations as the Fourier optics
        option above, but then proceeds by shooting photons from that result.  This can sometimes be
        a good option for OpticalPSFs, especially if the same OpticalPSF can be reused for may
        objects, since the Fourier part of the process would only be performed once in this case.

    3) Photon-shooting using the "geometric approximation".

        This is done if geometric_shooting=True when creating the PhaseScreenPSF, and method='phot'
        when calling drawImage.  In this case, a completely different algorithm is used make an
        image.  Photons are uniformly generated in the `Aperture` pupil, and then the phase gradient
        at that location is used to deflect each photon in the image plane.  This method, which
        corresponds to geometric optics, is broadly accurate for phase screens that vary slowly
        across the aperture, and is usually several orders of magnitude or more faster than Fourier
        optics (depending on the flux of the object, of course, but much faster even for rather
        bright flux levels).

        One short-coming of this method is that it neglects interference effects, i.e. diffraction.
        For `PhaseScreenList` that include at least one `AtmosphericScreen`, a correction, dubbed
        the "second kick", will automatically be applied to handle both the quickly varying modes
        of the screens and the diffraction pattern of the `Aperture`.  For PhaseScreenLists without
        an `AtmosphericScreen`, the correction is simply an Airy function.  Note that this
        correction can be overridden using the second_kick keyword argument, and also tuned to some
        extent using the kcrit keyword argument.

    Note also that calling drawImage on a PhaseScreenPSF that uses a `PhaseScreenList` with any
    uninstantiated `AtmosphericScreen` will perform that instantiation, and that the details of the
    instantiation depend on the drawing method used, and also the kcrit keyword argument to
    PhaseScreenPSF.  See the `AtmosphericScreen` docstring for more details.

    Parameters:
        screen_list:        `PhaseScreenList` object from which to create PSF.
        lam:                Wavelength in nanometers at which to compute PSF.
        t0:                 Time at which to start exposure in seconds.  [default: 0.0]
        exptime:            Time in seconds over which to accumulate evolving instantaneous PSF.
                            [default: 0.0]
        time_step:          Time interval in seconds with which to sample phase screens when
                            drawing using real-space or Fourier methods, or when using
                            photon-shooting without the geometric optics approximation.  Note
                            that the default value of 0.025 is fairly arbitrary.  For careful
                            studies, we recommend checking that results are stable when
                            decreasing time_step.  Also note that when drawing using
                            photon-shooting with the geometric optics approximation this
                            keyword is ignored, as the phase screen can be sampled
                            continuously in this case instead of at discrete intervals.
                            [default: 0.025]
        flux:               Flux of output PSF [default: 1.0]
        theta:              Field angle of PSF as a 2-tuple of `Angle` instances.
                            [default: (0.0*galsim.arcmin, 0.0*galsim.arcmin)]
        interpolant:        Either an Interpolant instance or a string indicating which
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use.  [default: galsim.Quintic()]
        scale_unit:         Units to use for the sky coordinates of the output profile.
                            [default: galsim.arcsec]
        ii_pad_factor:      Zero-padding factor by which to extend the image of the PSF when
                            creating the ``InterpolatedImage``.  See the ``InterpolatedImage``
                            docstring for more details.  [default: 1.5]
        suppress_warning:   If ``pad_factor`` is too small, the code will emit a warning telling
                            you its best guess about how high you might want to raise it.
                            However, you can suppress this warning by using
                            ``suppress_warning=True``.  [default: False]
        geometric_shooting: If True, then when drawing using photon shooting, use geometric
                            optics approximation where the photon angles are derived from the
                            phase screen gradient.  If False, then first draw using Fourier
                            optics and then shoot from the derived InterpolatedImage.
                            [default: True]
        aper:               `Aperture` to use to compute PSF(s).  [default: None]
        second_kick:        An optional second kick to also convolve by when using geometric
                            photon-shooting.  (This can technically be any `GSObject`, though
                            usually it should probably be a SecondKick object).  If None, then a
                            good second kick will be chosen automatically based on
                            ``screen_list``.  If False, then a second kick won't be applied.
                            [default: None]
        kcrit:              Critical Fourier scale (in units of 1/r0) at which to separate low-k
                            and high-k turbulence.  The default value was chosen based on
                            comparisons between Fourier optics and geometric optics with a second
                            kick correction.  While most values of kcrit smaller than the default
                            produce similar results, we caution the user to compare the affected
                            geometric PSFs against Fourier optics PSFs carefully before changing
                            this value.  [default: 0.2]
        fft_sign:           The sign (+/-) to use in the exponent of the Fourier kernel when
                            evaluating the Fourier optics PSF.  As of version 2.3, GalSim uses a
                            plus sign by default, which we believe to be consistent with, for
                            example, how Zemax computes a Fourier optics PSF on DECam.  Before
                            version 2.3, the default was a negative sign.  Input should be either
                            the string '+' or the string '-'.  [default: '+']
        gsparams:           An optional `GSParams` argument. [default: None]

    The following are optional keywords to use to setup the aperture if ``aper`` is not provided:

    Parameters:
        diam:               Aperture diameter in meters. [default: None]
        circular_pupil:     Adopt a circular pupil?  [default: True]
        obscuration:        Linear dimension of central obscuration as fraction of aperture
                            linear dimension. [0., 1.).  [default: 0.0]
        nstruts:            Number of radial support struts to add to the central obscuration.
                            [default: 0]
        strut_thick:        Thickness of support struts as a fraction of aperture diameter.
                            [default: 0.05]
        strut_angle:        `Angle` made between the vertical and the strut starting closest to it,
                            defined to be positive in the counter-clockwise direction; must be an
                            `Angle` instance. [default: 0. * galsim.degrees]
        oversampling:       Optional oversampling factor *in the image plane* for the PSF
                            eventually constructed using this `Aperture`.  Setting
                            ``oversampling < 1`` will produce aliasing in the PSF (not good).
                            [default: 1.0]
        pad_factor:         Additional multiple by which to extend the PSF image to avoid
                            folding.  [default: 1.0]
        pupil_plane_im:     The GalSim.Image, NumPy array, or name of file containing the pupil
                            plane image, to be used instead of generating one based on the
                            obscuration and strut parameters.  [default: None]
        pupil_angle:        If ``pupil_plane_im`` is not None, rotation angle for the pupil plane
                            (positive in the counter-clockwise direction).  Must be an `Angle`
                            instance. [default: 0. * galsim.degrees]
        pupil_plane_scale:  Sampling interval in meters to use for the pupil plane array.  In
                            most cases, it's a good idea to leave this as None, in which case
                            GalSim will attempt to find a good value automatically.  The
                            exception is when specifying the pupil arrangement via an image, in
                            which case this keyword can be used to indicate the sampling of that
                            image.  See also ``pad_factor`` for adjusting the pupil sampling
                            scale.  [default: None]
        pupil_plane_size:   Size in meters to use for the pupil plane array.  In most cases, it's
                            a good idea to leave this as None, in which case GalSim will attempt
                            to find a good value automatically.  See also ``oversampling`` for
                            adjusting the pupil size.  [default: None]
    """
    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    _default_iipf = 1.5

    def __init__(self, screen_list, lam, t0=0.0, exptime=0.0, time_step=0.025, flux=1.0,
                 theta=(0.0*arcsec, 0.0*arcsec), interpolant=None, scale_unit=arcsec,
                 ii_pad_factor=None, suppress_warning=False,
                 geometric_shooting=True, aper=None, second_kick=None, kcrit=0.2, fft_sign='+',
                 gsparams=None, _force_stepk=0., _force_maxk=0., _bar=None, **kwargs):
        # Hidden `_bar` kwarg can be used with astropy.console.utils.ProgressBar to print out a
        # progress bar during long calculations.

        if not isinstance(screen_list, PhaseScreenList):
            screen_list = PhaseScreenList(screen_list)

        if fft_sign not in ['+', '-']:
            raise GalSimValueError("Invalid fft_sign", fft_sign, allowed_values=['+','-'])

        self._screen_list = screen_list
        self.t0 = float(t0)
        self.lam = float(lam)
        self.exptime = float(exptime)
        self.time_step = float(time_step)
        if aper is None:
            # Check here for diameter.
            if 'diam' not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Diameter required if aperture not specified directly.", diam=None, aper=aper)
            aper = Aperture(lam=lam, screen_list=self._screen_list, gsparams=gsparams, **kwargs)
        elif gsparams is None:
            gsparams = aper.gsparams
        else:
            aper = aper.withGSParams(gsparams)
        self.aper = aper

        if not isinstance(theta[0], Angle) or not isinstance(theta[1], Angle):
            raise TypeError("theta must be 2-tuple of galsim.Angle's.")
        self.theta = theta
        self.interpolant = interpolant
        if isinstance(scale_unit, str):
            scale_unit = AngleUnit.from_name(scale_unit)
        self.scale_unit = scale_unit
        self._gsparams = GSParams.check(gsparams)
        self.scale = aper._sky_scale(self.lam, self.scale_unit)

        self._force_stepk = _force_stepk
        self._force_maxk = _force_maxk

        self._img = None

        if self.exptime < 0:
            raise GalSimRangeError("Cannot integrate PSF for negative time.", self.exptime, 0.)

        self._ii_pad_factor = ii_pad_factor if ii_pad_factor is not None else self._default_iipf

        self._bar = _bar if _bar else dict()  # with dict() _bar.update() is a trivial no op.
        self._flux = float(flux)
        self._suppress_warning = suppress_warning
        self._geometric_shooting = geometric_shooting
        self._kcrit = kcrit
        self._fft_sign = fft_sign
        # We'll set these more intelligently as needed below
        self._second_kick = second_kick
        self._screen_list._delayCalculation(self)
        self._finalized = False

    @lazy_property
    def _real_ii(self):
        ii = InterpolatedImage(
                self._img, x_interpolant=self.interpolant,
                _force_stepk=self._force_stepk, _force_maxk=self._force_maxk,
                pad_factor=self._ii_pad_factor,
                use_true_center=False, gsparams=self._gsparams)

        if not self._suppress_warning:
            specified_stepk = 2*np.pi/(self._img.array.shape[0]*self.scale)
            observed_stepk = ii.stepk

            if observed_stepk < specified_stepk:
                galsim_warn(
                    "The calculated stepk (%g) for PhaseScreenPSF is smaller than what was used "
                    "to build the wavefront (%g). This could lead to aliasing problems. "
                    "Increasing pad_factor is recommended."%(observed_stepk, specified_stepk))
        return ii

    @lazy_property
    def _dummy_ii(self):
        # If we need self._ii before we've done _prepareDraw, then build a placeholder that has
        # roughly the right properties. All we really need is for the stepk and maxk to be
        # correct, so use the force_ options to set them how we want.
        if self._force_stepk > 0.:
            stepk = self._force_stepk
        else:
            stepk = self._screen_list._getStepK(lam=self.lam, diam=self.aper.diam,
                                                obscuration=self.aper.obscuration,
                                                gsparams=self._gsparams)
        if self._force_maxk > 0.:
            maxk = self._force_maxk
        else:
            maxk = self.aper._getMaxK(self.lam, self.scale_unit)
        image = _Image(np.array([[self._flux]], dtype=float),
                       _BoundsI(1, 1, 1, 1), PixelScale(1.))
        interpolant = 'delta'  # Use delta so it doesn't contribute to stepk
        return InterpolatedImage(
                image, pad_factor=1.0, x_interpolant=interpolant,
                _force_stepk=stepk, _force_maxk=maxk)

    @property
    def _ii(self):
        if self._finalized:
            return self._real_ii
        else:
            return self._dummy_ii

    @property
    def kcrit(self):
        """The critical Fourier scale being used for this object.
        """
        return self._kcrit

    @property
    def fft_sign(self):
        """The sign (+/-) to use in the exponent of the Fourier kernel when evaluating the Fourier
        optics PSF.
        """
        return self._fft_sign

    @lazy_property
    def screen_kmax(self):
        """The maximum k value to use in the screen.  Typically `kcrit`/r0.
        """
        r0_500 = self._screen_list.r0_500_effective
        if r0_500 is None:
            return np.inf
        else:
            r0 = r0_500 * (self.lam/500)**(6./5)
            return self.kcrit / r0

    @lazy_property
    def second_kick(self):
        """Make a SecondKick object based on contents of screen_list and aper.
        """
        from .airy import Airy
        from .second_kick import SecondKick
        if self._second_kick is None:
            r0_500 = self._screen_list.r0_500_effective
            if r0_500 is None:  # No AtmosphericScreens in list
                return Airy(lam=self.lam, diam=self.aper.diam,
                            obscuration=self.aper.obscuration, gsparams=self._gsparams)
            else:
                r0 = r0_500 * (self.lam/500.)**(6./5)
                return SecondKick(
                        self.lam, r0, self.aper.diam, self.aper.obscuration,
                        kcrit=self.kcrit, scale_unit=self.scale_unit,
                        gsparams=self._gsparams)
        else:
            return self._second_kick

    @property
    def flux(self):
        """The flux of the profile.
        """
        return self._flux

    @property
    def screen_list(self):
        """The `PhaseScreenList` being used for this object.
        """
        return self._screen_list

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams: return self
        gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        aper = self.aper.withGSParams(gsparams)
        ret = self.__class__.__new__(self.__class__)
        ret.__dict__.update(self.__dict__)
        # Make sure we generate fresh versions of any attrs that depend on gsparams
        for attr in ['second_kick', '_real_ii', '_dummy_ii']:
            ret.__dict__.pop(attr, None)
        ret._gsparams = gsparams
        ret.aper = aper
        # Make sure we mark that we need to recalculate any previously finalized InterpolatedImage
        ret._finalized = False
        ret._screen_list._delayCalculation(ret)
        ret._img = None
        return ret

    def __str__(self):
        return ("galsim.PhaseScreenPSF(%s, lam=%s, exptime=%s)" %
                (self._screen_list, self.lam, self.exptime))

    def __repr__(self):
        outstr = ("galsim.PhaseScreenPSF(%r, lam=%r, exptime=%r, flux=%r, aper=%r, theta=%r, "
                  "interpolant=%r, scale_unit=%r, fft_sign=%r, gsparams=%r)")
        return outstr % (self._screen_list, self.lam, self.exptime, self.flux, self.aper,
                         self.theta, self.interpolant, self.scale_unit,
                         self._fft_sign, self.gsparams)

    def __eq__(self, other):
        # Even if two PSFs were generated with different sets of parameters, they will act
        # identically if their img, interpolant, stepk, maxk, pad_factor, fft_sign and gsparams
        # match.
        return (self is other or
                (isinstance(other, PhaseScreenPSF) and
                 self._screen_list == other._screen_list and
                 self.lam == other.lam and
                 self.aper == other.aper and
                 self.t0 == other.t0 and
                 self.exptime == other.exptime and
                 self.time_step == other.time_step and
                 self._flux == other._flux and
                 self.interpolant == other.interpolant and
                 self._force_stepk == other._force_stepk and
                 self._force_maxk == other._force_maxk and
                 self._ii_pad_factor == other._ii_pad_factor and
                 self._fft_sign == other._fft_sign and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.PhaseScreenPSF", tuple(self._screen_list), self.lam, self.aper,
                     self.t0, self.exptime, self.time_step, self._flux, self.interpolant,
                     self._force_stepk, self._force_maxk, self._ii_pad_factor, self._fft_sign,
                     self.gsparams))

    def _prepareDraw(self):
        # Trigger delayed computation of all pending PSFs.
        self._screen_list._prepareDraw()

    def _step(self):
        """Compute the current instantaneous PSF and add it to the developing integrated PSF."""
        from . import fft
        u = self.aper.u_illuminated
        v = self.aper.v_illuminated
        # This is where I need to make sure the screens are instantiated for FFT.
        self._screen_list.instantiate(check='FFT')
        wf = self._screen_list._wavefront(u, v, None, self.theta)
        expwf = np.exp((2j*np.pi/self.lam) * wf)
        expwf_grid = np.zeros_like(self.aper.illuminated, dtype=np.complex128)
        expwf_grid[self.aper.illuminated] = expwf
        # Note fft is '-' and ifft is '+' below
        if self._fft_sign == '+':
            ftexpwf = fft.ifft2(expwf_grid, shift_in=True, shift_out=True)
        else:
            ftexpwf = fft.fft2(expwf_grid, shift_in=True, shift_out=True)
        if self._img is None:
            self._img = np.zeros(self.aper.illuminated.shape, dtype=np.float64)
        self._img += np.abs(ftexpwf)**2
        self._bar.update()

    def _finalize(self):
        """Take accumulated integrated PSF image and turn it into a proper GSObject."""
        self._img *= self._flux / self._img.sum(dtype=float)
        b = _BoundsI(1,self.aper.npix,1,self.aper.npix)
        self._img = _Image(self._img, b, PixelScale(self.scale))

        self._finalized = True

    def __getstate__(self):
        d = self.__dict__.copy()
        # The SBProfile is picklable, but it is pretty inefficient, due to the large images being
        # written as a string.  Better to pickle the image and remake the InterpolatedImage.
        d.pop('_dummy_ii',None)
        d.pop('_real_ii',None)
        d.pop('second_kick',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        if not self._finalized:
            self._screen_list._delayCalculation(self)

    @property
    def _maxk(self):
        return self._ii.maxk

    @property
    def _stepk(self):
        return self._ii.stepk

    @property
    def _centroid(self):
        self._prepareDraw()
        return self._ii.centroid

    @property
    def _positive_flux(self):
        if self._geometric_shooting:
            return self._flux
        else:
            return self._ii.positive_flux

    @property
    def _negative_flux(self):
        if self._geometric_shooting:
            return 0.
        else:
            return self._ii.negative_flux

    @property
    def _flux_per_photon(self):
        if self._geometric_shooting:
            return 1.
        else:
            return self._calculate_flux_per_photon()

    @property
    def _max_sb(self):
        return self._ii.max_sb

    def _xValue(self, pos):
        self._prepareDraw()
        return self._ii._xValue(pos)

    def _kValue(self, kpos):
        self._prepareDraw()
        return self._ii._kValue(kpos)

    def _drawReal(self, image, jac=None, offset=(0.,0.), flux_scaling=1.):
        self._ii._drawReal(image, jac, offset, flux_scaling)

    def _shoot(self, photons, rng):
        from .photon_array import PhotonArray
        from .random import UniformDeviate

        if not self._geometric_shooting:
            self._prepareDraw()
            return self._ii._shoot(photons, rng)

        n_photons = len(photons)
        t = np.empty((n_photons,), dtype=float)
        ud = UniformDeviate(rng)
        ud.generate(t)
        t *= self.exptime
        t += self.t0
        u = self.aper.u_illuminated
        v = self.aper.v_illuminated
        pick = np.empty((n_photons,), dtype=float)
        ud.generate(pick)
        pick *= len(u)
        pick = pick.astype(int)
        u = u[pick]
        v = v[pick]

        # This is where the screens need to be instantiated for drawing with geometric photon
        # shooting.
        self._screen_list.instantiate(kmax=self.screen_kmax, check='phot')
        nm_to_arcsec = 1.e-9 * radians / arcsec
        if self._fft_sign == '+':
            nm_to_arcsec *= -1
        photons.x, photons.y = self._screen_list._wavefront_gradient(u, v, t, self.theta)
        photons.x *= nm_to_arcsec
        photons.y *= nm_to_arcsec
        photons.flux = self._flux / n_photons

        if self.second_kick:
            p2 = PhotonArray(len(photons))
            self.second_kick._shoot(p2, rng)
            photons.convolve(p2, rng)

    def _drawKImage(self, image, jac=None):
        self._ii._drawKImage(image, jac)

    @property
    def img(self):
        from .deprecated import depr
        depr('img', 2.1, '', "This functionality has been removed.")
        return self._img

    @property
    def finalized(self):
        from .deprecated import depr
        depr('finalized', 2.1, "This functionality has been removed.")
        return self._finalized

    @doc_inherit
    def withFlux(self, flux):
        if self._finalized:
            # Then it's probably not faster to rebuild with a different flux.
            return self.withScaledFlux(flux / self.flux)
        else:
            return PhaseScreenPSF(self._screen_list, lam=self.lam, exptime=self.exptime, flux=flux,
                                  aper=self.aper, theta=self.theta, interpolant=self.interpolant,
                                  scale_unit=self.scale_unit, gsparams=self.gsparams)


class OpticalPSF(GSObject):
    """A class describing aberrated PSFs due to telescope optics.  Its underlying implementation
    uses an InterpolatedImage to characterize the profile.

    The diffraction effects are characterized by the diffraction angle, which is a function of the
    ratio lambda / D, where lambda is the wavelength of the light and D is the diameter of the
    telescope.  The natural unit for this value is radians, which is not normally a convenient
    unit to use for other `GSObject` dimensions.  Assuming that the other sky coordinates you are
    using are all in arcsec (e.g. the pixel scale when you draw the image, the size of the galaxy,
    etc.), then you should convert this to arcsec as well::

        >>> lam = 700  # nm
        >>> diam = 4.0    # meters
        >>> lam_over_diam = (lam * 1.e-9) / diam  # radians
        >>> lam_over_diam *= 206265  # Convert to arcsec
        >>> psf = galsim.OpticalPSF(lam_over_diam, ...)

    To make this process a bit simpler, we recommend instead providing the wavelength and diameter
    separately using the parameters ``lam`` (in nm) and ``diam`` (in m).  GalSim will then convert
    this to any of the normal kinds of angular units using the ``scale_unit`` parameter::

        >>> psf = galsim.OpticalPSF(lam=lam, diam=diam, scale_unit=galsim.arcsec, ...)

    When drawing images, the scale_unit should match the unit used for the pixel scale or the WCS.
    e.g. in this case, a pixel scale of 0.2 arcsec/pixel would be specified as ``pixel_scale=0.2``.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention defined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976).  For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.  By default, the
    aberration coefficients indicate the amplitudes of _circular_ Zernike polynomials, which are
    orthogonal over a circle.  If you would like the aberration coefficients to instead be
    interpretted as the amplitudes of _annular_ Zernike polynomials, which are orthogonal over an
    annulus (see Mahajan, J. Opt. Soc. Am. 71, 1 (1981)), set the ``annular_zernike`` keyword
    argument to True.

    There are two ways to specify the geometry of the pupil plane, i.e., the obscuration disk size
    and the areas that will be illuminated outside of it.  The first way is to use keywords that
    specify the size of the obscuration, and the nature of the support struts holding up the
    secondary mirror (or prime focus cage, etc.).  These are taken to be rectangular obscurations
    extending from the outer edge of the pupil to the outer edge of the obscuration disk (or the
    pupil center if ``obscuration = 0.``).  You can specify how many struts there are (evenly spaced
    in angle), how thick they are as a fraction of the pupil diameter, and what angle they start at
    relative to the positive y direction.

    The second way to specify the pupil plane configuration is by passing in an image of it.  This
    can be useful for example if the struts are not evenly spaced or are not radially directed, as
    is assumed by the simple model for struts described above.  In this case, keywords related to
    struts are ignored; moreover, the ``obscuration`` keyword is used to ensure that the images are
    properly sampled (so it is still needed), but the keyword is then ignored when using the
    supplied image of the pupil plane.  Note that for complicated pupil configurations, it may be
    desireable to increase ``pad_factor`` for more fidelity at the expense of slower running time.
    The ``pupil_plane_im`` that is passed in can be rotated during internal calculations by
    specifying a ``pupil_angle`` keyword.

    If you choose to pass in a pupil plane image, it must be a square array in which the image of
    the pupil is centered.  The areas that are illuminated should have some value >0, and the other
    areas should have a value of precisely zero.  Based on what the OpticalPSF class thinks is the
    required sampling to make the PSF image, the image that is passed in of the pupil plane might be
    zero-padded during internal calculations.  The pixel scale of the pupil plane can be specified
    in one of three ways.  In descending order of priority, these are:

    1.  The ``pupil_plane_scale`` keyword argument (units are meters).
    2.  The ``pupil_plane_im.scale`` attribute (units are meters).
    3.  If (1) and (2) are both None, then the scale will be inferred by assuming that the
        illuminated pixel farthest from the image center is at a physical distance of self.diam/2.

    Note that if the scale is specified by either (1) or (2) above (which always includes specifying
    the pupil_plane_im as a filename, since the default scale then will be 1.0), then the
    lam_over_diam keyword must not be used, but rather the lam and diam keywords are required
    separately.  Finally, to ensure accuracy of calculations using a pupil plane image, we recommend
    sampling it as finely as possible.

    As described above, either specify the lam/diam ratio directly in arbitrary units::

        >>> optical_psf = galsim.OpticalPSF(lam_over_diam=lam_over_diam, defocus=0., ...)

    or, use separate keywords for the telescope diameter and wavelength in meters and nanometers,
    respectively::

        >>> optical_psf = galsim.OpticalPSF(lam=lam, diam=diam, defocus=0., ...)

    Either of these options initializes ``optical_psf`` as an OpticalPSF instance.

    Parameters:
        lam_over_diam:      Lambda / telescope diameter in the physical units adopted for ``scale``
                            (user responsible for consistency).  Either ``lam_over_diam``, or
                            ``lam`` and ``diam``, must be supplied.
        lam:                Lambda (wavelength) in units of nanometers.  Must be supplied with
                            ``diam``, and in this case, image scales (``scale``) should be
                            specified in units of ``scale_unit``.
        diam :              Telescope diameter in units of meters.  Must be supplied with
                            ``lam``, and in this case, image scales (``scale``) should be
                            specified in units of ``scale_unit``.
        tip:                Tip in units of incident light wavelength. [default: 0]
        tilt:               Tilt in units of incident light wavelength. [default: 0]
        defocus:            Defocus in units of incident light wavelength. [default: 0]
        astig1:             Astigmatism (like e2) in units of incident light wavelength.
                            [default: 0]
        astig2:             Astigmatism (like e1) in units of incident light wavelength.
                            [default: 0]
        coma1:              Coma along y in units of incident light wavelength. [default: 0]
        coma2:              Coma along x in units of incident light wavelength. [default: 0]
        trefoil1:           Trefoil (one of the arrows along y) in units of incident light
                            wavelength. [default: 0]
        trefoil2:           Trefoil (one of the arrows along x) in units of incident light
                            wavelength. [default: 0]
        spher:              Spherical aberration in units of incident light wavelength.
                            [default: 0]
        aberrations:        Optional keyword, to pass in a list, tuple, or NumPy array of
                            aberrations in units of reference wavelength (ordered according to
                            the Noll convention), rather than passing in individual values for each
                            individual aberration.  Note that aberrations[1] is piston (and not
                            aberrations[0], which is unused.)  This list can be arbitrarily long to
                            handle Zernike polynomial aberrations of arbitrary order.
        annular_zernike:    Boolean indicating that aberrations specify the amplitudes of annular
                            Zernike polynomials instead of circular Zernike polynomials.
                            [default: False]
        aper:               `Aperture` object to use when creating PSF.  [default: None]
        circular_pupil:     Adopt a circular pupil?  [default: True]
        obscuration:        Linear dimension of central obscuration as fraction of pupil linear
                            dimension, [0., 1.). This should be specified even if you are providing
                            a ``pupil_plane_im``, since we need an initial value of obscuration to
                            use to figure out the necessary image sampling. [default: 0]
        interpolant:        Either an Interpolant instance or a string indicating which interpolant
                            should be used.  Options are 'nearest', 'sinc', 'linear', 'cubic',
                            'quintic', or 'lanczosN' where N should be the integer order to use.
                            [default: galsim.Quintic()]
        oversampling:       Optional oversampling factor for the InterpolatedImage. Setting
                            ``oversampling < 1`` will produce aliasing in the PSF (not good).
                            Usually ``oversampling`` should be somewhat larger than 1.  1.5 is
                            usually a safe choice.  [default: 1.5]
        pad_factor:         Additional multiple by which to zero-pad the PSF image to avoid folding
                            compared to what would be employed for a simple `Airy`.  Note that
                            ``pad_factor`` may need to be increased for stronger aberrations, i.e.
                            those larger than order unity.  [default: 1.5]
        ii_pad_factor:      Zero-padding factor by which to extend the image of the PSF when
                            creating the ``InterpolatedImage``.  See the ``InterpolatedImage``
                            docstring for more details.  [default: 1.5]
        suppress_warning:   If ``pad_factor`` is too small, the code will emit a warning telling you
                            its best guess about how high you might want to raise it.  However,
                            you can suppress this warning by using ``suppress_warning=True``.
                            [default: False]
        geometric_shooting: If True, then when drawing using photon shooting, use geometric
                            optics approximation where the photon angles are derived from the
                            phase screen gradient.  If False, then first draw using Fourier
                            optics and then shoot from the derived InterpolatedImage.
                            [default: False]
        flux:               Total flux of the profile. [default: 1.]
        nstruts:            Number of radial support struts to add to the central obscuration.
                            [default: 0]
        strut_thick:        Thickness of support struts as a fraction of pupil diameter.
                            [default: 0.05]
        strut_angle:        `Angle` made between the vertical and the strut starting closest to it,
                            defined to be positive in the counter-clockwise direction; must be an
                            `Angle` instance. [default: 0. * galsim.degrees]
        pupil_plane_im:     The GalSim.Image, NumPy array, or name of file containing the pupil
                            plane image, to be used instead of generating one based on the
                            obscuration and strut parameters.  [default: None]
        pupil_angle:        If ``pupil_plane_im`` is not None, rotation angle for the pupil plane
                            (positive in the counter-clockwise direction).  Must be an `Angle`
                            instance. [default: 0. * galsim.degrees]
        pupil_plane_scale:  Sampling interval in meters to use for the pupil plane array.  In
                            most cases, it's a good idea to leave this as None, in which case
                            GalSim will attempt to find a good value automatically.  The
                            exception is when specifying the pupil arrangement via an image, in
                            which case this keyword can be used to indicate the sampling of that
                            image.  See also ``pad_factor`` for adjusting the pupil sampling scale.
                            [default: None]
        pupil_plane_size:   Size in meters to use for the pupil plane array.  In most cases, it's
                            a good idea to leave this as None, in which case GalSim will attempt
                            to find a good value automatically.  See also ``oversampling`` for
                            adjusting the pupil size.  [default: None]
        scale_unit:         Units to use for the sky coordinates when calculating lam/diam if these
                            are supplied separately.  Should be either a `galsim.AngleUnit` or a
                            string that can be used to construct one (e.g., 'arcsec', 'radians',
                            etc.).  [default: galsim.arcsec]
        fft_sign:           The sign (+/-) to use in the exponent of the Fourier kernel when
                            evaluating the Fourier optics PSF.  As of version 2.3, GalSim uses a
                            plus sign by default, which we believe to be consistent with, for
                            example, how Zemax computes a Fourier optics PSF on DECam.  Before
                            version 2.3, the default was a negative sign.  Input should be either
                            the string '+' or the string '-'.  [default: '+']
        gsparams:           An optional `GSParams` argument. [default: None]
    """
    _opt_params = {
        "diam": float,
        "defocus": float,
        "astig1": float,
        "astig2": float,
        "coma1": float,
        "coma2": float,
        "trefoil1": float,
        "trefoil2": float,
        "spher": float,
        "annular_zernike": bool,
        "circular_pupil": bool,
        "obscuration": float,
        "oversampling": float,
        "pad_factor": float,
        "suppress_warning": bool,
        "interpolant": str,
        "flux": float,
        "nstruts": int,
        "strut_thick": float,
        "strut_angle": Angle,
        "pupil_plane_im": str,
        "pupil_angle": Angle,
        "pupil_plane_scale": float,
        "pupil_plane_size": float,
        "scale_unit": str,
        "fft_sign": str}
    _single_params = [{"lam_over_diam": float, "lam": float}]

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    _default_iipf = 1.5  # The default ii_pad_factor, since we need to check it for the repr

    def __init__(self, lam_over_diam=None, lam=None, diam=None, tip=0., tilt=0., defocus=0.,
                 astig1=0., astig2=0., coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0.,
                 aberrations=None, annular_zernike=False,
                 aper=None, circular_pupil=True, obscuration=0., interpolant=None,
                 oversampling=1.5, pad_factor=1.5, ii_pad_factor=None, flux=1.,
                 nstruts=0, strut_thick=0.05, strut_angle=0.*radians,
                 pupil_plane_im=None, pupil_plane_scale=None, pupil_plane_size=None,
                 pupil_angle=0.*radians, scale_unit=arcsec, fft_sign='+', gsparams=None,
                 _force_stepk=0., _force_maxk=0.,
                 suppress_warning=False, geometric_shooting=False):
        from .phase_screens import OpticalScreen
        if fft_sign not in ['+', '-']:
            raise GalSimValueError("Invalid fft_sign", fft_sign, allowed_values=['+','-'])
        if isinstance(scale_unit, str):
            scale_unit = AngleUnit.from_name(scale_unit)
        # Need to handle lam/diam vs. lam_over_diam here since lam by itself is needed for
        # OpticalScreen.
        if lam_over_diam is not None:
            if lam is not None or diam is not None:
                raise GalSimIncompatibleValuesError(
                    "If specifying lam_over_diam, then do not specify lam or diam",
                    lam_over_diam=lam_over_diam, lam=lam, diam=diam)
            # For combination of lam_over_diam and pupil_plane_im with a specified scale, it's
            # tricky to determine the actual diameter of the pupil needed by Aperture.  So for now,
            # we just disallow this combination.  Please feel free to raise an issue at
            # https://github.com/GalSim-developers/GalSim/issues if you need this functionality.
            if pupil_plane_im is not None:
                if isinstance(pupil_plane_im, basestring):
                    # Filename, therefore specific scale exists.
                    raise GalSimIncompatibleValuesError(
                        "If specifying lam_over_diam, then do not specify pupil_plane_im as "
                        "as a filename.",
                        lam_over_diam=lam_over_diam, pupil_plane_im=pupil_plane_im)
                elif isinstance(pupil_plane_im, Image) and pupil_plane_im.scale is not None:
                    raise GalSimIncompatibleValuesError(
                        "If specifying lam_over_diam, then do not specify pupil_plane_im "
                        "with definite scale attribute.",
                        lam_over_diam=lam_over_diam, pupil_plane_im=pupil_plane_im)
                elif pupil_plane_scale is not None:
                    raise GalSimIncompatibleValuesError(
                        "If specifying lam_over_diam, then do not specify pupil_plane_scale. ",
                        lam_over_diam=lam_over_diam, pupil_plane_scale=pupil_plane_scale)
            lam = 500.  # Arbitrary
            diam = lam*1.e-9 / lam_over_diam * radians / scale_unit
        else:
            if lam is None or diam is None:
                raise GalSimIncompatibleValuesError(
                    "If not specifying lam_over_diam, then specify lam AND diam",
                    lam_over_diam=lam_over_diam, lam=lam, diam=diam)

        # Make the optical screen.
        self._screen = OpticalScreen(
                diam=diam, defocus=defocus, astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2,
                trefoil1=trefoil1, trefoil2=trefoil2, spher=spher, aberrations=aberrations,
                obscuration=obscuration, annular_zernike=annular_zernike, lam_0=lam)

        # Make the aperture.
        if aper is None:
            aper = Aperture(
                    diam, lam=lam, circular_pupil=circular_pupil, obscuration=obscuration,
                    nstruts=nstruts, strut_thick=strut_thick, strut_angle=strut_angle,
                    oversampling=oversampling, pad_factor=pad_factor,
                    pupil_plane_im=pupil_plane_im, pupil_angle=pupil_angle,
                    pupil_plane_scale=pupil_plane_scale, pupil_plane_size=pupil_plane_size,
                    gsparams=gsparams)
            self.obscuration = obscuration
        else:
            self.obscuration = aper.obscuration

        # Save for pickling
        self._lam = float(lam)
        self._flux = float(flux)
        self._interpolant = interpolant
        self._scale_unit = scale_unit
        self._gsparams = GSParams.check(gsparams)
        self._suppress_warning = suppress_warning
        self._geometric_shooting = geometric_shooting
        self._aper = aper
        self._force_stepk = _force_stepk
        self._force_maxk = _force_maxk
        self._ii_pad_factor = ii_pad_factor if ii_pad_factor is not None else self._default_iipf
        self._fft_sign = fft_sign

    @lazy_property
    def _psf(self):
        psf = PhaseScreenPSF(PhaseScreenList(self._screen), lam=self._lam, flux=self._flux,
                             aper=self._aper, interpolant=self._interpolant,
                             scale_unit=self._scale_unit, fft_sign=self._fft_sign,
                             gsparams=self._gsparams,
                             suppress_warning=self._suppress_warning,
                             geometric_shooting=self._geometric_shooting,
                             _force_stepk=self._force_stepk, _force_maxk=self._force_maxk,
                             ii_pad_factor=self._ii_pad_factor)
        psf._prepareDraw()  # No need to delay an OpticalPSF.
        return psf

    def __str__(self):
        screen = self._screen
        s = "galsim.OpticalPSF(lam=%s, diam=%s" % (screen.lam_0, self._aper.diam)
        if any(screen.aberrations):
            s += ", aberrations=[" + ",".join(str(ab) for ab in screen.aberrations) + "]"
        if self._aper._pupil_plane_im is None:
            s += self._aper._geometry_str()
        if screen.annular_zernike:
            s += ", annular_zernike=True"
            s += ", obscuration=%r"%self.obscuration
        if self._flux != 1.0:
            s += ", flux=%s" % self._flux
        s += ")"
        return s

    def __repr__(self):
        screen = self._screen
        s = "galsim.OpticalPSF(lam=%r, diam=%r" % (self._lam, self._aper.diam)
        s += ", aper=%r"%self._aper
        if any(screen.aberrations):
            s += ", aberrations=[" + ",".join(repr(ab) for ab in screen.aberrations) + "]"
        if screen.annular_zernike:
            s += ", annular_zernike=True"
            s += ", obscuration=%r"%self.obscuration
        if self._interpolant != None:
            s += ", interpolant=%r"%self._interpolant
        if self._scale_unit != arcsec:
            s += ", scale_unit=%r"%self._scale_unit
        if self._fft_sign != '+':
            s += ", fft_sign='-'"
        if self._gsparams != GSParams():
            s += ", gsparams=%r"%self._gsparams
        if self._flux != 1.0:
            s += ", flux=%r" % self._flux
        if self._force_stepk != 0.:
            s += ", _force_stepk=%r" % self._force_stepk
        if self._force_maxk != 0.:
            s += ", _force_maxk=%r" % self._force_maxk
        if self._ii_pad_factor != OpticalPSF._default_iipf:
            s += ", ii_pad_factor=%r" % self._ii_pad_factor
        s += ")"
        return s

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, OpticalPSF) and
                 self._lam == other._lam and
                 self._aper == other._aper and
                 self._screen == other._screen and
                 self._flux == other._flux and
                 self._interpolant == other._interpolant and
                 self._scale_unit == other._scale_unit and
                 self._force_stepk == other._force_stepk and
                 self._force_maxk == other._force_maxk and
                 self._ii_pad_factor == other._ii_pad_factor and
                 self._fft_sign == other._fft_sign and
                 self._gsparams == other._gsparams))

    def __hash__(self):
        return hash(("galsim.OpticalPSF", self._lam, self._aper, self._screen,
                     self._flux, self._interpolant, self._scale_unit, self._force_stepk,
                     self._force_maxk, self._ii_pad_factor, self._fft_sign, self._gsparams))

    def __getstate__(self):
        # The SBProfile is picklable, but it is pretty inefficient, due to the large images being
        # written as a string.  Better to pickle the psf and remake the PhaseScreenPSF.
        d = self.__dict__.copy()
        d.pop('_psf', None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return self._psf.maxk

    @property
    def _stepk(self):
        return self._psf.stepk

    @property
    def _centroid(self):
        return self._psf.centroid

    @property
    def _positive_flux(self):
        return self._psf.positive_flux

    @property
    def _negative_flux(self):
        return self._psf.negative_flux

    @property
    def _flux_per_photon(self):
        return self._psf._flux_per_photon

    @property
    def _max_sb(self):
        return self._psf.max_sb

    @property
    def fft_sign(self):
        return self._fft_sign

    def _xValue(self, pos):
        return self._psf._xValue(pos)

    def _kValue(self, kpos):
        return self._psf._kValue(kpos)

    def _drawReal(self, image, jac=None, offset=(0.,0.), flux_scaling=1.):
        self._psf._drawReal(image, jac, offset, flux_scaling)

    def _shoot(self, photons, rng):
        self._psf._shoot(photons, rng)

    def _drawKImage(self, image, jac=None):
        self._psf._drawKImage(image, jac)

    @doc_inherit
    def withFlux(self, flux):
        screen = self._screen
        return OpticalPSF(
                lam=self._lam, diam=self._aper.diam, aper=self._aper,
                aberrations=screen.aberrations, annular_zernike=screen.annular_zernike,
                flux=flux, _force_stepk=self._force_stepk, _force_maxk=self._force_maxk,
                ii_pad_factor=self._ii_pad_factor, fft_sign=self._fft_sign,
                gsparams=self._gsparams)
