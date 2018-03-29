# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
"""@file optics.py
Module containing the optical PSF generation routines.

Most of the contents of this file are just functions; they are used to generate galsim.OpticalPSF()
class instances (one of the GSObjects, also in this file).

Most methods in this file (except for the OpticalPSF class itself) are solely of use to developers
for generating arrays that may be useful in defining GSObjects with an optical component.  They will
not therefore be used in a typical image simulation workflow: users will find most of what they need
simply using the OpticalPSF class.

Glossary of key terms used in function names:

PSF = point spread function

OTF = optical transfer function = FT{PSF}

MTF = modulation transfer function = |FT{PSF}|

PTF = phase transfer function = p, where OTF = MTF * exp(i * p)

Wavefront = the amplitude and phase of the incident light on the telescope pupil, encoded as a
complex number. The OTF is the autocorrelation function of the wavefront.
"""


import numpy as np
import galsim
from galsim import GSObject
from galsim.deprecated import depr
from galsim.utilities import roll2d, kxky, rotate_xy


class OpticalPSF(GSObject):
    """A class describing aberrated PSFs due to telescope optics.  Its underlying implementation
    uses an InterpolatedImage to characterize the profile.

    The diffraction effects are characterized by the diffraction angle, which is a function of the
    ratio lambda / D, where lambda is the wavelength of the light and D is the diameter of the
    telescope.  The natural units for this value is radians, which is not normally a convenient
    unit to use for other GSObject dimensions.  Assuming that the other sky coordinates you are
    using are all in arcsec (e.g. the pixel scale when you draw the image, the size of the galaxy,
    etc.), then you should convert this to arcsec as well:

        >>> lam = 700  # nm
        >>> diam = 4.0    # meters
        >>> lam_over_diam = (lam * 1.e-9) / diam  # radians
        >>> lam_over_diam *= 206265  # Convert to arcsec
        >>> psf = galsim.OpticalPSF(lam_over_diam, ...)

    To make this process a bit simpler, we recommend instead providing the wavelength and diameter
    separately using the parameters `lam` (in nm) and `diam` (in m).  GalSim will then convert this
    to any of the normal kinds of angular units using the `scale_unit` parameter:

        >>> psf = galsim.OpticalPSF(lam=lam, diam=diam, scale_unit=galsim.arcsec, ...)

    When drawing images, the scale_unit should match the unit used for the pixel scale or the WCS.
    e.g. in this case, a pixel scale of 0.2 arcsec/pixel would be specified as `pixel_scale=0.2`.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention defined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976).  For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    There are two ways to specify the geometry of the pupil plane, i.e., the obscuration disk size
    and the areas that will be illuminated outside of it.  The first way is to use keywords that
    specify the size of the obscuration, and the nature of the support struts holding up the
    secondary mirror (or prime focus cage, etc.).  These are taken to be rectangular obscurations
    extending from the outer edge of the pupil to the outer edge of the obscuration disk (or the
    pupil center if `obscuration = 0.`).  You can specify how many struts there are (evenly spaced
    in angle), how thick they are as a fraction of the pupil diameter, and what angle they start at
    relative to the positive y direction.

    The second way to specify the pupil plane configuration is by passing in an image of it.  This
    can be useful for example if the struts are not evenly spaced or are not radially directed, as
    is assumed by the simple model for struts described above.  In this case, keywords related to
    struts are ignored; moreover, the `obscuration` keyword is used to ensure that the images are
    properly sampled (so it is still needed), but the keyword is then ignored when using the
    supplied image of the pupil plane.  The `pupil_plane_im` that is passed in can be rotated during
    internal calculations by specifying a `pupil_angle` keyword.  Also note that given how the
    `obscuration` keyword is used, the obscuration in the supplied pupil plane image should indeed
    be reasonably circular rather than having an arbitrary shape; however, minor deviations from a
    perfect circle do not cause significant problems.

    If you choose to pass in a pupil plane image, it must be a square array in which the image of
    the pupil is centered.  The areas that are illuminated should have some value >0, and the other
    areas should have a value of precisely zero.  Based on what the OpticalPSF class thinks is the
    required sampling to make the PSF image, the image that is passed in of the pupil plane might be
    zero-padded during internal calculations.  If the pupil plane image has a scale associated with
    it, that scale will be completely ignored; the scale is determined internally using basic
    physical considerations.  Finally, to ensure accuracy of calculations using a pupil plane image,
    we recommend sampling it as finely as possible.

    Initialization
    --------------

    As described above, either specify the lam/diam ratio directly in arbitrary units:

        >>> optical_psf = galsim.OpticalPSF(lam_over_diam=lam_over_diam, defocus=0., astig1=0.,
                                            astig2=0., coma1=0., coma2=0., trefoil1=0., trefoil2=0.,
                                            spher=0., aberrations=None, circular_pupil=True,
                                            obscuration=0., interpolant=None, oversampling=1.5,
                                            pad_factor=1.5, max_size=None, nstruts=0,
                                            strut_thick=0.05, strut_angle=0.*galsim.degrees,
                                            pupil_plane_im=None, pupil_angle=0.*galsim.degrees)

    or, use separate keywords for the telescope diameter and wavelength in meters and nanometers,
    respectively:

        >>> optical_psf = galsim.OpticalPSF(lam=lam, diam=diam, defocus=0., ...)

    Either of these options initializes `optical_psf` as an OpticalPSF instance.

    @param lam_over_diam    Lambda / telescope diameter in the physical units adopted for `scale`
                            (user responsible for consistency).  Either `lam_over_diam`, or `lam`
                            and `diam`, must be supplied.
    @param lam              Lambda (wavelength) in units of nanometers.  Must be supplied with
                            `diam`, and in this case, image scales (`scale`) should be specified in
                            units of `scale_unit`.
    @param diam             Telescope diameter in units of meters.  Must be supplied with
                            `lam`, and in this case, image scales (`scale`) should be specified in
                            units of `scale_unit`.
    @param defocus          Defocus in units of incident light wavelength. [default: 0]
    @param astig1           Astigmatism (like e2) in units of incident light wavelength.
                            [default: 0]
    @param astig2           Astigmatism (like e1) in units of incident light wavelength.
                            [default: 0]
    @param coma1            Coma along y in units of incident light wavelength. [default: 0]
    @param coma2            Coma along x in units of incident light wavelength. [default: 0]
    @param trefoil1         Trefoil (one of the arrows along y) in units of incident light
                            wavelength. [default: 0]
    @param trefoil2         Trefoil (one of the arrows along x) in units of incident light
                            wavelength. [default: 0]
    @param spher            Spherical aberration in units of incident light wavelength.
                            [default: 0]
    @param aberrations      Optional keyword, to pass in a list, tuple, or NumPy array of
                            aberrations in units of incident light wavelength (ordered according to
                            the Noll convention), rather than passing in individual values for each
                            individual aberration.  Currently GalSim supports aberrations from
                            defocus through third-order spherical (`spher`), which are aberrations
                            4-11 in the Noll convention, and hence `aberrations` should be an
                            object of length 12, with the 5th number (index 4) being the defocus,
                            and so on through index 11 corresponding to `spher`.  Orders 1-3
                            (piston, tip, and tilt), while not true optical aberrations, are
                            permitted to be non-zero and will be treated appropriately, while the
                            first number will be ignored. [default: None]
    @param circular_pupil   Adopt a circular pupil?  [default: True]
    @param obscuration      Linear dimension of central obscuration as fraction of pupil linear
                            dimension, [0., 1.). This should be specified even if you are providing
                            a `pupil_plane_im`, since we need an initial value of obscuration to use
                            to figure out the necessary image sampling. [default: 0]
    @param interpolant      Either an Interpolant instance or a string indicating which interpolant
                            should be used.  Options are 'nearest', 'sinc', 'linear', 'cubic',
                            'quintic', or 'lanczosN' where N should be the integer order to use.
                            [default: galsim.Quintic()]
    @param oversampling     Optional oversampling factor for the InterpolatedImage. Setting
                            `oversampling < 1` will produce aliasing in the PSF (not good).
                            Usually `oversampling` should be somewhat larger than 1.  1.5 is
                            usually a safe choice.  [default: 1.5]
    @param pad_factor       Additional multiple by which to zero-pad the PSF image to avoid folding
                            compared to what would be employed for a simple Airy.  Note that
                            `pad_factor` may need to be increased for stronger aberrations, i.e.
                            those larger than order unity.  [default: 1.5]
    @param suppress_warning If `pad_factor` is too small, the code will emit a warning telling you
                            its best guess about how high you might want to raise it.  However,
                            you can suppress this warning by using `suppress_warning=True`.
                            [default: False]
    @param max_size         Set a maximum size of the internal image for the optical PSF profile
                            in arcsec.  Sometimes the code calculates a rather large image size
                            to describe the optical PSF profile.  If you will eventually be
                            drawing onto a smallish postage stamp, you might want to save some
                            CPU time by setting `max_size` to be the size of your postage stamp.
                            [default: None]
    @param flux             Total flux of the profile. [default: 1.]
    @param nstruts          Number of radial support struts to add to the central obscuration.
                            [default: 0]
    @param strut_thick      Thickness of support struts as a fraction of pupil diameter.
                            [default:`0.05]
    @param strut_angle      Angle made between the vertical and the strut starting closest to it,
                            defined to be positive in the counter-clockwise direction; must be an
                            Angle instance. [default: 0. * galsim.degrees]
    @param pupil_plane_im   The GalSim.Image, NumPy array, or name of file containing the pupil
                            plane image, to be used instead of generating one based on the
                            obscuration and strut parameters.  Note that if the image is saved as
                            unsigned integers, you will get a warning about conversion to floats,
                            which is harmless. [default: None]
    @param pupil_angle      If `pupil_plane_im` is not None, rotation angle for the pupil plane
                            (positive in the counter-clockwise direction).  Must be an Angle
                            instance. [default: 0. * galsim.degrees]
    @param scale_unit       Units to use for the sky coordinates when calculating lam/diam if these
                            are supplied separately.  Should be either a galsim.AngleUnit or a
                            string that can be used to construct one (e.g., 'arcsec', 'radians',
                            etc.).  [default: galsim.arcsec]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    There are no additional methods for OpticalPSF beyond the usual GSObject methods.
    """
    _req_params = { }
    _opt_params = {
        "diam" : float ,
        "defocus" : float ,
        "astig1" : float ,
        "astig2" : float ,
        "coma1" : float ,
        "coma2" : float ,
        "trefoil1" : float ,
        "trefoil2" : float ,
        "spher" : float ,
        "circular_pupil" : bool ,
        "obscuration" : float ,
        "oversampling" : float ,
        "pad_factor" : float ,
        "suppress_warning" : bool ,
        "max_size" : float ,
        "interpolant" : str ,
        "flux" : float ,
        "nstruts" : int ,
        "strut_thick" : float ,
        "strut_angle" : galsim.Angle ,
        "pupil_plane_im" : str ,
        "pupil_angle" : galsim.Angle ,
        "scale_unit" : str }
    _single_params = [ { "lam_over_diam" : float , "lam" : float } ]
    _takes_rng = False

    def __init__(self, lam_over_diam=None, lam=None, diam=None, defocus=0., astig1=0., astig2=0.,
                 coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0., aberrations=None,
                 circular_pupil=True, obscuration=0., interpolant=None, oversampling=1.5,
                 pad_factor=1.5, suppress_warning=False, max_size=None, flux=1.,
                 nstruts=0, strut_thick=0.05, strut_angle=0.*galsim.degrees, pupil_plane_im=None,
                 pupil_angle=0.*galsim.degrees, scale_unit=galsim.arcsec, gsparams=None):

        depr("galsim.optics.OpticalPSF", 1.4, "galsim.OpticalPSF",
             "We have reimplemented OpticalPSF under the more general PhaseScreenPSF framework.  "
             "The new version can be accessed either from galsim.OpticalPSF or "
             "galsim.phase_psf.OpticalPSF")
        # Parse arguments: either lam_over_diam in arbitrary units, or lam in nm and diam in m.
        # If the latter, then get lam_over_diam in units of `scale_unit`, as specified in
        # docstring.
        if lam_over_diam is not None:
            if lam is not None or diam is not None:
                raise TypeError("If specifying lam_over_diam, then do not specify lam or diam")
        else:
            if lam is None or diam is None:
                raise TypeError("If not specifying lam_over_diam, then specify lam AND diam")
            # In this case we're going to use scale_unit, so parse it in case of string input:
            if isinstance(scale_unit, basestring):
                scale_unit = galsim.angle.get_angle_unit(scale_unit)
            lam_over_diam = (1.e-9*lam/diam)*(galsim.radians/scale_unit)

        # Choose scale for lookup table using Nyquist for optical aperture and the specified
        # oversampling factor
        scale_lookup = .5 * lam_over_diam / oversampling

        # Start with the stepk value for Airy:
        airy = galsim.Airy(lam_over_diam = lam_over_diam, obscuration = obscuration,
                           gsparams = gsparams)
        stepk_airy = airy.stepK()

        # Boost Airy image size by a user-specifed pad_factor to allow for larger, aberrated PSFs
        stepk = stepk_airy / pad_factor

        # Check the desired image size against max_size if provided
        twoR = 2. * np.pi / stepk  # The desired image size in arcsec
        if max_size is not None and twoR > max_size:
            twoR = max_size

        # Get a good FFT size.  i.e. 2^n or 3 * 2^n.
        npix = galsim._galsim.goodFFTSize(int(np.ceil(twoR / scale_lookup)))

        if aberrations is None:
            # Repackage the aberrations into a single array, to be passed in to all the utilities in
            # this file.  We do this instead of passing around the individual values, so that only
            # two pieces of code will have to be changed if we want to support higher aberrations.
            # (The changes would be here, and in the wavefront() routine below.)
            aberrations = np.zeros(12)
            aberrations[4] = defocus
            aberrations[5] = astig1
            aberrations[6] = astig2
            aberrations[7] = coma1
            aberrations[8] = coma2
            aberrations[9] = trefoil1
            aberrations[10] = trefoil2
            aberrations[11] = spher
        else:
            # Aberrations were passed in, so check that there are the right number of entries.
            if len(aberrations) > 12:
                raise ValueError("Cannot (yet) specify aberrations past index=11")
            if len(aberrations) <= 4:
                raise ValueError("Aberrations keyword must have length > 4")
            # Make sure no individual ones were passed in, since they will be ignored.
            if np.any(
                np.array([defocus,astig1,astig2,coma1,coma2,trefoil1,trefoil2,spher]) != 0):
                raise TypeError("Cannot pass in individual aberrations and array!")

            # Finally, just in case it was a tuple/list, make sure we end up with NumPy array:
            aberrations = np.array(aberrations)
            # And pad it out to length 12 if necessary.
            if len(aberrations) < 12:
                aberrations = np.append(aberrations, [0] * (12-len(aberrations)))

            # Check for non-zero elements in first value.  Probably a mistake.
            if aberrations[0] != 0.0:
                import warnings
                warnings.warn(
                    "Detected non-zero value in aberrations[0] -- this value is ignored!")

        self._lam_over_diam = lam_over_diam
        self._aberrations = aberrations.tolist()
        self._oversampling = oversampling
        self._pad_factor = pad_factor
        self._max_size = max_size
        self._circular_pupil = circular_pupil
        self._flux = flux
        self._obscuration = obscuration
        self._nstruts = nstruts
        self._strut_thick = strut_thick
        self._strut_angle = strut_angle
        self._pupil_plane_im = pupil_plane_im
        self._pupil_angle = pupil_angle
        self._interpolant = interpolant
        self._gsparams = gsparams

        # Make the psf image using this scale and array shape
        optimage = galsim.optics.psf_image(
            lam_over_diam=lam_over_diam, scale=scale_lookup, array_shape=(npix, npix),
            aberrations=aberrations, circular_pupil=circular_pupil, obscuration=obscuration,
            flux=flux, nstruts=nstruts, strut_thick=strut_thick, strut_angle=strut_angle,
            pupil_plane_im=pupil_plane_im, pupil_angle=pupil_angle, oversampling=oversampling)

        # Initialize the GSObject (InterpolatedImage)
        ii = galsim.InterpolatedImage(optimage, x_interpolant=interpolant,
                                      calculate_stepk=True, calculate_maxk=True,
                                      use_true_center=False, normalization='sb',
                                      gsparams=gsparams)
        # The above procedure ends up with a larger image than we really need, which
        # means that the default stepK value will be smaller than we need.
        # Hence calculate_stepk=True and calculate_maxk=True above.

        self._optimage = optimage
        self._serialize_stepk = ii._serialize_stepk
        self._serialize_maxk = ii._serialize_maxk

        self._sbp = ii._sbp

        if not suppress_warning:
            # Check the calculated stepk value.  If it is smaller than stepk, then there might
            # be aliasing.
            final_stepk = self.stepK()
            if final_stepk < stepk:
                import warnings
                warnings.warn(
                    "The calculated stepk (%g) for OpticalPSF is smaller "%final_stepk +
                    "than what was used to build the wavefront (%g). "%stepk +
                    "This could lead to aliasing problems. " +
                    "Using pad_factor >= %f is recommended."%(pad_factor * stepk / final_stepk))

    def __eq__(self, other):
        return (isinstance(other, galsim.OpticalPSF) and
                self._lam_over_diam == other._lam_over_diam and
                np.array_equal(self._aberrations, other._aberrations) and
                self._oversampling == other._oversampling and
                self._pad_factor == other._pad_factor and
                self._obscuration == other._obscuration and
                self._nstruts == other._nstruts and
                self._strut_thick == other._strut_thick and
                self._strut_angle == other._strut_angle and
                self._pupil_plane_im == other._pupil_plane_im and
                self._circular_pupil == other._circular_pupil and
                self._max_size == other._max_size and
                self._interpolant == other._interpolant and
                self._flux == other._flux and
                self._gsparams == other._gsparams)

    def __hash__(self):
        # Cache this in case self._pupil_plane_im is large.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.OpticalPSF", self._lam_over_diam, tuple(self._aberrations),
                               self._oversampling, self._pad_factor, self._obscuration,
                               self._nstruts, self._strut_thick, self._strut_angle,
                               self._pupil_angle, self._circular_pupil, self._max_size,
                               self._interpolant, self._flux, self._gsparams))
            if self._pupil_plane_im is not None:
                self._hash ^= hash((tuple(self._pupil_plane_im.array.ravel()),
                                    self._pupil_plane_im.bounds, self._pupil_plane_im.wcs))
        return self._hash

    def __repr__(self):
        s = 'galsim.OpticalPSF(lam_over_diam=%r, aberrations=%r, oversampling=%r, pad_factor=%r'%(
                self._lam_over_diam, self._aberrations, self._oversampling, self._pad_factor)
        if self._obscuration != 0.:
            s += ', obscuration=%r'%self._obscuration
        if self._nstruts != 0:
            s += ', nstruts=%r, strut_thick=%r, strut_angle=%r'%(
                    self._nstruts, self._strut_thick, self._strut_angle)
        if self._pupil_plane_im != None:
            s += ', pupil_plane_im=%r, pupil_angle=%r'%(
                    self._pupil_plane_im, self._pupil_angle)
        if self._circular_pupil == False:
            s += ', circular_pupil=False'
        if self._max_size is not None:
            s += ', max_size=%r'%self._max_size
        if self._interpolant is not None:
            s += ', interpolant=%r'%self._interpolant
        s += ', flux=%r, gsparams=%r)'%(self._flux, self._gsparams)
        return s

    def __str__(self):
        s = 'galsim.OpticalPSF(lam_over_diam=%s'%self._lam_over_diam
        # The aberrations look a bit nicer without the spaces, since it tightens up that
        # parameter, which makes it easier to distinguish from the others.
        ab_str = [ str(ab) for ab in self._aberrations ]
        s += ', aberrations=[' + ','.join(ab_str) + ']'
        if self._obscuration != 0.:
            s += ', obscuration=%s'%self._obscuration
        if self._nstruts != 0:
            s += ', nstruts=%s, strut_thick=%s, strut_angle=%s'%(
                    self._nstruts, self._strut_thick, self._strut_angle)
        if self._pupil_plane_im != None:
            s += ', pupil_plane_im=%s, pupil_angle=%s'%(
                    self._pupil_plane_im, self._pupil_angle)
        if self._circular_pupil == False:
            s += ', circular_pupil=False'
        if self._flux != 1.0:
            s += ', flux=%s'%self._flux
        s += ')'
        return s

    def __getstate__(self):
        # The SBProfile is picklable, but it is pretty inefficient, due to the large images being
        # written as a string.  Better to pickle the image and remake the InterpolatedImage.
        d = self.__dict__.copy()
        del d['_sbp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        ii =  galsim.InterpolatedImage(self._optimage, x_interpolant=self._interpolant,
                                       use_true_center=False, normalization='sb',
                                       _serialize_stepk=self._serialize_stepk,
                                       _serialize_maxk=self._serialize_maxk,
                                       gsparams=self._gsparams)
        self._sbp = ii._sbp


def _load_pupil_plane(pupil_plane_im, pupil_angle=0.*galsim.degrees, array_shape=None,
                     lam_over_diam=None, obscuration=None):
    """Set up the pupil plane based on using or loading a previously generated image.

    This routine also has to set up the array for the rho values associated with that image.

    If you want the routine to do an automated, approximate check for sufficient sampling, you
    should supply it with `lam_over_diam` and `obscuration` keywords that it can use.  Even so, this
    check does not check for adequate sampling of the struts, just the basic features like the
    obscuration.

    @param pupil_plane_im  The GalSim.Image, NumPy array, or name of file containing the pupil plane
                           image.  Note that if the image is saved as unsigned integers, you will
                           get a warning about conversion to floats, which is harmless.
    @param pupil_angle     Rotation angle for the pupil plane (positive in the counter-clockwise
                           direction).  Must be an Angle instance. [default: 0.*galsim.degrees]
    @param array_shape     The NumPy array shape required for the output image.  If None, then use
                           the shape of the input `pupil_plane_im`.  [default: None]
    @param lam_over_diam   The lam/diam defining the diffraction-limited PSF, to use for checks of
                           sampling as described in the docstring.  If not supplied, the tests will
                           not be carried out. [default: None]
    @param obscuration     The obscuration defining the diffraction-limited PSF, to use for checks of
                           sampling as described in the docstring.  If not supplied, the tests will
                           not be carried out. [default: None]

    @returns a tuple `(rho, in_pupil)`, the first of which is the coordinate of the pupil
    in unit disc-scaled coordinates for use by Zernike polynomials (as a complex number)
    for describing the wavefront across the pupil plane.  The array `in_pupil` is a vector of
    Bools used to specify where in the pupil plane described by `rho` is illuminated.  See also
    wavefront().
    """
    # Handle multiple types of input: NumPy array, galsim.Image, or string for filename with image.
    if isinstance(pupil_plane_im, np.ndarray):
        # Make it into an image.
        pupil_plane_im = galsim.Image(pupil_plane_im)
    elif isinstance(pupil_plane_im, galsim.Image):
        # Make sure not to overwrite input image.
        pupil_plane_im = pupil_plane_im.copy()
    else:
        # Read in image of pupil plane from file.
        pupil_plane_im = galsim.fits.read(pupil_plane_im)

    # Sanity checks
    if pupil_plane_im.array.shape[0] != pupil_plane_im.array.shape[1]:
        raise ValueError("We require square input pupil plane arrays!")
    if pupil_plane_im.array.shape[0] % 2 == 1:
        raise ValueError("Even-sized input arrays are required for the pupil plane!")

    # Pad image if necessary given the requested array shape
    if array_shape is not None:

        # If requested array shape is larger than the input one, then add some zero-padding to the
        # input image.
        if array_shape[0] > pupil_plane_im.array.shape[0]:
            border_size = int(0.5*(array_shape[0] - pupil_plane_im.array.shape[0]))
            new_im = galsim.Image(pupil_plane_im.bounds.withBorder(border_size))
            new_im[pupil_plane_im.bounds] = pupil_plane_im
            pupil_plane_im = new_im.copy()
        # If the requested shape is smaller than the input one, we do nothing at this stage.  All
        # internal calculations done by this routine will be carried out at higher resolution, only
        # returning to lower resolution once we have completely specified the wavefront (aberrations
        # etc.)

    # Figure out the rho array parameters here.  We do this now, before rotating the input image,
    # since the rotation can lead to small interpolation errors.  The basic idea is to find how far
    # from the center is the farthest pixel that is non-zero (i.e., that is illuminated).  This is
    # used to define the unit disk when setting the rho parameters.
    kx, ky = kxky(pupil_plane_im.array.shape)
    delta_k = (np.max(kx)-np.min(kx))/(pupil_plane_im.array.shape[0]-1)
    kx = (kx) * pupil_plane_im.array.shape[0]/(2.*np.pi)
    ky = (ky) * pupil_plane_im.array.shape[1]/(2.*np.pi)
    tot_k = np.sqrt(kx**2 + ky**2)
    tot_k = roll2d(tot_k, (tot_k.shape[0] // 2, tot_k.shape[1] // 2))
    max_in_pupil = max(tot_k[pupil_plane_im.array>0])

    # Next, deal with any requested rotations.
    if pupil_angle.rad == 0.:
        pp_arr = pupil_plane_im.array
    else:
        # Rotate the pupil plane image as required based on the `pupil_angle`, being careful to
        # ensure that the image is one of the allowed types.  We ignore the scale.
        int_im = galsim.InterpolatedImage(galsim.Image(pupil_plane_im, scale=1., dtype=np.float64),
                                          x_interpolant='linear', calculate_stepk=False,
                                          calculate_maxk=False)
        int_im = int_im.rotate(pupil_angle)
        new_im = galsim.ImageF(pupil_plane_im.array.shape[0], pupil_plane_im.array.shape[0])
        new_im = int_im.drawImage(image=new_im, scale=1., method='no_pixel')
        pp_arr = new_im.array
        # Restore hard edges that might have been lost during the interpolation.  To do this, we
        # check the maximum value of the entries.  Values after interpolation that are >half that
        # maximum value are kept as nonzero (True), but those that are <half the maximum value are
        # set to zero (False).
        max_pp_val = np.max(pp_arr)
        pp_arr[pp_arr<0.5*max_pp_val] = 0.

    # Turn it into a boolean type, so all values >0 are True and all values==0 are False.
    pp_arr = pp_arr.astype(bool)
    # Roll the pupil plane image so the center is in the corner, just like the outputs of
    # `generate_pupil_plane`.
    pp_arr = roll2d(pp_arr, (pp_arr.shape[0] // 2, pp_arr.shape[1] // 2))

    # Then set up the rho array appropriately. Given our estimate above for `max_in_pupil`, we can
    # get delta rho:
    drho = 1.0 / max_in_pupil
    # We want rho to go from negative to positive values before we roll it.
    rho_vec = np.linspace(-0.5*pp_arr.shape[0]*drho, 0.5*pp_arr.shape[0]*drho-drho,
                           num=pp_arr.shape[0])
    effective_oversampling = (np.max(rho_vec)+drho)/2.
    rho_x, rho_y = np.meshgrid(rho_vec, rho_vec)
    assert rho_x.shape == pp_arr.shape
    assert rho_y.shape == pp_arr.shape
    # Then roll it, consistent with the convention in `generate_pupil_plane`.
    rho_x = roll2d(rho_x, (pp_arr.shape[0] // 2, pp_arr.shape[1] // 2))
    rho_y = roll2d(rho_y, (pp_arr.shape[0] // 2, pp_arr.shape[1] // 2))
    rho = rho_x + 1j * rho_y

    if obscuration is not None and lam_over_diam is not None:
        # We do a basic check of the sampling now that we have rho.  First, we use the fact that the
        # right edge of the pupil should have rho_x=1, rho_y=0, which implies k_x =
        # 0.5*(k_{max,int}).  We can calculate k_{max,int} based on the Airy parameters, which tells
        # us the value of k_x at that position.  Knowing how many pixels that is from the edge of
        # the pupil plane image, we therefore know Delta k (the k-space sampling):
        kmax_internal = 2.*np.pi / lam_over_diam
        delta_k = 0.5*kmax_internal / (float(max_in_pupil)+1.)
        # We can compare this with the ideal spacing for an Airy with this lam/diam and obscuration:
        airy = galsim.Airy(lam_over_diam=lam_over_diam, obscuration=obscuration)
        stepk_airy = airy.stepK() # This has the same units as those for kmax_internal and delta_k
        if delta_k > stepk_airy:
            import warnings
            ratio = delta_k / stepk_airy
            warnings.warn("Input pupil plane image may not be sampled well enough!\n"
                          "Consider increasing sampling by a factor %f, and/or check OpticalPSF "
                          "outputs for signs of folding in real space."%ratio)

    return rho, pp_arr, effective_oversampling

def _generate_pupil_plane(array_shape=(256, 256), scale=1., lam_over_diam=2., circular_pupil=True,
                         obscuration=0., nstruts=0, strut_thick=0.05,
                         strut_angle=0.*galsim.degrees):
    """Generate a pupil plane, including a central obscuration such as caused by a secondary mirror.

    @param array_shape     The NumPy array shape desired for the output array.
    @param scale           Grid spacing of PSF in real space units.
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param circular_pupil  Adopt a circular pupil? [default: 0]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]

    @returns a tuple `(rho, in_pupil)`, the first of which is the coordinate of the pupil
    in unit disc-scaled coordinates for use by Zernike polynomials (as a complex number)
    for describing the wavefront across the pupil plane.  The array `in_pupil` is a vector of
    Bools used to specify where in the pupil plane described by `rho` is illuminated.  See also
    wavefront().
    """
    kmax_internal = scale * 2. * np.pi / lam_over_diam # INTERNAL kmax in units of array grid spacing
    # Build kx, ky coords
    kx, ky = kxky(array_shape)
    # Then define unit disc rho pupil coords for Zernike polynomials
    rho = (kx + 1j * ky) / (.5 * kmax_internal)
    rhosq = np.abs(rho)**2
    # Amazingly, the above line is faster than the following. (~ 35% faster)
    # See the longer comment about this in psf function.
    #rhosq = rho.real**2 + rho.imag**2

    # Cut out circular pupil if desired (default, square pupil optionally supported) and include
    # central obscuration
    if obscuration >= 1.:
        raise ValueError("Pupil fully obscured! obscuration ="+str(obscuration)+" (>= 1)")
    if circular_pupil:
        in_pupil = (rhosq < 1.)
        if obscuration > 0.:
            in_pupil *= rhosq >= obscuration**2  # * acts like "and" for boolean arrays
    else:
        in_pupil = (np.abs(kx) < .5 * kmax_internal) * (np.abs(ky) < .5 * kmax_internal)
        if obscuration > 0.:
            in_pupil *= ( (np.abs(kx) >= .5 * obscuration * kmax_internal) *
                          (np.abs(ky) >= .5 * obscuration * kmax_internal) )
    if nstruts > 0:
        if not isinstance(strut_angle, galsim.Angle):
            raise TypeError("Input kwarg strut_angle must be a galsim.Angle instance.")
        # Add the initial rotation if requested, converting to radians
        if strut_angle.rad != 0.:
            kxs, kys = rotate_xy(kx, ky, -strut_angle) # strut rotation +=ve, so coords
                                                                 # rotation -ve!
        else:
            kxs, kys = kx, ky
        # Define the angle between struts for successive use below
        rotang = 360. * galsim.degrees / float(nstruts)
        # Then loop through struts setting to zero in the pupil regions which lie under the strut
        in_pupil *= (
            (np.abs(kxs) >= .5 * strut_thick * kmax_internal) +
            ((kys < 0.) * (np.abs(kxs) < .5 * strut_thick * kmax_internal)))
        for istrut in range(nstruts)[1:]:
            kxs, kys = rotate_xy(kxs, kys, -rotang)
            in_pupil *= (
                (np.abs(kxs) >= .5 * strut_thick * kmax_internal) +
                ((kys < 0.) * (np.abs(kxs) < .5 * strut_thick * kmax_internal)))
    return rho, in_pupil

def wavefront(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
              circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
              strut_angle=0.*galsim.degrees, pupil_plane_im=None, pupil_angle=0.*galsim.degrees):
    """Return a complex, aberrated wavefront across a circular (default) or square pupil.

    Outputs a complex image (shape=`array_shape`) of a circular pupil wavefront of unit amplitude
    that can be easily transformed to produce an optical PSF with `lambda/D = lam_over_diam` on an
    output grid of spacing `scale`.  This routine would need to be modified in order to include
    higher order aberrations than `spher` (order 11 in Noll convention).

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    The pupil sample locations are arranged in standard DFT element ordering format, so that
    `(kx, ky) = (0, 0)` is the [0, 0] array element.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention defined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976). For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    @param array_shape     The NumPy array shape desired for the output array.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    @param pupil_plane_im  The GalSim.Image, NumPy array, or name of file containing the pupil
                           plane image, to be used instead of generating one based on the
                           obscuration and strut parameters.  Note that if the image is saved as
                           unsigned integers, you will get a warning about conversion to floats,
                           which is harmless. [default: None]
    @param pupil_angle     If `pupil_plane_im` is not None, rotation angle for the pupil plane
                           (positive in the counter-clockwise direction).  Must be an Angle
                           instance. [default: 0. * galsim.degrees]

    @returns the wavefront for `kx, ky` locations corresponding to `kxky(array_shape)`.
    """

    depr("wavefront", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    # Define the pupil coordinates and non-zero regions based on input kwargs.  These are either
    # generated automatically, or taken from an input image.
    if pupil_plane_im is not None:
        rho_all, in_pupil, effective_oversampling = \
            _load_pupil_plane(pupil_plane_im, pupil_angle, array_shape=array_shape,
                              lam_over_diam=lam_over_diam, obscuration=obscuration)

    else:
        rho_all, in_pupil = _generate_pupil_plane(
            array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam,
            circular_pupil=circular_pupil, obscuration=obscuration,
            nstruts=nstruts, strut_thick=strut_thick, strut_angle=strut_angle)
        effective_oversampling = None

    # Then make wavefront image
    wf = np.zeros(in_pupil.shape, dtype=complex)

    # It is much faster to pull out the elements we will use once, rather than use the
    # subscript each time.  At the end we will fill the appropriate part of wf with the
    # values calculated from this rho vector.
    rho = rho_all[in_pupil]
    rhosq = np.abs(rho)**2

    # Also check for aberrations:
    if aberrations is None:
        aberrations = np.zeros(12)

    # Old version for reference:

    # rho2 = rho * rho
    # rho3 = rho2 * rho
    # temp = np.zeros(rho.shape, dtype=complex)
    # Defocus:
    # temp += np.sqrt(3.) * (2. * rhosq - 1.) * defocus
    # Astigmatism:
    # temp += np.sqrt(6.) * ( astig1 * rho2.imag + astig2 * rho2.real )
    # Coma:
    # temp += np.sqrt(8.) * (3. * rhosq - 2.) * ( coma1 * rho.imag + coma2 * rho.real )
    # Trefoil (one of the arrows along x2)
    # temp += np.sqrt(8.) * ( trefoil1 * rho3.imag + trefoil2 * rho3.real )
    # Spherical aberration
    # temp += np.sqrt(5.) * (6. * rhosq**2 - 6. * rhosq + 1.) * spher

    # Faster to use Horner's method in rho:
    temp = (
            # Constant terms: includes defocus (4) and piston (1)
            -np.sqrt(3.) * aberrations[4] + aberrations[1]

            # Terms with rhosq, but no rho, rho**2, etc.: includes defocus (4) and spher (11)
            + rhosq * ( 2. * np.sqrt(3.) * aberrations[4]
                        - 6. * np.sqrt(5.) * aberrations[11]
                        + rhosq * (6. * np.sqrt(5.) * aberrations[11]) )

            # Now the powers of rho: includes tip (2), tilt (3), coma2 (8), coma1 (7), astig2 (6),
            # astig1 (5), trefoil2 (10), trefoil1 (9).
            # We eventually take the real part.
            + ( rho * ( 2. * (aberrations[2] - 1j * aberrations[3])
                + (rhosq-2./3.) * (3. * np.sqrt(8.) * (aberrations[8] - 1j * aberrations[7]))
                        + rho * ( (np.sqrt(6.) * (aberrations[6] - 1j * aberrations[5]))
                                   + rho * (np.sqrt(8.) * (aberrations[10] - 1j * aberrations[9]))
                                )
                      )
              ).real
    )

    wf[in_pupil] = np.exp(2j * np.pi * temp)

    return wf, effective_oversampling

def wavefront_image(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
                    circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
                    strut_angle=0.*galsim.degrees):
    """Return wavefront as a (real, imag) tuple of Image objects rather than complex NumPy
    array.

    Outputs a circular pupil wavefront of unit amplitude that can be easily transformed to produce
    an optical PSF with `lambda/diam = lam_over_diam` on an output grid of spacing `scale`.

    The Image output can be used to directly instantiate an InterpolatedImage, and its
    `scale` will reflect the spacing of the output grid in the system of units adopted for
    `lam_over_diam`.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    The pupil sample locations are arranged in standard DFT element ordering format, so that
    `(kx, ky) = (0, 0)` is the [0, 0] array element.  The `scale` of the output Image is correct in
    k space units.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention defined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976). For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    @param array_shape     The NumPy array shape desired for the output array.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    """
    depr("wavefront_image", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    array, effective_oversampling = wavefront(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle)
    if array_shape[0] != array_shape[1]:
        import warnings
        warnings.warn(
            "Wavefront Images' scales will not be correct in both directions for non-square "+
            "arrays, only square grids currently supported by galsim.Images.")
    scale = 2. * np.pi / array_shape[0]
    imreal = galsim.Image(np.ascontiguousarray(array.real.astype(np.float64)), scale=scale)
    imimag = galsim.Image(np.ascontiguousarray(array.imag.astype(np.float64)), scale=scale)
    return (imreal, imimag), effective_oversampling

def psf(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
        circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
        strut_angle=0.*galsim.degrees, flux=1., pupil_plane_im=None, pupil_angle=0.*galsim.degrees):
    """Return NumPy array containing circular (default) or square pupil PSF with low-order
    aberrations.

    The PSF is centred on the `array[array_shape[0] / 2, array_shape[1] / 2]` pixel by default, and
    uses surface brightness rather than flux units for pixel values, matching SBProfile.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    Ouput NumPy array is C-contiguous.

    @param array_shape     The NumPy array shape desired for the output array.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    @param flux            Total flux of the profile. [default: 1.]
    @param pupil_plane_im  The GalSim.Image, NumPy array, or name of file containing the pupil
                           plane image, to be used instead of generating one based on the
                           obscuration and strut parameters.  Note that if the image is saved as
                           unsigned integers, you will get a warning about conversion to floats,
                           which is harmless. [default: None]
    @param pupil_angle     If `pupil_plane_im` is not None, rotation angle for the pupil plane
                           (positive in the counter-clockwise direction).  Must be an Angle
                           instance. [default: 0. * galsim.degrees]
    """
    depr("psf", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    wf, effective_oversampling = wavefront(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle, pupil_plane_im=pupil_plane_im,
        pupil_angle=pupil_angle)

    ftwf = np.fft.fft2(wf)

    # MJ: You wouldn't think that using an abs here would be efficient, but I did some timing
    #     tests on my laptop, and of the three options:
    #         im = (ftwf * ftwf.conj()).real
    #         im = ftwf.real**2 + ftwf.imag**2
    #         im = np.abs(ftwf)**2
    #     the third one was fastest.
    #     Average times were about 0.0265, 0.0170, and 0.0105, respectively.
    #     I'm guessing numpy must do some kind of delayed calculation magic with the np.abs()
    #     function that lets them figure out that they don't need the sqrt here.
    im = np.abs(ftwf)**2

    # The roll operation below restores the c_contiguous flag, so no need for a direct action
    im = roll2d(im, (im.shape[0] // 2, im.shape[1] // 2))
    im *= (flux / (im.sum() * scale**2))

    return im, effective_oversampling

def psf_image(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
              circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
              strut_angle=0.*galsim.degrees, flux=1., pupil_plane_im=None,
              pupil_angle=0.*galsim.degrees, oversampling=1.5):
    """Return circular (default) or square pupil PSF with low-order aberrations as an Image.

    The PSF is centred on the `array[array_shape[0] / 2, array_shape[1] / 2] pixel` by default, and
    uses surface brightness rather than flux units for pixel values, matching SBProfile.

    The Image output can be used to directly instantiate an InterpolatedImage, and its
    `scale` will reflect the spacing of the output grid in the system of units adopted for
    `lam_over_diam`.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    @param array_shape     The NumPy array shape desired for the array view of the Image.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    @param flux            Total flux of the profile. [default flux=1.]
    @param pupil_plane_im  The GalSim.Image, NumPy array, or name of file containing the pupil
                           plane image, to be used instead of generating one based on the
                           obscuration and strut parameters.  Note that if the image is saved as
                           unsigned integers, you will get a warning about conversion to floats,
                           which is harmless. [default: None]
    @param pupil_angle     If `pupil_plane_im` is not None, rotation angle for the pupil plane
                           (positive in the counter-clockwise direction).  Must be an Angle
                           instance. [default: 0. * galsim.degrees]
    @param oversampling    Effective level of oversampling requested.
    """
    depr("psf_image", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    array, effective_oversampling = psf(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, flux=flux, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle, pupil_plane_im=pupil_plane_im,
        pupil_angle=pupil_angle)

    if effective_oversampling is not None:
        oversamp_ratio = effective_oversampling / oversampling
        # Here we compare the effective oversampling inferred from the image with the oversampling
        # that was requested by the user.  We do not place a scrict cut at 1.0 because the routine
        # doesn't start to immediately fail catastrophically the instant this ratio goes below 1,
        # particularly if the user has requested an oversampling above 1 (like the default 1.5) to
        # begin with.  But once the ratio gets below 0.9, this could get problematic, and anyway it
        # might mean the user supplied the wrong image by mistake, so we raise an exception.
        if oversamp_ratio < 0.9:
            raise RuntimeError("Supplied pupil plane image results in inadequate sampling "
                               "by factor of %f"%oversamp_ratio)
        im = galsim.Image(array.astype(np.float64), scale=scale/oversamp_ratio)
    else:
        im = galsim.Image(array.astype(np.float64), scale=scale)
    return im

def otf(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
        circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
        strut_angle=0.*galsim.degrees):
    """Return the complex OTF of a circular (default) or square pupil with low-order aberrations as
    a NumPy array.

    OTF array element ordering follows the DFT standard of `kxky(array_shape)`, and has
    `otf[0, 0] = 1+0j` by default.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    Output complex NumPy array is C-contiguous.

    @param array_shape     The NumPy array shape desired for the output array.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    """
    depr("otf", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    wf, _ = wavefront(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle)
    ftwf = np.fft.fft2(wf)
    otf = np.fft.ifft2(np.abs(ftwf)**2)
    # Make unit flux before returning
    return np.ascontiguousarray(otf) / otf[0, 0].real

def otf_image(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
              circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
              strut_angle=0.*galsim.degrees):
    """Return the complex OTF of a circular (default) or square pupil with low-order aberrations as
    a (real, imag) tuple of Image objects, rather than a complex NumPy array.

    OTF array element ordering follows the DFT standard of `kxky(array_shape)`, and has
    `otf[0, 0] = 1+0j` by default.  The `scale` of the output Image is correct in k space units.

    The Image output can be used to directly instantiate an InterpolatedImage, and its
    `scale` will reflect the spacing of the output grid in the system of units adopted for
    `lam_over_diam`.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    @param array_shape     The NumPy array shape desired for array views of Image tuple.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    """
    depr("otf_image", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    array = otf(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle)
    if array_shape[0] != array_shape[1]:
        import warnings
        warnings.warn(
            "OTF Images' scales will not be correct in both directions for non-square arrays, "+
            "only square grids currently supported by galsim.Images.")
    scale = 2. * np.pi / array_shape[0]
    imreal = galsim.Image(np.ascontiguousarray(array.real.astype(np.float64)), scale=scale)
    imimag = galsim.Image(np.ascontiguousarray(array.imag.astype(np.float64)), scale=scale)
    return (imreal, imimag)

def mtf(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
        circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
        strut_angle=0.*galsim.degrees):
    """Return NumPy array containing the MTF of a circular (default) or square pupil with low-order
    aberrations.

    MTF array element ordering follows the DFT standard of `kxky(array_shape)`, and has
    `mtf[0, 0] = 1` by default.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    Output double NumPy array is C-contiguous.

    @param array_shape     The NumPy array shape desired for the output array.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    """
    depr("mtf", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    return np.abs(otf(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        obscuration=obscuration, circular_pupil=circular_pupil, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle))

def mtf_image(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
              circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
              strut_angle=0.*galsim.degrees):
    """Return the MTF of a circular (default) or square pupil with low-order aberrations as an
    Image.

    MTF array element ordering follows the DFT standard of `kxky(array_shape)`, and has
    `mtf[0, 0] = 1` by default.  The `scale` of the output Image is correct in k space units.

    The Image output can be used to directly instantiate an InterpolatedImage, and its
    `scale` will reflect the spacing of the output grid in the system of units adopted for
    `lam_over_diam`.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    @param array_shape     The NumPy array shape desired for the array view of the Image.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.). [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    """
    depr("mtf_image", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    array = mtf(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle)
    if array_shape[0] != array_shape[1]:
        import warnings
        warnings.warn(
            "MTF Image scale will not be correct in both directions for non-square arrays, only "+
            "square grids currently supported by galsim.Images.")
    im = galsim.Image(array.astype(np.float64), scale = 2. * np.pi / array_shape[0])
    return im

def ptf(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
        circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
        strut_angle=0.*galsim.degrees):
    """Return NumPy array containing the PTF [radians] of a circular (default) or square pupil with
    low-order aberrations.

    PTF array element ordering follows the DFT standard of `kxky(array_shape)`, and has
    `ptf[0, 0] = 0`. by default.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    Output double NumPy array is C-contiguous.

    @param array_shape     The NumPy array shape desired for the output array.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.) [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    """
    depr("ptf", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    kx, ky = kxky(array_shape)
    k2 = (kx**2 + ky**2)
    ptf = np.zeros(array_shape)

    # INTERNAL kmax in units of array grid spacing
    kmax_internal = scale * 2. * np.pi / lam_over_diam

    # Try to handle where both real and imag tend to zero...
    ptf[k2 < kmax_internal**2] = np.angle(otf(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle)[k2 < kmax_internal**2])
    return ptf

def ptf_image(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
              circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
              strut_angle=0.*galsim.degrees):
    """Return the PTF [radians] of a circular (default) or square pupil with low-order aberrations
    as an Image.

    PTF array element ordering follows the DFT standard of `kxky(array_shape)`, and has
    `ptf[0, 0] = 0.` by default.  The `scale` of the output Image is correct in k space units.

    The Image output can be used to directly instantiate an InterpolatedImage, and its
    `scale` will reflect the spacing of the output grid in the system of units adopted for
    `lam_over_diam`.

    To ensure properly Nyquist sampled output any user should set `lam_over_diam >= 2. * scale`.

    @param array_shape     The NumPy array shape desired for the array view of the Image.
    @param scale           Grid spacing of PSF in real space units
    @param lam_over_diam   Lambda / telescope diameter in the physical units adopted for `scale`
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, `aberrations` should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  Adopt a circular pupil? [default: True]
    @param obscuration     Linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.) [default: 0]
    @param nstruts         Number of radial support struts to add to the central obscuration.
                           [default: 0]
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter.
                           [default: 0.05]
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be an
                           Angle instance. [default: 0. * galsim.degrees]
    """
    depr("ptf_image", 1.4, '',
         "This functionality has been removed.  If you have a need for it, please raise an issue.")

    array = ptf(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle)
    if array_shape[0] != array_shape[1]:
        import warnings
        warnings.warn(
            "PTF Image scale will not be correct in both directions for non-square arrays, only "+
            "square grids currently supported by galsim.Images.")
    im = galsim.Image(array.astype(np.float64), scale = 2. * np.pi / array_shape[0])
    return im
