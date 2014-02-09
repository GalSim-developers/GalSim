# Copyright 2012-2014 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file optics.py
Module containing the optical PSF generation routines.

Most of the contents of this file are just functions; they are used to generate galsim.OpticalPSF()
class instances (one of the GSObjects, also in this file).

Mostly they are solely of use to developers for generating arrays that may be useful in defining 
GSObjects with an optical component.  They will not therefore be used in a typical image simulation
workflow: users will find most of what they need simply using the OpticalPSF() class.

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
import utilities
from galsim import GSObject, goodFFTSize

class OpticalPSF(GSObject):
    """A class describing aberrated PSFs due to telescope optics.  Its underlying implementation
    uses an InterpolatedImage to characterize the profile.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention defined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976).  For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    You can also optionally specify that the secondary mirror (or prime focus cage, etc.) are held
    by some number of support struts.  These are taken to be rectangular obscurations extending from
    the outer edge of the pupil to the outer edge of the obscuration disk (or the pupil center if
    `obscuration = 0.`).  You can specify how many struts there are (evenly spaced in angle), how
    thick they are as a fraction of the pupil diameter, and what angle they start at relative to
    the positive y direction.

    Initialization
    --------------
    
        >>> optical_psf = galsim.OpticalPSF(lam_over_diam, defocus=0., astig1=0., astig2=0.,
                                            coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0.,
                                            aberrations=None, circular_pupil=True, obscuration=0.,
                                            interpolant=None, oversampling=1.5, pad_factor=1.5,
                                            max_size=None, nstruts=0, strut_thick=0.05,
                                            strut_angle=0.*galsim.degrees)

    Initializes optical_psf as a galsim.OpticalPSF() instance.

    @param lam_over_diam    Lambda / telescope diameter in the physical units adopted for scale 
                            (user responsible for consistency).
    @param defocus          Defocus in units of incident light wavelength.
    @param astig1           Astigmatism (like e2) in units of incident light wavelength.
    @param astig2           Astigmatism (like e1) in units of incident light wavelength.
    @param coma1            Coma along y in units of incident light wavelength.
    @param coma2            Coma along x in units of incident light wavelength.
    @param trefoil1         Trefoil (one of the arrows along y) in units of incident light
                            wavelength.
    @param trefoil2         Trefoil (one of the arrows along x) in units of incident light
                            wavelength.
    @param spher            Spherical aberration in units of incident light wavelength.
    @param aberrations      Optional keyword, to pass in a list, tuple, or NumPy array of
                            aberrations in units of incident light wavelength (ordered according to
                            the Noll convention), rather than passing in individual values for each
                            individual aberration.  Currently GalSim supports aberrations from
                            defocus through third-order spherical ('spher'), which are aberrations
                            4-11 in the Noll convention, and hence 'aberrations' should be an
                            object of length 12, with the first four numbers being ignored, the 5th
                            (index 4) being defocus, and so on through index 11 corresponding to
                            'spher'.
    @param circular_pupil   Adopt a circular pupil?  [default `circular_pupil = True`]
    @param obscuration      Linear dimension of central obscuration as fraction of pupil linear
                            dimension, [0., 1.).
    @param interpolant      Either an Interpolant2d (or Interpolant) instance or a string indicating
                            which interpolant should be used.  Options are 'nearest', 'sinc', 
                            'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the 
                            integer order to use. [default `interpolant = galsim.Quintic()`]
    @param oversampling     Optional oversampling factor for the InterpolatedImage. Setting 
                            oversampling < 1 will produce aliasing in the PSF (not good).
                            Usually oversampling should be somewhat larger than 1.  1.5 is 
                            usually a safe choice.
                            [default `oversampling = 1.5`]
    @param pad_factor       Additional multiple by which to zero-pad the PSF image to avoid folding
                            compared to what would be employed for a simple galsim.Airy 
                            [default `pad_factor = 1.5`].  Note that `pad_factor` may need to be 
                            increased for stronger aberrations, i.e. those larger than order unity.
    @param suppress_warning If pad_factor is too small, the code will emit a warning telling you
                            its best guess about how high you might want to raise it.  However,
                            you can suppress this warning by using suppress_warning=True.
                            [default `suppress_warning = False`]
    @param max_size         Set a maximum size of the internal image for the optical PSF profile
                            in arcsec.  Sometimes the code calculates a rather large image size
                            to describe the optical PSF profile.  If you will eventually be 
                            drawing onto a smallish postage stamp, you might want to save some
                            CPU time by setting max_size to be the size of your postage stamp.
                            [default `max_size = None`]
    @param flux             Total flux of the profile [default `flux=1.`].
    @param nstruts          Number of radial support struts to add to the central obscuration
                            [default `nstruts = 0`].
    @param strut_thick      Thickness of support struts as a fraction of pupil diameter
                            [default `strut_thick = 0.05`]
    @param strut_angle      Angle made between the vertical and the strut starting closest to it,
                            defined to be positive in the counter-clockwise direction; must be a
                            galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    @param gsparams         You may also specify a gsparams argument.  See the docstring for
                            galsim.GSParams using help(galsim.GSParams) for more information about
                            this option.

    Methods
    -------
    The OpticalPSF is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(), 
    applyShear() etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "lam_over_diam" : float }
    _opt_params = {
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
        "flux" : float,
        "nstruts" : int,
        "strut_thick" : float,
        "strut_angle" : galsim.Angle }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, lam_over_diam, defocus=0., astig1=0., astig2=0., coma1=0., coma2=0.,
                 trefoil1=0., trefoil2=0., spher=0., aberrations=None,
                 circular_pupil=True, obscuration=0., interpolant=None, oversampling=1.5,
                 pad_factor=1.5, suppress_warning=False, max_size=None, flux=1.,
                 nstruts=0, strut_thick=0.05, strut_angle=0.*galsim.degrees,
                 gsparams=None):

        
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
        if max_size != None and twoR > max_size:
            twoR = max_size

        # Get a good FFT size.  i.e. 2^n or 3 * 2^n.
        npix = goodFFTSize(int(np.ceil(twoR / scale_lookup)))

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

            # Check for non-zero elements in first 4 values.  Probably a mistake.
            if np.any(aberrations[0:4] != 0.0):
                import warnings
                warnings.warn(
                    "Detected non-zero value in aberrations[0:4] -- these values are ignored!")

        # Make the psf image using this scale and array shape
        optimage = galsim.optics.psf_image(
            lam_over_diam=lam_over_diam, scale=scale_lookup, array_shape=(npix, npix),
            aberrations=aberrations, circular_pupil=circular_pupil, obscuration=obscuration,
            flux=flux, nstruts=nstruts, strut_thick=strut_thick, strut_angle=strut_angle)
        
        # Initialize the GSObject (InterpolatedImage)
        GSObject.__init__(
            self, galsim.InterpolatedImage(optimage, x_interpolant=interpolant, scale=scale_lookup,
                                           calculate_stepk=True, calculate_maxk=True,
                                           use_true_center=False, normalization='sb',
                                           gsparams=gsparams))
        # The above procedure ends up with a larger image than we really need, which
        # means that the default stepK value will be smaller than we need.  
        # Hence calculate_stepk=True and calculate_maxk=True above.

        if not suppress_warning:
            # Check the calculated stepk value.  If it is smaller than stepk, then there might
            # be aliasing.
            final_stepk = self.SBProfile.stepK()
            if final_stepk < stepk:
                import warnings
                warnings.warn(
                    "The calculated stepk (%g) for OpticalPSF is smaller "%final_stepk +
                    "than what was used to build the wavefront (%g). "%stepk +
                    "This could lead to aliasing problems. " +
                    "Using pad_factor >= %f is recommended."%(pad_factor * stepk / final_stepk))



def generate_pupil_plane(array_shape=(256, 256), scale=1., lam_over_diam=2., circular_pupil=True,
                         obscuration=0., nstruts=0, strut_thick=0.05, 
                         strut_angle=0.*galsim.degrees):
    """Generate a pupil plane, including a central obscuration such as caused by a secondary mirror.

    @param array_shape     the NumPy array shape desired for the output array.
    @param scale           grid spacing of PSF in real space units.
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
 
    Returns a tuple (rho, in_pupil), the first of which is the coordinates of the pupil
    in unit disc-scaled coordinates for use by Zernike polynomials (as a complex number)
    for describing the wavefront across the pupil plane.  The array in_pupil is a vector of 
    Bools used to specify where in the pupil plane described by rho is illuminated.  See also 
    wavefront(). 
    """
    kmax_internal = scale * 2. * np.pi / lam_over_diam # INTERNAL kmax in units of array grid spacing
    # Build kx, ky coords
    kx, ky = utilities.kxky(array_shape)
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
            kxs, kys = utilities.rotate_xy(kx, ky, -strut_angle) # strut rotation +=ve, so coords
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
            kxs, kys = utilities.rotate_xy(kxs, kys, -rotang)
            in_pupil *= (
                (np.abs(kxs) >= .5 * strut_thick * kmax_internal) +
                ((kys < 0.) * (np.abs(kxs) < .5 * strut_thick * kmax_internal)))
    return rho, in_pupil

def wavefront(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
              circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
              strut_angle=0.*galsim.degrees):
    """Return a complex, aberrated wavefront across a circular (default) or square pupil.
    
    Outputs a complex image (shape=array_shape) of a circular pupil wavefront of unit amplitude
    that can be easily transformed to produce an optical PSF with lambda/D = lam_over_diam on an
    output grid of spacing scale.  This routine would need to be modified in order to include higher
    order aberrations than spher (order 11 in Noll convention).

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.
    
    The pupil sample locations are arranged in standard DFT element ordering format, so that
    (kx, ky) = (0, 0) is the [0, 0] array element.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention defined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976). For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    @param array_shape     the NumPy array shape desired for the output array.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].

    Outputs the wavefront for kx, ky locations corresponding to kxky(array_shape).
    """
    # Define the pupil coordinates and non-zero regions based on input kwargs
    rho_all, in_pupil = generate_pupil_plane(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam,
        circular_pupil=circular_pupil, obscuration=obscuration,
        nstruts=nstruts, strut_thick=strut_thick, strut_angle=strut_angle)

    # Then make wavefront image
    wf = np.zeros(array_shape, dtype=complex)

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
            # Constant terms: includes defocus (4)
            -np.sqrt(3.) * aberrations[4]

            # Terms with rhosq, but no rho, rho**2, etc.: includes defocus (4) and spher (11)
            + rhosq * ( 2. * np.sqrt(3.) * aberrations[4]
                        - 6. * np.sqrt(5.) * aberrations[11]
                        + rhosq * (6. * np.sqrt(5.) * aberrations[11]) )

            # Now the powers of rho: includes coma2 (8), coma1 (7), astig2 (6), astig1 (5), trefoil2
            # (10), trefoil1 (9).
            # We eventually take the real part
            + ( rho * ( (rhosq-2./3.) * (3. * np.sqrt(8.) * (aberrations[8] - 1j * aberrations[7]))
                        + rho * ( (np.sqrt(6.) * (aberrations[6] - 1j * aberrations[5]))
                                   + rho * (np.sqrt(8.) * (aberrations[10] - 1j * aberrations[9])) 
                                )
                      ) 
              ).real
    )

    wf[in_pupil] = np.exp(2j * np.pi * temp)

    return wf

def wavefront_image(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
                    circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
                    strut_angle=0.*galsim.degrees):
    """Return wavefront as a (real, imag) tuple of Image objects rather than complex NumPy
    array.

    Outputs a circular pupil wavefront of unit amplitude that can be easily transformed to produce
    an optical PSF with lambda/diam = lam_over_diam on an output grid of spacing scale.

    The Image output can be used to directly instantiate an InterpolatedImage, and its 
    scale will reflect the spacing of the output grid in the system of units adopted for 
    lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.
    
    The pupil sample locations are arranged in standard DFT element ordering format, so that
    (kx, ky) = (0, 0) is the [0, 0] array element.  The scale of the output Image is correct in
    k space units.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention defined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976). For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    @param array_shape     the NumPy array shape desired for the output array.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    """
    array = wavefront(
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
    return (imreal, imimag)

def psf(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
        circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
        strut_angle=0.*galsim.degrees, flux=1.):
    """Return NumPy array containing circular (default) or square pupil PSF with low-order 
    aberrations.

    The PSF is centred on the array[array_shape[0] / 2, array_shape[1] / 2] pixel by default, and
    uses surface brightness rather than flux units for pixel values, matching SBProfile.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.

    Ouput NumPy array is C-contiguous.

    @param array_shape     the NumPy array shape desired for the output array.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    @param flux            total flux of the profile [default flux=1.].
    """
    wf = wavefront(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle)

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
    im = utilities.roll2d(im, (array_shape[0] / 2, array_shape[1] / 2)) 
    im *= (flux / (im.sum() * scale**2))

    return im

def psf_image(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
              circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
              strut_angle=0.*galsim.degrees, flux=1.):
    """Return circular (default) or square pupil PSF with low-order aberrations as an Image.

    The PSF is centred on the array[array_shape[0] / 2, array_shape[1] / 2] pixel by default, and
    uses surface brightness rather than flux units for pixel values, matching SBProfile.

    The Image output can be used to directly instantiate an InterpolatedImage, and its 
    scale will reflect the spacing of the output grid in the system of units adopted for 
    lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.

    @param array_shape     the NumPy array shape desired for the array view of the Image.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    @param flux            total flux of the profile [default flux=1.].
    """
    array = psf(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        circular_pupil=circular_pupil, obscuration=obscuration, flux=flux, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle)
    im = galsim.Image(array.astype(np.float64), scale=scale)
    return im

def otf(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
        circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
        strut_angle=0.*galsim.degrees):
    """Return the complex OTF of a circular (default) or square pupil with low-order aberrations as
    a NumPy array.

    OTF array element ordering follows the DFT standard of kxky(array_shape), and has
    otf[0, 0] = 1+0j by default.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.

    Output complex NumPy array is C-contiguous.
    
    @param array_shape     the NumPy array shape desired for the output array.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    """
    wf = wavefront(
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

    OTF array element ordering follows the DFT standard of kxky(array_shape), and has
    otf[0, 0] = 1+0j by default.  The scale of the output Image is correct in k space units.

    The Image output can be used to directly instantiate an InterpolatedImage, and its 
    scale will reflect the spacing of the output grid in the system of units adopted for 
    lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.
    
    @param array_shape     the NumPy array shape desired for array views of Image tuple.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    """
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

    MTF array element ordering follows the DFT standard of kxky(array_shape), and has
    mtf[0, 0] = 1 by default.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.

    Output double NumPy array is C-contiguous.

    @param array_shape     the NumPy array shape desired for the output array.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    """
    return np.abs(otf(
        array_shape=array_shape, scale=scale, lam_over_diam=lam_over_diam, aberrations=aberrations,
        obscuration=obscuration, circular_pupil=circular_pupil, nstruts=nstruts,
        strut_thick=strut_thick, strut_angle=strut_angle))

def mtf_image(array_shape=(256, 256), scale=1., lam_over_diam=2., aberrations=None,
              circular_pupil=True, obscuration=0., nstruts=0, strut_thick=0.05,
              strut_angle=0.*galsim.degrees):
    """Return the MTF of a circular (default) or square pupil with low-order aberrations as an 
    Image.

    MTF array element ordering follows the DFT standard of kxky(array_shape), and has
    mtf[0, 0] = 1 by default.  The scale of the output Image is correct in k space units.

    The Image output can be used to directly instantiate an InterpolatedImage, and its 
    scale will reflect the spacing of the output grid in the system of units adopted for 
    lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.

    @param array_shape     the NumPy array shape desired for the array view of the Image.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.).
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    """
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

    PTF array element ordering follows the DFT standard of kxky(array_shape), and has
    ptf[0, 0] = 0. by default.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.

    Output double NumPy array is C-contiguous.

    @param array_shape     the NumPy array shape desired for the output array.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick     Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle     Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    """
    kx, ky = utilities.kxky(array_shape)
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

    PTF array element ordering follows the DFT standard of kxky(array_shape), and has
    ptf[0, 0] = 0. by default.  The scale of the output Image is correct in k space units.

    The Image output can be used to directly instantiate an InterpolatedImage, and its 
    scale will reflect the spacing of the output grid in the system of units adopted for 
    lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * scale.

    @param array_shape     the NumPy array shape desired for the array view of the Image.
    @param scale           grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for scale 
                           (user responsible for consistency).
    @param aberrations     NumPy array containing the supported aberrations in units of incident
                           light wavelength, ordered according to the Noll convention: defocus,
                           astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher.  Since these are
                           aberrations 4-11 in the Noll convention, 'aberrations' should have length
                           12, with defocus corresponding to index 4 and so on.  The first four
                           numbers in this array will be ignored.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    @param nstruts         Number of radial support struts to add to the central obscuration
                           [default `nstruts = 0`].
    @param strut_thick      Thickness of support struts as a fraction of pupil diameter
                           [default `strut_thick = 0.05`].
    @param strut_angle        Angle made between the vertical and the strut starting closest to it,
                           defined to be positive in the counter-clockwise direction; must be a
                           galsim.Angle instance [default `strut_angle = 0. * galsim.degrees`].
    """
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
