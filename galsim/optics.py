# Copyright 2012, 2013 The GalSim developers:
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

These are just functions; they are used to generate galsim.OpticalPSF() class instances (see 
base.py).   

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
from galsim import GSObject

class OpticalPSF(GSObject):
    """A class describing aberrated PSFs due to telescope optics.  Has an SBInterpolatedImage in the
    SBProfile attribute.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention definined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976).  For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    Initialization
    --------------
    
        >>> optical_psf = galsim.OpticalPSF(lam_over_diam, defocus=0., astig1=0., astig2=0.,
                                            coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0.,
                                            circular_pupil=True, obscuration=0., interpolant=None,
                                            oversampling=1.5, pad_factor=1.5)

    Initializes optical_psf as a galsim.OpticalPSF() instance.

    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?  [default `circular_pupil = True`]
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    @param interpolant     Either an Interpolant2d (or Interpolant) instance or a string indicating
                           which interpolant should be used.  Options are 'nearest', 'sinc', 
                           'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the 
                           integer order to use. [default `interpolant = galsim.Quintic()`]
    @param oversampling    Optional oversampling factor for the SBInterpolatedImage table 
                           [default `oversampling = 1.5`], setting oversampling < 1 will produce 
                           aliasing in the PSF (not good).
    @param pad_factor      Additional multiple by which to zero-pad the PSF image to avoid folding
                           compared to what would be employed for a simple galsim.Airy 
                           [default `pad_factor = 1.5`].  Note that `pad_factor` may need to be 
                           increased for stronger aberrations, i.e. those larger than order unity.
    @param flux            Total flux of the profile [default `flux=1.`].
    @param gsparams        You may also specify a gsparams argument.  See the docstring for
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
        "interpolant" : str ,
        "flux" : float }
    _single_params = []
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, lam_over_diam, defocus=0.,
                 astig1=0., astig2=0., coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0., 
                 circular_pupil=True, obscuration=0., interpolant=None, oversampling=1.5,
                 pad_factor=1.5, flux=1., gsparams=None):

        # Currently we load optics, noise etc in galsim/__init__.py, but this might change (???)
        import galsim.optics
        
        # Choose dx for lookup table using Nyquist for optical aperture and the specified
        # oversampling factor
        dx_lookup = .5 * lam_over_diam / oversampling
        
        # We need alias_threshold here, so don't wait to make this a default GSParams instance
        # if the user didn't specify anything else.
        if not gsparams:
            gsparams = galsim.GSParams()

        # Use a similar prescription as SBAiry to set Airy stepK and thus reference unpadded image
        # size in physical units
        stepk_airy = min(
            gsparams.alias_threshold * .5 * np.pi**3 * (1. - obscuration) / lam_over_diam,
            np.pi / 5. / lam_over_diam)
        
        # Boost Airy image size by a user-specifed pad_factor to allow for larger, aberrated PSFs,
        # also make npix always *odd* so that opticalPSF lookup table array is correctly centred:
        npix = 1 + 2 * (np.ceil(pad_factor * (np.pi / stepk_airy) / dx_lookup)).astype(int)
        
        # Make the psf image using this dx and array shape
        optimage = galsim.optics.psf_image(
            lam_over_diam=lam_over_diam, dx=dx_lookup, array_shape=(npix, npix), defocus=defocus,
            astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1,
            trefoil2=trefoil2, spher=spher, circular_pupil=circular_pupil, obscuration=obscuration,
            flux=flux)
        
        # If interpolant not specified on input, use a Quintic interpolant
        if interpolant is None:
            quintic = galsim.Quintic(tol=1e-4)
            self.interpolant = galsim.InterpolantXY(quintic)
        else:
            self.interpolant = galsim.utilities.convert_interpolant_to_2d(interpolant)

        # Initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBInterpolatedImage(optimage, xInterp=self.interpolant, dx=dx_lookup,
                                             gsparams=gsparams))

        # The above procedure ends up with a larger image than we really need, which
        # means that the default stepK value will be smaller than we need.  
        # Thus, we call the function calculateStepK() to refine the value.
        self.SBProfile.calculateStepK()
        self.SBProfile.calculateMaxK()


def generate_pupil_plane(array_shape=(256, 256), dx=1., lam_over_diam=2., circular_pupil=True,
                         obscuration=0.):
    """Generate a pupil plane, including a central obscuration such as caused by a secondary mirror.

    @param array_shape     the NumPy array shape desired for the output array.
    @param dx              grid spacing of PSF in real space units.
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)

    Returns a tuple (rho, theta, in_pupil), the first two of which are the coordinates of the
    pupil in unit disc-scaled coordinates for use by Zernike polynomials for describing the
    wavefront across the pupil plane.  The array in_pupil is a vector of Bools used to specify
    where in the pupil plane described by rho, theta is illuminated.  See also optics.wavefront. 
    """
    kmax_internal = dx * 2. * np.pi / lam_over_diam # INTERNAL kmax in units of array grid spacing
    # Build kx, ky coords
    kx, ky = utilities.kxky(array_shape)
    # Then define unit disc rho and theta pupil coords for Zernike polynomials
    rho = np.sqrt((kx**2 + ky**2) / (.5 * kmax_internal)**2)
    theta = np.arctan2(ky, kx)
    # Cut out circular pupil if desired (default, square pupil optionally supported) and include 
    # central obscuration
    if obscuration >= 1.:
        raise ValueError("Pupil fully obscured! obscuration ="+str(obscuration)+" (>= 1)")
    if circular_pupil:
        in_pupil = (rho < 1.)
        if obscuration > 0.:
            in_pupil = in_pupil * (rho >= obscuration)  # * acts like "and" for boolean arrays
    else:
        in_pupil = (np.abs(kx) < .5 * kmax_internal) * (np.abs(ky) < .5 * kmax_internal)
        if obscuration > 0.:
            in_pupil = in_pupil * (
                (np.abs(kx) >= .5 * obscuration * kmax_internal) *
                (np.abs(ky) >= .5 * obscuration * kmax_internal))
    return rho, theta, in_pupil

def wavefront(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0.,
              circular_pupil=True, obscuration=0.):
    """Return a complex, aberrated wavefront across a circular (default) or square pupil.
    
    Outputs a complex image (shape=array_shape) of a circular pupil wavefront of unit amplitude
    that can be easily transformed to produce an optical PSF with lambda/D = lam_over_diam on an
    output grid of spacing dx.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.
    
    The pupil sample locations are arranged in standard DFT element ordering format, so that
    (kx, ky) = (0, 0) is the [0, 0] array element.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention definined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976). For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    @param array_shape     the NumPy array shape desired for the output array.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    
    Outputs the wavefront for kx, ky locations corresponding to kxky(array_shape).
    """
    # Define the pupil coordinates and non-zero regions based on input kwargs
    rho, theta, in_pupil = generate_pupil_plane(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, circular_pupil=circular_pupil,
        obscuration=obscuration)
    pi = np.pi # minor but saves Python checking the entire np. namespace every time I need pi    
    # Then make wavefront image
    wf = np.zeros(array_shape, dtype=complex)
    wf[in_pupil] = 1.
    # Defocus
    wf[in_pupil] *= np.exp(2j * pi * defocus * np.sqrt(3.) * (2. * rho[in_pupil]**2 - 1.))
    # Astigmatism (like e2)
    wf[in_pupil] *= np.exp(2j * pi * astig1 * np.sqrt(6.) * rho[in_pupil]**2
                           * np.sin(2. * theta[in_pupil]))
    # Astigmatism (like e1)
    wf[in_pupil] *= np.exp(2j * pi * astig2 * np.sqrt(6.) * rho[in_pupil]**2
                           * np.cos(2. * theta[in_pupil]))
    # Coma along x2
    wf[in_pupil] *= np.exp(2j * pi * coma1 * np.sqrt(8.) * (3. * rho[in_pupil]**2 - 2.)
                           * rho[in_pupil] * np.sin(theta[in_pupil]))
    # Coma along x1
    wf[in_pupil] *= np.exp(2j * pi * coma2 * np.sqrt(8.) * (3. * rho[in_pupil]**2 - 2.)
                           * rho[in_pupil]* np.cos(theta[in_pupil]))
    # Trefoil (one of the arrows along x2)
    wf[in_pupil] *= np.exp(2j * pi * trefoil1 * np.sqrt(8.) * rho[in_pupil]**3
                           * np.sin(3. * theta[in_pupil]))
    # Trefoil (one of the arrows along x1)
    wf[in_pupil] *= np.exp(2j * pi * trefoil2 * np.sqrt(8.) * rho[in_pupil]**3
                           * np.cos(3. * theta[in_pupil]))
    # Spherical aberration
    wf[in_pupil] *= np.exp(2j * pi * spher * np.sqrt(5.)
                           * (6. * rho[in_pupil]**4 - 6. * rho[in_pupil]**2 + 1.))

    return wf

def wavefront_image(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0.,
                    astig1=0., astig2=0., coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0.,
                    circular_pupil=True, obscuration=0.):
    """Return wavefront as a (real, imag) tuple of ImageViewD objects rather than complex NumPy
    array.

    Outputs a circular pupil wavefront of unit amplitude that can be easily transformed to produce
    an optical PSF with lambda/diam = lam_over_diam on an output grid of spacing dx.

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted
    for lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.
    
    The pupil sample locations are arranged in standard DFT element ordering format, so that
    (kx, ky) = (0, 0) is the [0, 0] array element.  The scale of the output ImageViewD is correct in
    k space units.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention definined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976). For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    @param array_shape     the NumPy array shape desired for the output array.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    """
    array = wavefront(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher,
        circular_pupil=circular_pupil, obscuration=obscuration)
    imreal = galsim.ImageViewD(np.ascontiguousarray(array.real.astype(np.float64)))
    imimag = galsim.ImageViewD(np.ascontiguousarray(array.imag.astype(np.float64)))
    if array_shape[0] != array_shape[1]:
        import warnings
        warnings.warn(
            "Wavefront Images' scales will not be correct in both directions for non-square "+
            "arrays, only square grids currently supported by galsim.Images.")
    imreal.setScale(2. * np.pi / array_shape[0])
    imimag.setScale(2. * np.pi / array_shape[0])
    return (imreal, imimag)

def psf(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0., coma1=0.,
        coma2=0., trefoil1=0., trefoil2=0., spher=0., circular_pupil=True, obscuration=0., flux=1.):
    """Return NumPy array containing circular (default) or square pupil PSF with low-order 
    aberrations.

    The PSF is centred on the array[array_shape[0] / 2, array_shape[1] / 2] pixel by default, and
    uses surface brightness rather than flux units for pixel values, matching SBProfile.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.

    Ouput NumPy array is C-contiguous.

    @param array_shape     the NumPy array shape desired for the output array.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    @param flux            total flux of the profile [default flux=1.]
    """
    wf = wavefront(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher,
        circular_pupil=circular_pupil, obscuration=obscuration)
    ftwf = np.fft.fft2(wf)  # I think this (and the below) is quicker than np.abs(ftwf)**2
    # The roll operation below restores the c_contiguous flag, so no need for a direct action
    im = utilities.roll2d((ftwf * ftwf.conj()).real, (array_shape[0] / 2, array_shape[1] / 2)) 
    return im * (flux / (im.sum() * dx**2))

def psf_image(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0., circular_pupil=True,
              obscuration=0., flux=1.):
    """Return circular (default) or square pupil PSF with low-order aberrations as an ImageViewD.

    The PSF is centred on the array[array_shape[0] / 2, array_shape[1] / 2] pixel by default, and
    uses surface brightness rather than flux units for pixel values, matching SBProfile.

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted
    for lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.

    @param array_shape     the NumPy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    @param flux            total flux of the profile [default flux=1.]
    """
    array = psf(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher,
        circular_pupil=circular_pupil, obscuration=obscuration, flux=flux)
    im = galsim.ImageViewD(array.astype(np.float64))
    im.setScale(dx)
    return im

def otf(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0., coma1=0.,
        coma2=0., trefoil1=0., trefoil2=0., spher=0., circular_pupil=True, obscuration=0.):
    """Return the complex OTF of a circular (default) or square pupil with low-order aberrations as
    a NumPy array.

    OTF array element ordering follows the DFT standard of kxky(array_shape), and has
    otf[0, 0] = 1+0j by default.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.

    Output complex NumPy array is C-contiguous.
    
    @param array_shape     the NumPy array shape desired for the output array.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    """
    wf = wavefront(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher, 
        circular_pupil=circular_pupil, obscuration=obscuration)
    ftwf = np.fft.fft2(wf)  # I think this (and the below) is quicker than np.abs(ftwf)**2
    otf = np.fft.ifft2((ftwf * ftwf.conj()).real)
    # Make unit flux before returning
    return np.ascontiguousarray(otf) / otf[0, 0].real

def otf_image(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0., circular_pupil=True,
              obscuration=0.):
    """Return the complex OTF of a circular (default) or square pupil with low-order aberrations as 
    a (real, imag) tuple of ImageViewD objects, rather than a complex NumPy array.

    OTF array element ordering follows the DFT standard of kxky(array_shape), and has
    otf[0, 0] = 1+0j by default.  The scale of the output ImageViewD is correct in k space units.

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted
    for lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.
    
    @param array_shape     the NumPy array shape desired for array views of ImageViewD tuple.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    """
    array = otf(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher,
        circular_pupil=circular_pupil, obscuration=obscuration)
    imreal = galsim.ImageViewD(np.ascontiguousarray(array.real.astype(np.float64)))
    imimag = galsim.ImageViewD(np.ascontiguousarray(array.imag.astype(np.float64)))
    if array_shape[0] != array_shape[1]:
        import warnings
        warnings.warn(
            "OTF Images' scales will not be correct in both directions for non-square arrays, "+
            "only square grids currently supported by galsim.Images.")
    imreal.setScale(2. * np.pi / array_shape[0])
    imimag.setScale(2. * np.pi / array_shape[0])
    return (imreal, imimag)

def mtf(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0., coma1=0.,
        coma2=0., trefoil1=0., trefoil2=0., spher=0., circular_pupil=True, obscuration=0.):
    """Return NumPy array containing the MTF of a circular (default) or square pupil with low-order
    aberrations.

    MTF array element ordering follows the DFT standard of kxky(array_shape), and has
    mtf[0, 0] = 1 by default.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.

    Output double NumPy array is C-contiguous.

    @param array_shape     the NumPy array shape desired for the output array.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    """
    return np.abs(otf(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher,
        obscuration=obscuration, circular_pupil=circular_pupil))

def mtf_image(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0., circular_pupil=True,
              obscuration=0.):
    """Return the MTF of a circular (default) or square pupil with low-order aberrations as an 
    ImageViewD.

    MTF array element ordering follows the DFT standard of kxky(array_shape), and has
    mtf[0, 0] = 1 by default.  The scale of the output ImageViewD is correct in k space units.

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted
    for lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.

    @param array_shape     the NumPy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    """
    array = mtf(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher,
        circular_pupil=circular_pupil, obscuration=obscuration)
    im = galsim.ImageViewD(array.astype(np.float64))
    if array_shape[0] != array_shape[1]:
        import warnings
        warnings.warn(
            "MTF Image scale will not be correct in both directions for non-square arrays, only "+
            "square grids currently supported by galsim.Images.")
    im.setScale(2. * np.pi / array_shape[0])
    return im

def ptf(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0., coma1=0.,
        coma2=0., trefoil1=0., trefoil2=0., spher=0., circular_pupil=True, obscuration=0.):
    """Return NumPy array containing the PTF [radians] of a circular (default) or square pupil with
    low-order aberrations.

    PTF array element ordering follows the DFT standard of kxky(array_shape), and has
    ptf[0, 0] = 0. by default.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.

    Output double NumPy array is C-contiguous.

    @param array_shape     the NumPy array shape desired for the output array.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    """
    kx, ky = utilities.kxky(array_shape)
    k2 = (kx**2 + ky**2)
    ptf = np.zeros(array_shape)
    kmax_internal = dx * 2. * np.pi / lam_over_diam # INTERNAL kmax in units of array grid spacing
    # Try to handle where both real and imag tend to zero...
    ptf[k2 < kmax_internal**2] = np.angle(otf(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher,
        circular_pupil=circular_pupil, obscuration=obscuration)[k2 < kmax_internal**2])
    return ptf

def ptf_image(array_shape=(256, 256), dx=1., lam_over_diam=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0., circular_pupil=True,
              obscuration=0.):
    """Return the PTF [radians] of a circular (default) or square pupil with low-order aberrations
    as an ImageViewD.

    PTF array element ordering follows the DFT standard of kxky(array_shape), and has
    ptf[0, 0] = 0. by default.  The scale of the output ImageViewD is correct in k space units.

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted
    for lam_over_diam.

    To ensure properly Nyquist sampled output any user should set lam_over_diam >= 2. * dx.

    @param array_shape     the NumPy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    """
    array = ptf(
        array_shape=array_shape, dx=dx, lam_over_diam=lam_over_diam, defocus=defocus, astig1=astig1,
        astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1, trefoil2=trefoil2, spher=spher,
        circular_pupil=circular_pupil, obscuration=obscuration)
    im = galsim.ImageViewD(array.astype(np.float64))
    if array_shape[0] != array_shape[1]:
        import warnings
        warnings.warn(
            "PTF Image scale will not be correct in both directions for non-square arrays, only "+
            "square grids currently supported by galsim.Images.")
    im.setScale(2. * np.pi / array_shape[0])
    return im
