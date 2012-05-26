import numpy as np
import galsim

"""@file optics.py @brief Module containing the optical PSF generation routines.

These are just functions; they are used to generate galsim.Optics() class instances (see base.py).

Glossary of key terms used in function names:

PSF = point spread function

OTF = optical transfer function = FT{PSF}

MTF = modulation transfer function = |FT{PSF}|

PTF = phase transfer function = p, where OTF = MTF * exp(i * p)

Wavefront = the amplitude and phase of the incident light on the telescope pupil, encoded as a
complex number. The OTF is the autocorrelation function of the wavefront.

TODO: Implement a centrally-obstructed pupil plane (e.g. such as is caused by secondary mirrors).
"""

def roll2d(image, (iroll, jroll)):
    """Perform a 2D roll (circular shift) on a supplied 2D numpy array, conveniently.

    Parameters
    ----------
    @param image            the numpy array to be circular shifted.
    @param (iroll, jroll)   the roll in the i and j dimensions, respectively.

    @returns the rolled image.
    """
    return np.roll(np.roll(image, jroll, axis=1), iroll, axis=0)

def kxky(array_shape=(256, 256)):
    """Return the tuple kx, ky corresponding to the DFT of a unit integer-sampled array of input
    shape.
    
    Uses the SBProfile conventions for Fourier space, so k varies in approximate range (-pi, pi].
    Uses the most common DFT element ordering conventions (and those of FFTW), so that `(0, 0)`
    array element corresponds to `(kx, ky) = (0, 0)`.

    See also the docstring for np.fftfreq, which uses the same DFT convention, and is called here,
    but misses a factor of pi.
    
    Adopts Numpy array index ordering so that the trailing axis corresponds to kx, rather than the
    leading axis as would be expected in IDL/Fortran.  See docstring for numpy.meshgrid which also
    uses this convention.

    Parameters
    ----------
    @param array_shape   the Numpy array shape desired for `kx, ky`. 
    """
    k_xaxis = np.fft.fftfreq(array_shape[1]) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(array_shape[0]) * 2. * np.pi
    return np.meshgrid(k_xaxis, k_yaxis)

def wavefront(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., spher=0., circular_pupil=True):
    """Return a complex, aberrated wavefront across a circular (default) or square pupil.
    
    Outputs a complex image (shape=array_shape) of a circular pupil wavefront of unit amplitude
    that can be easily transformed to produce an optical PSF with lambda/D = lam_over_D on an
    output grid of spacing dx.

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.
    
    The pupil sample locations are arranged in standard DFT element ordering format, so that
    (kx, ky) = (0, 0) is the [0, 0] array element.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the definitions given here:
    http://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the output array.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_D      lambda / D in the physical units adopted for dx (user responsible for 
                           consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    
    Outputs the wavefront for kx, ky locations corresponding to kxky(array_shape).
    """
    pi = np.pi # minor but saves Python checking the entire np. namespace every time I need pi
    kmax_internal = dx * 2. * pi / lam_over_D # INTERNAL kmax in units of array grid spacing
    # Build kx, ky coords
    kx, ky = kxky(array_shape)
    # Then define unit disc rho and theta pupil coords for Zernike polynomials
    rho = np.sqrt((kx**2 + ky**2) / (.5 * kmax_internal)**2)
    theta = np.arctan2(ky, kx)
    # Cut out circular pupil if desired (default)
    if circular_pupil:
        in_pupil = (rho < 1.)
    else:
        in_pupil = (np.abs(kx) <= .5 * kmax) * (np.abs(ky) <= .5 * kmax_internal)
    # Then make wavefront image
    wf = np.zeros(array_shape, dtype=complex)
    wf[in_pupil] = 1.
    # Defocus
    wf[in_pupil] *= np.exp(2j * pi * defocus * (2. * rho[in_pupil]**2 - 1.))
    # Astigmatism (like e1)
    wf[in_pupil] *= np.exp(2j * pi * astig1 * rho[in_pupil]**2 * np.cos(2. * theta[in_pupil]))
    # Astigmatism (like e2)
    wf[in_pupil] *= np.exp(2j * pi * astig2 * rho[in_pupil]**2 * np.sin(2. * theta[in_pupil]))
    # Coma along x1
    wf[in_pupil] *= np.exp(2j * pi * coma1 * (3. * rho[in_pupil]**2 - 2.) * rho[in_pupil]
                           * np.cos(theta[in_pupil]))
    # Coma along x2
    wf[in_pupil] *= np.exp(2j * pi * coma2 * (3. * rho[in_pupil]**2 - 2.) * rho[in_pupil]
                           * np.sin(theta[in_pupil]))
    # Spherical abberation
    wf[in_pupil] *= np.exp(2j * pi * spher * (6. * rho[in_pupil]**4 - 6. * rho[in_pupil]**2 + 1.))
    return wf

def wavefront_image(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0.,
                    coma1=0., coma2=0., spher=0., circular_pupil=True):
    """@brief Return wavefront as a (real, imag) tuple of ImageViewD objects rather than complex
    numpy array.

    Outputs a circular pupil wavefront of unit amplitude that can be easily transformed to produce
    an optical PSF with lambda/D = lam_over_D on an output grid of spacing dx.

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.
    
    The pupil sample locations are arranged in standard DFT element ordering format, so that
    (kx, ky) = (0, 0) is the [0, 0] array element.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the definitions given here:
    http://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the output array.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    """
    array = wavefront(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D, defocus=defocus,
                      astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                      circular_pupil=circular_pupil)
    return (galsim.ImageViewD(np.ascontiguousarray(array.real.astype(np.float64))),
            galsim.ImageViewD(np.ascontiguousarray(array.imag.astype(np.float64))))

def psf(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0., coma1=0.,
        coma2=0., spher=0., circular_pupil=True, obs=None):
    """@brief Return numpy array containing circular (default) or square pupil PSF with low-order
    aberrations.

    The PSF is centred on the array[array_shape[0] / 2, array_shape[1] / 2] pixel by default, and
    uses surface brightness rather than flux units for pixel values, matching SBProfile.

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.

    Ouput numpy array is C-contiguous.

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the output array.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obs             add a central obstruction due to secondary mirror?
    """
    if obs == None:  # TODO: Build a secondary mirror obstruction function!
        wf = wavefront(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D, defocus=defocus,
                       astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                       circular_pupil=circular_pupil)
    else:
        raise NotImplementedError('Secondary mirror obstruction not yet implemented')
    ftwf = np.fft.fft2(wf)  # I think this (and the below) is quicker than np.abs(ftwf)**2
    # The roll operation below restores the c_contiguous flag, so no need for a direct action
    im = roll2d((ftwf * ftwf.conj()).real, (array_shape[0] / 2, array_shape[1] / 2)) 
    return im / im.sum() / dx**2

def psf_image(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., spher=0., circular_pupil=True, obs=None):
    """@brief Return circular (default) or square pupil PSF with low-order aberrations as an
    ImageViewD.

    The PSF is centred on the array[array_shape[0] / 2, array_shape[1] / 2] pixel by default, and
    uses surface brightness rather than flux units for pixel values, matching SBProfile.

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the array view on the Image.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obs             add a central obstruction due to secondary mirror?
    """
    array = psf(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D, defocus=defocus,
                astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                circular_pupil=circular_pupil, obs=obs)
    return galsim.ImageViewD(array.astype(np.float64))

def otf(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0., coma1=0.,
        coma2=0., spher=0., circular_pupil=True, obs=None):
    """@brief Return the complex OTF of a circular (default) or square pupil with low-order
    aberrations as a numpy array.

    OTF array element ordering follows the DFT standard of kxky(array_shape).

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.

    Output complex numpy array is C-contiguous.
    
    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the output array.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obs             add a central obstruction due to secondary mirror?
    """
    if obs == None:  # TODO: Build a secondary mirror obstruction function!
        wf = wavefront(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D, defocus=defocus,
                       astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                       circular_pupil=circular_pupil)
    else:
        raise NotImplementedError('Secondary mirror obstruction not yet implemented')
    ftwf = np.fft.fft2(wf)  # I think this (and the below) is quicker than np.abs(ftwf)**2
    otf = np.fft.ifft2((ftwf * ftwf.conj()).real)
    # Make unit flux before returning
    return np.ascontiguousarray(otf) / otf[0, 0].real

def otf_image(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., spher=0., kmax=np.pi, circular_pupil=True, obs=None):
    """@brief Return the complex OTF of a circular (default) or square pupil with low-order
    aberrations as a (real, imag) tuple of ImageViewD objects rather than a complex numpy array.

    OTF array element ordering follows the DFT standard of kxky(array_shape).

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.
    
    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for array views into image.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obs             add a central obstruction due to secondary mirror?
    """
    array = otf(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D, defocus=defocus,
                astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                circular_pupil=circular_pupil, obs=obs)
    return (galsim.ImageViewD(np.ascontiguousarray(array.real.astype(np.float64))),
            galsim.ImageViewD(np.ascontiguousarray(array.imag.astype(np.float64))))

def mtf(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0., coma1=0.,
        coma2=0., spher=0., circular_pupil=True, obs=None):
    """@brief Return numpy array containing the MTF of a circular (default) or square pupil with
    low-order aberrations.

    MTF array element ordering follows the DFT standard of kxky(array_shape).

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.

    Output double numpy array is C-contiguous.

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the output array.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obs             add a central obstruction due to secondary mirror?
    """
    return np.abs(otf(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D, defocus=defocus,
                      astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                      obs=obs, circular_pupil=circular_pupil))

def mtf_image(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., spher=0., circular_pupil=True, obs=None):
    """@brief Return the MTF of a circular (default) or square pupil with low-order aberrations as
    an ImageViewD.

    MTF array element ordering follows the DFT standard of kxky(array_shape).

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the array view on the Image.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obs             add a central obstruction due to secondary mirror?
    """
    array = mtf(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D, defocus=defocus,
                astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                circular_pupil=circular_pupil, obs=obs)
    return galsim.ImageViewD(array.astype(np.float64))

def ptf(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0., coma1=0.,
        coma2=0., spher=0., circular_pupil=True, obs=None):
    """@brief Return numpy array containing the PTF [radians] of a circular (default) or square
    pupil with low-order aberrations.

    PTF array element ordering follows the DFT standard of kxky(array_shape).

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.

    Output double numpy array is C-contiguous.

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the output array.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obs             add a central obstruction due to secondary mirror?
    """
    kx, ky = kxky(array_shape)
    k2 = (kx**2 + ky**2)
    ptf = np.zeros(array_shape)
    kmax_internal = dx * 2. * np.pi / lam_over_D # INTERNAL kmax in units of array grid spacing
    # Try to handle where both real and imag tend to zero...
    ptf[k2 < kmax_internal**2] = np.angle(otf(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D,
                                              defocus=defocus, astig1=astig1, astig2=astig2,
                                              coma1=coma1, coma2=coma2, spher=spher,
                                              circular_pupil=circular_pupil, obs=obs)[k2 <
                                                                            kmax_internal**2])
    return ptf

def ptf_image(array_shape=(256, 256), dx=1., lam_over_D=2., defocus=0., astig1=0., astig2=0.,
              coma1=0., coma2=0., spher=0., kmax=np.pi, circular_pupil=True, obs=None):
    """@brief Return the PTF [radians] of a circular (default) or square pupil with low-order
    aberrations as an ImageViewD.

    PTF array element ordering follows the DFT standard of kxky(array_shape).

    To ensure properly Nyquist sampled output any user should set lam_over_D >= 2. * dx.

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the array view on the Image.
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?
    @param obs             add a central obstruction due to secondary mirror?
    """
    array = ptf(array_shape=array_shape, dx=dx, lam_over_D=lam_over_D, defocus=defocus,
                astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                circular_pupil=circular_pupil, obs=obs)
    return galsim.ImageViewD(array.astype(np.float64))

