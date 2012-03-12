"""Module containing the optical PSF generation routines.
"""
import numpy as np

def roll2d(image, (iroll, jroll)):
    """Perform a 2D roll (circular shift) on a supplied 2D image, conveniently.
    """
    return np.roll(np.roll(image, jroll, axis=1), iroll, axis=0)

def kxky(shape=(256, 256)):
    """Output the tuple kx, ky corresponding to the DFT of a unit integer-sampled array of input
    shape.
    
    Uses the SBProfile conventions for Fourier space, so k varies in approximate range (-\pi, \pi].
    Uses the most common DFT element ordering conventions (and those of FFTW), so that (0, 0) array
    element corresponds to (kx, ky) = (0, 0).

    See also the docstring for np.fftfreq, which uses the same DFT convention, and is called here,
    but misses a factor of pi.
    
    Adopts Numpy array index ordering so that the trailing axis corresponds to kx, rather than the
    leading axis as would be expected in IDL/Fortran.  See docstring for numpy.meshgrid which also
    uses this convention.
    """
    k_xaxis = np.fft.fftfreq(shape[1]) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(shape[0]) * 2. * np.pi
    return np.meshgrid(k_xaxis, k_yaxis)

def wavefront(shape=(256, 256), defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
              kmax=np.pi, circular_pupil=True):
    """Construct a complex, aberrated wavefront across a circular pupil (default) or square
    array extent.
    
    Outputs a complex image (shape=shape) of a circular pupil wavefront that will produce a PSF
    with bandlimit kmax.  We adopt the conventions of SBProfile so that the Nyquist frequency of
    an image with unit integer pixel spacing is pi.

    The default output is a circular pupil of radius pi/2 in k-space (=kmax/2), i.e. one
    which fills to a diameter of half the array dimension.  The OTF thus has maximum radius pi in
    k-space at this default kmax, and so the output PSF will be fully sampled (just).

    Typically, therefore, kmax should be chosen to be less than pi and will depend on the pixel
    scale desired in the output.
    
    The pupil sample locations are arranged in standard DFT element ordering format, so that
    (kx, ky) = (0, 0) is the [0, 0] array element.
	
    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the definitions given here:
    http://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations
    
    Outputs the wavefront for kx, ky locations corresponding to kxky(shape), in a C-contiguous
    array ordering.
    """
    pi = np.pi # minor but saves Python checking the entire np. namespace every time I need pi
    # Build kx, ky coords
    kx, ky = kxky(shape)
    # Then define unit disc rho and theta pupil coords for Zernike polynomials
    rho = np.sqrt((kx**2 + ky**2) / (.5 * kmax)**2)
    theta = np.arctan2(ky, kx)
    # Cut out circular pupil if desired (default)
    if circular_pupil:
        in_pupil = (rho < 1.)
    else:
        in_pupil = (np.abs(kx) <= .5 * kmax) * (np.abs(ky) <= .5 * kmax)
    # Then make wavefront image
    wf = np.zeros(shape, dtype=complex)
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

def psf(shape=(256, 256), defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
        kmax=np.pi, circular_pupil=True, secondary=None):
    """Generate an image of a circular (default) or square pupil PSF with specified low-order
    wavefront aberrations.

    Image has unit total flux, and is centred on the image[shape[0] / 2, shape[1] / 2] pixel,
    by default.  Function is bandlimited at kmax (default = pi; Nyquist frequency).

    Ouput numpy array is C-contiguous.
    """
    if secondary == None:  # TODO: Build a secondary mirror obstruction function!
	wf = wavefront(shape=shape, defocus=defocus, astig1=astig1, astig2=astig2, coma1=coma1,
		       coma2=coma2, spher=spher, kmax=kmax, circular_pupil=circular_pupil)
    else:
	raise NotImplementedError('Secondary mirror obstruction not yet implemented')
    ftwf = np.fft.fft2(wf)  # I think this (and the below) is quicker than np.abs(ftwf)**2
    # The roll operation below restores the c_contiguous flag, so no need for a direct action
    im = roll2d((ftwf * ftwf.conj()).real, (shape[0] / 2, shape[1] / 2)) 
    return im / im.sum()

def otf(shape=(256, 256), defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
        kmax=np.pi, circular_pupil=True, secondary=None):
    """Generate the complex OTF of a circular (default) or square pupil with specified low-order
    wavefront aberrations.

    OTF has otf[0, 0] = 1+0j by default, and array element ordering follows the DFT standard of
    kxky(shape).  Function is bandlimited at kmax (default = pi; Nyquist frequency).

    Output complex numpy array is C-contiguous, but real and imaginary parts from otf.real or
    otf.imag will not be.
    """
    if secondary == None:  # TODO: Build a secondary mirror obstruction function!
	wf = wavefront(shape=shape, defocus=defocus, astig1=astig1, astig2=astig2, coma1=coma1,
		       coma2=coma2, spher=spher, kmax=kmax, circular_pupil=circular_pupil)
    else:
	raise NotImplementedError('Secondary mirror obstruction not yet implemented')
    ftwf = np.fft.fft2(wf)  # I think this (and the below) is quicker than np.abs(ftwf)**2
    otf = np.fft.ifft2((ftwf * ftwf.conj()).real)
    # Make C contiguous and unit flux before returning
    return np.ascontiguousarray(otf) / otf[0, 0].real

def mtf(shape=(256, 256), defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
        kmax=np.pi, circular_pupil=True, secondary=None):
    """Generate the MTF of a circular (default) or square pupil with specified low-order
    wavefront aberrations.

    MTF has mtf[0, 0] = 1 by default, and array element ordering follows the DFT standard of
    kxky(shape).  Function is bandlimited at kmax (default = pi; Nyquist frequency).

    Output float numpy array is C-contiguous.
    """
    return np.abs(otf(shape=shape, defocus=defocus, astig1=astig1, astig2=astig2, coma1=coma1,
		      coma2=coma2, spher=spher, kmax=kmax, circular_pupil=circular_pupil))

def ptf(shape=(256, 256), defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
        kmax=np.pi, circular_pupil=True, secondary=None):
    """Generate the PTF (in radians) of a circular (default) or square pupil with specified
    low-order wavefront aberrations.

    PTF has ptf[0, 0] = 0 by default, and array element ordering follows the DFT standard of
    kxky(shape).  Function is bandlimited at kmax (default = pi; Nyquist frequency).

    Output float numpy array is C-contiguous.
    """
    kx, ky = kxky(shape)
    k2 = (kx**2 + ky**2)
    ptf = np.zeros(shape)
    # Try to handle where both real and imag tend to zero...
    ptf[k2 < kmax**2] = np.angle(otf(shape=shape, defocus=defocus, astig1=astig1, astig2=astig2,
                                 coma1=coma1, coma2=coma2, spher=spher, kmax=kmax,
                                 circular_pupil=circular_pupil)[k2 < kmax**2])
    return ptf

