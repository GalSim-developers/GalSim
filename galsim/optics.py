"""Module containing the optical PSF generation routines.
"""
import numpy as np

def roll2d(image, nroll):
    """Perform a 2D roll (circular shift) on a supplied 2D image, concisely.
    
    Will probably not do what you want it to if the supplied image is not square!
    """
    return np.roll(np.roll(image, nroll, axis=1), nroll, axis=0)

def kxky(shape=(256, 256)):
    """Output the tuple kx, ky corresponding to the DFT of a unit integer-sampled array of input
    shape.
    
    Uses the SBProfile conventions for Fourier space, so k varies in approximate range (-\pi, \pi].
    Uses the most common DFT element ordering conventions (and those of FFTW), so that (0, 0) array
    element corresponds to (kx, ky) = (0, 0).
    
    Also adopts C/Python array ordering so that the trailing axis corresponds to kx, rather than the
    leading axis as would be expected in IDL/Fortran.  See docstring for numpy.meshgrid which also
    uses this convention.
    """
    kxax = (np.arange(shape[1], dtype=float) - .5 * float(shape[1])) * 2. * np.pi / float(shape[1])
    kyax = (np.arange(shape[0], dtype=float) - .5 * float(shape[0])) * 2. * np.pi / float(shape[0])
    kx, ky = np.meshgrid(kxax, kyax)
    kx = np.roll(kx, shape[1] / 2, axis=1)
    ky = np.roll(ky, shape[0] / 2, axis=0)
    return kx, ky

def wavefront(shape=(256, 256), defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
			  kmax=np.pi, circular_pupil=True):
    """Construct a complex, aberrated wavefront across a circular pupil (default) or full array.
    
    Outputs a complex image (shape=shape) of a circular pupil wavefront of radius kmax.  We adopt the
    conventions of SBProfile so that the Nyquist frequency of an image with unit integer pixel
    spacing is \pi.  The default output is a circular pupil of radius \pi in k-space, i.e. one which
    fills to the edge of the array but does not cross it.
    
    The pupil sample locations are arranged in standard DFT element ordering format, so that
    (kx, ky) = (0, 0) is the [0, 0] array element.
	
    Input abberation coefficients are assumed to be supplied in units of wavelength, and correspond
    to the defintions given here:
    http://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations
    
    Outputs the wavefront for kx, ky locations corresponding to kxky(shape).
    """
    pi = np.pi # minor but saves Python checking the entire np. namespace every time I need pi
	# Build kx, ky coords
    kx, ky = kxky(shape)
	# Then define unit disc rho and theta pupil coords for Zernike polynomials
    rho = np.sqrt((kx**2 + ky**2) / kmax**2)
    theta = np.arctan2(ky, kx)
	# Cut out circular pupil if desired (default)
    if circular_pupil:
        in_pupil = (rho <= 1.)
    else:
        in_pupil = np.ones(shape, dtype=bool)
	# Then make wavefront image
	print shape
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

#def cobstruct(pupil, kobs=None):
#	"""Place a circular obstruction (e.g. secondary mirror) in the middle of a supplied image.
#
#	Sets all regions of supplied pupil with |k| < kobs [assumed to be laid out in s



