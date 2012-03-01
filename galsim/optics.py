import numpy as np

"""
Module containing the optical PSF generation routines
"""

def roll2d(image, nroll):
	"""Perform a 2D roll (circular shift) on a supplied 2D image, concisely.

	Will probably not do what you want it to if the supplied image is not square!
	"""
	return np.roll(np.roll(image, nroll, axis=1), nroll, axis=0)

def wavefront(defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0., kmax=32., npix = 64,
			  circular_pupil=True):
	"""Construct a complex, aberrated wavefront across a circular pupil (default) or full image.

	Outputs a complex (npix, npix) image of a circular pupil of radius kmax (in pixels), in standard
	DFT element ordering format.

	Input abberation coefficients are assumed to be supplied in units of 1/kmax.
	"""
	# Build coords
	kx, ky = np.meshgrid(np.arange(npix, dtype=float) - .5 * float(npix),
					   np.arange(npix, dtype=float) - .5 * float(npix))
    # Move origin to 0, 0 as per DFT convention
	kx = roll2d(x, npix / 2)
	ky = roll2d(y, npix / 2)
	rho = np.sqrt((kx**2 + ky**2) / kmax**2)
	theta = np.arctan2(ky, kx)
	# Cut out circular pupil if desired (default)
    if circular_pupil:
	    in_pupil = (rho < 1.)
	else:
		in_pupil = np.ones((npix, npix), dtype=bool)
	# Then make wavefront image
	wf = np.zeros((npix, npix), dtype=complex)
	wf[in_pupil] = 1.
	# Defocus
	wf[in_pupil] *= np.exp(1j * defocus * (2. * rho[in_pupil]**2 - 1.))
	# Astigmatism (like e1)
	wf[in_pupil] *= np.exp(1j * astig1 * rho[in_pupil]**2 * np.cos(2. * theta[in_pupil]))
	# Astigmatism (like e2)
	wf[in_pupil] *= np.exp(1j * astig2 * rho[in_pupil]**2 * np.sin(2. * theta[in_pupil]))
	# Coma along x1
	wf[in_pupil] *= np.exp(1j * coma1 * (3. * rho[in_pupil]**2 - 2.) * rho[in_pupil]
						   * np.cos(theta[in_pupil]))
	# Coma along x2
	wf[in_pupil] *= np.exp(1j * coma2 * (3. * rho[in_pupil]**2 - 2.) * rho[in_pupil]
						   * np.sin(theta[in_pupil]))
	# Spherical abberation
	wf[in_pupil] *= np.exp(1j * spher * (6. * rho[in_pupil]**4 - 6. * rho[in_pupil]**2 + 1.))
	return wf
