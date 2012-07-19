import numpy as np
import galsim

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

def g1g2_to_e1e2(g1, g2):
    """@brief Convenience function for going from (g1, g2) -> (e1, e2).
    """
    # Conversion:
    # e = (a^2-b^2) / (a^2+b^2)
    # g = (a-b) / (a+b)
    # b/a = (1-g)/(1+g)
    # e = (1-(b/a)^2) / (1+(b/a)^2)
    gsq = g1*g1 + g2*g2
    if gsq > 0.:
        g = np.sqrt(gsq)
        boa = (1-g) / (1+g)
        e = (1 - boa*boa) / (1 + boa*boa)
        e1 = g1 * (e/g)
        e2 = g2 * (e/g)
        return e1, e2
    elif gsq == 0.:
        return 0., 0.
    else:
        raise ValueError("Input |g|^2 < 0, cannot convert.")

def rand_arr(shape, deviate):
    """@brief Function to make a 2d array of random deviates (of any sort)
    """
    if len(shape) is not 2:
        raise ValueError("Can only make a 2d array from this function!")
    tmp_img = galsim.ImageD(shape[0], shape[1])
    deviate.applyTo(tmp_img.view())
    return tmp_img.array
