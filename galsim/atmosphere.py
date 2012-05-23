
import numpy as np

import galsim

"""\file atmosphere.py Simple atmospheric PSF generation routines
"""


class DoubleGaussian(galsim.Add):
    """Double Gaussian, which is the sum of two SBProfile Gaussian profiles
    """
    def __init__(self, flux1=1., flux2=1., sigma1=None, sigma2=None, fwhm1=None, fwhm2=None):
        sblist = []
        # Note: we do not have to check for improper args (0 or 2 radii specified) because this is
        # done in the C++
        sblist.append(galsim.Gaussian(flux1, sigma=sigma1, fwhm=fwhm1))
        sblist.append(galsim.Gaussian(flux2, sigma=sigma2, fwhm=fwhm2))
        galsim.Add.__init__(self, sblist)

def atmospheric_mtf(lam_over_r0, array_shape=(256, 256)):
    """Atmospheric modulation transfer function for long exposures with Kolmogorov turbulence. 
    """
    k_xaxis = np.fft.fftfreq(array_shape[0])
    k_yaxis = np.fft.fftfreq(array_shape[1])
    kx, ky = np.meshgrid(k_xaxis, k_yaxis)
    amtf = np.exp(-3.44 * (lam_over_r0 * np.hypot(kx, ky))**(5. / 3.))
    return amtf

def psf(lam_over_r0, array_shape=(256,256)):
    """Create an array with a long exposure Kolmogorov PSF. 
    
    Parameters
    ----------
    @param lam_over_r0        lambda (wavelength) divided by the Fried parameter r0. Should be
                              provided in units of pixels
    @param array_shape        The numpy array shape desired for the output array
    """

    amtf = atmospheric_mtf(lam_over_r0, array_shape)
    ftmtf = np.fft.fft2(amtf)
    im = galsim.optics.roll2d((ftmtf * ftmtf.conj()).real, (array_shape[0] / 2, array_shape[1] / 2))
    return im / im.sum()
    
def psf_image(lam_over_r0, array_shape=(256, 256), dx=1.):
    array = psf(lam_over_r0, array_shape=array_shape)
    return galsim.ImageViewD(array.astype(np.float64) / dx**2)
