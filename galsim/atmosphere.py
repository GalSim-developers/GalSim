
import numpy as np

import galsim
import utilities

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

def kolmogorov_mtf(array_shape=(256, 256), dx=1., lam_over_r0=1.):
    """@brief Return the atmospheric modulation transfer function for long exposures with 
    Kolmogorov turbulence as a numpy array. 

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted (user responsible for 
                           consistency), where r0 is the Fried parameter. The FWHM of the Kolmogorov
                           PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108). Typical 
                           values for the Fried parameter are on the order of 10 cm for most 
                           observatories and up to 20 cm for excellent sites. The values are 
                           usually quoted at lambda = 500 nm and r0 depends weakly on wavelength
                           [r0 ~ lambda^(-6/5)].
    """
    # This is based on the ALIAS_THRESHOLD 0.005 in src/SBProfile.cpp and galsim/base.py
    kmax_internal = 1.2954 / lam_over_r0 * dx
    kx, ky = galsim.utilities.kxky(array_shape)
    amtf = np.exp(-3.442 * (np.hypot(kx, ky) / kmax_internal / np.pi)**(5. / 3.))
    return amtf

def kolmogorov_mtf_image(array_shape=(256, 256), dx=1., lam_over_r0=1.):
    """@brief Return the atmospheric modulation transfer function for long exposures with 
    Kolmogorov turbulence as an ImageViewD. 

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted (user responsible for 
                           consistency), where r0 is the Fried parameter. The FWHM of the Kolmogorov
                           PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108). Typical 
                           values for the Fried parameter are on the order of 10 cm for most 
                           observatories and up to 20 cm for excellent sites. The values are 
                           usually quoted at lambda = 500 nm and r0 depends weakly on wavelength
                           [r0 ~ lambda^(-6/5)].
    """
    amtf = kolmogorov_mtf(array_shape=array_shape, dx=dx, lam_over_r0=lam_over_r0)
    return galsim.ImageViewD(amtf.astype(np.float64))

def kolmogorov_psf(array_shape=(256,256), dx=1., lam_over_r0=1.):
    """@brief Return numpy array containing long exposure Kolmogorov PSF.
    
    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted (user responsible for 
                           consistency), where r0 is the Fried parameter. The FWHM of the Kolmogorov
                           PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108). Typical 
                           values for the Fried parameter are on the order of 10 cm for most 
                           observatories and up to 20 cm for excellent sites. The values are 
                           usually quoted at lambda = 500 nm and r0 depends on wavelength
                           [r0 ~ lambda^(-6/5)].
    """
    amtf = kolmogorov_mtf(array_shape=array_shape, dx=dx, lam_over_r0=lam_over_r0)
    ftmtf = np.fft.fft2(amtf)
    im = galsim.utilities.roll2d((ftmtf * ftmtf.conj()).real, (array_shape[0] / 2,
                                                               array_shape[1] / 2))
    return im / (im.sum() * dx**2)
    
def kolmogorov_psf_image(array_shape=(256, 256), dx=1., lam_over_r0=1.):
    """@brief Return long exposure Kolmogorov PSF as an ImageViewD.

    Parameters
    ----------
    @param array_shape     the Numpy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted for dx (user responsible for 
                           consistency). r0 is the Fried parameter. Typical values for the 
                           Fried parameter are on the order of 10 cm for most observatories.
    """
    array = kolmogorov_psf(array_shape=array_shape, dx=dx, lam_over_r0=lam_over_r0)
    return galsim.ImageViewD(array.astype(np.float64))
