
import numpy as np

import galsim
import utilities

"""@file atmosphere.py @brief Module containing simple atmospheric PSF generation routines.

These are just functions; they are used to generate galsim.AtmosphericPSF() class instances (see 
base.py).   

Mostly they are solely of use to developers for generating arrays that may be useful in defining 
GSObjects with a Kolmogorov atmospheric PSF profile.  They will not therefore be used in a typical
image simulation workflow: users will find most of what they need simply using the Kolmogorov()
(preferred) or AtmosphericPSF() class.

Glossary of key terms used in function names:

PSF = point spread function

MTF = modulation transfer function = |FT{PSF}|
"""

def kolmogorov_mtf(array_shape=(256, 256), dx=1., lam_over_r0=1.):
    """@brief Return the atmospheric modulation transfer function for long exposures with 
    Kolmogorov turbulence as a numpy array. 

    @param array_shape     the Numpy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted for dx (user responsible for 
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

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted.
    for lam_over_r0.

    @param array_shape     the Numpy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted for dx (user responsible for 
                           consistency), where r0 is the Fried parameter. The FWHM of the Kolmogorov
                           PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108). Typical 
                           values for the Fried parameter are on the order of 10 cm for most 
                           observatories and up to 20 cm for excellent sites. The values are 
                           usually quoted at lambda = 500 nm and r0 depends weakly on wavelength
                           [r0 ~ lambda^(-6/5)].
    """
    amtf = kolmogorov_mtf(array_shape=array_shape, dx=dx, lam_over_r0=lam_over_r0)
    return galsim.ImageViewD(amtf.astype(np.float64))

def kolmogorov_psf(array_shape=(256,256), dx=1., lam_over_r0=1., flux=1.):
    """@brief Return numpy array containing long exposure Kolmogorov PSF.
    
    @param array_shape     the Numpy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted for dx (user responsible for 
                           consistency), where r0 is the Fried parameter. The FWHM of the Kolmogorov
                           PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108). Typical 
                           values for the Fried parameter are on the order of 10 cm for most 
                           observatories and up to 20 cm for excellent sites. The values are 
                           usually quoted at lambda = 500 nm and r0 depends on wavelength
                           [r0 ~ lambda^(-6/5)].
    @param flux            total flux of the profile [default flux=1.]
    """

    amtf = kolmogorov_mtf(array_shape=array_shape, dx=dx, lam_over_r0=lam_over_r0)
    ftmtf = np.fft.fft2(amtf)
    im = galsim.utilities.roll2d((ftmtf * ftmtf.conj()).real, (array_shape[0] / 2,
                                                               array_shape[1] / 2))
    return im * (flux / (im.sum() * dx**2))
    
def kolmogorov_psf_image(array_shape=(256, 256), dx=1., lam_over_r0=1., flux=1.):
    """@brief Return long exposure Kolmogorov PSF as an ImageViewD.

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted.
    for lam_over_diam.

    @param array_shape     the Numpy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted for dx (user responsible for 
                           consistency). r0 is the Fried parameter. Typical values for the 
                           Fried parameter are on the order of 10 cm for most observatories.
    @param flux            total flux of the profile [default flux=1.]
    """
    array = kolmogorov_psf(array_shape=array_shape, dx=dx, lam_over_r0=lam_over_r0, flux=flux)
    return galsim.ImageViewD(array.astype(np.float64))
