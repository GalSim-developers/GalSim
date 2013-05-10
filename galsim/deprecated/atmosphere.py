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
"""@file atmosphere.py
Module containing simple atmospheric PSF generation routines.

These are just functions; they are used to generate galsim.AtmosphericPSF() class instances (see 
base.py).   

Mostly they are solely of use to developers for generating arrays that may be useful in defining 
GSObjects with a Kolmogorov atmospheric PSF profile.  They will not therefore be used in a typical
image simulation workflow.  In future it is planned to implemenent, in this module, a stochastic
atmospheric model with multiple turbulent layers.

Glossary of key terms used in function names:

PSF = point spread function

MTF = modulation transfer function = |FT{PSF}|
"""


import numpy as np
import galsim
import galsim.utilities
from galsim import GSObject


class AtmosphericPSF(GSObject):
    """Base class for long exposure Kolmogorov PSF.  Currently deprecated: use Kolmogorov.

    Initialization
    --------------
    Example:    

        >>> atmospheric_psf = galsim.AtmosphericPSF(lam_over_r0, interpolant=None, oversampling=1.5)
    
    Initializes atmospheric_psf as a galsim.AtmosphericPSF() instance.  This class is currently
    deprecated in favour of the newer Kolmogorov class which does not require grid FFTs.  However,
    it is retained as a placeholder for a future AtmosphericPSF which will model the turbulent
    atmosphere stochastically.

    @param lam_over_r0     lambda / r0 in the physical units adopted for apparent sizes (user 
                           responsible for consistency), where r0 is the Fried parameter.  The FWHM
                           of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 
                           108). Typical  values for the Fried parameter are on the order of 10cm 
                           for most observatories and up to 20cm for excellent sites.  The values 
                           are usually quoted at lambda = 500nm and r0 depends on wavelength as
                           [r0 ~ lambda^(-6/5)].
    @param fwhm            FWHM of the Kolmogorov PSF.
                           Either `fwhm` or `lam_over_r0` (and only one) must be specified.
    @param interpolant     Either an Interpolant2d (or Interpolant) instance or a string indicating
                           which interpolant should be used.  Options are 'nearest', 'sinc', 
                           'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the 
                           integer order to use. [default `interpolant = galsim.Quintic()`]
    @param oversampling    Optional oversampling factor for the SBInterpolatedImage table 
                           [default `oversampling = 1.5`], setting `oversampling < 1` will produce 
                           aliasing in the PSF (not good).
    @param flux            Total flux of the profile [default `flux=1.`]
    @param gsparams        You may also specify a gsparams argument.  See the docstring for
                           galsim.GSParams using help(galsim.GSParams) for more information about
                           this option.

    Methods
    -------
    The AtmosphericPSF is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.

    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "oversampling" : float , "interpolant" : str , "flux" : float }
    _single_params = [ { "lam_over_r0" : float , "fwhm" : float } ]
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, lam_over_r0=None, fwhm=None, interpolant=None, oversampling=1.5, flux=1.,
                 gsparams=None):

        # The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
        if lam_over_r0 is None :
            if fwhm is not None :
                lam_over_r0 = fwhm / 0.976
            else:
                raise TypeError("Either lam_over_r0 or fwhm must be specified for AtmosphericPSF")
        else :
            if fwhm is None:
                fwhm = 0.976 * lam_over_r0
            else:
                raise TypeError(
                        "Only one of lam_over_r0 and fwhm may be specified for AtmosphericPSF")
        # Set the lookup table sample rate via FWHM / 2 / oversampling (BARNEY: is this enough??)
        dx_lookup = .5 * fwhm / oversampling

        # Fold at 10 times the FWHM
        stepk_kolmogorov = np.pi / (10. * fwhm)

        # Odd array to center the interpolant on the centroid. Might want to pad this later to
        # make a nice size array for FFT, but for typical seeing, arrays will be very small.
        npix = 1 + 2 * (np.ceil(np.pi / stepk_kolmogorov)).astype(int)
        atmoimage = kolmogorov_psf_image(
            array_shape=(npix, npix), dx=dx_lookup, lam_over_r0=lam_over_r0, flux=flux)
        
        # Run checks on the interpolant and build default if None
        if interpolant is None:
            quintic = galsim.Quintic(tol=1e-4)
            self.interpolant = galsim.InterpolantXY(quintic)
        else:
            self.interpolant = galsim.utilities.convert_interpolant_to_2d(interpolant)

        # Then initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBInterpolatedImage(atmoimage, xInterp=self.interpolant, dx=dx_lookup,
                                             gsparams=gsparams))

        # The above procedure ends up with a larger image than we really need, which
        # means that the default stepK value will be smaller than we need.  
        # Thus, we call the function calculateStepK() to refine the value.
        self.SBProfile.calculateStepK()
        self.SBProfile.calculateMaxK()

    def getHalfLightRadius(self):
        # TODO: This seems like it would not be impossible to calculate
        raise NotImplementedError("Half light radius calculation not yet implemented for "+
                                  "Atmospheric PSF objects (could be though).")



def kolmogorov_mtf(array_shape=(256, 256), dx=1., lam_over_r0=1.):
    """Return the atmospheric modulation transfer function for long exposures with Kolmogorov 
    turbulence as a NumPy array. 

    @param array_shape     the NumPy array shape desired for the array view of the ImageViewD.
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
    """Return the atmospheric modulation transfer function for long exposures with Kolmogorov 
    turbulence as an ImageViewD.

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted.
    for lam_over_r0.

    @param array_shape     the NumPy array shape desired for the array view of the ImageViewD.
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
    """Return NumPy array containing long exposure Kolmogorov PSF.
    
    @param array_shape     the NumPy array shape desired for the array view of the ImageViewD.
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
    """Return long exposure Kolmogorov PSF as an ImageViewD.

    The ImageView output can be used to directly instantiate an SBInterpolatedImage, and its 
    .getScale() method will reflect the spacing of the output grid in the system of units adopted.
    for lam_over_diam.

    @param array_shape     the NumPy array shape desired for the array view of the ImageViewD.
    @param dx              grid spacing of PSF in real space units
    @param lam_over_r0     lambda / r0 in the physical units adopted for dx (user responsible for 
                           consistency). r0 is the Fried parameter. Typical values for the 
                           Fried parameter are on the order of 10 cm for most observatories.
    @param flux            total flux of the profile [default flux=1.]
    """
    array = kolmogorov_psf(array_shape=array_shape, dx=dx, lam_over_r0=lam_over_r0, flux=flux)
    return galsim.ImageViewD(array.astype(np.float64))
