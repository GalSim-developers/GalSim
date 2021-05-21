# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

import numpy as np
import sys

from .image import Image
from .errors import GalSimRangeError, GalSimValueError, GalSimIncompatibleValuesError, galsim_warn

def applyNonlinearity(self, NLfunc, *args):
    """
    Applies the given non-linearity function (``NLfunc``) on the `Image` instance directly.

    This routine can transform the image in a non-linear manner specified by the user. However,
    the typical kind of non-linearity one sees in astronomical images is voltage non-linearity,
    also sometimes known as 'classical non-linearity', refers to the non-linearity in
    charge-to-voltage conversion process. This arises as charge gets integrated at the junction
    capacitance of the pixel node. Voltage non-linearity decreases signals at higher signal
    levels, causing the attenuation of brighter pixels. The image should include both the
    signal from the astronomical objects as well as the background level. Other detectors effects
    such as dark current and persistence (not currently included in GalSim) would also occur
    before the inclusion of nonlinearity.

    The argument ``NLfunc`` is a callable function (for example a lambda function, a
    `galsim.LookupTable`, or a user-defined function), possibly with arguments that need to be given
    as subsequent arguments to the ``applyNonlinearity`` function (after the ``NLfunc`` argument).
    ``NLfunc`` should be able to take a 2d NumPy array as input, and return a NumPy array of the
    same shape.  It should be defined such that it outputs the final image with nonlinearity
    included (i.e., in the limit that there is no nonlinearity, the function should return the
    original image, NOT zero). The image should be in units of electrons when this routine is being
    used to generate classical non-linearity. When used for other purposes, the units can be in
    electrons or in ADU, as found appropriate by the user.

    Examples::

        >>> f = lambda x: x + (1.e-7)*(x**2)
        >>> img.applyNonlinearity(f)

        >>> f = lambda x, beta1, beta2: x - beta1*x*x + beta2*x*x*x
        >>> img.applyNonlinearity(f, 1.e-7, 1.e-10)

    On calling the method, the `Image` instance ``img`` is transformed by the user-defined function
    ``f`` with ``beta1`` = 1.e-7 and ``beta2`` = 1.e-10.

    Parameters:
        NLfunc:  The function that maps the input image pixel values to the output image pixel
                 values.
        *args:   Any subsequent arguments are passed along to the NLfunc function.

    """

    # Extract out the array from Image since not all functions can act directly on Images
    result = NLfunc(self.array,*args)
    if not isinstance(result, np.ndarray):
        raise GalSimValueError("NLfunc does not return a NumPy array.", NLfunc)
    if self.array.shape != result.shape:
        raise GalSimValueError("NLfunc does not return a NumPy array of the same shape as input.",
                               NLfunc)
    self.array[:,:] = result


def addReciprocityFailure(self, exp_time, alpha, base_flux):
    r"""
    Accounts for the reciprocity failure and includes it in the original `Image` directly.

    Reciprocity, in the context of photography, is the inverse relationship between the incident
    flux (I) of a source object and the exposure time (t) required to produce a given response (p)
    in the detector, i.e., p = I*t. At very low (also at high) levels of incident flux, deviation
    from this relation is observed, leading to reduced sensitivity at low flux levels. The pixel
    response to a high flux is larger than its response to a low flux. This flux-dependent non-
    linearity is known as 'Reciprocity Failure' and is known to happen in photographic films since
    1893. Interested users can refer to http://en.wikipedia.org/wiki/Reciprocity_(photography)

    CCDs are not known to suffer from this effect. HgCdTe detectors that are used for near infrared
    astrometry, although to an extent much lesser than the photographic films, are found to
    exhibit reciprocity failure at low flux levels. The exact mechanism of this effect is unknown
    and hence we lack a good theoretical model. Many models that fit the empirical data exist and
    a common relation is

    .. math::
            \frac{p_R}{p} = \left(1 + \alpha \log_{10}\left(\frac{p}{t}\right)
                               - \alpha \log_{10}\left(\frac{p^\prime}{t^\prime}\right)\right)

    where :math:`t` is the exposure time (in units of seconds), :math:`p` is the pixel response
    (in units of electrons) and :math:`p_R` is the response if the reciprocity relation fails to
    hold. :math:`p^\prime/t^\prime` is the count rate (in electrons/second) corresponding to the
    photon flux (base flux) at which the detector is calibrated to have its nominal gain.
    \alpha is the parameter in the model, measured in units of per decade and varies with detectors
    and the operating temperature. The functional form for the reciprocity failure is motivated
    empirically from the tests carried out on H2RG detectors.

    See for reference Fig. 1 and Fig. 2 of http://arxiv.org/abs/1106.1090. Since :math:`p_R/p`
    remains close to unity over a wide range of flux, we convert this relation to a power law by
    approximating :math:`p_R/p \approx 1 + \log(p_R/p)`. This gives a relation that is better
    behaved than the logarithmic relation at low flux levels.

    .. math::
            \frac{p_R}{p} = \left(\frac{p/t}{p^\prime/t^\prime}\right)^\frac{\alpha}{\log(10)}.

    Because of how this function is defined, the input image must have non-negative pixel
    values for the resulting image to be well-defined. Negative pixel values result in 'nan's.
    The image should be in units of electrons, or if it is in ADU, then the value passed to
    exp_time should be the exposure time divided by the nominal gain. The image should include
    both the signal from the astronomical objects as well as the background level.  The addition of
    nonlinearity should occur after including the effect of reciprocity failure.

    Parameters:
        exp_time:   The exposure time (t) in seconds, which goes into the expression for
                    reciprocity failure given in the docstring.
        alpha:      The alpha parameter in the expression for reciprocity failure, in
                    units of 'per decade'.
        base_flux:  The flux (:math:`p^\prime/t^\prime`) at which the gain is calibrated to have
                    its nominal value.
    """

    if alpha < 0.:
        raise GalSimRangeError("Invalid value of alpha, must be >= 0",
                               alpha, 0, None)
    if exp_time < 0.:
        raise GalSimRangeError("Invalid value of exp_time, must be >= 0",
                               exp_time, 0, None)
    if base_flux < 0.:
        raise GalSimRangeError("Invalid value of base_flux, must be >= 0",
                               base_flux, 0, None)

    if np.any(self.array<0):
        galsim_warn("One or more pixel values are negative and will be set as 'nan'.")

    p0 = exp_time*base_flux
    a = alpha/np.log(10)
    self.applyNonlinearity(lambda x,x0,a: (x**(a+1))/(x0**a), p0, a)

def applyIPC(self, IPC_kernel, edge_treatment='extend', fill_value=None, kernel_nonnegativity=True,
    kernel_normalization=True):
    """
    Applies the effect of interpixel capacitance to the `Image` instance.

    In NIR detectors, the quantity that is sensed is not the charge as in CCDs, but a voltage that
    relates to the charge present within each pixel. The voltage read at a given pixel location is
    influenced by the charges present in the neighboring pixel locations due to capacitive
    coupling of sense nodes.

    This interpixel capacitance is approximated as a linear effect that can be described by a 3x3
    kernel that is convolved with the image. The kernel must be an `Image` instance and could be
    intrinsically anisotropic. A sensible kernel must have non-negative entries and must be
    normalized such that the sum of the elements is 1, in order to conserve the total charge.
    The (1,1) element of the kernel is the contribution to the voltage read at a pixel from the
    electrons in the pixel to its bottom-left, the (1,2) element of the kernel is the contribution
    from the charges to its left and so on.

    The argument 'edge_treatment' specifies how the edges of the image should be treated, which
    could be in one of the three ways:

    1. 'extend': The kernel is convolved with the zero-padded image, leading to a larger
       intermediate image. The central portion of this image is returned.  [default]
    2. 'crop': The kernel is convolved with the image, with the kernel inside the image completely.
       Pixels at the edges, where the center of the kernel could not be placed, are set to the
       value specified by 'fill_value'. If 'fill_value' is not specified or set to 'None', then
       the pixel values in the original image are retained. The user can make the edges invalid
       by setting fill_value to numpy.nan.
    3. 'wrap': The kernel is convolved with the image, assuming periodic boundary conditions.

    The size of the image array remains unchanged in all three cases.

    Parameters:
        IPC_kernel:            A 3x3 `Image` instance that is convolved with the `Image` instance
        edge_treatment:        Specifies the method of handling edges and should be one of
                               'crop', 'extend' or 'wrap'. See above for details.
                               [default: 'extend']
        fill_value:            Specifies the value (including nan) to fill the edges with when
                               edge_treatment is 'crop'. If unspecified or set to 'None', the
                               original pixel values are retained at the edges. If
                               edge_treatment is not 'crop', then this is ignored.
        kernel_nonnegativity:  Specify whether the kernel should have only non-negative
                               entries.  [default: True]
        kernel_normalization:  Specify whether to check and enforce correct normalization for
                               the kernel.  [default: True]
    """

    # IPC kernel has to be a 3x3 Image
    ipc_kernel = IPC_kernel.array
    if not ipc_kernel.shape==(3,3):
        raise GalSimValueError("IPC kernel must be an Image instance of size 3x3.", IPC_kernel)

    # Check for non-negativity of the kernel
    if kernel_nonnegativity and (ipc_kernel<0).any():
        raise GalSimValueError("IPC kernel must not contain negative entries", IPC_kernel)

    # Check and enforce correct normalization for the kernel
    if kernel_normalization and abs(ipc_kernel.sum()-1) > 10.*np.finfo(ipc_kernel.dtype.type).eps:
        galsim_warn("The entries in the IPC kernel did not sum to 1. Scaling the kernel to "
                    "ensure correct normalization.")
        IPC_kernel = IPC_kernel/ipc_kernel.sum()

    # edge_treatment can be 'extend', 'wrap' or 'crop'
    if edge_treatment=='crop':
        # Simply re-label the array of the Image instance
        pad_array = self.array
    elif edge_treatment=='extend':
        # Copy the array of the Image instance and pad with zeros
        pad_array = np.zeros((self.array.shape[0]+2,self.array.shape[1]+2))
        pad_array[1:-1,1:-1] = self.array
    elif edge_treatment=='wrap':
        # Copy the array of the Image instance and pad with zeros initially
        pad_array = np.zeros((self.array.shape[0]+2,self.array.shape[1]+2))
        pad_array[1:-1,1:-1] = self.array
        # and wrap around the edges
        pad_array[0,:] = pad_array[-2,:]
        pad_array[-1,:] = pad_array[1,:]
        pad_array[:,0] = pad_array[:,-2]
        pad_array[:,-1] = pad_array[:,1]
    else:
        raise GalSimValueError("Invalid edge_treatment.", edge_treatment,
                               ('extend', 'wrap', 'crop'))

    # Generating different segments of the padded array
    center = pad_array[1:-1,1:-1]
    top = pad_array[2:,1:-1]
    bottom = pad_array[:-2,1:-1]
    left = pad_array[1:-1,:-2]
    right = pad_array[1:-1,2:]
    topleft = pad_array[2:,:-2]
    bottomright = pad_array[:-2,2:]
    topright = pad_array[2:,2:]
    bottomleft = pad_array[:-2,:-2]

    # Ensure that the origin is (1,1)
    kernel = IPC_kernel.view()
    kernel.setOrigin(1,1)

    # Generating the output array, with 2 rows and 2 columns lesser than the padded array
    # Image values have been used to make the code look more intuitive
    out_array = kernel(1,3)*topleft + kernel(2,3)*top + kernel(3,3)*topright + \
        kernel(1,2)*left + kernel(2,2)*center + kernel(3,2)*right + \
        kernel(1,1)*bottomleft + kernel(2,1)*bottom + kernel(3,1)*bottomright

    if edge_treatment=='crop':
        self.array[1:-1,1:-1] = out_array
        #Explicit edge effects handling with filling the edges with the value given in fill_value
        if fill_value is not None:
            self.array[0,:] = fill_value
            self.array[-1,:] = fill_value
            self.array[:,0] = fill_value
            self.array[:,-1] = fill_value
    else:
        self.array[:,:] = out_array

def applyPersistence(self,imgs,coeffs):
    """
    Applies the effects of persistence to the `Image` instance.

    Persistence refers to the retention of a small fraction of the signal after resetting the
    imager pixel elements. The persistence signal of a previous exposure is left in the pixel even
    after several detector resets. This effect is most likely due to charge traps in the material.
    Laboratory tests on the Roman Space Telescope CMOS detectors show that if exposures and
    readouts are taken in a fixed cadence, the persistence signal can be given as a linear
    combination of prior pixel values that can be added to the current image.

    This routine takes in a list of `Image` instances and adds them to `Image` weighted by the
    values passed on to 'coeffs'. The pixel values of the `Image` instances in the list must
    correspond to the electron counts before the readout. This routine does NOT keep track of
    realistic dither patterns. During the image simulation process, the user has to queue a list of
    previous `Image` instances (imgs) outside the routine by inserting the latest image in the
    beginning of the list and deleting the oldest image. The values in 'coeffs' tell how much of
    each `Image` is to be added. This usually remains constant in the image generation process.

    Parameters:
        imgs:       A list of previous `Image` instances that still persist.
        coeffs:     A list of floats that specifies the retention factors for the corresponding
                    `Image` instances listed in 'imgs'.
    """
    if not len(imgs)==len(coeffs):
        raise GalSimIncompatibleValuesError("The length of 'imgs' and 'coeffs' must be the same",
                                            imgs=imgs, coeffs=coeffs)
    for img,coeff in zip(imgs,coeffs):
        self += coeff*img

def quantize(self):
    """
    Rounds the pixel values in an image to integer values, while preserving the type of the data.

    At certain stages in the astronomical image generation process, detectors effectively round to
    the nearest integer.  The exact stage at which this happens depends on the type of device (CCD
    vs. NIR detector).  For example, for H2RG detectors, quantization happens in two stages: first,
    when detecting a certain number of photons, corresponding to the sum of background and signal
    multiplied by the QE and including reciprocity failure.  After this, a number of other processes
    occur (e.g., nonlinearity, IPC, read noise) that could result in non-integer pixel values, only
    rounding to an integer at the stage of analog-to-digital conversion.

    Because we cannot guarantee that quantization will always be the last step in the process, the
    quantize() routine does not actually modify the type of the image to 'int'.  However, users can
    easily do so by doing::

        >>> image.quantize()
        >>> int_image = galsim.Image(image, dtype=int)

    """
    self.applyNonlinearity(np.round)


Image.applyNonlinearity = applyNonlinearity
Image.addReciprocityFailure = addReciprocityFailure
Image.applyIPC = applyIPC
Image.applyPersistence = applyPersistence
Image.quantize = quantize
