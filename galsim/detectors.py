# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
# list of conditions, and the disclaimer given in the accompanying LICENSE file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions, and the disclaimer given in the documentation
# and/or other materials provided with the distribution.
#
"""@file detectors.py

Module with routines to simulate CCD and NIR detector effects like nonlinearity, reciprocity
failure, interpixel capacitance, etc.
"""

import galsim
import numpy
import sys

def applyNonlinearity(self, NLfunc, *args):
    """
    Applies the given non-linearity function (`NLfunc`) on the Image instance directly.

    The image should include both the signal from the astronomical objects as well as the
    background level.  Other detectors effects such as dark current and persistence (not currently
    included in GalSim) would also occur before the inclusion of nonlinearity.

    The argument `NLfunc` is a callable function (for example a lambda function, a
    galsim.LookupTable, or a user-defined function), possibly with arguments that need to be given
    as subsequent arguments to the `applyNonlinearity` function (after the `NLfunc` argument).
    `NLfunc` should be able to take a 2d NumPy array as input, and return a NumPy array of the same
    shape.  It should be defined such that it outputs the final image with nonlinearity included
    (i.e., in the limit that there is no nonlinearity, the function should return the original
    image, NOT zero). The functional form of `NLfunc` must be such that it can operate on the Image
    instance in whatever units (ADU or e-) it is specified in.

    Calling with no parameter:
    -------------------------

        >>> f = lambda x: x + (1.e-7)*(x**2)
        >>> img.applyNonlinearity(f)

    Calling with 1 or more parameters:
    ---------------------------------

        >>> f = lambda x, beta1, beta2: x - beta1*x*x + beta2*x*x*x
        >>> img.applyNonlinearity(f, 1.e-7, 1.e-10)

    On calling the method, the Image instance `img` is transformed by the user-defined function `f`
    with `beta1` = 1.e-7 and `beta2` = 1.e-10.

    @param NLfunc    The function that maps the input image pixel values to the output image pixel
                     values. 
    @param *args     Any subsequent arguments are passed along to the NLfunc function.

    """

    # Extract out the array from Image since not all functions can act directly on Images
    result = NLfunc(self.array,*args)
    if not isinstance(result, numpy.ndarray):
        raise ValueError("NLfunc does not return a NumPy array.")
    if self.array.shape != result.shape:
        raise ValueError("NLfunc does not return a NumPy array of the same shape as input!")
    self.array[:,:] = result


def addReciprocityFailure(self, exp_time, alpha, unity_gain_flux = 1.0):
    """
    Accounts for the reciprocity failure and corrects the original Image for it directly.

    Reciprocity, in the context of photography, is the inverse relationship between the incident
    flux (I) of a source object and the exposure time (T) required to produce a given response (E)
    in the detector, i.e., E = I*t. At very low (also at high) levels of incident flux, deviation
    from this relation is observed, leading to reduced sensitivity at low flux levels. The pixel
    response to a high flux is larger than its response to a low flux. This flux-dependent non-
    linearity is known as 'Reciprocity Failure' and is known to happen in photographic films since
    1893. Interested users can refer to http://en.wikipedia.org/wiki/Reciprocity_(photography)

    CCDs are not known to suffer from this effect. HgCdTe detectors that are used for near infrared
    astrometry, although to an extent much lesser than the photographic films, are found to
    exhibit reciprocity failure at low flux levels. The exact mechanism of this effect is unknown
    and hence we lack a good theoretical model. Many models that fit the empirical data exist and
    a common relation is

            pR/p = (1 + alpha*log10(p/T) - alpha*log10(p'/T'))

    where T is the exposure time (in units of seconds), p is the pixel response (in units of
    electrons) and pR is the response if the reciprocity relation were to hold. p'/T' is count
    rate (in electrons/second) corresponding to the photon flux (base flux) at which the detector
    is calibrated to have its nominal gain. alpha is the parameter in the model, measured in units
    of per decade. The functional form for the reciprocity failure is motivated empirically from
    the tests carried out on H2RG detectors. See for reference Fig. 1 and Fig. 2 of
    http://arxiv.org/abs/1106.1090. Since pR/p remains close to unity over a wide range of flux,
    we convert this relation to a power law by approximating (pR/p)-1 ~ log(pR/p). This gives a
    relation that is better behaved than the logarithmic relation at low flux levels.

            pR/p = ((p/T)/(p'/T'))^(alpha/log(10)).

    Because of how this function is defined, the input image must have strictly positive pixel
    values for the resulting image to be well-defined. Negative pixel values result in 'nan's.
    The image should be in units of electrons, or if it is in ADU, then the value passed to
    exp_time should be the exposure time divided by the nominal gain. The image should include
    both the signal from the astronomical objects as well as the background level.  The addition of
    nonlinearity should occur after including the effect of reciprocity failure.

    Calling
    -------

        >>>  img.addReciprocityFailure(exp_time, alpha, unity_gain_flux)

    @param exp_time         The exposure time in seconds, which goes into the expression for
                            reciprocity failure given in the docstring.
    @param alpha            The alpha parameter in the expression for reciprocity failure, in
                            units of 'per decade'.
    @param unity_gain_flux  The flux at which the gain is calibrated to have its nominal value.
    
    @returns None
    """

    if not isinstance(alpha, float) or alpha < 0.:
        raise ValueError("Invalid value of alpha, must be float >= 0")
    if not (isinstance(exp_time, float) or isinstance(exp_time, int)) or exp_time < 0.:
        raise ValueError("Invalid value of exp_time, must be float or int >= 0")
    if not (isinstance(unity_gain_flux, float) or isinstance(unity_gain_flux,int)) or \
        unity_gain_flux < 0.:
        raise ValueError("Invalid value of unity_gain_flux, must be float or int >= 0")

    # Old log formalism
    # -----------------
    # if numpy.any(self.array < sys.float_info.min*float(exp_time)*float(unity_gain_flux)):
    #     import warnings
    #     warnings.warn("At least one element of image/exp_time is too close to 0 or negative.")
    #     warnings.warn("Floating point errors might occur.")

    # self.array[:,:] = self.array*(1.0 + alpha*numpy.log10(self.array/(float(exp_time))) - alpha*
    # numpy.log10(float(unity_gain_flux)))

    # Old formalism: pR = p*(1+alpha*log10(p/T)-alpha*log10(p'/T'))
    #               (or) (pR/p) = 1 + alpha*log10(p/T)-alpha*log10(p'/T'))
    # New formalism: (pR-p)/p ~ log10(pR/p)*ln(10) = log10((p/T)/(p'/T'))^alpha
    #                (or) (pR-p)/p = ((p/T)/(p'/T'))^(alpha/ln(10))

    if numpy.any(self.array<0):
        import warnings
        warnings.warn("One or more pixel values are negative and will be set as 'nan'.")

    p = self.array
    self.array[:,:] = p*(((p/exp_time)/unity_gain_flux)**(alpha/numpy.log(10)))

galsim.Image.applyNonlinearity = applyNonlinearity
galsim.Image.addReciprocityFailure = addReciprocityFailure
