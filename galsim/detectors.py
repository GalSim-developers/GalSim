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
# list of conditions, and the disclaimer given in the accompanying LICENSE
# file.
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

def applyNonlinearity(self, NLfunc, args=None):
    """
    Applies the given non-linearity function (`NLfunc`) to the image, and returns a new image of
    the same datatype.

    The image should be in units of electrons (not ADU), and should include both the signal from
    the astronomical objects as well as the background level.  Other detectors effects such as dark
    current and persistence (not currently included in GalSim) would also occur before the
    inclusion of nonlinearity.

    The argument `NLfunc` is a callable function (for example a lambda function, a
    galsim.LookupTable, or a user-defined function), possibly with arguments that need to be given
    as input using the `args` keyword.  If the `NLfunc` has more than one parameter to be
    specified, then those parameter values need to be given as a list to applyNonlinearity.
    `NLfunc` should be able to take a 2d NumPy array as input, and return a NumPy array of the
    same shape.  It should be defined such that it outputs the final image with nonlinearity
    included (i.e., in the limit that there is no nonlinearity, the function should return the
    original image, NOT zero).

    Calling with no parameter
    -------

        >>> f = lambda x: x + (1e-7)*(x**2)
        >>> imgNL = img.applyNonlinearity(f)

    Calling with 1 parameter
    ------

        >>> f = lambda x,beta: x + beta*(x**2)
        >>> beta = 1e-7
        >>> imgNL = img.applyNonlinearity(f,beta)

    Calling with 1 or more parameters
    -------

        >>> f = lambda x, beta1, beta2: x - beta1*x*x + beta2*x*x*x
        >>> args = [1e-7,1e-10]
        >>> imgNL = img.applyNonlinearity(f, *args)

    On output, the Image instance `imgNL` is the transformation of the Image instance `img` given
    by the user-defined function `f` with user-defined parameters of 1e-7 and 1e-10.

    @param NLfunc    The function that maps the input image pixel values to the output image pixel
                     values. 
    @param args      Any necessary arguments required by `NLfunc`, which must take either a single
                     non-iterable argument or a tuple containing multiple parameters.

    @returns a new Image with the nonlinearity effects included.
    """

    # Extract out the array from Image since not all functions can act directly on Images
    if args == None:                                   #no parameter
        img_nl = NLfunc(self.array) 
    elif type(args) if list or type(args) is tuple:    # 1 or more parameter
        img_nl = NLfunc(self.array,*args)
    else:                                              # 1 parameter
        img_nl = NLfunc(self.array,args)

    if not isinstance(img_nl, numpy.ndarray) or self.array.shape != img_nl.shape:
        raise ValueError("Image shapes are inconsistent after applying nonlinearity function!")

    # Enforce consistency of bounds and scale between input, output Images.
    img_nl = galsim.Image(img_nl, xmin=self.xmin, ymin=self.ymin)
    img_nl.scale = self.scale
    return img_nl

def addReciprocityFailure(self, exp_time=200., alpha=0.0065):
    """
    For the given image, returns a new image that includes the effects of reciprocity failure.

    The reciprocity failure results in mapping the original image to a new one that is equal to the
    original `im` multiplied by `(1+alpha*log10(im/exp_time))`, where the parameter `alpha` and the
    exposure time are given as keyword arguments.  Because of how this function is defined, the
    input image must have strictly positive pixel values.

    The image should be in units of electrons (not ADU), and should include both the signal from
    the astronomical objects as well as the background level.  The addition of nonlinearity should
    occur after including the effect of reciprocity failure.

    Calling
    -------

        >>> new_image = img.addReciprocityFailure(exp_time, alpha)

    @param exp_time  The exposure time in seconds, which goes into the expression for reciprocity
                     failure given in the docstring. [default: 200]
    @param alpha     The alpha parameter in the expression for reciprocity failure.
                     [default: 0.0065]

    @returns a new Image with the effects of reciprocity failure included.
    """
    if not isinstance(alpha, float) or alpha < 0.:
        raise ValueError("Invalid value of alpha, must be float >= 0")
    if not (isinstance(exp_time, float) or isinstance(exp_time, int)) or exp_time < 0.:
        raise ValueError("Invalid value of exp_time, must be float >= 0")

    # Extracting the array out since log won't operate on the Image.
    arr_in = 1.0*self.array/float(exp_time)

    if numpy.any(arr_in < sys.float_info.min):
        import warnings
        warnings.warn("At least one element of image/exp_time is too close to 0 or negative.")
        warnings.warn("Floating point errors might occur.")

    arr_out = self.array*(1.0 + alpha*numpy.log10(arr_in))

    # Enforce consistency of bounds and scale between input, output Images.
    im_out = galsim.Image(arr_out, xmin=self.xmin, ymin=self.ymin)
    im_out.scale = self.scale
    return im_out


galsim.Image.applyNonlinearity = applyNonlinearity
galsim.Image.addReciprocityFailure = addReciprocityFailure
