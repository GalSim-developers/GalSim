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

def applyNonlinearity(self, NLfunc, args=None):
    """
    Applies the given non-linearity function (`NLfunc`) to the image, and returns a new image of the
    same datatype.

    The image should be in units of electrons (not ADU), and should include both the signal from the
    astronomical objects as well as the background level.  Other detectors effects such as dark
    current and persistence (not currently included in GalSim) would also occur before the inclusion
    of nonlinearity.

    The argument `NLfunc` is a callable function (for example a lambda function, a
    galsim.LookupTable, or a user-defined function), possibly with arguments that need to be given
    as input using the `args` keyword.  `NLfunc` should be able to take a 2d NumPy array as input,
    and return a NumPy array of the same shape.  It should be defined such that it outputs the final
    image with nonlinearity included (i.e., in the limit that there is no nonlinearity, the function
    should return the original image, NOT zero).

    @param NLfunc    The function that maps the input image pixel values to the output image pixel
                     values.
    @param args      Any necessary arguments required by `NLfunc`.

    @returns a new Image with the nonlinearity effects included.
    """

    # Extract out the array from Image since not all functions can act directly on Images
    if args != None:
        img_nl = NLfunc(self.array, args) 
    else:
        img_nl = NLfunc(self.array)

    if not isinstance(img_nl, numpy.ndarray) or self.array.shape != img_nl.shape:
        raise ValueError("Image shapes are inconsistent after applying nonlinearity function!")

    img_nl = galsim.Image(img_nl, xmin=self.xmin, ymin=self.ymin)
    img_nl.scale = self.scale
    return img_nl

def addReciprocityFailure(self, exp_time=200, alpha=0.0065):
    """
    For the given image, returns a new image that includes the effects of reciprocity failure.

    The reciprocity failure results in mapping the original image to a new one that is equal to the
    original `im` multiplied by `(1+alpha*log10(im/exp_time))`, where the parameter `alpha` and the
    exposure time are given as keyword arguments.

    The image should be in units of electrons (not ADU), and should include both the signal from
    the astronomical objects as well as the background level.  The addition of nonlinearity should
    occur after reciprocity failure.

    Calling
    -------

        >>> new_image = img.addRecipFail(exp_time, alpha)

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

    # Extracting the array out since log won't operate on Image
    arr_in = self.array
    arr_out = arr_in*(1.0 + alpha*numpy.log10(1.0*arr_in/float(exp_time)))

    im_out = galsim.Image(arr_out, xmin=self.xmin, ymin=self.ymin)
    im_out.scale = self.scale
    return im_out


galsim.Image.applyNonlinearity = applyNonlinearity
galsim.Image.addReciprocityFailure = addReciprocityFailure
