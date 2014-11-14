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

def applyNonlinearity(self, NLfunc, *args):
    """
    Applies the given non-linearity function (`NLfunc`) on the Image instance directly.

    The image should be in units of electrons (not ADU), and should include both the signal from
    the astronomical objects as well as the background level.  Other detectors effects such as dark
    current and persistence (not currently included in GalSim) would also occur before the
    inclusion of nonlinearity.

    The argument `NLfunc` is a callable function (for example a lambda function, a
    galsim.LookupTable, or a user-defined function), possibly with arguments that need to be given
    as subsequent arguments given to the `applyNonlinearity` function (after the `NLfunc` argument.
    `NLfunc` should be able to take a 2d NumPy array as input, and return a NumPy array of the
    same shape.  It should be defined such that it outputs the final image with nonlinearity
    included (i.e., in the limit that there is no nonlinearity, the function should return the
    original image, NOT zero).

    Calling with no parameter
    -------

        >>> f = lambda x: x + (1e-7)*(x**2)
        >>> img.applyNonlinearity(f)

    Calling with 1 or more parameters
    -------

        >>> f = lambda x, beta1, beta2: x - beta1*x*x + beta2*x*x*x
        >>> img.applyNonlinearity(f, 1.e-7, 1.e-10)

    On calling the method, the Image instance `img` is transformed by the user-defined function `f` with `beta1` = 1.e-7 and `beta2` = 1.e-10.

    @param NLfunc    The function that maps the input image pixel values to the output image pixel
                     values. 
    @param *args     Any subsequent arguments are passed along to the NLfunc function.

    """

    # Extract out the array from Image since not all functions can act directly on Images
    self.array[:,:] = (NLfunc(self.array, *args))


def addReciprocityFailure(self, exp_time, alpha):
    """
    Accounts for the reciprocity failure and corrects the original Image for it directly.

    The reciprocity failure results in mapping the original image to a new one that is equal to the
    original `im` multiplied by `(1+alpha*log10(im/exp_time))`, where the parameter `alpha` and the
    exposure time are given as keyword arguments.  Because of how this function is defined, the
    input image must have strictly positive pixel values for the resulting image to be well defined.

    The image should be in units of electrons (not ADU), and should include both the signal from
    the astronomical objects as well as the background level.  The addition of nonlinearity should
    occur after including the effect of reciprocity failure.

    Calling
    -------

        >>>  img.addReciprocityFailure(exp_time, alpha)

    @param exp_time  The exposure time in seconds, which goes into the expression for reciprocity failure given in the docstring. 
    @param alpha     The alpha parameter in the expression for reciprocity failure, in units of 'per decade'. 
    
    @returns None
    """

    if not isinstance(alpha, float) or alpha < 0.:
        raise ValueError("Invalid value of alpha, must be float >= 0")
    if not (isinstance(exp_time, float) or isinstance(exp_time, int)) or exp_time < 0.:
        raise ValueError("Invalid value of exp_time, must be float >= 0")

    if numpy.any(self.array < sys.float_info.min*float(exp_time)):
        import warnings
        warnings.warn("At least one element of image/exp_time is too close to 0 or negative.")
        warnings.warn("Floating point errors might occur.")

    self.array[:,:] = self.array*(1.0 + alpha*numpy.log10(self.array/float(exp_time)))

def applyIPC(self, IPC_kernel, edge_effects=None):
    """
    docstring
    """

    # IPC kernel has to be a 3x3 numpy array
    if not isinstance(IPC_kernel,numpy.ndarray):
        raise ValueError("IPC_kernel must be a NumPy array.")
    if not len(IPC_kernel) is 3 or IPC_kernel.size is 9:
        raise ValueError("IPC kernel must be a NumPy array of size 3x3.")

    # Warning the user about default edge effect handling
    if edge_effects is None:
        import warnings
        warnings.warn("No value for edge_effects specified. Choosing the default option 'crop'. The size of the Image instance will remain unchanged. ")
        edge_effects = 'crop'

    # edge_effects can be 'extend', 'warp' or 'crop'
    else if not edge_effects is 'extend' or edge_effects is 'warp' or edge_effects is 'crop':
        raise ValueError("edge_effects has to be one of 'extend', 'warp' or 'crop'. ")

    center = self.array[1:-1,1:-1]
    top = self.array[:-2,1:-1]
    bottom = self.array[2:,1:-1]
    left = self.array[1:-1,:-2]
    right = self.array[1:-1,2:]
    topleft = self.array[:-2,:-2]
    bottomright = self.array[2:,2:]
    topright = self.array[:-2,2:]
    bottomleft = self.array[2:,:-2]

    self.array[1:-1,1:-1] = IPC_kernel[0,0]*topleft + IPC_kernel[0,1]*top + IPC_kernel[0,2]*topright + IPC_kernel[1,0]*left + IPC_kernel[1,1]*center + IPC_kernel[1,2]*right + IPC_kernel[2,0]*bottomleft + IPC_kernel[2,1]*bottom + IPC_kernel[2,2]*bottomright

    #Edges
    # OTHER OPTIONS WILL BE ADDED AFTER TESTING THE EFFICIENCY
    self.array[0,:] = 0.0
    self.array[-1,:] = 0.0
    self.array[:,0] = 0.0
    self.array[:,-1] = 0.0

galsim.Image.applyNonlinearity = applyNonlinearity
galsim.Image.addReciprocityFailure = addReciprocityFailure