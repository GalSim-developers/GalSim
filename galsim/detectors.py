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
failure, interpixel capacitance, etc. """

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

    On calling the method, the Image instance `img` is transformed by the user-defined function
    `f` with `beta1` = 1.e-7 and `beta2` = 1.e-10.

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
    input image must have strictly positive pixel values for the resulting image to be well-defined

    The image should be in units of electrons (not ADU), and should include both the signal from
    the astronomical objects as well as the background level.  The addition of nonlinearity should
    occur after including the effect of reciprocity failure.

    Calling
    -------

        >>>  img.addReciprocityFailure(exp_time, alpha)

    @param exp_time  The exposure time in seconds, which goes into the expression for reciprocity
    failure given in the docstring. 
    @param alpha     The alpha parameter in the expression for reciprocity failure, in units of
    'per decade'. 
    
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

def applyIPC(self, IPC_kernel, edge_treatment='extend', fill_value=None, kernel_nonnegativity=True,
    kernel_normalization=True):
    """
    Applies the effect of interpixel capacitance to the Image instance.

    In NIR detectors, the quantity that is sensed in not the charge as in CCDs, but a voltage that
    relates to the charge present within each pixel. The voltage read at a given pixel location is
    influenced by the charges present in the neighboring pixel locations due to capacitive
    coupling of sense nodes.

    This interpixel capacitance is approximated as a linear effect that can be described by a 3x3
    kernel that is convolved with the image. The kernel could be intrinsically anisotropic. A
    sensible kernel must have non-negative entries and must be normalized such that the sum of the
    elements is 1, in order to conserve the total charge.

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

    Calling
    -------

        >>> img.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='extend',
            kernel_nonnegativity=True, kernel_normalization=True)

    @param IPC_kernel              A 3x3 NumPy array that is convolved with the Image instance
    @param edge_treatment          Specifies the method of handling edges and should be one of
                                   'crop', 'extend' or 'wrap'. See above for details.
                                   [default: 'extend']
    @param fill_value              Specifies the value (including nan) to fill the edges with when
                                   edge_treatment is 'crop'. If unspecified or set to 'None', the
                                   original pixel values are retained at the edges. If
                                   edge_treatment is not 'crop', then this is ignored.
    @param kernel_nonnegativity    Specify whether the kernel should have only non-negative entries
                                   . [default: True]
    @param kernel_normalization    Specify whether to check and enforce correct normalization for
                                   the kernel.  [default: True]

    @returns None
    """

    # IPC kernel has to be a 3x3 numpy array
    if not isinstance(IPC_kernel,numpy.ndarray):
        raise ValueError("IPC_kernel must be a NumPy array.")
    if not IPC_kernel.shape==(3,3):
        raise ValueError("IPC kernel must be a NumPy array of size 3x3.")

    # Check for non-negativity of the kernel
    if kernel_nonnegativity is True:
        if (IPC_kernel<0).any() is True:
            raise ValueError("IPC kernel must not contain negative entries")

    # Check and enforce correct normalization for the kernel
    if kernel_normalization is True:
        if IPC_kernel.sum() != 1.0:
            print IPC_kernel.sum()
            import warnings
            warnings.warn("The entries in the kernel did not sum to 1. Scaling the kernel to "
                "ensure correct normalization.")
            IPC_kernel = IPC_kernel/IPC_kernel.sum() 

    # edge_treatment can be 'extend', 'wrap' or 'crop'
    if edge_treatment is 'crop':
        # Simply re-label the array of the Image instance
        pad_array = self.array
    elif edge_treatment is 'extend':
        # Copy the array of the Image instance and pad with zeros
        pad_array = numpy.zeros((self.array.shape[0]+2,self.array.shape[1]+2))
        pad_array[1:-1,1:-1] = self.array
    elif edge_treatment is 'wrap':
        # Copy the array of the Image instance and pad with zeros initially
        pad_array = numpy.zeros((self.array.shape[0]+2,self.array.shape[1]+2))
        pad_array[1:-1,1:-1] = self.array
        # and wrap around the edges
        pad_array[0,:] = pad_array[-2,:]
        pad_array[-1,:] = pad_array[1,:]
        pad_array[:,0] = pad_array[:,-2]
        pad_array[:,-1] = pad_array[:,1]
    else:
        raise ValueError("edge_treatment has to be one of 'extend', 'wrap' or 'crop'. ")

    #Generating different segments of the padded array
    center = pad_array[1:-1,1:-1]
    top = pad_array[:-2,1:-1]
    bottom = pad_array[2:,1:-1]
    left = pad_array[1:-1,:-2]
    right = pad_array[1:-1,2:]
    topleft = pad_array[:-2,:-2]
    bottomright = pad_array[2:,2:]
    topright = pad_array[:-2,2:]
    bottomleft = pad_array[2:,:-2]

    #Generating the output array, with 2 rows and 2 columns lesser than the padded array
    out_array = IPC_kernel[0,0]*topleft + IPC_kernel[0,1]*top + IPC_kernel[0,2]*topright +
        IPC_kernel[1,0]*left + IPC_kernel[1,1]*center + IPC_kernel[1,2]*right + IPC_kernel[2,0]*
        bottomleft + IPC_kernel[2,1]*bottom + IPC_kernel[2,2]*bottomright

    if edge_treatment is 'crop':
        self.array[1:-1,1:-1] = out_array
        #Explicit edge effects handling with filling the edges with the value given in fill_value
        if fill_value is not None:
            self.array[0,:] = fill_value
            self.array[-1,:] = fill_value
            self.array[:,0] = fill_value
            self.array[:,-1] = fill_value
    else:
        self.array[:,:] = out_array

galsim.Image.applyNonlinearity = applyNonlinearity
galsim.Image.addReciprocityFailure = addReciprocityFailure
galsim.Image.applyIPC = applyIPC
