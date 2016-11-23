# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
#
"""@file fft.py
Functional equivalents of (some of) the np.fft functions, but using FFTW.

These should be drop in replacements for np.fft.* functions.  e.g.

    >>> karray = galsim.fft.fft2(xarray)

is functionally equivalent to

    >>> karray = np.fft.fft2(xarray)

but should be a bit faster.

Note that the GalSim versions often only implement the normal use case without many of the
advanced options available with the numpy functions.  This is mostly laziness on our part --
we only implemented the functions that we needed.  If your usage requires some option available
in the numpy version, feel free to post a feature request on our GitHub page.
"""

import galsim
import numpy as np

def fft2(a, shift_in=False, shift_out=False):
    """Compute the 2-dimensional discrete Fourier Transform.

    For valid inputs, the result is equivalent to numpy.fft.fft2(a), but usually faster.

        >>> ka1 = numpy.fft.fft2(a)
        >>> ka2 = galsim.fft.fft2(a)

    Restrictions on this version vs the numpy version:

        - The input array must be 2-dimensional.
        - It must be square.
        - The size in each direction must be even.
        - If it has a real dtype, it will be coerced to numpy.float64.
        - If it hsa a complex dtype, it will be coerced to numpy.complex128.

    The returned array will be complex with dtype numpy.complex128.

    If shift_in is True, then this is equivalent to applying numpy.fft.fftshift to the input.

        >>> ka1 = numpy.fft.fft2(numpy.fft.fftshift(a))
        >>> ka2 = galsim.fft.fft2(a, shift_in=True)

    If shift_out is True, then this is equivalent to applying numpy.fft.fftshift to the output.

        >>> ka1 = numpy.fft.fftshift(numpy.fft.fft2(a))
        >>> ka2 = galsim.fft.fft2(a, shift_out=True)

    @param a            The input array to be transformed
    @param shift_in     Whether to shift the input array so that the center is moved to (0,0).
    @param shift_out    Whether to shift the output array so that the center is moved to (0,0).

    @returns a complex numpy array
    """
    s = a.shape
    if len(s) != 2:
        raise ValueError("Input array must be 2D.")
    N = s[0]
    if N != s[1]:
        raise ValueError("Input array must be square.")

    No2 = N // 2
    if a.dtype.kind == 'c':
        a = a.astype(np.complex128)
        xim = galsim._galsim.ConstImageViewC(a, -No2, -No2)
    else:
        a = a.astype(np.float64)
        xim = galsim._galsim.ConstImageViewD(a, -No2, -No2)
    return xim.cfft(shift_in=shift_in, shift_out=shift_out).array


def ifft2(a, shift_in=False, shift_out=False):
    """Compute the 2-dimensional inverse discrete Fourier Transform.

    For valid inputs, the result is equivalent to numpy.fft.ifft2(a), but usually faster.

        >>> a1 = numpy.fft.ifft2(ka)
        >>> a2 = galsim.fft.ifft2(ka)

    Restrictions on this version vs the numpy version:

        - The array must be 2-dimensional.
        - It must be square.
        - The size in each direction must be even.
        - The array is assumed to be Hermitian, which means the k values with kx<0 are assumed
          to be equal to the conjuate of their inverse.  This will always be the case if
          a is an output of fft2 (with a real input array).
          i.e. for kx >= N/2, ky > 0: a[ky, kx] == a[N-ky, N-kx].conjugate()
               for kx >= N/2, ky = 0: a[0, kx] == a[0, N-kx].conjugate()
          Only the elements a[:,0:N/2+1] are accessed by this function.
        - If it has a real dtype, it will be coerced to numpy.float64.
        - If it hsa a complex dtype, it will be coerced to numpy.complex128.

    The returned array will be real with dtype numpy.float64.

    If shift_in is True, then this is equivalent to applying numpy.fft.fftshift to the input.

        >>> a1 = numpy.fft.ifft2(numpy.fft.fftshift(ka))
        >>> a2 = galsim.fft.ifft2(ka, shift_in=True)

    If shift_out is True, then this is equivalent to applying numpy.fft.fftshift to the output.

        >>> a1 = numpy.fft.fftshift(numpy.fft.ifft2(ka))
        >>> a2 = galsim.fft.ifft2(ka, shift_out=True)

    @param a            The input array to be transformed
    @param shift_in     Whether to shift the input array so that the center is moved to (0,0).
    @param shift_out    Whether to shift the output array so that the center is moved to (0,0).

    @returns a real numpy array
    """
    s = a.shape
    if len(s) != 2:
        raise ValueError("Input array must be 2D.")
    N = s[0]
    if N != s[1]:
        raise ValueError("Input array must be square.")

    No2 = N // 2
    if a.dtype.kind == 'c':
        a = a.astype(np.complex128)
        xim = galsim._galsim.ConstImageViewC(a, -No2, -No2)
    else:
        a = a.astype(np.float64)
        xim = galsim._galsim.ConstImageViewD(a, -No2, -No2)
    kim = xim.cfft(inverse=True, shift_in=shift_in, shift_out=shift_out)
    kar = kim.array
    return kar


def rfft2(a, shift_in=False, shift_out=False):
    """Compute the one-dimensional discrete Fourier Transform for real input.

    For valid inputs, the result is equivalent to numpy.fft.rfft2(a), but usually faster.

        >>> ka1 = numpy.fft.rfft2(a)
        >>> ka2 = galsim.fft.rfft2(a)

    Restrictions on this version vs the numpy version:

        - The input array must be 2-dimensional.
        - If it does not have dtype numpy.float64, it will be cerced to numpy.float64.
        - It must be square.
        - The size in each direction must be even.

    The returned array will be complex with dtype numpy.complex128.

    If shift_in is True, then this is equivalent to applying numpy.fft.fftshift to the input.

        >>> ka1 = numpy.fft.rfft2(numpy.fft.fftshift(a))
        >>> ka2 = galsim.fft.rfft2(a, shift_in=True)

    If shift_out is True, then this is equivalent to applying numpy.fft.fftshift to the output.

        >>> ka1 = numpy.fft.fftshift(numpy.fft.rfft2(a),axes=(0,))
        >>> ka2 = galsim.fft.rfft2(a, shift_out=True)

    @param a            The input array to be transformed
    @param shift_in     Whether to shift the input array so that the center is moved to (0,0).
    @param shift_out    Whether to shift the output array so that the center is moved to (0,0).

    @returns a complex numpy array
    """
    s = a.shape
    if len(s) != 2:
        raise ValueError("Input array must be 2D.")
    N = s[0]
    if N != s[1]:
        raise ValueError("Input array must be square.")

    No2 = N // 2
    a = a.astype(np.float64)
    xim = galsim._galsim.ConstImageViewD(a, -No2, -No2)
    kim = xim.rfft(shift_in=shift_in, shift_out=shift_out)
    kar = kim.array
    return kar


def irfft2(a, shift_in=False, shift_out=False):
    """Compute the 2-dimensional inverse FFT of a real array.

    For valid inputs, the result is equivalent to numpy.fft.irfft2(a), but usually faster.

        >>> a1 = numpy.fft.irfft2(ka)
        >>> a2 = galsim.fft.irfft2(ka)

    Restrictions on this version vs the numpy version:

        - The array must be 2-dimensional.
        - If it does not have dtype numpy.complex128, it will be cerced to numpy.complex128.
        - It must have shape (N, N/2+1).
        - The size in the y direction (axis=0) must be even.

    The returned array will be real with dtype numpy.float64.

    If shift_in is True, then this is equivalent to applying numpy.fft.fftshift to the input.

        >>> a1 = numpy.fft.irfft2(numpy.fft.fftshift(a, axes=(0,)))
        >>> a2 = galsim.fft.irfft2(a, shift_in=True)

    If shift_out is True, then this is equivalent to applying numpy.fft.fftshift to the output.

        >>> a1 = numpy.fft.fftshift(numpy.fft.irfft2(a))
        >>> a2 = galsim.fft.irfft2(a, shift_out=True)

    @param a            The input array to be transformed
    @param shift_in     Whether to shift the input array so that the center is moved to (0,0).
    @param shift_out    Whether to shift the output array so that the center is moved to (0,0).

    @returns a real numpy array
    """
    s = a.shape
    if len(s) != 2:
        raise ValueError("Input array must be 2D.")
    N = s[0]
    No2 = N // 2
    if No2+1 != s[1]:
        raise ValueError("Input array must have shape (N, N/2+1).")

    a = a.astype(np.complex128)
    kim = galsim._galsim.ConstImageViewC(a, 0, -No2)
    xim = kim.irfft(shift_in=shift_in, shift_out=shift_out)
    xar = xim.array
    return xar


