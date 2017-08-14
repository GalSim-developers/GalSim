# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
        - The size in each direction must be even. (Ideally 2^k or 3*2^k for speed, but this is
          not required.)
        - If it has a real dtype, it will be coerced to numpy.float64.
        - If it has a complex dtype, it will be coerced to numpy.complex128.

    The returned array will be complex with dtype numpy.complex128.

    If shift_in is True, then this is equivalent to applying numpy.fft.fftshift to the input.

        >>> ka1 = numpy.fft.fft2(numpy.fft.fftshift(a))
        >>> ka2 = galsim.fft.fft2(a, shift_in=True)

    If shift_out is True, then this is equivalent to applying numpy.fft.fftshift to the output.

        >>> ka1 = numpy.fft.fftshift(numpy.fft.fft2(a))
        >>> ka2 = galsim.fft.fft2(a, shift_out=True)

    @param a            The input array to be transformed
    @param shift_in     Whether to shift the input array so that the center is moved to (0,0).
                        [default: False]
    @param shift_out    Whether to shift the output array so that the center is moved to (0,0).
                        [default: False]

    @returns a complex numpy array
    """
    s = a.shape
    if len(s) != 2:
        raise ValueError("Input array must be 2D.  Got shape=%s"%str(s))
    M, N = s
    Mo2 = M // 2
    No2 = N // 2

    if M != Mo2*2 or N != No2*2:
        raise ValueError("Input array must have even sizes. Got shape=%s"%str(s))

    if a.dtype.kind == 'c':
        a = a.astype(np.complex128, copy=False)
        xim = galsim._galsim.ConstImageViewCD(a, -No2, -Mo2)
        kar = xim.cfft(shift_in=shift_in, shift_out=shift_out).array
    else:
        a = a.astype(np.float64, copy=False)
        xim = galsim._galsim.ConstImageViewD(a, -No2, -Mo2)

        # This works, but it's a bit slower.
        #kar = xim.cfft(shift_in=shift_in, shift_out=shift_out).array

        # Faster to start with rfft2 version
        rkar = xim.rfft(shift_in=shift_in, shift_out=shift_out).array
        # This only returns kx >= 0.  Fill out the full image.
        kar = np.empty( (M,N), dtype=np.complex128)
        if shift_out:
            kar[:,No2:N] = rkar[:,0:No2]
            kar[0,0:No2] = rkar[0,No2:0:-1].conjugate()
            kar[1:Mo2,0:No2] = rkar[M-1:Mo2:-1,No2:0:-1].conjugate()
            kar[Mo2:M,0:No2] = rkar[Mo2:0:-1,No2:0:-1].conjugate()
        else:
            kar[:,0:No2] = rkar[:,0:No2]
            kar[0,No2:N] = rkar[0,No2:0:-1].conjugate()
            kar[1:M,No2:N] = rkar[M-1:0:-1,No2:0:-1].conjugate()
    return kar


def ifft2(a, shift_in=False, shift_out=False):
    """Compute the 2-dimensional inverse discrete Fourier Transform.

    For valid inputs, the result is equivalent to numpy.fft.ifft2(a), but usually faster.

        >>> a1 = numpy.fft.ifft2(ka)
        >>> a2 = galsim.fft.ifft2(ka)

    Restrictions on this version vs the numpy version:

        - The array must be 2-dimensional.
        - The size in each direction must be even. (Ideally 2^k or 3*2^k for speed, but this is
          not required.)
        - The array is assumed to be Hermitian, which means the k values with kx<0 are assumed
          to be equal to the conjuate of their inverse.  This will always be the case if
          a is an output of fft2 (with a real input array).
          i.e. for kx >= N/2, ky > 0: a[ky, kx] == a[N-ky, N-kx].conjugate()
               for kx >= N/2, ky = 0: a[0, kx] == a[0, N-kx].conjugate()
          Only the elements a[:,0:N/2+1] are accessed by this function.
        - If it has a real dtype, it will be coerced to numpy.float64.
        - If it has a complex dtype, it will be coerced to numpy.complex128.

    The returned array will be real with dtype numpy.float64.

    If shift_in is True, then this is equivalent to applying numpy.fft.fftshift to the input.

        >>> a1 = numpy.fft.ifft2(numpy.fft.fftshift(ka))
        >>> a2 = galsim.fft.ifft2(ka, shift_in=True)

    If shift_out is True, then this is equivalent to applying numpy.fft.fftshift to the output.

        >>> a1 = numpy.fft.fftshift(numpy.fft.ifft2(ka))
        >>> a2 = galsim.fft.ifft2(ka, shift_out=True)

    @param a            The input array to be transformed
    @param shift_in     Whether to shift the input array so that the center is moved to (0,0).
                        [default: False]
    @param shift_out    Whether to shift the output array so that the center is moved to (0,0).
                        [default: False]

    @returns a real numpy array
    """
    s = a.shape
    if len(s) != 2:
        raise ValueError("Input array must be 2D.  Got shape=%s"%str(s))
    M,N = s
    Mo2 = M // 2
    No2 = N // 2

    if M != Mo2*2 or N != No2*2:
        raise ValueError("Input array must have even sizes. Got shape=%s"%str(s))

    if a.dtype.kind == 'c':
        a = a.astype(np.complex128, copy=False)
        xim = galsim._galsim.ConstImageViewCD(a, -No2, -Mo2)
    else:
        a = a.astype(np.float64, copy=False)
        xim = galsim._galsim.ConstImageViewD(a, -No2, -Mo2)
    return xim.cfft(inverse=True, shift_in=shift_in, shift_out=shift_out).array


def rfft2(a, shift_in=False, shift_out=False):
    """Compute the one-dimensional discrete Fourier Transform for real input.

    For valid inputs, the result is equivalent to numpy.fft.rfft2(a), but usually faster.

        >>> ka1 = numpy.fft.rfft2(a)
        >>> ka2 = galsim.fft.rfft2(a)

    Restrictions on this version vs the numpy version:

        - The input array must be 2-dimensional.
        - If it does not have dtype numpy.float64, it will be coerced to numpy.float64.
        - The size in each direction must be even. (Ideally 2^k or 3*2^k for speed, but this is
          not required.)

    The returned array will be complex with dtype numpy.complex128.

    If shift_in is True, then this is equivalent to applying numpy.fft.fftshift to the input.

        >>> ka1 = numpy.fft.rfft2(numpy.fft.fftshift(a))
        >>> ka2 = galsim.fft.rfft2(a, shift_in=True)

    If shift_out is True, then this is equivalent to applying numpy.fft.fftshift to the output.

        >>> ka1 = numpy.fft.fftshift(numpy.fft.rfft2(a),axes=(0,))
        >>> ka2 = galsim.fft.rfft2(a, shift_out=True)

    @param a            The input array to be transformed
    @param shift_in     Whether to shift the input array so that the center is moved to (0,0).
                        [default: False]
    @param shift_out    Whether to shift the output array so that the center is moved to (0,0).
                        [default: False]

    @returns a complex numpy array
    """
    s = a.shape
    if len(s) != 2:
        raise ValueError("Input array must be 2D.  Got shape=%s"%str(s))
    M,N = s
    Mo2 = M // 2
    No2 = N // 2

    if M != Mo2*2 or N != No2*2:
        raise ValueError("Input array must have even sizes. Got shape=%s"%str(s))

    a = a.astype(np.float64, copy=False)
    xim = galsim._galsim.ConstImageViewD(a, -No2, -Mo2)
    return xim.rfft(shift_in=shift_in, shift_out=shift_out).array


def irfft2(a, shift_in=False, shift_out=False):
    """Compute the 2-dimensional inverse FFT of a real array.

    For valid inputs, the result is equivalent to numpy.fft.irfft2(a), but usually faster.

        >>> a1 = numpy.fft.irfft2(ka)
        >>> a2 = galsim.fft.irfft2(ka)

    Restrictions on this version vs the numpy version:

        - The array must be 2-dimensional.
        - If it does not have dtype numpy.complex128, it will be coerced to numpy.complex128.
        - It must have shape (M, N/2+1).
        - The size M must be even. (Ideally 2^k or 3*2^k for speed, but this is not required.)

    The returned array will be real with dtype numpy.float64.

    If shift_in is True, then this is equivalent to applying numpy.fft.fftshift to the input.

        >>> a1 = numpy.fft.irfft2(numpy.fft.fftshift(a, axes=(0,)))
        >>> a2 = galsim.fft.irfft2(a, shift_in=True)

    If shift_out is True, then this is equivalent to applying numpy.fft.fftshift to the output.

        >>> a1 = numpy.fft.fftshift(numpy.fft.irfft2(a))
        >>> a2 = galsim.fft.irfft2(a, shift_out=True)

    @param a            The input array to be transformed
    @param shift_in     Whether to shift the input array so that the center is moved to (0,0).
                        [default: False]
    @param shift_out    Whether to shift the output array so that the center is moved to (0,0).
                        [default: False]

    @returns a real numpy array
    """
    s = a.shape
    if len(s) != 2:
        raise ValueError("Input array must be 2D.  Got shape=%s"%str(s))
    M,No2 = s
    No2 -= 1  # s is (M,No2+1)
    Mo2 = M // 2

    if M != Mo2*2:
        raise ValueError("Input array must have even sizes. Got shape=%s"%str(s))

    a = a.astype(np.complex128, copy=False)
    kim = galsim._galsim.ConstImageViewCD(a, 0, -Mo2)
    return kim.irfft(shift_in=shift_in, shift_out=shift_out).array


