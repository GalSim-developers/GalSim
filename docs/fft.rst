Fourier Transforms
==================

In the C++ layer we use `FFTW <http://www.fftw.org/>`_ for our 2D Fourier transforms.
This package is generally faster than numpy fft functions.  So for at least a subset
of the functionality available in the numpy versions, we have implemented python functions
that call out to the backend C++ FFTW functions.  These should all be basically drop
in replacements for the same-named ``numpy.fft`` functions.

.. autofunction:: galsim.fft.fft2
.. autofunction:: galsim.fft.ifft2
.. autofunction:: galsim.fft.rfft2
.. autofunction:: galsim.fft.irfft2

