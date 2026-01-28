Fourier Transforms
==================

In the C++ layer we use `FFTW <http://www.fftw.org/>`_ for our 2D Fourier transforms.
This package is generally faster than numpy fft functions.  So for at least a subset
of the functionality available in the numpy versions, we have implemented python functions
that call out to the backend C++ FFTW functions.

These should be drop in replacements for np.fft.* functions.  e.g.::

    >>> karray = galsim.fft.fft2(xarray)

is functionally equivalent to::

    >>> karray = np.fft.fft2(xarray)

but should be a bit faster.

.. note::
    The GalSim versions often only implement the normal use case without many of the
    advanced options available with the numpy functions.  This is mostly laziness on our part --
    we only implemented the functions that we needed.  If your usage requires some option available
    in the numpy version, feel free to post a feature request on our GitHub page.


.. autofunction:: galsim.fft.fft2
.. autofunction:: galsim.fft.ifft2
.. autofunction:: galsim.fft.rfft2
.. autofunction:: galsim.fft.irfft2

