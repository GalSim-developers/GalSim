Arbitrary Profiles
==================

If none of the above classes seem appropriate, it is possible to define any arbitrary
profile by an image and then interpolating between the pixel positions to define the
surface brightness at any arbitrary location.  Similarly, one can define the image in
Fourier space, or use shapelets, which are a different complete basis set, instead of
using pixels.

Interpolated Images
-------------------

.. autoclass:: galsim.InterpolatedImage
    :members:
    :show-inheritance:

.. autofunction:: galsim._InterpolatedImage

Interpolated Fourier-space Images
---------------------------------

.. autoclass:: galsim.InterpolatedKImage
    :members:
    :show-inheritance:

.. autofunction:: galsim._InterpolatedKImage

Shapelet Decomposition
----------------------

.. autoclass:: galsim.Shapelet
    :members:
    :show-inheritance:


