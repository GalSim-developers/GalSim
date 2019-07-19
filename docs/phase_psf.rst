
Phase-screen PSFs
=================

We have available a more complicated kind of PSF model that tries to correctly model the
wavefront as it passed through various "screens" such as the atmosphere or optics.
It has a number of ancillary helper functions and classes associated with it to
define things like the aperture and the effect of the various screens.

.. note::

    `OpticalPSF` is technically a kind of `PhaseScreenPSF`, but if you only want the
    optical model, you generally don't need to bother with building any of the screens
    manually.  The `OpticalPSF` class constructor will handle this for you.

.. autoclass:: galsim.PhaseScreenPSF
    :members:

.. autoclass:: galsim.Aperture
    :members:

.. autoclass:: galsim.PhaseScreenList
    :members:

.. autoclass:: galsim.OpticalScreen
    :members:

.. autoclass:: galsim.AtmosphericScreen
    :members:

.. autofunction:: galsim.Atmosphere

