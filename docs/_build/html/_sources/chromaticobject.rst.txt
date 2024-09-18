Chromatic Profiles
==================

The `ChromaticObject` class and its various subclasses Define wavelength-dependent surface
brightness profiles.

Implementation is done by constructing `GSObject` instances as functions of wavelength.
The `ChromaticObject.drawImage` method then integrates over wavelength while also multiplying by a
throughput function (a `galsim.Bandpass` instance).

These classes can be used to simulate galaxies with color gradients, observe a given galaxy through
different filters, or implement wavelength-dependent point spread functions.

So far photon-shooting a `ChromaticObject` is not implemented, but there is ongoing work to
include this option in GalSim, as the current FFT drawing method is very slow.  So these are not
yet particularly useful for large image simulations, especially ones including many faint sources.


.. autoclass:: galsim.ChromaticObject
    :members:

    .. automethod:: galsim.ChromaticObject.__mul__

.. autoclass:: galsim.ChromaticAtmosphere
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticOpticalPSF
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticAiry
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticRealGalaxy
    :members:
    :show-inheritance:

.. autoclass:: galsim.InterpolatedChromaticObject
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticSum
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticConvolution
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticDeconvolution
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticAutoConvolution
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticAutoCorrelation
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticTransformation
    :members:
    :show-inheritance:

.. autoclass:: galsim.ChromaticFourierSqrtProfile
    :members:
    :show-inheritance:
