
Noise Generators
================

GalSim has a number of different noise models one can use for adding noise to an `Image`:

* `GaussianNoise` adds Gaussian noise with a specified :math:`\sigma` (or variance) to each pixel.
* `PoissonNoise` treats each pixel value as the expectation value for the number of incident
  photons in the pixel, and implements a Poisson process drawing a realization of the observed
  number of photons in each pixel.
* `CCDNoise` combines the two above noise types to implement Poisson photon noise plus Gaussian
  read noise.
* `VariableGaussianNoise` is like `GaussianNoise`, but allows for a different variance in each pixel.
* `DeviateNoise` adds noise according to any of the various `Random Deviates` implemented in GalSim.

These are all subclasses of the base class `BaseNoise`, which mostly just defines the common
API for all of these classes.

.. autoclass:: galsim.BaseNoise
    :members:

    .. automethod:: galsim.BaseNoise.__mul__
    .. automethod:: galsim.BaseNoise.__div__

.. autoclass:: galsim.GaussianNoise
    :members:
    :show-inheritance:

.. autoclass:: galsim.PoissonNoise
    :members:
    :show-inheritance:

.. autoclass:: galsim.CCDNoise
    :members:
    :show-inheritance:

.. autoclass:: galsim.VariableGaussianNoise
    :members:
    :show-inheritance:

.. autoclass:: galsim.DeviateNoise
    :members:
    :show-inheritance:
