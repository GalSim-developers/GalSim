
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
    :special-members:

.. autoclass:: galsim.GaussianNoise
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.PoissonNoise
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.CCDNoise
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.VariableGaussianNoise
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.DeviateNoise
    :members:
    :special-members:
    :show-inheritance:
