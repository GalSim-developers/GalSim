
Noise and Random Values
#######################

An important part of generating realistic images is to add appropriate levels of noise.
When simulating objects, you may also want the actual objects being drawn to be random in some
way.  GalSim has a number of classes to help inject these kinds of randomness into your
simulations.

Random Variates
===============

.. autoclass:: galsim.BaseDeviate
    :members:
    :special-members:

.. autoclass:: galsim.UniformDeviate
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.GaussianDeviate
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.PoissonDeviate
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.BinomialDeviate
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.Chi2Deviate
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.GammaDeviate
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.WeibullDeviate
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.DistDeviate
    :members:
    :special-members:
    :show-inheritance:

Noise Generators
================

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

.. autoclass:: galsim.DeviateNoise
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.VariableGaussianNoise
    :members:
    :special-members:
    :show-inheritance:

Correlated Noise Generators
===========================

.. autoclass:: galsim.correlatednoise._BaseCorrelatedNoise
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.CorrelatedNoise
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.UncorrelatedNoise
    :members:
    :special-members:
    :show-inheritance:

.. autofunction:: galsim.getCOSMOSNoise

