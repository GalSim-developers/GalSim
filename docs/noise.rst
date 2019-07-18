
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

.. autoclass:: galsim.UniformDeviate
    :members:

.. autoclass:: galsim.GaussianDeviate
    :members:

.. autoclass:: galsim.PoissonDeviate
    :members:

.. autoclass:: galsim.BinomialDeviate
    :members:

.. autoclass:: galsim.Chi2Deviate
    :members:

.. autoclass:: galsim.GammaDeviate
    :members:

.. autoclass:: galsim.WeibullDeviate
    :members:

.. autoclass:: galsim.DistDeviate
    :members:

Noise Generators
================

.. autoclass:: galsim.BaseNoise
    :members:

.. autoclass:: galsim.GaussianNoise
    :members:

.. autoclass:: galsim.PoissonNoise
    :members:

.. autoclass:: galsim.CCDNoise
    :members:

.. autoclass:: galsim.DeviateNoise
    :members:

.. autoclass:: galsim.VariableGaussianNoise
    :members:

Correlated Noise Generators
===========================

.. autoclass:: galsim.correlatednoise._BaseCorrelatedNoise
    :members:

.. autoclass:: galsim.CorrelatedNoise
    :members:

.. autoclass:: galsim.UncorrelatedNoise
    :members:

.. autofunction:: galsim.getCOSMOSNoise

