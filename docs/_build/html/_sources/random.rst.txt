
Noise and Random Values
#######################

An important part of generating realistic images is to add appropriate levels of noise.
When simulating objects, you may also want the actual objects being drawn to be random in some
way.  GalSim has a number of classes to help inject these kinds of randomness into your
simulations.

* `Random Deviates` describes how to generate pseudo-random numbers according to a variety of
  different probability distribution functions.
* `Noise Generators` describes how to add noise to images according a a few different noise models.
* `Correlated Noise` describes both how to add correlated noise to images and also ways to remove
  (or reduce) existing correlations from an image.

.. toctree::
    :maxdepth: 2

    deviate
    noise
    corr_noise

