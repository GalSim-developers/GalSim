
Random Deviates
===============

GalSim can produce random values according to a variety of probability distributions:

* `UniformDeviate` implements :math:`p(x) = 1` for :math:`0 \le x < 1`.
* `GaussianDeviate` implements :math:`p(x) = \frac{1}{\sqrt{2\pi\sigma}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}`.
* `PoissonDeviate` implements :math:`p(x) = \frac{e^{-\mu}\mu^x}{x!}` for integer :math:`x > 0`.
* `BinomialDeviate` implements :math:`p(x) = {N \choose x}p^k(1-p)^{N-x}` for integer :math:`0 \le x \le N`.
* `Chi2Deviate` implements :math:`p(x) = \frac{x^{(n/2)-1}e^{-x/2}}{\Gamma(n/2)2^{n/2}}` for :math:`x > 0`.
* `GammaDeviate` implements :math:`p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)}` for :math:`x > 0`.
* `WeibullDeviate` implements :math:`p(x) = \frac{a}{b}\left(\frac{x}{b}\right)^{a-1}e^{-\left(\frac{x}{b}\right)^a}` for :math:`x \ge 0`.
* `DistDeviate` implements any arbitrary, user-supplied :math:`p(x)`.

These are all subclasses of the base class `BaseDeviate`, which implements the underlying
pseudo-random number generator using the Boost libraries Mersenne twister.

We have fixed the implementation of this to Boost version 1.48.0, the relevant files of which are
bundled with the GalSim distribution, so that random numbers produced by GalSim simulations are
deterministic across different user platforms and operating systems.  These Boost files are
included with GalSim, so the user does not need to have Boost installed on their system.

There are ways to connect various different deviate objects to use the same underlying
`BaseDeviate`, which is often important for producing deterministic simulations given a particular
random number seed.  See the docstring of `BaseDeviate` for details.

.. note::

    We have put some care into the way we seed the random number generator such that it is
    safe to start several random number sequences seeded by sequential seeds.  This is already
    supposed to be the case for the Boost Mersenne Twister implementation, but we add some extra
    (probably overly paranoid) steps to ensure this by seeding one pseudo-rng, skip a few values,
    and then use that to seed the actual pseudo-rng that we will use.

    This means you can start the rngs for sequential images or even galaxies with sequential seed
    values and there will not be any measurable correlations in the results.  This can greatly
    ease the ability to split work across multiple processes and still achieve deterministic
    results.

.. autoclass:: galsim.BaseDeviate
    :members:

    .. automethod:: galsim.BaseDeviate._seed
    .. automethod:: galsim.BaseDeviate._reset

.. autoclass:: galsim.UniformDeviate
    :members:
    :show-inheritance:

    .. automethod:: galsim.UniformDeviate.__call__

.. autoclass:: galsim.GaussianDeviate
    :members:
    :show-inheritance:

    .. automethod:: galsim.GaussianDeviate.__call__

.. autoclass:: galsim.PoissonDeviate
    :members:
    :show-inheritance:

    .. automethod:: galsim.PoissonDeviate.__call__

.. autoclass:: galsim.BinomialDeviate
    :members:
    :show-inheritance:

    .. automethod:: galsim.BinomialDeviate.__call__

.. autoclass:: galsim.Chi2Deviate
    :members:
    :show-inheritance:

    .. automethod:: galsim.Chi2Deviate.__call__

.. autoclass:: galsim.GammaDeviate
    :members:
    :show-inheritance:

    .. automethod:: galsim.GammaDeviate.__call__

.. autoclass:: galsim.WeibullDeviate
    :members:
    :show-inheritance:

    .. automethod:: galsim.WeibullDeviate.__call__

.. autoclass:: galsim.DistDeviate
    :members:
    :show-inheritance:

    .. automethod:: galsim.DistDeviate.__call__
