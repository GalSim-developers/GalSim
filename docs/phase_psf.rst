
Phase-screen PSFs
=================

We have available a more complicated kind of PSF model that tries to correctly model the
wavefront as it passed through various "screens" such as the atmosphere or optics.
It has a number of ancillary helper functions and classes associated with it to
define things like the aperture and the effect of the various screens.

For PSFs drawn using real-space or Fourier methods, these utilities essentially evaluate the
Fourier optics diffraction equation:

.. math::
    PSF(x,y) = \int \left| {\cal F}\left(A(u, v) e^{i \phi(u, v, x, y, t)} \right) \right|^2 dt

where :math:`x`, :math:`y` are focal plane coordinates and :math:`u`, :math:`v` are
pupil plane coordinates, :math:`A` is the aperture, :math:`\phi` is the phase of the
incident wavefront, and :math:`\cal F` is the Fourier transform operator.

For ``method='phot'``, two possible strategies are available.

1. The first strategy is to draw the PSF using Fourier methods into an `InterpolatedImage`,
   and then shoot photons from that profile.  This strategy has good accuracy, but can be
   computationally expensive, particularly for atmospheric PSFs that need to be built up in
   small increments to simulate a finite exposure time.
2. The second strategy, which can be significantly faster, especially for atmospheric PSFs,
   is to use the geometric optics approximation.  This approximation has good accuracy for
   atmospheric PSFs, so we make it the default for `PhaseScreenPSF`.  The accuracy is somewhat
   less good for purely optical PSFs though, so the default behavior for OpticalPSF is to use
   the first strategy.  The ``geometric_shooting`` keyword can be used in both cases to
   override the default.

The main classes of note are:

`Aperture`
    Class representing the illuminated region of pupil.

`AtmosphericScreen`
    Class implementing phase(u, v, x, y, t) for von Karman type turbulence, with possibly evolving
    "non-frozen-flow" phases.

`OpticalScreen`
    Class implementing optical aberrations using Zernike polynomial expansions in the wavefront.

`PhaseScreenList`
    Python sequence type to hold multiple phase screens, for instance to simulate turbulence at
    different altitudes, or self-consistently model atmospheric and optical phase aberrations.
    A key method is `PhaseScreenList.makePSF`, which will take the list of phase screens, add
    them together linearly (Fraunhofer approximation), and evaluate the above diffraction equation
    to yield a `PhaseScreenPSF` object.

`PhaseScreenPSF`
    A `GSObject` holding the evaluated PSF from a set of phase screens.

`OpticalPSF`
    A `GSObject` for optical PSFs with potentially complicated pupils and Zernike aberrations.

    .. note::

        `OpticalPSF` is technically a kind of `PhaseScreenPSF`, but if you only want the
        optical model, you generally don't need to bother with building any of the screens
        manually.  The `OpticalPSF` class constructor will handle this for you.

`SecondKick`
    A `GSObject` describing the high-k turbulence portion of an atmospheric PSF convolved by
    an `Airy` PSF.  When using photon shooting with a `PhaseScreenPSF`, small scale (high-k)
    features are not well approximated by the geometric optics approximation, since this is
    where Fourier effects such as diffraction become important.  The `SecondKick` class can
    compensate by simulating the appropriate behavior at high k values.

`Atmosphere`
    Convenience function to quickly assemble multiple `AtmosphericScreen` instances into a
    `PhaseScreenList`.


.. autoclass:: galsim.Aperture
    :members:

.. autoclass:: galsim.AtmosphericScreen
    :members:

.. autoclass:: galsim.OpticalScreen
    :members:

.. autoclass:: galsim.PhaseScreenList
    :members:

.. autoclass:: galsim.PhaseScreenPSF
    :members:
    :show-inheritance:

.. autoclass:: galsim.SecondKick
    :members:
    :show-inheritance:

.. autofunction:: galsim.Atmosphere

.. autofunction:: galsim.phase_screens.initWorkerArgs

.. autofunction:: galsim.phase_screens.initWorker
