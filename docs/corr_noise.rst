
Correlated Noise
================

The pixel noise in real astronomical images can be correlated, so we have some functionality
to enable simulating this effect in GalSim.

* `BaseCorrelatedNoise` defines the correlated noise according to an input `GSObject`.  This is
  the base class for the other ways of defining correlated noise, and implements most of the
  useful methods of these classes.

  .. warning::

        This `GSObject` profile **must** have two-fold rotational symmetry to represent a physical
        correlation function, and this requirement is not enforced by GalSim.  Users need to
        ensure this fact in their calling code.

* `CorrelatedNoise` computes the correlated noise of in input `Image`, e.g. a blank patch of sky
  in an image similar to the one you want to simulate.
* `UncorrelatedNoise` is a `BaseCorrelatedNoise` that start with no correlations.  This can then
  be sheared or convolved with other profiles to induce correlations.
* `getCOSMOSNoise` is a function the returns a `BaseCorrelatedNoise` corresponding to the
  correlated noise found in the HST COSMOS F814W coadd images used by `RealGalaxy`.

While adding correlated noise to images is a useful feature, this functionality was originally
implemented in GalSim in order to *remove* correlations.  Specifically, the `RealGalaxy` class
treats HST images as surface brightness profiles.  These images have correlated noise that resulted
from the drizzle image processing.  Furthermore, when sheared or convolved by a PSF, the
noise in these images becomes even more correlated.  If uncorrected, these pixel correlations
can lead to biases in weak lensing shear estimates of the rendered galaxies.

To produce more accurate image simulations using these galaxies as the underlying models (and
especially avoid the weak lensing shear biases), it is often useful to "whiten" the correlated
noise that results form these manipulations.

* `Image.whitenNoise` takes a `BaseCorrelatedNoise` instance as the estimate of the correlated
  noise already in an image, and attempts to add more noise to result in uncorrelated "white"
  noise.
* `Image.symmetrizeNoise` similarly adds noise, but only attempts to produce a noise profile with
  4-fold (or generically any order) symmetry, which results in less added noise while still
  achieving the goal of not having correlated noise bias weak lensing shear measurements.

.. autoclass:: galsim.BaseCorrelatedNoise
    :members:
    :show-inheritance:

.. autoclass:: galsim.CorrelatedNoise
    :members:
    :show-inheritance:

.. autoclass:: galsim.UncorrelatedNoise
    :members:
    :show-inheritance:

.. autofunction:: galsim.getCOSMOSNoise

