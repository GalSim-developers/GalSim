Sensor Models
=============

The `Sensor` classes implement the process of turning a set of photons incident at the surface
of the detector in the focal plane into an image with counts of electrons in each pixel.

The `Sensor` class itself implements the simplest possible sensor model, which just converts each
photon into an electron in whatever pixel is below the location where the photon hits.
However, it also serves as a base class for other classes that implement more sophisticated
treatments of the photon to electron conversion and the drift from the conversion layer to the
bottom of the detector.


.. autoclass:: galsim.Sensor
    :members:

.. autoclass:: galsim.SiliconSensor
    :members:
    :show-inheritance:

