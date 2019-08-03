Charge Deflection Model
=======================

We include in GalSim a basic implemetation of the Antilogus charge deflection model.
Probably at this point, there are better implementations of this model, so this might not
be very useful for most users.  However, if you just want a quick and dirty way to simulate
the so-called "Brighter-Fatter Effect", you can use this.

.. note::
    A better implementation of brighter-fatter is now available as part of the `SiliconSensor`
    model.  It does not use the Antilogus model at all, but rather tries to simulate the
    underlying physics of the effect.

.. autoclass:: galsim.cdmodel.BaseCDModel
    :members:
    :special-members:

.. autoclass:: galsim.cdmodel.PowerLawCD
    :members:
    :special-members:
    :show-inheritance:

