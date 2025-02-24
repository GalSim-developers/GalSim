Charge Deflection Model
=======================

We include in GalSim a basic implementation of the Antilogus charge deflection model.
Probably at this point, there are better implementations of this model, so this might not
be very useful for most users.  However, if you just want a quick and dirty way to simulate
the so-called "Brighter-Fatter Effect", you can use this.

.. note::
    A better implementation of brighter-fatter is now available as part of the `SiliconSensor`
    model.  It does not use the Antilogus model at all, but rather tries to simulate the
    underlying physics of the effect.

.. autoclass:: galsim.cdmodel.BaseCDModel
    :members:

    .. automethod:: galsim.cdmodel.BaseCDModel.__init__

.. autoclass:: galsim.cdmodel.PowerLawCD
    :members:
    :show-inheritance:

    .. automethod:: galsim.cdmodel.PowerLawCD.__init__
