
Angles and Coordinates
######################

All the code we use for handling angles and celestial coordinates are now
in the LSSTDESC.Coord package:

https://github.com/LSSTDESC/Coord

An earlier version of this code was originally implemented in GalSim, so we
still import the relevant classes into the ``galsim`` namespace, so for example
``gasim.Angle`` is a valid alias for ``coord.Angle``.  You may therefor use either namespace
for your use of these classes.

Angles
======

.. autoclass:: galsim.Angle
    :members:

.. autofunction:: galsim._Angle

.. autoclass:: galsim.AngleUnit
    :members:


Celestial Coordinates
=====================

.. autoclass:: galsim.CelestialCoord
    :members:

