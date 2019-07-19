
DES Meds
========

Module for generating DES Multi-Epoch Data Structures (MEDS) in GalSim.

This module defines the `MultiExposureObject` class for representing multiple exposure data for a single object. The `WriteMEDS` function can be used to write a list of `MultiExposureObject` instances to a single MEDS file.

Importing this module also adds these data structures to the config framework, so that MEDS file output can subsequently be simulated directly using a config file.

.. autoclass:: galsim.des.MultiExposureObject
   :members:
.. autoclass:: galsim.des.MEDSBuilder
   :members:
.. autoclass:: galsim.des.OffsetBuilder
   :members:

.. autofunction:: galsim.des.WriteMEDS
