
DES Meds
========

Module for generating DES Multi-Epoch Data Structures (MEDS) in GalSim.

This module defines the `MultiExposureObject` class for representing multiple exposure data for a single object. The `WriteMEDS` function can be used to write a list of `MultiExposureObject` instances to a single MEDS file.

Importing this module also adds these data structures to the config framework, so that MEDS file output can subsequently be simulated directly using a config file.

.. autoclass:: galsim.des.MultiExposureObject
    :members:
    :special-members:

.. autoclass:: galsim.des.MEDSBuilder
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.des.OffsetBuilder
    :members:
    :special-members:
    :show-inheritance:

.. autofunction:: galsim.des.WriteMEDS
