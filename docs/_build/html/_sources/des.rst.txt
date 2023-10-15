
The DES Module
##############

The galsim.des module contains some functionality specific developed for the use of GalSim in
simulations of the Dark Energy Survey.  However, both PSFEx and MEDS files are used for other
surveys besides DES, so both `DES_PSFEx` and `MEDSBuilder` may be relevant to users outside of
DES.

.. note::
    To use this module, you must separately ``import galsim.des``.  These functions are
    not automatically imported when you ``import galsim``.

DES PSF models
--------------

.. autoclass:: galsim.des.DES_PSFEx
    :members:
    :show-inheritance:

.. autoclass:: galsim.des.DES_Shapelet
    :members:
    :show-inheritance:


Writing to MEDS Files
---------------------

This module defines the `MultiExposureObject` class for representing multiple exposure data for a single object. The `WriteMEDS` function can be used to write a list of `MultiExposureObject` instances to a single MEDS file.

Importing this module also adds these data structures to the config framework, so that MEDS file output can subsequently be simulated directly using a config file.

.. autoclass:: galsim.des.MultiExposureObject
    :members:

.. autoclass:: galsim.des.MEDSBuilder
    :members:
    :show-inheritance:

.. autoclass:: galsim.des.OffsetBuilder
    :members:
    :show-inheritance:

.. autofunction:: galsim.des.WriteMEDS
