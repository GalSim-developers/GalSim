"Real" Galaxies
===============

The `RealGalaxy` class uses images of galaxies from real astrophysical data (e.g. the Hubble Space
Telescope), along with a PSF model of the optical properties of the telescope that took these
images, to simulate new galaxy images with a different (must be larger) telescope PSF.  A
description of the simulation method can be found in Section 5 of Mandelbaum et al. (2012; MNRAS,
540, 1518), although note that the details of the implementation in Section 7 of that work are not
relevant to the more recent software used here.

The `RealGalaxyCatalog` class stores all required information about a real galaxy simulation
training sample and accompanying PSF model.

For information about downloading GalSim-readable `RealGalaxyCatalog` data in FITS format, see the
RealGalaxy Data Download page on the GalSim Wiki:

https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data

The `COSMOSCatalog` class is also based on the above `RealGalaxyCatalog`, and has functionality
for defining a "sky scene", i.e., a galaxy sample with reasonable properties that can then be
placed throughout a large image.

.. note::
   Currently, this only includes routines for making a COSMOS-based galaxy sample, but it could be
   expanded to include star samples as well.


.. autoclass:: galsim.RealGalaxy
    :members:
    :special-members:
    :show-inheritance:

.. autoclass:: galsim.RealGalaxyCatalog
    :members:
    :special-members:

.. autoclass:: galsim.COSMOSCatalog
    :members:
    :special-members:


