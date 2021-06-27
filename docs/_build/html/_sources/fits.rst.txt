Interfacing with FITS Files
===========================

As many astronomical images are stored as FITS files, GalSim includes functionality for
reading and writing these files with GalSim `Image` instances.

We include routines for reading and writing an individual `Image` to/from FITS files, and also
routines for handling multiple `Image` instances in a single FITS file.

We also have a wrapper around the FITS header information to make it work more like a Python
``dict``, called `FitsHeader`.

.. note::
    These routines are largely wrappers of the astropy.io.fits package.  They are still fairly
    useful for connecting GalSim objects with the AstroPy API.  However, they used to be critically
    important for providing a stable API across different PyFITS and then AstroPy versions.
    For instance, now the ``astropy.io.fits.Header`` API is very similar to our own `FitsHeader`,
    but we used to have many checks for different PyFITS and AstroPy versions to call things in
    different ways while maintaining an intuitive front-end user interface.


Reading FITS Files
------------------

.. autofunction:: galsim.fits.read

.. autofunction:: galsim.fits.readMulti

.. autofunction:: galsim.fits.readCube

.. autofunction:: galsim.fits.readFile

.. autofunction:: galsim.fits.closeHDUList

Writing FITS Files
------------------

.. autofunction:: galsim.fits.write

.. autofunction:: galsim.fits.writeMulti

.. autofunction:: galsim.fits.writeCube

.. autofunction:: galsim.fits.writeFile

FITS Headers
------------

.. autoclass:: galsim.fits.FitsHeader
    :members:
