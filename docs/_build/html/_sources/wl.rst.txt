
Weak Lensing
############

GalSim was originally built for the Great03 weak lensing challenge.  As such, it includes
various classes and routines for accurately handling and constructing weak lensing shear and
magnification.

* The `Shear` class is our basic object for handling and manipulating shear values.
* `PowerSpectrum` can be used to generate shear and convergence fields according to an input
  power spectrum function.
* `NFWHalo` can generate tangential shear profiles around an NFW halo mass profile.
* `galsim.pse.PowerSpectrumEstimator` can be used to estimate a shear power spectrum from
  gridded shear values.


.. toctree::
    :maxdepth: 2

    shear
    powerspectrum
    nfwhalo
    pse
