Changes from v0.3 to current version: 
------------------------------------

* Option for shears from a power spectrum: use a tabulated P(k), either input as arrays or read in
  from a file, for example from a cosmological shear power spectrum calculator.  Also, the
  `PowerSpectrum` class now properly handles conversions between different units for P(k) and the
  galaxy positions at which we are calculating shear.  Finally, an important bug in how the shears
  were generated from the power spectrum (which resulted in issues with overall normalization) was
  fixed. (Issue #305)

* An optimization of the InterpolatedImage constructor.  (Issue #305)

* A work-around for a pyfits bug that made our Rice-compressed output files (using pyfits)
  unreadable by ds9.  (Issue #305)

* There is now a python interface to C++ tables that can be used for interpolation in a more general
  context. (Issue #305)

* A free function added that gives the user quick access to the 2D spatial correlation function of
  noise in HST COSMOS F814W weak lensing science images (e.g. Leauthaud et al 2007).  Calling
  `galsim.correlatednoise.get_COSMOS_CorrFunc()` returns a correlation function that shares all the
  methods of a `galsim.ImageCorrFunc()`, and optional keyword inputs allow the physical scale of the
  function to be set correctly.
