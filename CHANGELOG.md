Changes from v0.3 to current version: 
------------------------------------

* Option for shears from a power spectrum: use a tabulated P(k), either input as arrays or read in from a
  file, for example from a cosmological shear power spectrum calculator.  The PowerSpectrum class
  now properly handles conversions between different units for P(k) and the galaxy positions at
  which we are calculating shear.  (Issue #305)

* An optimization of the InterpolatedImage constructor.  (Issue #305)

* A work-around for a pyfits bug that made our Rice-compressed output files (using pyfits)
  unreadable by ds9.  (Issue #305)

* There is now a python interface to C++ tables that can be used for interpolation in a more general
  context. (Issue #305)
