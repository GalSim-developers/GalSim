Changes from v0.3 to current version: 
------------------------------------

* Option for shear power spectra: use a tabulated P(k), either input as arrays or read in from a
  file, for example from a cosmological shear power spectrum calculator.  This work also involved
  making a python interface to C++ tables that can be used for interpolation in a more general
  context. (Issue #305)
