Changes from v0.3 to current version: 
------------------------------------

* When making GSObjects out of real images that have noise, it is possible to pad those images with
  a noise field (either correlated or uncorrelated) so that there is not an abrupt change of
  properties in the noise field when crossing the border into the padding region.  (Issue #238)

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

* Added a DistDeviate class that generates pseudo-random numbers from a user-defined probability
  distribution. (Issue #306)
  
* A free function added that gives the user quick access to the 2D spatial correlation function of
  noise in HST COSMOS F814W weak lensing science images (e.g. Leauthaud et al 2007).  Calling
  `galsim.correlatednoise.get_COSMOS_CorrFunc(...)` returns a correlation function that shares all
  the methods of a `galsim.ImageCorrFunc()`, and optional keyword inputs allow the point variance
  and physical scale of the function to be set correctly. (Issue #345)

* Added an addNoiseSNR() method to Image classes that adds noise such that the image has a 
  specified signal-to-noise ratio. (Issue #349)

* Made a new Noise hierarchy, and moved CCDNoise to that rather than have it be a BaseDeviate.
  There are also now GaussianNoise, PoissonNoise, and DeviateNoise classes. (Issue #349)

* Small bug fix for multiple operations on GSObjects when drawing images using photon shooting,
  which affects combinations like shifting and shearing.  (Issue #359)

* New demo11 script that includes getting PowerSpectrum shears from a file, uses an 
  InterpolatedImage for the PSF, uses DistDeviate (RandomDistribution in the config)
  for the sizes, and applies correlated noise such as found in COSMOS to the output image.
  (Issues #305, #306, #345)

* Minor bug fix for correct directory handling in the executable versions of the demo scripts, i.e.
  those generated in the `bin/` subdirectory using `scons examples`. (Issue #345)
