Changes from v0.4 to current version:
------------------------------------

* Added document describing the operations being carried out by the lensing engine when it draws
  shears according to a user-specified power spectrum. (Issue #248)

* Added the ability to draw lensing shears and convergences self-consistently
  from the same input shear power spectrum.  (Issue #304)

* Added a utility that can take an input set of shears on a grid, and
  reconstruct the convergence.  (Issue #304)

* Added the ability to modify parameters that control the precise rendering of GSObjects using the
  new GSParams class. (Issue #343)

* Added Shapelet class (sub-class of GSObject) for describing shapelet profiles. (Issue #350)

* Made various speed improvements related to drawing images, both in real and Fourier space. 
  (Issue #350)

* Changed `obj.draw()` to return the added_flux in addition to the image in parallel to existing
  behavior of `drawShoot`. (Issue #350)

* Added des module that add some DES-specific types and paves the way for adding similar modules
  for other telescopes/surveys.  Specifically, there are classes for the two ways that DES measures
  PSFs: DES_Shapelet and DES_PSFEx, demoed in examples/des.py and examples/des.yaml. (Issue #350)

* Enabled InputCatalog to read FITS catalogs. (Issue #350)

* Added FitsHeader class and config option. (Issue #350)

* Added the ability to read/write to a specific HDU rather than assuming the first hdu should 
  be used. (Issue #350)

* The `ImageCorrFunc` has been superseded by the `CorrelatedNoise`, which like the `GaussianNoise`,
  `PoissonNoise` etc. classes inherits from the `BaseNoise`.  The class contains all the correlation
  information represented by the `ImageCorrFunc`, as well as the random number generator required
  to apply noise. (Issue #352)

* Similarly the get_COSMOS_CorrFunc() is replaced by the getCOSMOSNoise() function, which now
  initializes a Noise model with a stored random number generator. (Issue #352)

* Bug fixed in the generation of correlated noise fields; formerly these erroneously had
  two-fold rotational symmetry.  (Issue #352)

* The correlated noise classes now have an applyWhiteningTo() method.  The purpose of this
  function is to add noise to images that contain correlated noise; the power spectrum of the added 
  noise is specifically designed to result in white (uncorrelated) noise in the final image.
  (Issue #352)

* Added the ability to modify algorithmic parameter settings for the moments and shape measurement
  routines using the new HSMParams class. (Issue #365)

* Changed the default centering convention for even-sized images to be in the actual center, 
  rather than 1/2 pixel up and to the right of the center.  This behavior can be turned off with
  a use_true_center=False keyword in draw or drawShoot. (Issue #380)

* Fixed bug in draw routine that led to spurious shears when an object is shifted.  Probably also
  whenever the profile is not radially symmetric in fact. (Issue #380)

* Added index_convention option in config to allow for images with (0,0) as the origin rather
  than the usual (1,1). (Issue #380)

* Changed the name of the center item for the Scattered image type to image_pos, and added a
  new sky_pos which can instead specify the location of the object in sky coordinates (typically 
  arcsec) relative to the image center. (Issue #380)

* Added LINKFLAGS to the list of SCons options to pass flags to the linker. (Issue #380)

* Added a new script, galsim/pse.py, that contains a PowerSpectrumEstimator class that can be used
  to estimate the shear power spectrum from a set of shears defined on a grid.  The main
  functionality of PowerSpectrumEstimator actually does not require an installed version of GalSim,
  just Python 2.6 or 2.7 and NumPy. (Issue #382)

* Added ability to truncate Sersic profiles with optional trunc parameter. (Issue #388)

* Added trefoil to optical aberration. (Issue #390)
