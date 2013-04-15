Changes from v0.4 to current version:
------------------------------------

* Added document describing the operations being carried out by the lensing engine when it draws
  shears according to a user-specified power spectrum. (Issue #248)

* Added the ability to draw lensing shears and convergences self-consistently
  from the same input shear power spectrum.  (Issue #304)

* Added a utility that can take an input set of shears on a grid, and
  reconstruct the convergence.  (Issue #304)

* Added Shapelet class (sub-class of GSObject) for describing shapelet profiles. (Issue #350)

* The `ImageCorrFunc` has been superseded by the `CorrelatedNoise`, which like the `GaussianNoise`,
  `PoissonNoise` etc. classes inherits from the `BaseNoise`.  The class contains all the correlation
  information represented by the `ImageCorrFunc`, as well as the random number generator required
  to apply noise (Issue #352).

* Similarly the get_COSMOS_CorrFunc() is replaced by the getCOSMOSNoise() function, which now
  initializes a Noise model with a stored random number generator (Issue #352).

* Bug fixed in the generation of correlated noise fields (Issue #352); formerly these erroneously 
  had two-fold rotational symmetry.

* The correlated noise classes now have an applyWhiteningTo() method.  The purpose of this
  function is to add noise to images that contain correlated noise; the power spectrum of the added 
  noise is specifically designed to result in white (uncorrelated) noise in the final image (Issue
  #352).

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

* Added a new script, examples/pse.py, that contains a PowerSpectrumEstimator class that can be used
  to estimate the shear power spectrum from a set of shears defined on a grid.  The main
  functionality of PowerSpectrumEstimator actually does not require an installed version of GalSim,
  just Python 2.6 or 2.7 and NumPy.  (Issue #382)
