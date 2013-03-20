Changes from v0.4 to current version:
------------------------------------

- The `ImageCorrFunc` has been superseded by the `CorrelatedNoise`, which like the `GaussianNoise`,
  `PoissonNoise` etc. classes inherits from the `BaseNoise`.  The class contains all the correlation
  information represented by the `ImageCorrFunc`, as well as the random number generator required
  to apply noise (Issue #352).

- Similarly the get_COSMOS_CorrFunc() is replaced by the get_COSMOS_CorrelatedNoise() function,
  which now initializes a Noise model with a stored random number generator (Issue #352).

- Bug fixed in the generation of correlated noise fields (Issue #352); formerly these erroneously 
  had two-fold rotational symmetry.

- The correlated noise classes now have an applyWhiteningTo() method.  The purpose of this
  function is to add noise to images that contain correlated noise; the power spectrum of the added 
  noise is specifically designed to result in white (uncorrelated) noise in the final image (Issue
  #352).

- Added Shapelet class (sub-class of GSObject) for describing shapelet profiles. (Issue #350)

- Made various speed improvements related to drawing images, both in real and Fourier space. 
  (Issue #350)

- Changed `obj.draw()` to return the added_flux in addition to the image in parallel to existing
  behavior of `drawShoot`. (Issue #350)

- Added des module that add some DES-specific types and paves the way for adding similar modules
  for other telescopes/surveys.  Specifically, there are classes for the two ways that DES measures
  PSFs: DES_Shapelet and DES_PSFEx, demoed in examples/des.py and examples/des.yaml. (Issue #350)

- Enabled InputCatalog to read FITS catalogs. (Issue #350)

- Added FitsHeader class and config option. (Issue #350)

- Added the ability to read/write to a specific HDU rather than assuming the first hdu should 
  be used. (Issue #350)
