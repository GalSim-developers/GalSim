Changes from v0.4 to current version:
------------------------------------

* The `ImageCorrFunc` has been superseded by the `CorrelatedNoise`, which like the `GaussianNoise`,
  `PoissonNoise` etc. classes inherits from the `BaseNoise`.  The class contains all the correlation
  information represented by the `ImageCorrFunc`, as well as the random number generator required
  to apply noise (Issue #352).

* Similarly the get_COSMOS_CorrFunc() is replaced by the get_COSMOS_CorrelatedNoise() function,
  which now initializes a Noise model with a stored random number generator (Issue #352).

* A bug is fixed in the generation of correlated noise fields (Issue #352); formerly these
  erroneously had two-fold rotational symmetry.

* The correlated noise classes now have an applyWhiteningTo() method.  The purpose of this
  function is to add noise to images that contain correlated noise; the power spectrum of the added 
  noise is specifically designed to result in white (uncorrelated) noise in the final image (Issue
  #352).
