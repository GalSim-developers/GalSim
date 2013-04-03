Changes from v0.4 to current version:
------------------------------------

- Added the ability to draw lensing shears and convergences self-consistently
  from the same input shear power spectrum.  (Issue #304)

- Added a utility that can take an input set of shears on a grid, and
  reconstruct the convergence.  (Issue #304)

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

- Added the ability to change the parameter settings for the moments and shape measurement routines
  using the new HSMParams class.  (Issue #365)
