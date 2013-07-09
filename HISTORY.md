Below is a summary of the major changes with each new tagged version of GalSim.
Each version also includes various other minor changes and bug fixes, which are 
not listed here for brevity.  See the CHANGLELOG.md files associated with each 
version for a more complete list.

v0.5
----

* Updates to functionality of specific classes:
  * Added Shapelet class (sub-class of GSObject) for describing shapelet
    profiles. (Issue #350)
  * Improved speed and accuracy of non-truncated Moffat. (Issue #407)
  * Added ability to specify Sersic profiles by the scale radius. (Issue #420)
  * Added ability to truncate Sersic profiles with optional trunc
    parameter. (Issue #388)
  * Added trefoil to optical aberration. (Issue #390)
  * Added a simple prescription for adding diffraction spikes, due to secondary
    mirror / instrument support struts obscuring the pupil, to the
    OpticalPSF. (Issue #302)

* Updates to lensing engine:
  * Added document describing the operations being carried out by the lensing
    engine when it draws shears according to a user-specified power
    spectrum. (Issue #248)
  * Added the ability to draw lensing shears and convergences self-consistently
    from the same input shear power spectrum. (Issue #304)
  * Added a utility that can take an input set of shears on a grid, and
    reconstruct the convergence. (Issue #304)
  * Added `kmin_factor` and `kmax_factor` parameters to PowerSpectrum
    `buildGrid` function. (Issue #377)
  * Added a new script, galsim/pse.py, that contains a PowerSpectrumEstimator
    class that can be used to estimate the shear power spectrum from a set of
    shears defined on a grid.  The main functionality of PowerSpectrumEstimator
    actually does not require an installed version of GalSim, just Python 2.6 or
    2.7 and NumPy. (Issue #382)

* Sped up the HSM module (Issue #340), changed HSM routines to require an
  explicit `galsim.hsm.` prefix, and edited some of the function names.
  e.g. `galsim.EstimateShearHSM` -> `galsim.hsm.EstimateShear`. (Issue #372)

* Added the ability to modify parameters that control the precise rendering of
  GSObjects using the new GSParams class. (Issue #343, #426) Similarly, added
  the ability to modify algorithmic parameter settings for the moments and shape
  measurement routines using the new HSMParams class. (Issue #365)

* Made various speed improvements related to drawing images, both in real and
  Fourier space.  (Issue #350)

* Changed `obj.draw()` to return the added_flux in addition to the image in
  parallel to existing behavior of `drawShoot`. (Issue #350)

* Added des module that add some DES-specific types and paves the way for adding
  similar modules for other telescopes/surveys.  Specifically, there are classes
  for the two ways that DES measures PSFs: DES_Shapelet and DES_PSFEx, demoed in
  examples/des.py and examples/des.yaml. (Issue #350)

* The `ImageCorrFunc` has been superseded by the `CorrelatedNoise`, which like
  the `GaussianNoise`, `PoissonNoise` etc. classes inherits from the
  `BaseNoise`.  The class contains all the correlation information represented
  by the `ImageCorrFunc`, as well as the random number generator required to
  apply noise. Similarly the get_COSMOS_CorrFunc() is replaced by the
  getCOSMOSNoise() function, which now initializes a Noise model with a stored
  random number generator. The correlated noise classes now have an
  applyWhiteningTo() method.  (Issue #352)

* Changed the default centering convention for even-sized images to be in the
  actual center, rather than 1/2 pixel up and to the right of the center.  This
  behavior can be turned off with a use_true_center=False keyword in draw or
  drawShoot. (Issue #380)

* Updates to config:
  * Added index_convention option in config to allow for images with (0,0) as the
    origin rather than the usual (1,1). (Issue #380)
  * Changed the name of the center item for the Scattered image type to image_pos,
    and added a new sky_pos which can instead specify the location of the object
    in sky coordinates (typically arcsec) relative to the image center. (Issue
    #380)

* Added LINKFLAGS to the list of SCons options to pass flags to the
  linker. (Issue #380)

* File i/o:
  * Enabled InputCatalog to read FITS catalogs. (Issue #350)
  * Added FitsHeader class and config option. (Issue #350)
  * Added the ability to read/write to a specific HDU rather than assuming the
    first hdu should be used. (Issue #350)
  * Fix some errors in code and documentation of the fits module related to
    writing to an HDUList.  (Issue #417)
  * Add new function `galsim.fits.writeFile`. (Issue #417)

* Basic bug fixes:
  * Fixed some bugs in the Sersic class that were leading to low level ringing
    features in the images when drawn with FFTs.  (Issue #426)
  * Fixed some issues with image arithmetic (failure to check/respect shape and
    scale). (Issue #419)
  * Fix bugs in `obj.drawK()` function. (Issue #407)
  * Fixed some bugs in the InterpolatedImage class, one related to padding of
    images with noise fields and one related to real-space convolution.  (Issues
    #389, #432)
  * Fixed bug in draw routine that led to spurious shears when an object is
    shifted or otherwise not radially symmetric. (Issue #380)
  * Bug fixed in the generation of correlated noise fields; formerly these
    erroneously had two-fold rotational symmetry. (Issue #352)

* Deprecated `AtmosphericPSF` and `Ellipse` classes. (Issue #372) (They will
  both eventually be replaced by significantly different functionality, so they
  should not be used.)

v0.4
----

* Added ability to pad images for InterpolatedImage or RealGalaxy with either
  correlated or uncorrelated noise.  (Issue #238)

* Added python-level LookupTable class which wraps the existing C++ Table 
  class. (Issue #305)

* Lensing engine updates: (Issue #305)
  - Added the option of drawing shears from a tabulated P(k)
  - Added ability to handle conversions between different angular units.
  - Fixed an important bug in the normalization of the generated shears.

* Added a DistDeviate class. (Issue #306)

* Added `galsim.correlatednoise.get_COSMOS_CorrFunc(...)`. (Issue #345)

* Added im.addNoiseSNR(). (Issue #349)

* Made a new Noise hierarchy for CCDNoise (no longer a BaseDeviate), 
  GaussianNoise, PoissonNoise, DeviateNoise. (Issue #349)

* Added demo11 script. (Issues #305, #306, #345)

v0.3
----

* Fixed several bugs in the Sersic class that had been causing ringing. 
  (Issues #319, #330)

* Added support for reading and writing compressed fits files. (Issue #299)

* Added InterpolatedImage class to wrap existing C++ level SBInterpolatedImage. 
  (Issue #333)

* Added a new class structure for representing 2D correlation functions, used 
  to describe correlated noise in images. (Issue #297).

* Add FormattedStr option for string values in config files.  (Issue #315)

* Added obj.drawK() to the python layer. (Issue #319)

* Fixed several sources of memory leaks. (Issue #327)

* Updated the moments and PSF correction code to use the Image class and TMV;
  to handle weight and bad pixel maps for the input Images; and to run ~2-3 
  times faster.  (Issues #331, #332)

* Fixed bug in config RandomCircle when using inner_radius option.

v0.2
----

* Significant revamping and commenting of demos, including both python and 
  config versions (Issues #243, #285, #289).

* Added python-level int1d function to wrap C++-level integrator, which
  allowed us to remove our dependency on scipy.  (Issue #288)

* Significant expansion in `config` functionality, using YAML/JSON format 
  config files (Issues #291, #295).

* Fixed some bugs in Image handling (including bugs related to duplicate 
  numpy.int32 types), and made Image handling generally more robust (Issues 
  #293, #294).

* Fixed bug in wrapping of angles (now not automatic -- use wrap() explicitly).

v0.1
----

Initial version of GalSim that had nearly all the functionality we eventually 
want.
