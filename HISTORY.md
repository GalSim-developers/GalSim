Below is a summary of the major changes with each new tagged version of GalSim.
Each version also includes various other minor changes and bug fixes, which are 
not listed here for brevity.  See the CHANGLELOG.md files associated with each 
version for a more complete list.  Issue numbers related to each change are 
given in parentheses.

v1.0
----

* Notable bug fixes and improvements:
  * Fixed bug in the rendering of shifted images. (#424)
  * Improved the fidelity of the Lanczos `conserve_dc=True` option. (#442)
  * Switched default interpolant for RealGalaxy to Quintic, since it was
    found to be more accurate in general. (#442)
  * Fixed a bug in InterpolatedImage calculateStepK function. (#454)
  * Fixed a bug in Image class resize function. (#461)
  * Sped up OpticalPSF and RealGalaxy significantly. (#466, #474)
  * Fixed a bug in the fourier rendering of truncated Sersic profiles. (#470)

* Added `galsim` executable (instead of `galsim_yaml`, `galsim_json`). (#460)

* Updates to functionality of specific classes and functions:
  * Updated the allowed range for Sersic n to 0.3 -- 6.2. (#325)
  * Made RealGalaxy objects keep track of their (correlated) noise. (#430)
  * Changed noise padding options for RealGalaxy and InterpolatedImage. (#430)
  * Added VariableGaussianNoise class. (#430)
  * Added offset parameter to both draw and drawShoot. (#439)
  * Changed the name of InputCatalog to just Catalog. (#449)
  * Added Dict class. (#449)

* Updates to config options:
  * Added RealGalaxyOriginal galaxy type. (#389)
  * Added whiten option for RealGalaxy objects. (#430)
  * Added `Current` type. (#430)
  * Added offset option in image field. (#449)
  * Added the ability to have multiple input catalogs, dicts, etc. (#449)
  * Added `signal_to_noise` option for PSFs when there is no galaxy. (#459)
  * Added `output.skip` and `output.noclobber` options. (#474)

* Updates to des module:
  * Added MEDS file output. (#376)
  * Removed des module from default imports of GalSim.  Now need to import
    galsim.des explicitly or load with `galsim -m des ...` (#460)


v0.5
----

* Updates to functionality of specific classes:
  * Added Shapelet class. (#350)
  * Added ability to truncate Sersic profiles. (#388)
  * Added trefoil and struts to OpticalPSF. (#302, #390)

* Updates to lensing engine:
  * Added document describing how lensing engine code works. (#248)
  * Added ability to draw (gamma,kappa) from same power spectrum. (#304)
  * Added kmin_factor and kmax_factor parameters to buildGrid. (#377)
  * Added PowerSpectrumEstimator class in pse module. (#382)

* Added GSParams (#343, #426) and HSMParams (#365) classes.

* Added des module and example scripts. (#350)

* Added applyWhiteningTo method to CorrelatedNoise class. (#352)

* Changed the default centering convention for even-sized images to be in the
  actual center, rather than 1/2 pixel off-center. (#380)

* Updates to config:
  * Added index_convention option. (#380)
  * Changed the name of the center item for the Scattered image type to 
    image_pos, and added a new sky_pos item. (#380)

* Added LINKFLAGS SCons option. (#380)

* File I/O:
  * Enabled InputCatalog to read FITS catalogs. (#350)
  * Added FitsHeader class and config option. (#350)
  * Added the ability to read/write to a specific HDU. (#350)
  * Fix some errors related to writing to an HDUList. (#417)
  * Add new function galsim.fits.writeFile. (#417)

* Basic bug fixes:
  * Fixed ringing when Sersic objectss were drawn with FFTs. (#426)
  * Fixed bugs in obj.drawK() function. (#407)
  * Fixed bugs with InterpolatedImage objects. (#389, #432)
  * Fixed bug in draw routine for shifted objects. (#380)
  * Fixed bug in the generation of correlated noise fields. (#352)

* Deprecated AtmosphericPSF and Ellipse classes. (#372) 


v0.4
----

* Added ability to pad images for InterpolatedImage or RealGalaxy with either
  correlated or uncorrelated noise. (#238)

* Added python-level LookupTable class which wraps the existing C++ Table 
  class. (#305)

* Lensing engine updates: (#305)
  - Added the option of drawing shears from a tabulated P(k)
  - Added ability to handle conversions between different angular units.
  - Fixed an important bug in the normalization of the generated shears.

* Added a DistDeviate class. (#306)

* Added galsim.correlatednoise.get_COSMOS_CorrFunc(...). (#345)

* Added im.addNoiseSNR(). (#349)

* Made a new Noise hierarchy for CCDNoise (no longer a BaseDeviate), 
  GaussianNoise, PoissonNoise, DeviateNoise. (#349)

* Added demo11 script. (#305, #306, #345)


v0.3
----

* Fixed several bugs in the Sersic class that had been causing ringing. 
  (#319, #330)

* Added support for reading and writing compressed fits files. (#299)

* Added InterpolatedImage class to wrap existing C++ level SBInterpolatedImage. 
  (#333)

* Added a new class structure for representing 2D correlation functions, used 
  to describe correlated noise in images. (#297).

* Add FormattedStr option for string values in config files. (#315)

* Added obj.drawK() to the python layer. (#319)

* Fixed several sources of memory leaks. (#327)

* Updated the moments and PSF correction code to use the Image class and TMV;
  to handle weight and bad pixel maps for the input Images; and to run ~2-3 
  times faster. (#331, #332)

* Fixed bug in config RandomCircle when using inner_radius option.


v0.2
----

* Significant revamping and commenting of demos, including both python and 
  config versions (#243, #285, #289).

* Added python-level int1d function to wrap C++-level integrator, which
  allowed us to remove our dependency on scipy. (#288)

* Significant expansion in config functionality, using YAML/JSON format 
  config files (#291, #295).

* Fixed some bugs in Image handling (including bugs related to duplicate 
  numpy.int32 types), and made Image handling generally more robust (#293, #294).

* Fixed bug in wrapping of angles (now not automatic -- use wrap() explicitly).


v0.1
----

Initial version of GalSim with most of the basic functionality.
