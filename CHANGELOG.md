Changes from v0.5 to current version:
------------------------------------

* Updated the allowed range for Sersic n to 0.3 -- 6.2.  Beyond this range we find that the 
  code has numerical problems leading to significant inaccuracies.  (Issue #325)

* Added MEDS file output to the des module.  Also made it easier for modules such as this to add 
  new input and output formats to the config structure.  (Issue #376)

* Fixed a bug in the rendering of shifted images.  (Issue #424)

* Added the offset parameter to the draw and drawShoot commands, and also to the constructor 
  of InterpolatedImage.  (Issue #439)

* Added scale as a constructor parameter for Images.  (Issue #439)

* Improved the ability of Lanczos interpolants to conserve a DC input flux (with the 
  `conserve_dc=True` parameter).  (Issue #442)

* Switched default interpolant for `RealGalaxy` to `Quintic`.  (Issue #442)

* Added the ability to have multiple input catalogs in the same config file.  (Issue #449)

* Added a Dict class to read in a python dictionary from a file.  This can be YAML, JSON, or 
  Pickle formats.  (Issue #449)

* Changed the name of InputCatalog to just Catalog.  (Issue #449)

* Added offset as a config option for the image field, which applies the given offset in pixels 
  when calling the draw command.  (Issue #449)

* Added `galsim` executable to be preferred over either `galsim_yaml` or `galsim_json`, although 
  the old names are still valid for backwards compatibility.  (Issue #460)

* Removed `des` module from default imports of GalSim.  Now you need to explicitly write
  `import galsim.des` to use that functionality.  For runs using the galsim executable, you would 
  now run `galsim -m des config_file`.  (Issue #460)

