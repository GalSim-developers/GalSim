Changes from v0.5 to current version:
------------------------------------

* Updated the allowed range for Sersic n to 0.3 -- 6.2.  Beyond this range we find that the 
  code has numerical problems leading to significant inaccuracies.  (Issue #325)

* Added MEDS file output to the des module.  Also made it easier for modules such as this to add 
  new input and output formats to the config structure.  (Issue #376)

* Fixed a bug in the rendering of shifted images.  (Issue #424)

* Made RealGalaxy objects keep track of their (correlated) noise.  Functions like applyShear,
  Convolve, and such automatically keep this updated so the final object also knows the current
  noise profile so it can be appropriately whitened.  (Issue #430)

* Changed the noise padding options for RealGalaxy and InterpolatedImage somewhat.  Now there is
  a parameter `noise_pad_size` which (optionally) sets a minimum size to pad out the image with
  the same noise as is in the main image.  This is required to make sure whitening will work
  correctly.  (Outside of this noise padding, `pad_factor` still sets the amount of zero-padding
  to use to remove the ghost images in the FFT.)  (Issue #430)

* Added whiten option to config for RealGalaxy objects to whiten the image.  (Issue #430)

* Added VariableGaussianNoise to apply Gaussian noise with a variable sigma across the image.
  (Issue #430)

* Added `Current` type in config to use the current value of some other item in the config file
  as part of a calculation.  Example of this is in demo11.yaml.  (Issue #430)

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

* Fixed a bug in InterpolatedImage calculateStepK function for noisy images.  (Issue #454)

* Added 'signal_to_noise' option in config for PSFs when there is not also a galaxy.  (Issue #459)

* Added `galsim` executable to be preferred over either `galsim_yaml` or `galsim_json`, although 
  the old names are still valid for backwards compatibility.  (Issue #460)

* Removed `des` module from default imports of GalSim.  Now you need to explicitly write
  `import galsim.des` to use that functionality.  For runs using the galsim executable, you would 
  now run `galsim -m des config_file`.  This also allows you to develop your own (external to 
  GalSim) module and use it with galsim.  Just use `galsim -m [your_module]` here.  (Issue #460)

* Fixed bug in Image class resize function.  (Issue #461)

* Sped up the OpticalPSF calculations significantly.  (Issue #466)
