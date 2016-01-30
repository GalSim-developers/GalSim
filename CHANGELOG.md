Changes from v1.3 to v1.4
=========================

API Changes
-----------

- Changed the default shift and/or offset for the output.psf field in a config
  file to not do any shift or offset.  It had been the default to match what
  was applied to the galaxy (cf. demo5).  However, we thought that was probably
  not the most intuitive default.  Now, matching the galaxy is still possible,
  but requires explicit specification of output.psf.shift = "galaxy" or
  output.psf.offset = "galaxy". (#691)


Bug Fixes
---------

- Improved ability of galsim.fits.read to handle invalid but fixable FITS
  headers. (#602)
- Fixed bug in des module related to building meds file with wcs taken from
  the input images. (#654)
- Improved ability of ChromaticObjects to find fiducial achromatic profiles
  and wavelengths with non-zero flux. (#680)
- Fixed a bug in some of the WCS classes if the RA/Dec axes in the FITS header
  are reversed (which is allowed by the FITS standard). (#681)
- Fixed a bug in the way Images are instantiated for certain combinations of
  ChromaticObjects and image-setup keyword arguments (#683)
- Added ability to manipulate the width of the moment-measuring weight function
  for the KSB shear estimation method of the galsim.hsm package. (#686)
- Fixed bug in the (undocumented) function COSMOSCatalog._makeSingleGalaxy,
  where the resulting object did not set the index attribute properly. (#694)


New Features
------------

- Added OutputCatalog class, which can be used to keep track of and then output
  truth information.  cf. demos 9 and 10. (#301, #691)
- Added methods calculateHLR, calculateMomentRadius, and calculateFWHM to both
  GSObject and Image. (#308)
- Added BoundsI.numpyShape() to easily get the numpy shape that corresponds
  to a given bounds instance. (#654)
- Changed `galsim.fits.writeMulti` to allow any of the "image"s to be
  already-built hdus, which are included as is. (#691)
- Added optional `wcs` argument to `Image.resize()`. (#691)
- Added `BaseDeviate.discard(n)` and `BaseDeviate.raw()`. (#691)
- Added `sersic_prec` option to COSMOSCatalog.makeGalaxy(). (#691)


Updates to galsim executable
----------------------------

- Dropped default verbosity from 2 to 1, since for real simulations, 2 is
  usually too much output. (#691)
- Added ability to easily split the total work into several jobs with
  galsim -n njobs -j jobnum. (#691)
- Added galsim -p to perform profiling on the run. (#691)


New config features
-------------------

- Added ability to write truth catalogs using output.truth field. (#301, #691)
- Improved the extensibility of the config parsing.  It is now easier to write
  custom image types, object types, value types, etc. and register them with
  the config parser.  The code with the new type definitions should be given
  as a module for the code to import using the new 'modules' top-level
  config field. (#691)
- Added the 'template' option to read another config file and use either the 
  whole file as a template or just a given field from the file. (#691)
- Made '$' and '@' shorthand for 'Eval' and 'Current' types respectively in
  string values.  e.g. '$(@image.pixel_scale) * 2' would be parsed to mean
  2 times the current value of image.pixel_scale.  (#691)
- Allowed gsobjects to be referenced from Current types. (#691)
- Added x,f specification for a RandomDistribution. (#691)
- Added a new 'stamp' top level field and moved some of the items that had
  belonged in 'image' over to 'stamp'.  Notably, 'draw_method', 'offset', and
  'gsparams', among other less commonly used parameters.  However, for
  backwards compatibility, they are all still allowed in the image field
  as well. (#691)

