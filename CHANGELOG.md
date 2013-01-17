Changes from v0.2 to v0.3
-------------------------

* Fixed bug in config RandomCircle when using inner_radius option.

* Fixed bug in config when trying to draw objects whose postage stamp falls entirely off the 
  main image.

* Fixed treatment of duplicate numpy.int32 types on some systems where the old check was not
  sufficient.

* Fix warnings in boost::random stuff on some systems (Issue #250)

* Minor changes in the python interface to the outputs of the moments and shape estimation routines
  (the HSMShapeData class).  (Issue #296, #316, #332)

* Support reading and writing compressed fits files. (Issue #299)

* Add FormattedStr option for string values in config files.  (Issue #315)

* On systems where a different C++ compiler was used for GalSim and for python, C++ exceptions show
  up with a non-informative error message.  While this is not easily fixable, there is now a test
  for this incompatibility when compiling, which results in a warning being generated.
  (Issue #317)

* Made default poisson_flux value = False when n_photons is explicitly given.  (Issue #319)

* It is now possible to draw the Fourier images of GSObjects using the `drawK()` method, which was
  always available in C++ but now is visible in python as well. (Issue #319)

* Several bug fixes in the Fourier space parameters of the Sersic surface brightness profile, which
  improves some issues with ringing in images composed of Sersic profiles on their own or combined
  with other profiles. (Issues #319, #330)

* A minor fix for some issues with image types in `fits.writeCube()`. (Issue #320)

* Minor change in the keywords related to directory specification for RealGalaxyCatalog.
  (Issue #322)

* Fixed several sources of memory leaks, with the most significant being in the moments and shape
  estimation software and a minor one in CppShear. (Issue #327)

* There is a useful new option when compiling, MEMTEST, that makes it easy to check for memory
  leaks in the C++ side of GalSim. (Issue #327)

* Enable copying Images of different types. (Issue #327)

* The moments and PSF correction code was updated to use the Image class and TMV. The python
  interface to this software was also updated so that it can handle weight and bad pixel maps for
  the input Images.  Finally, an optimization was introduced that typically speeds up these routines
  by a factor of 2-3.  (Issues #331, #332)

* There is a new base class, InterpolatedImage, which contains an SBInterpolatedImage.  Users who
  wish to take some arbitrary input image and manipulate it (shear, convolved, etc.) can use this
  base class as a way of easily carrying out those operations.  (Issue #333)
