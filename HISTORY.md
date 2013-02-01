Below is a history of tagged versions of GalSim called `vX.X`.  All tags of the form
`milestoneN` were from earlier stages of development and are not currently recommended for use.

v0.3
----

Major changes from v0.2:
* Support reading and writing compressed fits files. (Issue #299)

* The moments and PSF correction code was updated to use the Image class and TMV; to handle weight
  and bad pixel maps for the input Images; and to run typically 2-3 times faster.  (Issues #331,
  #332)

* There is a new base class, InterpolatedImage, for users who wish to take some arbitrary input
  image and manipulate it.  (Issue #333)

* There is a new class structure for representing 2D correlation functions, used to describe 
  correlated noise in images. (Issue #297).

Minor changes from v0.2:
* Minor changes in the python interface to the outputs of the moments and shape estimation routines
  (the HSMShapeData class).  (Issue #296, #316, #332)

* Add FormattedStr option for string values in config files.  (Issue #315)

* On systems where a different C++ compiler was used for GalSim and for python, C++ exceptions show
  up with a non-informative error message.  While this is not easily fixable, there is now a test
  for this incompatibility when compiling, which results in a warning being generated.
  (Issue #317)

* Made default poisson_flux value = False when n_photons is explicitly given to drawShoot. 
  (Issue #319)

* It is now possible to draw the Fourier images of GSObjects using the `drawK()` method, which was
  always available in C++ but now is visible in python as well. (Issue #319)

* Minor change in the keywords related to directory specification for RealGalaxyCatalog.
  (Issue #322)

* There is a useful new option when compiling, MEMTEST, that makes it easy to check for memory
  leaks in the C++ side of GalSim. (Issue #327)

* Enable copying Images of different types. (Issue #327)


Bug fixes from v0.2:
* Fixed bug in config RandomCircle when using inner_radius option.

* Fixed bug in config when trying to draw objects whose postage stamp falls entirely off the 
  main image.

* Fixed treatment of duplicate numpy.int32 types on some systems where the old check was not
  sufficient.

* Fix warnings in boost::random stuff on some systems (Issue #250)

* Several bug fixes in the Fourier space parameters of the Sersic surface brightness profile, which
  improves some issues with ringing in images composed of Sersic profiles on their own or combined
  with other profiles. (Issues #319, #330)

* A minor fix for some issues with image types in `fits.writeCube()`. (Issue #320)

* Fixed several sources of memory leaks, with the most significant being in the moments and shape
  estimation software and a minor one in CppShear. (Issue #327)

v0.2
----

Major changes from v0.1:

* Significant revamping and commenting of demos, including both python and config versions 
  (Issues #243, #285, #289).

* Fixed build problems with MKL (Issue #261).

* Removed scipy dependence from the lensing engine, replacing it with functions from the C++ integ
  package which are now exposed at the python layer (Issues #288, #291).

* Significant expansion in `config` functionality, using YAML/JSON format config files 
  (Issues #291, #295).

* Fixed some bugs in Image handling (including bugs related to duplicate numpy.int32 types), 
  and made Image handling generally more robust (Issues #293, #294).

* Minor fixes to build process to make it work with fink.  Version released via fink.

* Fixed bug in wrapping of angles (now not automatic -- use wrap() explicitly).

v0.1
----

Initial version of GalSim that had nearly all the functionality we eventually want.
