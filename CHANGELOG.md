Below are the most significant differences between the current version of GalSim on the master
branch of the GitHub repository, and the last tagged version (v0.2):

* Several bug fixes in the Fourier space parameters of the Sersic surface brightness profile, which
  improves some issues with ringing in images composed of Sersic profiles on their own or combined
  with other profiles.

* Fixed several sources of memory leaks, with the most significant being in the moments and shape
  estimation software and a minor one in CppShear.

* There's a useful new option when compiling, MEMTEST, that makes it easy to check for memory leaks
  in the C++ side of GalSim.

* Minor change in the keywords related to directory specification for RealGalaxyCatalog.

* It is now possible to draw the Fourier images of GSObjects using the `drawK()` method, which was
  always available in C++ but now is visible in python as well.

* A minor fix for some issues with image types in `fits.writeCube()`.

* On systems where a different C++ compiler was used for GalSim and for python, C++ exceptions show
  up with a non-informative error message.  While this is not easily fixable, there is now a test
  for this incompatibility when compiling, which results in a warning being generated.

* Minor changes in the python interface to the outputs of the moments and shape estimation routines
  (the HSMShapeData class).
