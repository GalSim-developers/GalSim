Below is a summary of the major changes with each new tagged version of GalSim.
Each version also includes various other minor changes and bug fixes, which are 
not listed here for brevity.  See the CHANGLELOG.md files associated with each 
version for a more complete list.

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
