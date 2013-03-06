Below is a summary of the major changes with each new tagged version of GalSim.
Each version also includes various other minor changes and bug fixes, which are 
not listed here for brevity.  See the CHANGLELOG.md files associated with each 
version for a more complete list.

v0.4
----

* When making GSObjects out of real images that have noise, it is possible to pad those images with
  a noise field (either correlated or uncorrelated) so that there is not an abrupt change of
  properties in the noise field when crossing the border into the padding region.  (Issue #238)

* There is now a python interface to C++ tables that can be used for interpolation in a more general
  context. (Issue #305)

* Lensing engine updates: Introduced the option of drawing shears from a tabulated P(k), for example
  from a cosmological shear power spectrum calculator.  Also, the `PowerSpectrum` class now properly
  handles conversions between different angular units.  Finally, an important bug in how the shears
  were generated from the power spectrum (which resulted in significant issues with overall
  normalization) was fixed. (Issue #305)

* Added a DistDeviate class that generates pseudo-random numbers from a user-defined probability
  distribution. (Issue #306)

* Added a free function (`galsim.correlatednoise.get_COSMOS_CorrFunc(...)`) that gives the user
  quick access to the 2D spatial correlation function of noise in HST COSMOS F814W weak lensing
  science images (e.g. Leauthaud et al 2007). (Issue #345)

* Made a new Noise hierarchy, and moved CCDNoise to that rather than have it be a BaseDeviate.
  There are also now GaussianNoise, PoissonNoise, and DeviateNoise classes. (Issue #349)

v0.3
----

* Several bug fixes in the Fourier space treatment of the Sersic surface brightness profile, which
  improves some issues with ringing in images composed of Sersic profiles on their own or combined
  with other profiles. (Issues #319, #330)

* Support reading and writing compressed fits files. (Issue #299)

* There is a new base class, InterpolatedImage, for users who wish to take some arbitrary input
  image and manipulate it.  (Issue #333)

* There is a new class structure for representing 2D correlation functions, used to describe 
  correlated noise in images. (Issue #297).

* Add FormattedStr option for string values in config files.  (Issue #315)

* Added obj.drawK() to the python layer. (Issue #319)

* Fixed several sources of memory leaks, with the most significant being in the moments and shape
  estimation software and a minor one in CppShear. (Issue #327)

* The moments and PSF correction code was updated to use the Image class and TMV; to handle weight
  and bad pixel maps for the input Images; and to run typically 2-3 times faster.  (Issues #331,
  #332)

* Fixed bug in config RandomCircle when using inner_radius option.

v0.2
----

* Significant revamping and commenting of demos, including both python and config versions 
  (Issues #243, #285, #289).

* Significant expansion in `config` functionality, using YAML/JSON format config files 
  (Issues #291, #295).

* Removed scipy dependence from the lensing engine, replacing it with functions from the C++ integ
  package which are now exposed at the python layer (Issues #288, #291).

* Fixed some bugs in Image handling (including bugs related to duplicate numpy.int32 types), 
  and made Image handling generally more robust (Issues #293, #294).

* Fixed bug in wrapping of angles (now not automatic -- use wrap() explicitly).

v0.1
----

Initial version of GalSim that had nearly all the functionality we eventually want.
