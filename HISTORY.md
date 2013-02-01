Below is a summary of the major changes with each new tagged version of GalSim.
Each version also includes various other minor changes and bug fixes, which are 
not listed here for brevity.  See the CHANGLELOG.md files associated with each 
version for a more complete list.

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
