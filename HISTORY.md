Below is a history of tagged versions of GalSim called `vX.X`.  All tags of the form
`milestoneN` were from earlier stages of development and are not currently recommended for use.

v0.2
----

Major changes from v0.1:

* Significant revamping and commenting of demos, including both python and config versions (Issues
  243, 285, 289).

* Removed scipy dependence from the lensing engine, replacing it with functions from the C++ integ
  package which are now exposed at the python layer (Issues 288, 291).

* Fixed some bugs in Image handling (including bugs related to duplicate numpy.int32 types), 
  and made Image handling generally more robust (Issues 293, 294).

* Significant expansion in `config` functionality, using YAML/JSON format config files (Issues 291,
  295).

* Minor fixes to build process to make it work with fink.  Version released via fink.

* Fixed build problems with MKL (Issue 261).

* Fixed bug in wrapping of angles (now not automatic -- use wrap() explicitly).

v0.1
----

Initial version of GalSim that had nearly all the functionality we eventually want.
