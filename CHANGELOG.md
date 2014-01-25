Changes from v1.0.0 to v1.0.1:
--------------------------

* Fixed some bugs in the config machinery when files have varying numbers
  of objects. (#487)
  - If the number of objects is dependent on a Dict value, the code had been
    erroneously using the Dict for the previous file.
  - Sequences that index on the object number had not necessarily been
    starting at the first object number in a file when the files had varying 
    numbers of objects.
  - There had not been a random number generator available in the config
    for items at the file-level scope.  So if you wanted nobjects to be
    a random variate, that had not been possible.

* Fixed a bug in config where objects that are considered "safe" (that is,
  unchanging from object to object -- the psf profile, for example) during 
  the processing of a given file were not being correctly invalidated for 
  the next file in a multi-file run if the next file uses a different 
  catalog for instance.

* Support astropy.io.fits in lieu of stand-alone pyfits module. (#488)
  This is where pyfits will live going forward.  So we now support both
  the astropy distribution as well as the legacy pyfits module.

* Fixed a bug in the drawing of a Pixel all by itself.  It had been rendered
  incorrectly, although the bug did not affect the rendering of other profiles
  convolved by a Pixel. (#497)
