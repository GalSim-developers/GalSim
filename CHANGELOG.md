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
