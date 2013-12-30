Changes from v1.0 to v1.1:
--------------------------

Updates to config options:
* Added a new image.retry_failures item that can be set so that if the 
  construction of a GSObject fails for any reason, you can ask it to retry.
  An example of this functionality has been added to demo8. (Issue #482)
* Added a new output.retry_io item that can be set so that if the output write 
  command fails (due to hard drive overloading for example), then it will wait 
  a second and try again. (Issue #482)
* Changed the sequence indexing within an image to always start at 0, rather 
  than use obj_num (which continues increasing through all objects in the run).
  Functionally, this would usually only matter if the number of objects per
  file or image is not a constant.  If the number of objects is constant, the 
  automatic looping of the sequencing index essentially does this for you.
  (Issue #487)
* Added Sum type for value types for which it makes sense: float, int, angle,
  shear, position. (Issue #457)
* Allowed the user to modify or add config parameters from the command line. 
  (Issue #479)
