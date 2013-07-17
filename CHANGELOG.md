Changes from v0.5 to current version:
------------------------------------

* Fixed a bug in the rendering of shifted images.  (Issue #424)

* Added the offset parameter to the draw and drawShoot commands, and also to the constructor 
  of InterpolatedImage.  (Issue #439)

* Added scale as a constructor parameter for Images.  (Issue #439)

* Improved the ability of Lanczos interpolants to conserve a DC input flux (with the 
  `conserve_dc=True` parameter).  (Issue #442)

* Switched default interpolant for `RealGalaxy` to `Quintic`.  (Issue #442)
