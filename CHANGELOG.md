Changes from v1.3 to v1.4
=========================

Bug Fixes
---------

- Fixed a bug in some of the WCS classes if the RA/Dec axes in the FITS header
  are reversed (which is allowed by the FITS standard). (#681)
- Improved ability of ChromaticObjects to find fiducial achromatic profiles
  and wavelengths with non-zero flux. (#680)
- Fixed a bug in the way Images are instantiated for certain combinations of
  ChromaticObjects and image-setup keyword arguments (#683)
- Added ability to manipulate the width of the moment-measuring weight function
  for the KSB shear estimation method of the galsim.hsm package. (#686)
