Below are the most significant differences between the current version of GalSim on the master
branch of the GitHub repository, and the last tagged version (v0.2):

* Several bug fixes in the Fourier space parameters of the Sersic surface brightness profile, which
  improves some issues with ringing in images composed of Sersic profiles on their own or combined
  with other profiles.

* Fixed several sources of memory leaks, with the most significant being in the moments and shape
  estimation software.
