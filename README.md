@mainpage

# GalSim: The modular galaxy image simulation toolkit

For installation instructions, please see the file INSTALL.md in the main
repository directory. 

There are tagged versions of the code corresponding to specific project 
milestones and releases.  

If interested in code from before v0.1, the tagged milestone versions may be the
best ones to try if you want a stable version with specific capabilities.  

(For more info on tags see the "Milestone tags" Section below and docs/git.txt)


How to get started
------------------

1. We assume that all steps in `INSTALL.md` are complete.

2. Optional, but recommended whenever you try a new version of the code: run the
unit tests to make sure that there are no errors.  You can do this by running
`scons tests`.

3. Optional: run `doxygen` to generate documentation, using `Doxyfile` in the
main repository directory to specify all doxygen settings.  You can 
alternatively view the documentation on GitHub at 
http://galsim-developers.github.com/GalSim/


Demonstration scripts
---------------------

Once these steps are completed, there are a number of scripts in `examples/` 
that demonstrate how the code can be used.  These labelled `demo1.py`-`demo8.py`

As the project develops through further versions, and adds further
capabilities to the software, further demo scripts will be added to `examples/`
to illustrate and showcase what GalSim can do.


Reference documentation
-----------------------

For an overview of GalSim workflow and tools, please see 
`doc/GalSim_Quick_Reference.pdf` in the GalSim repository.

For the Python GSObject classes used to describe astronomical objects, see 
galsim.base in the doxygen documentation.


Additional scripts
------------------

While the demo scripts can be run from the command-line while sitting in
`examples/` without any arguments, the remaining scripts are auxiliary utilities
that take various command-line arguments, which are always explained in comments
at the top of the file.

* ShootInterpolated.py is a script that takes as input a filename for a FITS
image, which it will simulate (optionally sheared and/or resampled) via
photon-shooting.

* MeasMoments.py can be used to measure the adaptive moments (best-fit
elliptical Gaussian) for a FITS image.

* MeasShape.py can be used to carry out PSF correction using one of four
methods, given FITS images of the galaxy and PSF.


Milestone tags
--------------

After every GalSim general milestone we tagged a snapshot of the code at that 
moment, with the tag name `milestoneN` where N is the milestone number.

You can see the available tags using the command `git tag -l` at terminal from 
within the repository.

The version of the code at any given milestone can then be checked out using the
tag name, e.g.:

    $ git checkout milestone2

This will then update your directory tree to the snapshot of the code at the 
milestone requested.  (You will also get a message about being in a "detached" 
HEAD state.  That is normal.)
