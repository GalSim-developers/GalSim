#!/bin/tcsh
#
# series of commands used by Rachel to generate the test images in ./SBProfile_comparison_images/
#
# The script itself includes the path to the original SBProfile executable on her machine, so cannot
# be run by others.  But it is in the repository to serve as a record of how the test images were
# produced, and if others want to run it, they can change the path appropriately for their system.
#
# Important notes:
## not currently testing SBLaguerre, SBPixel classes

#################################
# File and directory definitions#
#################################
set origexec = ~/great3/gary-code/great3/SBDraw
set outdir = ./SBProfile_comparison_images

#########################################################################################
# First set of tests: a single SBProfile of each kind, no operations carried out on them#
#########################################################################################
# Gaussian
$origexec "gauss 1" $outdir/gauss_1.fits 0.2

# Exponential
$origexec "exp 1" $outdir/exp_1.fits 0.2

# Sersic
$origexec "sersic 3 1" $outdir/sersic_3_1.fits 0.2

# Airy
$origexec "airy 0.8 0.1" $outdir/airy_.8_.1.fits 0.2

# Box
$origexec "box 1" $outdir/box_1.fits 0.2

# Moffat
$origexec "moffat 2 5 1" $outdir/moffat_2_5.fits 0.2

#################################################
# Second set of tests: operations on SBProfiles #
#################################################




