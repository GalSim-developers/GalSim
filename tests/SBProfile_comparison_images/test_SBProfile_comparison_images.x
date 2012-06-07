#!/bin/tcsh
#
# series of commands used by Rachel to generate the test images in ./SBProfile_comparison_images/
#
# The script itself includes the path to the original SBProfile executable on her machine, so cannot
# be run by others.  But it is in the repository to serve as a record of how the test images were
# produced, and if others want to run it, they can change the path appropriately for their system.
#
# Important notes:
## not currently testing SBLaguerre, SBInterpolatedImage classes

##################################
# File and directory definitions #
##################################
set origexec = ~/great3/gary-code/great3/SBDraw
set outdir = ./SBProfile_comparison_images

##########################################################################################
# First set of tests: a single SBProfile of each kind, no operations carried out on them #
##########################################################################################

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

# A Gaussian profile with a small shear
$origexec "gauss 1 S 0.02 0.02" $outdir/gauss_smallshear.fits 0.2

# A Sersic profile with a significant shear
$origexec "sersic 4 1 S 0 0.5" $outdir/sersic_largeshear.fits 0.2

# A Moffat convolved with a box
$origexec "(moffat 1.5 4 1) * (box 0.2)" $outdir/moffat_convolve_box.fits 0.2

# A Gaussian profile with a small shear, convolved with a box
$origexec "(gauss 1 S 0.04 0) * (box 0.2)" $outdir/gauss_smallshear_convolve_box.fits 0.2

# A rotated Sersic profile
$origexec "sersic 2.5 1 S 0.2 0 R 45.0" $outdir/sersic_ellip_rotated.fits 0.2

# A magnified exponential profile
$origexec "exp 1 D 1.5" $outdir/exp_mag.fits 0.2

# Adding two profiles together
$origexec "(gauss 1 F 0.75) + (gauss 3 F 0.25)" $outdir/double_gaussian.fits 0.2

# Shifting a box profile (translation)
$origexec "box 0.2 T 0.2 -0.2" $outdir/box_shift.fits 0.2

# Double the flux of what was initially a Sersic profile with unit flux
$origexec "sersic 3 1 F 2" $outdir/sersic_doubleflux.fits 0.2

