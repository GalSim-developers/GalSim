# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from __future__ import print_function
import numpy as np
import os
import sys

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# set up any necessary info for tests
# Note that changes here should match changes to test image files
image_dir = './inclined_exponential_images'

# Values here are strings, so the filenames will be sure to work (without truncating zeros)
image_inc_angles = ("1.3", "0.2", "0.01", "0.1", "0.78")
image_scale_radii = ("3.0", "3.0", "3.0", "2.0", "2.0")
image_scale_heights = ("0.5", "0.5", "0.5", "1.0", "0.5")
image_pos_angles = ("0.0", "0.0", "0.0", "-0.2", "-0.2")
image_nx = 64
image_ny = 64
oversampling = 1.0

@timer
def test_inclined_exponential():
    """Test that the inclined exponential profile matches the results from Lance Miller's code."""
    
    for inc_angle, scale_radius, scale_height, pos_angle in zip(image_inc_angles,image_scale_radii,
                                                                image_scale_heights,image_pos_angles):
        image_filename = "galaxy_"+inc_angle+"_"+scale_radius+"_"+scale_height+"_"+pos_angle+".fits"
        image = galsim.fits.read(image_filename, image_dir)
        
        # Get float values for the details
        inc_angle=float(inc_angle)
        scale_radius=float(scale_radius)/oversampling
        scale_height=float(scale_height)/oversampling
        pos_angle=float(pos_angle)
        
        # Now make a test image
        test_profile = galsim.InclinedExponential(inc_angle*galsim.radians,scale_radius,scale_height,
                                                  gsparams=galsim.GSParams(maximum_fft_size=5000))
        
        # Rotate it by the position angle
        test_profile = test_profile.rotate(pos_angle*galsim.radians)
        
        # Draw it onto an image
        test_image = galsim.Image(image_nx,image_ny,scale=1.0)
        test_profile.drawImage(test_image,offset=(0.5,0.5)) # Offset to match Lance's
        
        # Compare to the example - Due to the different fourier transforms used, some offset is expected,
        # so we just compare in the core to two decimal places
        
        image_core = image.array[image_ny//2-2:image_ny//2+3, image_nx//2-2:image_nx//2+3]
        test_image_core = test_image.array[image_ny//2-2:image_ny//2+3, image_nx//2-2:image_nx//2+3]
        
        ratio_core = image_core / test_image_core
        
        # galsim.fits.write(test_image,"test_"+image_filename,image_dir)
        
        np.testing.assert_array_almost_equal(ratio_core, np.mean(ratio_core)*np.ones_like(ratio_core), decimal = 2,
                                             err_msg = "Error in comparison of inclined exponential profile to samples.",
                                             verbose=True)

# @timer
# def test_ne():
#     """ Check that equality/inequality works as expected."""
#     rgc = galsim.RealGalaxyCatalog(catalog_file, dir=image_dir)
#     gsp = galsim.GSParams(folding_threshold=1.1e-3)
# 
#     gals = [galsim.RealGalaxy(rgc, index=0),
#             galsim.RealGalaxy(rgc, index=1),
#             galsim.RealGalaxy(rgc, index=0, x_interpolant='Linear'),
#             galsim.RealGalaxy(rgc, index=0, k_interpolant='Linear'),
#             galsim.RealGalaxy(rgc, index=0, flux=1.1),
#             galsim.RealGalaxy(rgc, index=0, pad_factor=1.1),
#             galsim.RealGalaxy(rgc, index=0, noise_pad_size=5.0),
#             galsim.RealGalaxy(rgc, index=0, gsparams=gsp)]
#     all_obj_diff(gals)


if __name__ == "__main__":
    test_inclined_exponential()
