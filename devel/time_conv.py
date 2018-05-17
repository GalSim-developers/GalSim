# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import galsim
import cProfile, pstats
import numpy as np

def main():

    # Using very low accuracy GSParams here for speed
    gsparams = galsim.GSParams(minimum_fft_size=256,
                               folding_threshold=0.1,
                               kvalue_accuracy=1e-3,
                               stepk_minimum_hlr=2.5,)

    # Note - we actually use an interpolated image instead; just putting this in
    # so you can run the code without needing that file
    psf_prof = galsim.OpticalPSF(lam=725, # nm
                                 diam=1.2, # m
                                 defocus=0,
                                 obscuration=0.33,
                                 nstruts=3,
                                 gsparams=gsparams)

    pixel_scale = 0.02
    convolved_image = galsim.Image(256,256, scale=pixel_scale)
    convolved_image.setCenter(0,0)

    # Do this once here to get the right kimage size/shape and wrap size.
    gal_prof = galsim.Sersic(n=4, half_light_radius=0.3, gsparams=gsparams)
    convolved_prof = galsim.Convolve(gal_prof, psf_prof, gsparams=gsparams)

    psf_kimage, wrap_size = convolved_prof.drawFFT_makeKImage(convolved_image)

    # Draw the PSF onto the kimage.
    psf_prof._drawKImage(psf_kimage)

    # Use the same size/shape for the galaxy part.
    gal_kimage = psf_kimage.copy()
    convolved_image2 = convolved_image.copy()

    for _i in range(1000):

        gal_prof = galsim.Sersic(n=4,half_light_radius=0.3,gsparams=gsparams)

        # Account for the fact that this is an even sized image.  The drawFFT function will
        # draw the profile centered on the nominal (integer) center pixel, which (since this is
        # an even-sized image) actuall +0.5,+0.5 from the true center.
        gal_prof_cen = gal_prof._shift(galsim.PositionD(-0.5*pixel_scale, -0.5*pixel_scale))

        # Draw just the galaxy profile in k-space
        gal_prof_cen._drawKImage(gal_kimage)

        # Multiply by the (constant) PSF kimage
        gal_kimage.array[:,:] *= psf_kimage.array

        # Finish the draw process
        gal_prof.drawFFT_finish(convolved_image, gal_kimage, wrap_size, add_to_image=False)

        if False:
            # Check that we get the same thing as the normal draw procedure
            convolved_prof = galsim.Convolve(gal_prof,psf_prof,gsparams=gsparams)
            # Using no pixel method here since we plan to use a PSF profile
            # which already includes the pixel response
            convolved_prof.drawImage(convolved_image2, method='no_pixel')
            max_diff = np.max(np.abs(convolved_image.array - convolved_image2.array))
            print('max diff = ',max_diff)
            assert(max_diff < 1.e-8)


if __name__ == "__main__":
    #cProfile.runctx("main()",{},{"main":main},filename="convolve_time_test.prof")
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('time')
    ps.print_stats(20)
