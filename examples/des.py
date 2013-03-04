# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""
Draw DES PSFs at the locations of observed galaxies.

This demo probably isn't so useful as an actual program, but it does showcase how to 
use the DES module that comes with GalSim, which can be modified to do what you actually 
need. 

It works on a full DES exposure with 62 chip images and the files that are output by the 
DESDM and WL pipelines.  The DESDM pipeline produces a catalog of detected objects for each
image, and also an interpolated PSF using Emmanuel Bertin's PSFEx code, which are stored in 
*_psfcat.psf files.  The WL pipeline produces a different estimate of the PSF using Mike Jarvis's
shapelet code, which are stored in *_fitpsf.fits files.

This script reads the appropriate files for each chip and builds two images, one for each kind
of PSF estimate, and draws an image of the PSF at the location of each galaxy.  Normally, you
would probably want to draw these with no noise on individual postage stamps or something like
that.
"""

import galsim
import os
import math

def main(argv):
    # For the file names, I pick a particular exposure.  The directory structure corresponds 
    # to where the files are stored on folio at UPenn.

    #img_dir = '/data3/DECAM/SV/DECam_154912'
    #wl_dir = '/data3/DECAM/wl/DECam_00154912_wl'
    #img_dir = '/Users/Mike/Astro/des/SV/DECam_00154912_wl'
    #wl_dir = '/Users/Mike/Astro/des/SV/DECam_00154912_wl'
    img_dir = 'des_data'
    wl_dir = 'des_data'
    root = 'DECam_00154912' 
    out_dir = 'output'

    #nchips = 62
    nchips = 1
    pixel_scale = 0.264      # arcsec / pixel
    sky_level = 16000        # ADU/arcsec^2

    # The random seed, so the results are deterministic
    random_seed = 1339201           

    x_col = 'X_IMAGE'
    y_col = 'Y_IMAGE'
    flux_col = 'FLUX_AUTO'

    # Make output directory if not already present.
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for chipnum in range(1,nchips+1):
        print 'Start chip ',chipnum

        image_file = '%s_%02d.fits.fz'%(root,chipnum)
        cat_file = '%s_%02d_cat.fits'%(root,chipnum)
        psfex_file = '%s_%02d_psfcat.psf'%(root,chipnum)
        fitpsf_file = '%s_%02d_fitpsf.fits'%(root,chipnum)
        fitpsf_image_file = '%s_%02d_fitpsf_image.fits'%(root,chipnum)
    
        image = galsim.fits.read(image_file, dir=img_dir)
        cat = galsim.InputCatalog(cat_file, hdu=2, dir=img_dir)
        #psfex = galsim.des.DES_PSFEx(psfex_file)
        fitpsf = galsim.des.DES_Shapelet(fitpsf_file, dir=wl_dir)

        # Setup the images:
        fitpsf_image = galsim.ImageF(image.bounds)
        fitpsf_image.scale = pixel_scale

        nobj = cat.nobjects
        print 'Catalog has ',nobj,' objects'

        for k in range(nobj):
            # The usual random number generator using a different seed for each galaxy.
            # I'm not actually using the rng for object creation (everything comes from # the 
            # input files), but the rng that matches the config version is here just in case.
            rng = galsim.BaseDeviate(random_seed+k)

            # Get the position from the galaxy catalog
            x = cat.getFloat(k,x_col)
            y = cat.getFloat(k,y_col)
            ix = int(math.floor(x+0.5))
            iy = int(math.floor(y+0.5))
            dx = x-ix
            dy = y-iy
            image_pos = galsim.PositionD(x,y)
            print 'pos = ',image_pos

            # Also get the flux of the galaxy from the catalog
            flux = cat.getFloat(k,flux_col)

            # Define the pixel
            pix = galsim.Pixel(pixel_scale)

            if not fitpsf.bounds.includes(image_pos):
                print '...not in fitpsf.bounds'
                continue

            # Define the PSF profile
            psf = fitpsf.getPSF(image_pos)
            psf.setFlux(flux)

            # Make the final image, convolving with pix
            final = galsim.Convolve([pix,psf])

            # Apply partial-pixel shift 
            final.applyShift(dx*pixel_scale,dy*pixel_scale)

            # Draw the postage stamp image
            stamp = final.draw(dx=pixel_scale)[0]

            # Recenter the stamp at the desired position:
            stamp.setCenter(ix,iy)

            # Find overlapping bounds
            bounds = stamp.bounds & fitpsf_image.bounds
            fitpsf_image[bounds] += stamp[bounds]

        rng = galsim.BaseDeviate(random_seed+nobj)
        sky_level_pixel = sky_level * pixel_scale**2
        noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
        fitpsf_image.addNoise(noise)

        # Now write the images to disk.
        fitpsf_image.write(fitpsf_image_file, dir=out_dir)

if __name__ == "__main__":
    import sys
    main(sys.argv)
