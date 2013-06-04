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
import numpy as np
import os
import sys
import pyfits

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# set up any necessary info for tests
### Note: changes to either of the tests below might require regeneration of the catalog and image
### files that are saved here.  Modify with care!!!
image_dir = './real_comparison_images'
catalog_file = os.path.join(image_dir,'test_catalog.fits')

ind_fake = 1 # index of mock galaxy (Gaussian) in catalog
fake_gal_fwhm = 0.7 # arcsec
fake_gal_shear1 = 0.29 # shear representing intrinsic shape component 1
fake_gal_shear2 = -0.21 # shear representing intrinsic shape component 2; note non-round, to detect
              # possible issues with x<->y or others that might not show up using circular galaxy
fake_gal_flux = 1000.0
fake_gal_orig_PSF_fwhm = 0.1 # arcsec
fake_gal_orig_PSF_shear1 = 0.0
fake_gal_orig_PSF_shear2 = -0.07

targ_pixel_scale = [0.18, 0.25] # arcsec
targ_PSF_fwhm = [0.7, 1.0] # arcsec
targ_PSF_shear1 = [-0.03, 0.0]
targ_PSF_shear2 = [0.05, -0.08]
targ_applied_shear1 = 0.06
targ_applied_shear2 = -0.04

sigma_to_fwhm = 2.0*np.sqrt(2.0*np.log(2.0)) # multiply sigma by this to get FWHM for Gaussian
fwhm_to_sigma = 1.0/sigma_to_fwhm

ind_real = 0 # index of real galaxy in catalog
shera_file = 'real_comparison_images/shera_result.fits'
shera_target_PSF_file = 'real_comparison_images/shera_target_PSF.fits'
shera_target_pixel_scale = 0.24
shera_target_flux = 1000.0

# some helper functions
def ellip_to_moments(e1, e2, sigma):
    a_val = (1.0 + e1) / (1.0 - e1)
    b_val = np.sqrt(a_val - (0.5*(1.0+a_val)*e2)**2)
    mxx = a_val * (sigma**2) / b_val
    myy = (sigma**2) / b_val
    mxy = 0.5 * e2 * (mxx + myy)
    return mxx, myy, mxy

def moments_to_ellip(mxx, myy, mxy):
    e1 = (mxx - myy) / (mxx + myy)
    e2 = 2*mxy / (mxx + myy)
    sig = (mxx*myy - mxy**2)**(0.25)
    return e1, e2, sig

def test_real_galaxy_ideal():
    """Test accuracy of various calculations with fake Gaussian RealGalaxy vs. ideal expectations"""
    import time
    t1 = time.time()
    # read in faked Gaussian RealGalaxy from file
    rgc = galsim.RealGalaxyCatalog(catalog_file, image_dir)
    rg = galsim.RealGalaxy(rgc, index = ind_fake)

    ## for the generation of the ideal right answer, we need to add the intrinsic shape of the
    ## galaxy and the lensing shear using the rule for addition of distortions which is ugly, but oh
    ## well:
    (d1, d2) = galsim.utilities.g1g2_to_e1e2(fake_gal_shear1, fake_gal_shear2)
    (d1app, d2app) = galsim.utilities.g1g2_to_e1e2(targ_applied_shear1, targ_applied_shear2)
    denom = 1.0 + d1*d1app + d2*d2app
    dapp_sq = d1app**2 + d2app**2
    d1tot = (d1 + d1app + d2app/dapp_sq*(1.0 - np.sqrt(1.0-dapp_sq))*(d2*d1app - d1*d2app))/denom
    d2tot = (d2 + d2app + d1app/dapp_sq*(1.0 - np.sqrt(1.0-dapp_sq))*(d1*d2app - d2*d1app))/denom

    # convolve with a range of Gaussians, with and without shear (note, for this test all the
    # original and target ePSFs are Gaussian - there's no separate pixel response so that everything
    # can be calculated analytically)
    for tps in targ_pixel_scale:
        for tpf in targ_PSF_fwhm:
            for tps1 in targ_PSF_shear1:
                for tps2 in targ_PSF_shear2:
                    print 'tps,tpf,tps1,tps2 = ',tps,tpf,tps1,tps2
                    # make target PSF
                    targ_PSF = galsim.Gaussian(fwhm = tpf)
                    targ_PSF.applyShear(g1=tps1, g2=tps2)
                    # simulate image
                    sim_image = galsim.simReal(
                            rg, targ_PSF, tps, 
                            g1 = targ_applied_shear1, g2 = targ_applied_shear2,
                            rand_rotate = False, target_flux = fake_gal_flux)
                    # galaxy sigma, in units of pixels on the final image
                    sigma_ideal = (fake_gal_fwhm/tps)*fwhm_to_sigma
                    # compute analytically the expected galaxy moments:
                    mxx_gal, myy_gal, mxy_gal = ellip_to_moments(d1tot, d2tot, sigma_ideal)
                    # compute analytically the expected PSF moments:
                    targ_PSF_e1, targ_PSF_e2 = galsim.utilities.g1g2_to_e1e2(tps1, tps2)
                    targ_PSF_sigma = (tpf/tps)*fwhm_to_sigma
                    mxx_PSF, myy_PSF, mxy_PSF = ellip_to_moments(
                            targ_PSF_e1, targ_PSF_e2, targ_PSF_sigma)
                    # get expected e1, e2, sigma for the PSF-convolved image
                    tot_e1, tot_e2, tot_sigma = moments_to_ellip(
                            mxx_gal+mxx_PSF, myy_gal+myy_PSF, mxy_gal+mxy_PSF)

                    # compare with images that are expected
                    expected_gaussian = galsim.Gaussian(
                            flux = fake_gal_flux, sigma = tps*tot_sigma)
                    expected_gaussian.applyShear(e1 = tot_e1, e2 = tot_e2)
                    expected_image = galsim.ImageD(
                            sim_image.array.shape[0], sim_image.array.shape[1])
                    expected_gaussian.draw(expected_image, dx = tps)
                    printval(expected_image,sim_image)
                    np.testing.assert_array_almost_equal(
                        sim_image.array, expected_image.array, decimal = 3,
                        err_msg = "Error in comparison of ideal Gaussian RealGalaxy calculations")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_real_galaxy_saved():
    """Test accuracy of various calculations with real RealGalaxy vs. stored SHERA result"""
    import time
    t1 = time.time()
    # read in real RealGalaxy from file
    rgc = galsim.RealGalaxyCatalog(catalog_file, image_dir)
    rg = galsim.RealGalaxy(rgc, index = ind_real)

    # read in expected result for some shear
    shera_image = galsim.fits.read(shera_file)
    shera_target_PSF_image = galsim.fits.read(shera_target_PSF_file)

    # simulate the same galaxy with GalSim
    sim_image = galsim.simReal(rg, shera_target_PSF_image, shera_target_pixel_scale,
                               g1 = targ_applied_shear1, g2 = targ_applied_shear2,
                               rand_rotate = False, target_flux = shera_target_flux)

    # there are centroid issues when comparing Shera vs. SBProfile outputs, so compare 2nd moments
    # instead of images
    sbp_res = sim_image.FindAdaptiveMom()
    shera_res = shera_image.FindAdaptiveMom()

    np.testing.assert_almost_equal(sbp_res.observed_shape.e1,
                                   shera_res.observed_shape.e1, 2,
                                   err_msg = "Error in comparison with SHERA result: e1")
    np.testing.assert_almost_equal(sbp_res.observed_shape.e2,
                                   shera_res.observed_shape.e2, 2,
                                   err_msg = "Error in comparison with SHERA result: e2")
    np.testing.assert_almost_equal(sbp_res.moments_sigma, shera_res.moments_sigma, 2,
                                   err_msg = "Error in comparison with SHERA result: sigma")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_real_galaxy_ideal()
    test_real_galaxy_saved()
