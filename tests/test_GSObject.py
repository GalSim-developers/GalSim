import os
import sys
import numpy as np
# import galsim even if path not yet added to PYTHONPATH env variable (e.g. by full install)
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

imgdir = os.path.join(".", "SBProfile_comparison_images")

# Test values taken from test_SBProfile.py... and modified slightly.
# for radius tests - specify half-light-radius, FHWM, sigma to be compared with high-res image (with
# pixel scale chosen iteratively until convergence is achieved, beginning with test_dx)
test_hlr = 1.9
test_fwhm = 1.9
test_sigma = 1.9
test_scale = 1.9
test_sersic_n = [1.4, 2.6]
test_lor0 = 1.9

# moffat params and reference values
test_beta = 2.5
test_trunc = 13.

moffat_ref_fwhm_from_scale = 2.1479511706648715 # test_scale * 2 sqrt(2**(1 / test_beta) - 1)
moffat_ref_hlr_from_scale = 1.4522368913645236  # calculated from SBProfile (regression test only)

moffat_ref_scale_from_fwhm = 1.680671352916542 # test_scale /( 2 sqrt(2**(1 / test_beta) - 1) )
moffat_ref_hlr_from_fwhm = 1.285657994217926   # calculated from SBProfile (regression test only)

moffat_ref_scale_from_hlr = 2.494044174293422  # calculated from SBProfile (regression test only)
moffat_ref_fwhm_from_hlr = 2.8195184757176097 # calculated from SBProfile (regression test only)

# atmospheric params and reference values
test_oversampling = 1.7

atmos_ref_fwhm_from_lor0 = test_lor0 * 0.976
atmos_ref_lor0_from_fwhm = test_fwhm / 0.976

 
# for flux normalization tests
test_flux = 1.9

# decimal point to go to for parameter value comparisons
param_decimal = 15

def test_gaussian_param_consistency():
    # init with sigma and flux
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma, test_sigma, decimal=param_decimal,
        err_msg="Starting sigma param and derived attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_sigma * np.sqrt(2. * np.log(2.)), decimal=param_decimal,
        err_msg="Starting sigma param and derived half_light_radius attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, test_sigma * 2. * np.sqrt(2. * np.log(2.)), decimal=param_decimal,
        err_msg="Starting sigma param and derived fwhm attribute inconsistent.")

    # init with fwhm and flux
    obj = galsim.Gaussian(fwhm=test_fwhm, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, test_fwhm, decimal=param_decimal,
        err_msg="Starting fwhm param and derived attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_fwhm / 2., decimal=param_decimal,
        err_msg="Starting fwhm param and derived half_light_radius attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma, test_fwhm /  (2. * np.sqrt(2. * np.log(2.))), decimal=param_decimal,
        err_msg="Starting fwhm param and derived sigma attribute inconsistent.")

    # init with hlr and flux
    obj = galsim.Gaussian(half_light_radius=test_hlr, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_hlr, decimal=param_decimal,
        err_msg="Starting half_light_radius param and derived attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, test_hlr * 2., decimal=param_decimal,
        err_msg="Starting half_light_radius param and derived fwhm attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma, test_hlr /  np.sqrt(2. * np.log(2.)), decimal=param_decimal,
        err_msg="Starting half_light_radius param and derived sigma attribute inconsistent.")

def test_moffat_param_consistency():
    # init with scale_radius and flux
    obj = galsim.Moffat(
        scale_radius=test_scale, flux=test_flux, trunc=test_trunc, beta=test_beta)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.trunc, test_trunc, decimal=param_decimal,
        err_msg="Trunc param and trunc attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.beta, test_beta, decimal=param_decimal,
        err_msg="Beta param and beta attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.scale_radius, test_scale, decimal=param_decimal,
        err_msg="Starting scale_radius param and derived attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, moffat_ref_hlr_from_scale, decimal=param_decimal,
        err_msg="Starting scale_radius param and derived half_light_radius attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, moffat_ref_fwhm_from_scale, decimal=param_decimal,
        err_msg="Starting scale_radius param and derived fwhm attribute inconsistent.")

    # init with fwhm and flux
    obj = galsim.Moffat(
        fwhm=test_fwhm, flux=test_flux, trunc=test_trunc, beta=test_beta)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.trunc, test_trunc, decimal=param_decimal,
        err_msg="Trunc param and trunc attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.beta, test_beta, decimal=param_decimal,
        err_msg="Beta param and beta attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, test_fwhm, decimal=param_decimal,
        err_msg="Starting fwhm param and derived attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, moffat_ref_hlr_from_fwhm, decimal=param_decimal,
        err_msg="Starting fwhm param and derived half_light_radius attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.scale_radius, moffat_ref_scale_from_fwhm, decimal=param_decimal,
        err_msg="Starting fwhm param and derived scale_radius attribute inconsistent.")

    # init with hlr and flux
    obj = galsim.Moffat(
        half_light_radius=test_hlr, flux=test_flux, trunc=test_trunc, beta=test_beta)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.trunc, test_trunc, decimal=param_decimal,
        err_msg="Trunc param and trunc attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.beta, test_beta, decimal=param_decimal,
        err_msg="Beta param and beta attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_hlr, decimal=param_decimal,
        err_msg="Starting half_light_radius param and derived attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, moffat_ref_fwhm_from_hlr, decimal=param_decimal,
        err_msg="Starting half_light_radius param and derived fwhm attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.scale_radius, moffat_ref_scale_from_hlr, decimal=param_decimal,
        err_msg="Starting half_light_radius param and derived scale_radius attribute inconsistent.")

def test_atmos_param_consistency():
    # init with lam_over_r0 and flux
    obj = galsim.AtmosphericPSF(
        lam_over_r0=test_lor0, flux=test_flux, oversampling=test_oversampling)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.oversampling, test_oversampling, decimal=param_decimal,
        err_msg="Oversampling param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_r0, test_lor0, decimal=param_decimal,
        err_msg="Lambda / r0 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, atmos_ref_fwhm_from_lor0, decimal=param_decimal,
        err_msg="Starting lambda / r0 param and derived fwhm attribute inconsistent.")

    # init with FWHM and flux
    obj = galsim.AtmosphericPSF(fwhm=test_fwhm, flux=test_flux, oversampling=test_oversampling)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.oversampling, test_oversampling, decimal=param_decimal,
        err_msg="Oversampling param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_r0, atmos_ref_lor0_from_fwhm, decimal=param_decimal,
        err_msg="Starting FWHM param and derived lambda / r0 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, test_fwhm, decimal=param_decimal,
        err_msg="FWHM param and attribute inconsistent.")

    
    

