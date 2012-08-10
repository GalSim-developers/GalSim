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

# for flux normalization tests
test_flux = 1.9

# Moffat params and reference values
test_beta = 2.5
test_trunc = 13.

moffat_ref_fwhm_from_scale = 2.1479511706648715 # test_scale * 2 sqrt(2**(1 / test_beta) - 1)
moffat_ref_hlr_from_scale = 1.4522368913645236  # calculated from SBProfile (regression test only)

moffat_ref_scale_from_fwhm = 1.680671352916542 # test_scale /( 2 sqrt(2**(1 / test_beta) - 1) )
moffat_ref_hlr_from_fwhm = 1.285657994217926   # calculated from SBProfile (regression test only)

moffat_ref_scale_from_hlr = 2.494044174293422  # calculated from SBProfile (regression test only)
moffat_ref_fwhm_from_hlr = 2.8195184757176097 # calculated from SBProfile (regression test only)

# AtmosphericPSF params and reference values
test_lor0 = 1.9
test_oversampling = 1.7

atmos_ref_fwhm_from_lor0 = test_lor0 * 0.976
atmos_ref_lor0_from_fwhm = test_fwhm / 0.976

# Airy params and reference values
test_loD = 1.9
test_obscuration = 0.32

airy_ref_hlr_from_loD = test_loD * 0.5348321477242647
airy_ref_fwhm_from_loD = test_loD * 1.028993969962188

airy_ref_loD_from_hlr = test_hlr / 0.5348321477242647
airy_ref_fwhm_from_hlr = test_hlr * 1.028993969962188 / 0.5348321477242647

airy_ref_hlr_from_fwhm = test_fwhm * 0.5348321477242647 / 1.028993969962188
airy_ref_loD_from_fwhm = test_fwhm / 1.028993969962188

# OpticalPSF test params (only a selection)
test_defocus = -0.7
test_astig1 = 0.03
test_astig2 = -0.04

# Exponential reference values
exponential_ref_hlr_from_scale = test_scale * 1.6783469900166605
exponential_ref_scale_from_hlr = test_hlr / 1.6783469900166605

# DoubleGaussian test params
test_sigma1 = test_sigma
test_sigma2 = test_sigma * 1.3
test_fwhm1 = test_fwhm
test_fwhm2 = test_fwhm * 1.3
test_hlr1 = test_hlr
test_hlr2 = test_hlr * 1.3
test_flux1 = test_flux
test_flux2 = 0.7 * test_flux

# decimal point to go to for parameter value comparisons
param_decimal = 13


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

def test_airy_param_consistency():
    # init with lam_over_D and flux with obs = 0.
    obj = galsim.Airy(lam_over_D=test_loD, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_D, test_loD, decimal=param_decimal,
        err_msg="Lambda / D param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, airy_ref_hlr_from_loD, decimal=param_decimal,
        err_msg="Starting Lambda / D param and derived half_light_radius attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, airy_ref_fwhm_from_loD, decimal=param_decimal,
        err_msg="Starting Lambda / D param and derived FWHM attribute inconsistent.")

    # init with HLR and flux with obs = 0.
    obj = galsim.Airy(half_light_radius=test_hlr, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_hlr, decimal=param_decimal,
        err_msg="Half light radius param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_D, airy_ref_loD_from_hlr, decimal=param_decimal,
        err_msg="Starting half_light_radius param and derived Lambda / D attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, airy_ref_fwhm_from_hlr, decimal=param_decimal,
        err_msg="Starting half_light_radius param and derived FWHM attribute inconsistent.")

    # init with FWHM and flux with obs = 0.
    obj = galsim.Airy(fwhm=test_fwhm, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, test_fwhm, decimal=param_decimal,
        err_msg="FWHM param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_D, airy_ref_loD_from_fwhm, decimal=param_decimal,
        err_msg="Starting FWHM param and derived Lambda / D attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, airy_ref_hlr_from_fwhm, decimal=param_decimal,
        err_msg="Starting FWHM param and derived half_light_radius attribute inconsistent.")

    # Now test with obscuration !=0, only basic tests possible at present
    # init with lam_over_D and flux with obs = test_obscuration
    obj = galsim.Airy(lam_over_D=test_loD, flux=test_flux, obscuration=test_obscuration)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_D, test_loD, decimal=param_decimal,
        err_msg="Lambda / D param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.obscuration, test_obscuration, decimal=param_decimal,
        err_msg="Obscuration param and attribute inconsistent.")

def test_opticalpsf_param_consistency():
    # init with a few test params then do very simple tests
    obj = galsim.OpticalPSF(
        lam_over_D=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_D, test_loD, decimal=param_decimal,
        err_msg="Lambda / D param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.defocus, test_defocus, decimal=param_decimal,
        err_msg="Defocus param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.astig1, test_astig1, decimal=param_decimal,
        err_msg="Astig1 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.astig2, test_astig2, decimal=param_decimal,
        err_msg="Astig2 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.oversampling, test_oversampling, decimal=param_decimal,
        err_msg="Oversampling param and attribute inconsistent.")

def test_sersic_param_consistency():
    # loop through sersic n
    for test_n in test_sersic_n:
        # init with n, scale_radius and flux
        obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux)
        np.testing.assert_almost_equal(
            obj.n, test_n, decimal=param_decimal,
            err_msg="Sersic n param and attribute inconsistent.")
        np.testing.assert_almost_equal(
            obj.half_light_radius, test_hlr, decimal=param_decimal,
            err_msg="Sersic half_light_radius param and attribute inconsistent.")
        np.testing.assert_almost_equal(
            obj.flux, test_flux, decimal=param_decimal,
            err_msg="Flux param and attribute inconsistent.")

def test_exponential_param_consistency():
    # init with scale_radius and flux
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.scale_radius, test_scale, decimal=param_decimal,
        err_msg="Exponential scale_radius param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, exponential_ref_hlr_from_scale, decimal=param_decimal,
        err_msg="Exponential scale_radius param and derived half_light_radius attribute "+
        "inconsistent.")
    # init with half_light_radius and flux
    obj = galsim.Exponential(half_light_radius=test_hlr, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_hlr, decimal=param_decimal,
        err_msg="Exponential half_light_radius param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.scale_radius, exponential_ref_scale_from_hlr, decimal=param_decimal,
        err_msg="Exponential half_light_radius param and derived scale_radius attribute "+
        "inconsistent.")

def test_devaucouleurs_param_consistency():
    # init with half_light_radius and flux
    obj = galsim.Exponential(half_light_radius=test_hlr, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_hlr, decimal=param_decimal,
        err_msg="DeVaucouleurs half_light_radius param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")

def test_doublegaussian_param_consistency():
    # init with sigma1, sigma2 and flux1, flux2
    obj = galsim.DoubleGaussian(
        sigma1=test_sigma1, sigma2=test_sigma2, flux1=test_flux1, flux2=test_flux2)
    np.testing.assert_almost_equal(
        obj.flux1, test_flux1, decimal=param_decimal,
        err_msg="Flux1 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.flux2, test_flux2, decimal=param_decimal,
        err_msg="Flux2 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma1, test_sigma1, decimal=param_decimal,
        err_msg="Starting sigma1 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma2, test_sigma2, decimal=param_decimal,
        err_msg="Starting sigma2 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm1, test_sigma1 * 2. * np.sqrt(2. * np.log(2.)), decimal=param_decimal,
        err_msg="Starting sigma1 param and derived fwhm1 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm2, test_sigma2 * 2. * np.sqrt(2. * np.log(2.)), decimal=param_decimal,
        err_msg="Starting sigma2 param and derived fwhm2 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius1, test_sigma1 * np.sqrt(2. * np.log(2.)), decimal=param_decimal,
        err_msg="Starting sigma1 param and derived half_light_radius1 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius2, test_sigma2 * np.sqrt(2. * np.log(2.)), decimal=param_decimal,
        err_msg="Starting sigma2 param and derived half_light_radius2 attribute inconsistent.")

    # init with fwhm1, fwhm2 and flux1, flux2
    obj = galsim.DoubleGaussian(
        fwhm1=test_fwhm1, fwhm2=test_fwhm2, flux1=test_flux1, flux2=test_flux2)
    np.testing.assert_almost_equal(
        obj.flux1, test_flux1, decimal=param_decimal,
        err_msg="Flux1 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.flux2, test_flux2, decimal=param_decimal,
        err_msg="Flux2 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma1, test_fwhm1 / (2. * np.sqrt(2. * np.log(2.))), decimal=param_decimal,
        err_msg="Starting fwhm1 param and derived sigma1 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma2, test_fwhm2 / (2. * np.sqrt(2. * np.log(2.))), decimal=param_decimal,
        err_msg="Starting fwhm2 param and derived sigma 2attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm1, test_fwhm1, decimal=param_decimal,
        err_msg="Starting fwhm1 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm2, test_fwhm2, decimal=param_decimal,
        err_msg="Starting fwhm2 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius1, test_fwhm1 / 2., decimal=param_decimal,
        err_msg="Starting fwhm1 param and derived half_light_radius1 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius2, test_fwhm2 / 2., decimal=param_decimal,
        err_msg="Starting fwhm2 param and derived half_light_radius2 attribute inconsistent.")

        # init with fwhm1, fwhm2 and flux1, flux2
    obj = galsim.DoubleGaussian(
        half_light_radius1=test_hlr1, half_light_radius2=test_hlr2, flux1=test_flux1,
        flux2=test_flux2)
    np.testing.assert_almost_equal(
        obj.flux1, test_flux1, decimal=param_decimal,
        err_msg="Flux1 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.flux2, test_flux2, decimal=param_decimal,
        err_msg="Flux2 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma1, test_hlr1 / (np.sqrt(2. * np.log(2.))), decimal=param_decimal,
        err_msg="Starting half_light_radius1 param and derived sigma1 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.sigma2, test_hlr2 / (np.sqrt(2. * np.log(2.))), decimal=param_decimal,
        err_msg="Starting half_light_radius2 param and derived sigma2 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius1, test_hlr1, decimal=param_decimal,
        err_msg="Starting half_light_radius1 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius2, test_hlr2, decimal=param_decimal,
        err_msg="Starting half_light_radius2 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm1, test_hlr1 * 2., decimal=param_decimal,
        err_msg="Starting half_light_radius1 param and derived fwhm1 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm2, test_hlr2 * 2., decimal=param_decimal,
        err_msg="Starting half_light_radius2 param and derived fwhm2 attribute inconsistent.")

def test_add_param_consistency():
    # only really the flux param to test, but will do this with a couple of difference sorts of
    # GSObjects in the Add just to check, and for list and dual argument initialization
    # Sersic and Exp first
    obj1 = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = galsim.Sersic(n=3.5, half_light_radius=test_hlr, flux=2. * test_flux)
    # test dual argument initializtion
    add = galsim.Add(obj1, obj2)
    np.testing.assert_almost_equal(
        add.flux, 3. * test_flux, decimal=param_decimal,
        err_msg="Add total flux for dual argument initialization inconsistent with Sersic and Exp "+
        "inputs.")
    # test 2-entry list initialization
    add = galsim.Add([obj1, obj2])
    np.testing.assert_almost_equal(
        add.flux, 3. * test_flux, decimal=param_decimal,
        err_msg="Add total flux for double-entry list initialization inconsistent with Sersic and "+
        "Exp inputs.")
    # make a third object, dVc
    obj3 = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux = 3. * test_flux)
    # test 3-entry list initialization
    add = galsim.Add([obj1, obj2, obj3])
    np.testing.assert_almost_equal(
        add.flux, 6. * test_flux, decimal=param_decimal,
        err_msg="Add total flux for triple-entry list initialization inconsistent with Sersic and "+
        "Exp and DeVauc inputs.")

    # Then try a couple of SBInterpolatedImage-type classes
    obj1 = galsim.OpticalPSF(
        lam_over_D=test_loD, astig1=test_astig1, defocus=test_defocus, astig2=test_astig2,
        flux=test_flux, oversampling=test_oversampling)
    obj2 = galsim.AtmosphericPSF(
        lam_over_r0=test_lor0, flux=2. * test_flux, oversampling=test_oversampling)
    # test dual argument initializtion
    add = galsim.Add(obj1, obj2)
    np.testing.assert_almost_equal(
        add.flux, 3. * test_flux, decimal=param_decimal,
        err_msg="Add total flux for dual argument initialization inconsistent with OpticalPSF "
        "and AtmosphericPSF inputs.")
    # test 2-entry list initialization
    add = galsim.Add([obj1, obj2])
    np.testing.assert_almost_equal(
        add.flux, 3. * test_flux, decimal=param_decimal,
        err_msg="Add total flux for double-entry list initialization inconsistent with OpticalPSF "+
        "and AtmosphericPSF inputs.")
    # make a third object, a Moffat for something totally different
    obj3 = galsim.Moffat(half_light_radius=test_hlr, flux = 3. * test_flux, beta=test_beta,
                         trunc=test_trunc)
    # test 3-entry list initialization
    add = galsim.Add([obj1, obj2, obj3])
    np.testing.assert_almost_equal(
        add.flux, 6. * test_flux, decimal=param_decimal,
        err_msg="Add total flux for triple-entry list initialization inconsistent with OpticalPSF "+
        "and AtmosphericPSF and Moffat inputs.")
        


    
