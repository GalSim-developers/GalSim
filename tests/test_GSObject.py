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

# AtmosphericPSF / Kolmogorov params and reference values
test_lor0 = 1.9
test_oversampling = 1.7

atmos_ref_fwhm_from_lor0 = test_lor0 * 0.976
atmos_ref_lor0_from_fwhm = test_fwhm / 0.976

kolmo_ref_fwhm_from_lor0 = test_lor0 * 0.975865
kolmo_ref_lor0_from_fwhm = test_fwhm / 0.975865

kolmo_ref_hlr_from_lor0 = test_lor0 * 0.554811
kolmo_ref_lor0_from_hlr = test_hlr / 0.554811

kolmo_ref_fwhm_from_hlr = test_hlr * 0.975865 / 0.554811
kolmo_ref_hlr_from_fwhm = test_fwhm * 0.554811 / 0.975865

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

# decimal point to go to for parameter value comparisons
param_decimal = 12


def test_gaussian_param_consistency():
    """Test consistency of Gaussian parameters.
    """
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

def test_gaussian_flux_scaling():
    """Test flux scaling for Gaussian.
    """
    # init with sigma and flux only (should be ok given last tests)
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

def test_moffat_param_consistency():
    """Test consistency of Moffat parameters.
    """
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

def test_moffat_flux_scaling():
    """Test flux scaling for Moffat.
    """
    # init with scale_radius only (should be ok given last tests)
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

def test_atmos_param_consistency():
    """Test consistency of AtmosphericPSF parameters.
    """
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

def test_atmos_flux_scaling():
    """Test flux scaling for AtmosphericPSF.
    """
    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

def test_kolmo_param_consistency():
    """Test consistency of Kolmogorov parameters.
    """
    # init with lam_over_r0 and flux
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_r0, test_lor0, decimal=param_decimal,
        err_msg="Lambda / r0 param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, kolmo_ref_fwhm_from_lor0, decimal=param_decimal,
        err_msg="Starting lambda / r0 param and derived fwhm attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, kolmo_ref_hlr_from_lor0, decimal=param_decimal,
        err_msg="Starting lambda / r0 param and derived half_light_radius attribute inconsistent.")

    # init with FWHM and flux
    obj = galsim.Kolmogorov(fwhm=test_fwhm, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_r0, kolmo_ref_lor0_from_fwhm, decimal=param_decimal,
        err_msg="Starting FWHM param and derived lambda / r0 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, test_fwhm, decimal=param_decimal,
        err_msg="FWHM param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, kolmo_ref_hlr_from_fwhm, decimal=param_decimal,
        err_msg="Starting FWHM param and derived half_light_radius attribute inconsistent.")

    # init with HLR and flux
    obj = galsim.Kolmogorov(half_light_radius=test_hlr, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.lam_over_r0, kolmo_ref_lor0_from_hlr, decimal=param_decimal,
        err_msg="Starting half light radius param and derived lambda / r0 attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.fwhm, kolmo_ref_fwhm_from_hlr, decimal=param_decimal,
        err_msg="Starting half light radius param FWHM and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_hlr, decimal=param_decimal,
        err_msg="Half light radius param and attribute inconsistent.")

def test_kolmo_flux_scaling():
    """Test flux scaling for Kolmogorov.
    """
    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

def test_airy_param_consistency():
    """Test consistency of Airy parameters.
    """
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

def test_airy_flux_scaling():
    """Test flux scaling for Airy.
    """
    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.Airy(lam_over_D=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Airy(lam_over_D=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Airy(lam_over_D=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Airy(lam_over_D=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Airy(lam_over_D=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

def test_opticalpsf_param_consistency():
    """Test consistency of OpticalPSF parameters.
    """
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

def test_opticalpsf_flux_scaling():
    """Test flux scaling for OpticalPSF.
    """
    # init
    obj = galsim.OpticalPSF(
        lam_over_D=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.OpticalPSF(
        lam_over_D=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.OpticalPSF(
        lam_over_D=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.OpticalPSF(
        lam_over_D=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.OpticalPSF(
        lam_over_D=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

def test_sersic_param_consistency():
    """Test consistency of Sersic parameters.
    """
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

def test_sersic_flux_scaling():
    """Test flux scaling for Sersic.
    """
    # loop through sersic n
    for test_n in test_sersic_n:
        # init with hlr and flux only (should be ok given last tests)
        obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux)
        obj *= 2.
        np.testing.assert_almost_equal(
            obj.flux, test_flux * 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __imul__.")
        obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux)
        obj /= 2.
        np.testing.assert_almost_equal(
            obj.flux, test_flux / 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __idiv__.")
        obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux)
        obj2 = obj * 2.
        # First test that original obj is unharmed... (also tests that .copy() is working)
        np.testing.assert_almost_equal(
            obj.flux, test_flux, decimal=param_decimal,
            err_msg="Flux param inconsistent after __rmul__ (original).")
        # Then test new obj2 flux
        np.testing.assert_almost_equal(
            obj2.flux, test_flux * 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __rmul__ (result).")
        obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux)
        obj2 = 2. * obj
        # First test that original obj is unharmed... (also tests that .copy() is working)
        np.testing.assert_almost_equal(
            obj.flux, test_flux, decimal=param_decimal,
            err_msg="Flux param inconsistent after __mul__ (original).")
        # Then test new obj2 flux
        np.testing.assert_almost_equal(
            obj2.flux, test_flux * 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __mul__ (result).")
        obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux)
        obj2 = obj / 2.
        # First test that original obj is unharmed... (also tests that .copy() is working)
        np.testing.assert_almost_equal(
             obj.flux, test_flux, decimal=param_decimal,
             err_msg="Flux param inconsistent after __div__ (original).")
        # Then test new obj2 flux
        np.testing.assert_almost_equal(
            obj2.flux, test_flux / 2., decimal=param_decimal,
            err_msg="Flux param inconsistent after __div__ (result).")

def test_exponential_param_consistency():
    """Test consistency of Exponential parameters.
    """
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

def test_exponential_flux_scaling():
    """Test flux scaling for Exponential.
    """
    # init with scale and flux only (should be ok given last tests)
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")   

def test_devaucouleurs_param_consistency():
    """Test consistency of DeVaucouleurs parameters.
    """
    # init with half_light_radius and flux
    obj = galsim.Exponential(half_light_radius=test_hlr, flux=test_flux)
    np.testing.assert_almost_equal(
        obj.half_light_radius, test_hlr, decimal=param_decimal,
        err_msg="DeVaucouleurs half_light_radius param and attribute inconsistent.")
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param and attribute inconsistent.")

def test_devaucouleurs_flux_scaling():
    """Test flux scaling for DeVaucouleurs.
    """
    # init with half_light_radius and flux only (should be ok given last tests)
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

def test_add_param_consistency():
    """Test consistency of Add parameters for a number of different different sorts of constituent
    GSObjects.
    """
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

def test_add_flux_scaling():
    """Test flux scaling for Add.
    """
    # init with Gaussian and Exponential only (should be ok given last tests)
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
        
def test_convolve_param_consistency():
    """Test consistency of Convolve parameters for a number of different different sorts of
    constituent GSObjects.
    """
    # only really the flux param to test, but will do this with a couple of difference sorts of
    # GSObjects in the Add just to check, and for list and dual argument initialization
    # Sersic and Exp first
    obj1 = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = galsim.Sersic(n=3.5, half_light_radius=test_hlr, flux=2. * test_flux)
    # test dual argument initializtion
    convolve = galsim.Convolve(obj1, obj2)
    np.testing.assert_almost_equal(
        convolve.flux, 2. * test_flux**2, decimal=param_decimal,
        err_msg="Convolve total flux for dual argument initialization inconsistent with Sersic "+
        "and Exp inputs.")
    # test 2-entry list initialization
    convolve = galsim.Convolve([obj1, obj2])
    np.testing.assert_almost_equal(
        convolve.flux, 2. * test_flux**2, decimal=param_decimal,
        err_msg="Convolve total flux for double-entry list initialization inconsistent with "+
        "Sersic and Exp inputs.")
    # make a third object, dVc
    obj3 = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux = 3. * test_flux)
    # test 3-entry list initialization
    convolve = galsim.Convolve([obj1, obj2, obj3])
    np.testing.assert_almost_equal(
        convolve.flux, 6. * test_flux**3, decimal=param_decimal,
        err_msg="Convolve total flux for triple-entry list initialization inconsistent with "+
        "Sersic and Exp and DeVauc inputs.")

    # Then try a couple of SBInterpolatedImage-type classes
    obj1 = galsim.OpticalPSF(
        lam_over_D=test_loD, astig1=test_astig1, defocus=test_defocus, astig2=test_astig2,
        flux=test_flux, oversampling=test_oversampling)
    obj2 = galsim.AtmosphericPSF(
        lam_over_r0=test_lor0, flux=2. * test_flux, oversampling=test_oversampling)
    # test dual argument initializtion
    convolve = galsim.Convolve(obj1, obj2)
    np.testing.assert_almost_equal(
        convolve.flux, 2. * test_flux**2, decimal=param_decimal,
        err_msg="Convolve total flux for dual argument initialization inconsistent with OpticalPSF "
        "and AtmosphericPSF inputs.")
    # test 2-entry list initialization
    add = galsim.Convolve([obj1, obj2])
    np.testing.assert_almost_equal(
        convolve.flux, 2. * test_flux**2, decimal=param_decimal,
        err_msg="Convolve total flux for double-entry list initialization inconsistent with "+
        "OpticalPSF and AtmosphericPSF inputs.")
    # make a third object, a Moffat for something totally different
    obj3 = galsim.Moffat(half_light_radius=test_hlr, flux = 3. * test_flux, beta=test_beta,
                         trunc=test_trunc)
    # test 3-entry list initialization
    convolve = galsim.Convolve([obj1, obj2, obj3])
    np.testing.assert_almost_equal(
        convolve.flux, 6. * test_flux**3, decimal=param_decimal,
        err_msg="Convolve total flux for triple-entry list initialization inconsistent with "+
        "OpticalPSF and AtmosphericPSF and Moffat inputs.")

def test_convolve_flux_scaling():
    """Test flux scaling for Convolve.
    """
    # init with Gaussian and DeVauc only (should be ok given last tests)
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

def test_gaussian_data():
    """Test copying and sharing of parameter data store for Gaussian.
    """
    # As the data copying code is shared among all GSObjects, it's probably sufficient to simply
    # test a few of the broadly different types.
    # Gaussian, init with sigma and flux
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    cobj = obj.copy()
    np.testing.assert_equal(
        obj._data, cobj._data, err_msg="Object _data store not consistent after copy.")
    for key, value in cobj._data.iteritems():
        cobj._data[key] = "foo"
    # Check that this systematic ruining of the _data store is not reflected in the original:
    assert obj._data != cobj._data

def test_moffat_data():
    """Test copying and sharing of parameter data store for Moffat.
    """
    # As the data copying code is shared among all GSObjects, it's probably sufficient to simply
    # test a few of the broadly different types.
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    cobj = obj.copy()
    np.testing.assert_equal(
        obj._data, cobj._data, err_msg="Object _data store not consistent after copy.")
    for key, value in cobj._data.iteritems():
        cobj._data[key] = "foo"
    # Check that this systematic ruining of the _data store is not reflected in the original:
    assert obj._data != cobj._data

def test_opticalpsf_data():
    """Test copying and sharing of parameter data store for OpticalPSF.
    """
    # As the data copying code is shared among all GSObjects, it's probably sufficient to simply
    # test a few of the broadly different types.
    obj = galsim.OpticalPSF(
        lam_over_D=test_loD, astig1=test_astig1, astig2=test_astig2, defocus=test_defocus,
        oversampling=test_oversampling, flux=test_flux)
    cobj = obj.copy()
    np.testing.assert_equal(
        obj._data, cobj._data, err_msg="Object _data store not consistent after copy.")
    for key, value in cobj._data.iteritems():
        cobj._data[key] = "foo"
    # Check that this systematic ruining of the _data store is not reflected in the original:
    assert obj._data != cobj._data

def test_add_data():
    """Test copying and sharing of parameter data store for Data.
    """
    # As the data copying code is shared among all GSObjects, it's probably sufficient to simply
    # test a few of the broadly different types.
    obj1 = galsim.OpticalPSF(
        lam_over_D=test_loD, astig1=test_astig1, astig2=test_astig2, defocus=test_defocus,
        oversampling=test_oversampling, flux=test_flux)
    obj2 = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj = galsim.Add(obj1, obj2)
    cobj = obj.copy()
    np.testing.assert_equal(
        obj._data, cobj._data, err_msg="Object _data store not consistent after copy.")
    # As _data dict is empty for add, put in a new item by hand
    cobj._data["foo"] = "bar"
    # Check that this systematic ruining of the _data store is not reflected in the original:
    assert obj._data != cobj._data


if __name__ == "__main__":
    test_gaussian_param_consistency()
    test_gaussian_flux_scaling()
    test_moffat_param_consistency()
    test_moffat_flux_scaling()
    test_atmos_param_consistency()
    test_atmos_flux_scaling()
    test_kolmo_param_consistency()
    test_kolmo_flux_scaling()
    test_airy_param_consistency()
    test_airy_flux_scaling()
    test_opticalpsf_param_consistency()
    test_opticalpsf_flux_scaling()
    test_sersic_param_consistency()
    test_sersic_flux_scaling()
    test_exponential_param_consistency()
    test_exponential_flux_scaling()
    test_devaucouleurs_param_consistency()
    test_devaucouleurs_flux_scaling()
    test_doublegaussian_param_consistency()
    test_doublegaussian_flux_scaling()
    test_add_param_consistency()
    test_add_flux_scaling()
    test_convolve_param_consistency()
    test_convolve_flux_scaling()
    test_gaussian_data()
    test_moffat_data()
    test_opticalpsf_data()
    test_add_data()
