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
test_sersic_trunc = [0., 11.]

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

# decimal point to go to for precise image comparisons
image_decimal_precise = 15

# parameters for shift tests
int_shift_x = 7
int_shift_y = 3

n_pix_x = 50
n_pix_y = 60

delta_sub = 30

n_photons_low = 10



def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_gaussian_flux_scaling():
    """Test flux scaling for Gaussian.
    """
    import time
    t1 = time.time()
    # init with sigma and flux only (should be ok given last tests)
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_moffat_flux_scaling():
    """Test flux scaling for Moffat.
    """
    import time
    t1 = time.time()
    # init with scale_radius only (should be ok given last tests)
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Moffat(scale_radius=test_scale, beta=test_beta, trunc=test_trunc, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_atmos_flux_scaling():
    """Test flux scaling for AtmosphericPSF.
    """
    import time
    t1 = time.time()
    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.AtmosphericPSF(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_kolmo_flux_scaling():
    """Test flux scaling for Kolmogorov.
    """
    import time
    t1 = time.time()
    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Kolmogorov(lam_over_r0=test_lor0, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_airy_flux_scaling():
    """Test flux scaling for Airy.
    """
    import time
    t1 = time.time()
    # init with lam_over_r0 and flux only (should be ok given last tests)
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Airy(lam_over_diam=test_loD, flux=test_flux, obscuration=test_obscuration)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_opticalpsf_flux_scaling():
    """Test flux scaling for OpticalPSF.
    """
    import time
    t1 = time.time()
    # init
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, defocus=test_defocus,
        astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sersic_flux_scaling():
    """Test flux scaling for Sersic.
    """
    import time
    t1 = time.time()
    # loop through sersic n
    for test_n in test_sersic_n:
        # loop through sersic truncation
        for test_trunc in test_sersic_trunc:
            # init with hlr and flux only (should be ok given last tests)
            obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux, trunc=test_trunc)
            obj *= 2.
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __imul__.")
            obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux, trunc=test_trunc)
            obj /= 2.
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux / 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __idiv__.")
            obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux, trunc=test_trunc)
            obj2 = obj * 2.
            # First test that original obj is unharmed... (also tests that .copy() is working)
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (original).")
            # Then test new obj2 flux
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __rmul__ (result).")
            obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux, trunc=test_trunc)
            obj2 = 2. * obj
            # First test that original obj is unharmed... (also tests that .copy() is working)
            np.testing.assert_almost_equal(
                obj.getFlux(), test_flux, decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (original).")
            # Then test new obj2 flux
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux * 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __mul__ (result).")
            obj = galsim.Sersic(test_n, half_light_radius=test_hlr, flux=test_flux, trunc=test_trunc)
            obj2 = obj / 2.
            # First test that original obj is unharmed... (also tests that .copy() is working)
            np.testing.assert_almost_equal(
                 obj.getFlux(), test_flux, decimal=param_decimal,
                 err_msg="Flux param inconsistent after __div__ (original).")
            # Then test new obj2 flux
            np.testing.assert_almost_equal(
                obj2.getFlux(), test_flux / 2., decimal=param_decimal,
                err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_exponential_flux_scaling():
    """Test flux scaling for Exponential.
    """
    import time
    t1 = time.time()
    # init with scale and flux only (should be ok given last tests)
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Exponential(scale_radius=test_scale, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")   
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_devaucouleurs_flux_scaling():
    """Test flux scaling for DeVaucouleurs.
    """
    import time
    t1 = time.time()
    # init with half_light_radius and flux only (should be ok given last tests)
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_add_flux_scaling():
    """Test flux scaling for Add.
    """
    import time
    t1 = time.time()
    # init with Gaussian and Exponential only (should be ok given last tests)
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_convolve_flux_scaling():
    """Test flux scaling for Convolve.
    """
    import time
    t1 = time.time()
    # init with Gaussian and DeVauc only (should be ok given last tests)
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Convolve(
        [galsim.Gaussian(sigma=test_sigma, flux=np.sqrt(test_flux)),
         galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=np.sqrt(test_flux))])
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_integer_shift_fft():
    """Test if applyShift works correctly for integer shifts using draw method.
    """
    import time
    t1 = time.time()

    gal = galsim.Gaussian(sigma=test_sigma)
    pix = galsim.Pixel(1.)
    psf = galsim.Airy(lam_over_diam=test_hlr)

    # shift galaxy only
 
    final=galsim.Convolve([gal, psf, pix])
    img_center = galsim.ImageD(n_pix_x,n_pix_y)
    final.draw(img_center,dx=1)

    gal.applyShift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf, pix])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    final.draw(img_shift,dx=1)

    sub_center = img_center.array[
        (n_pix_y - delta_sub) / 2 : (n_pix_y + delta_sub) / 2,
        (n_pix_x - delta_sub) / 2 : (n_pix_x + delta_sub) / 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) / 2  + int_shift_y : (n_pix_y + delta_sub) / 2  + int_shift_y,
        (n_pix_x - delta_sub) / 2  + int_shift_x : (n_pix_x + delta_sub) / 2  + int_shift_x]

    np.testing.assert_array_almost_equal(
        sub_center, sub_shift, decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with shifted Galaxy only")

    # shift PSF only

    gal = galsim.Gaussian(sigma=test_sigma)
    psf.applyShift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf, pix])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    final.draw(img_shift,dx=1)

    sub_center = img_center.array[
        (n_pix_y - delta_sub) / 2 : (n_pix_y + delta_sub) / 2,
        (n_pix_x - delta_sub) / 2 : (n_pix_x + delta_sub) / 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) / 2  + int_shift_y : (n_pix_y + delta_sub) / 2  + int_shift_y,
        (n_pix_x - delta_sub) / 2  + int_shift_x : (n_pix_x + delta_sub) / 2  + int_shift_x]
    np.testing.assert_array_almost_equal(
        sub_center, sub_shift,  decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with only PSF shifted ")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_integer_shift_photon():
    """Test if applyShift works correctly for integer shifts using drawShoot method.
    """
    import time
    t1 = time.time()

    seed = 10
    gal = galsim.Gaussian(sigma=test_sigma)
    pix = galsim.Pixel(1.)
    psf = galsim.Airy(lam_over_diam=test_hlr)

    # shift galaxy only
 
    final=galsim.Convolve([gal, psf, pix])
    img_center = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawShoot(img_center,dx=1,rng=test_deviate,n_photons=n_photons_low)

    gal.applyShift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf, pix])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawShoot(img_shift,dx=1,rng=test_deviate,n_photons=n_photons_low)
    
    sub_center = img_center.array[
        (n_pix_y - delta_sub) / 2 : (n_pix_y + delta_sub) / 2,
        (n_pix_x - delta_sub) / 2 : (n_pix_x + delta_sub) / 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) / 2  + int_shift_y : (n_pix_y + delta_sub) / 2  + int_shift_y,
        (n_pix_x - delta_sub) / 2  + int_shift_x : (n_pix_x + delta_sub) / 2  + int_shift_x]


    np.testing.assert_array_almost_equal(
        sub_center, sub_shift, decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with shifted Galaxy only")

    # shift PSF only

    gal = galsim.Gaussian(sigma=test_sigma)
    psf.applyShift(dx=int_shift_x,dy=int_shift_y)
    final=galsim.Convolve([gal, psf, pix])
    img_shift = galsim.ImageD(n_pix_x,n_pix_y)
    test_deviate = galsim.BaseDeviate(seed)
    final.drawShoot(img_shift,dx=1,rng=test_deviate,n_photons=n_photons_low)

    sub_center = img_center.array[
        (n_pix_y - delta_sub) / 2 : (n_pix_y + delta_sub) / 2,
        (n_pix_x - delta_sub) / 2 : (n_pix_x + delta_sub) / 2]
    sub_shift = img_shift.array[
        (n_pix_y - delta_sub) / 2  + int_shift_y : (n_pix_y + delta_sub) / 2  + int_shift_y,
        (n_pix_x - delta_sub) / 2  + int_shift_x : (n_pix_x + delta_sub) / 2  + int_shift_x]
    np.testing.assert_array_almost_equal(
        sub_center, sub_shift,  decimal=image_decimal_precise,
        err_msg="Integer shift failed for FFT rendered Gaussian GSObject with only PSF shifted ")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



if __name__ == "__main__":
    test_gaussian_flux_scaling()
    test_moffat_flux_scaling()
    test_atmos_flux_scaling()
    test_kolmo_flux_scaling()
    test_airy_flux_scaling()
    test_opticalpsf_flux_scaling()
    test_sersic_flux_scaling()
    test_exponential_flux_scaling()
    test_devaucouleurs_flux_scaling()
    test_add_flux_scaling()
    test_convolve_flux_scaling()
    test_integer_shift_photon()
    test_integer_shift_fft()
