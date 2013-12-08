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

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def refraction_in_pixels(wave, zenith_angle, pixel_scale):
    """Compute the shift in PSF centroid (in pixels) due to refraction.

    @param wave   Wavelength in nanometers (Array)
    @param zenith_angle  zenith angle as a galsim.Angle
    @param pixel_scale   pixel scale as a galsim.Angle (angle per pixel)
    """

    shift = galsim.dcr.atmosphere_refraction_angle(wave, zenith_angle)
    return (shift / galsim.radians) / pixel_scale.rad()

def relative_seeing(wave):
    """Compute the relative size of the seeing as a function of wavelength.

    Kolmogorov turbulence predicts that FWHM of the seeing kernel will scale like lambda^{-1/5}.
    This function returns the size of the seeing relative to the seeing at 500nm.

    @param wave  Wavelength in nanometers (Array)
    @returns     Seeing FWHM relative to 500nm
    """
    return (wave/500)**(-0.2)

def direct_computation(galaxy_n, galaxy_hlr, galaxy_e1, galaxy_e2, galaxy_wave, galaxy_photons,
                       PSF_hlr, PSF_beta, PSF_e1, PSF_e2, zenith_angle,
                       shear_g1, shear_g2,
                       filter_wave, filter_throughput,
                       pixel_scale=0.2*galsim.arcsec, stamp_size=31):
    """Compute a test image of a galaxy using a chromatic PSF directly, i.e. without using the
    galsim.chromatic module.

    @param galaxy_wave        Wavelength array for galaxy spectrum in nanometers
    @param galaxy_photons     Flux(?) array for galaxy spectral density in units of photons per
                              nanometer.
    @param zenith_angle       as a galsim.Angle
    @param filter_wave        Wavelength array describing filter throughput in nanometers
    @param filter_throughput  Dimensionless throughput array.  I.e., probability to detect photon.
    @param pixel_scale        Arcseconds per pixel as a galsim.Angle.  Default 0.2 corresponds to
                              LSST camera.
    @returns                  GalSim ImageD
    """
    # do intermediate calculations using filter wavelength array, linearly interpolate when necessary
    wave = filter_wave
    throughput = filter_throughput
    detected_photons = np.interp(wave, galaxy_wave, galaxy_photons) * throughput

    # make galaxy
    obj = galsim.Sersic(n=galaxy_n, half_light_radius=galaxy_hlr)
    obj.applyShear(e1=galaxy_e1, e2=galaxy_e2)
    obj.applyShear(g1=shear_g1, g2=shear_g2)

    # make effective PSF
    mPSFs = [] # collection of flux-scaled monochromatic PSFs
    # normalize position to that at middle of r-band: ~610nm
    r610 = refraction_in_pixels(610, zenith_angle, pixel_scale)

    # pre-compute shifts and radii in parallel
    shifts = refraction_in_pixels(wave, zenith_angle, pixel_scale) - r610
    hlrs = PSF_hlr * relative_seeing(wave)

    # build up effective PSF one wavelength at a time
    dwave = wave[1]-wave[0]
    for w, p, hlr, shift in zip(wave, detected_photons, hlrs, shifts):
        mPSF = galsim.Moffat(flux=p*dwave, beta=PSF_beta, half_light_radius=hlr)
        mPSF.applyShear(e1=PSF_e1, e2=PSF_e2)
        mPSF.applyShift((0, shift))
        mPSFs.append(mPSF)
    PSF = galsim.Add(mPSFs)

    # finish off with pixel and create image
    pixel = galsim.Pixel(pixel_scale / galsim.arcsec)
    gal = galsim.Convolve([obj, PSF, pixel])
    image = galsim.ImageD(stamp_size, stamp_size, pixel_scale / galsim.arcsec)
    gal.draw(image=image)
    return image

def galsim_computation(galaxy_n, galaxy_hlr, galaxy_e1, galaxy_e2, galaxy_wave, galaxy_photons,
                       PSF_hlr, PSF_beta, PSF_e1, PSF_e2, zenith_angle,
                       shear_g1, shear_g2,
                       filter_wave, filter_throughput,
                       pixel_scale=0.2*galsim.arcsec, stamp_size=31):
    """Compute a test image of a galaxy using the galsim.chromatic module.

    @param galaxy_wave        Wavelength array for galaxy spectrum in nanometers
    @param galaxy_photons     Flux(?) array for galaxy spectral density in units of photons per
                              nanometer.
    @param zenith_angle       as a galsim.Angle
    @param filter_wave        Wavelength array describing filter throughput in nanometers
    @param filter_throughput  Dimensionless throughput array.  I.e., probability to detect photon.
    @param pixel_scale        Arcseconds per pixel.  Default 0.2 corresponds to LSST camera.
    @returns                  GalSim ImageD
    """

    obj = galsim.ChromaticBaseObject(galsim.Sersic, galaxy_wave, galaxy_photons,
                                     n=galaxy_n, half_light_radius=galaxy_hlr)
    obj.applyShear(e1=galaxy_e1, e2=galaxy_e2)
    obj.applyShear(g1=shear_g1, g2=shear_g2)

    r610 = refraction_in_pixels(610, zenith_angle, pixel_scale)
    dilate_fn = lambda wave: (0, refraction_in_pixels(wave, zenith_angle, pixel_scale) - r610)

    PSF = galsim.ChromaticShiftAndDilate(galsim.Moffat,
                                        shift_fn=dilate_fn,
                                        dilate_fn=relative_seeing,
                                        beta=PSF_beta,
                                        half_light_radius=PSF_hlr)
    PSF.applyShear(e1=PSF_e1, e2=PSF_e2)

    pixel = galsim.Pixel(pixel_scale / galsim.arcsec)
    gal = galsim.ChromaticConvolve([obj, PSF, pixel])
    image = galsim.ImageD(stamp_size, stamp_size, pixel_scale / galsim.arcsec)
    gal.draw(filter_wave, filter_throughput, image=image)
    return image

def test_direct_vs_galsim():
    """Compare two images, one created directly but without galsim.chromatic module, and one using
    the galsim.chromatic module.
    """
    galaxy_n = 4.0
    galaxy_hlr = 1.0 # arcsec
    galaxy_e1 = 0.4
    galaxy_e2 = 0.2
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "../examples/data/"))
    galaxy_wave, galaxy_flambda = np.genfromtxt(os.path.join(datapath, 'CWW_Im_ext.sed')).T
    galaxy_photons = galaxy_flambda * galaxy_wave # ergs -> N_photons
    galaxy_photons *= 2.e-7 # Manually adjusting to have peak of ~1 count
    galaxy_wave /= 10 # Angstrom -> nm
    PSF_hlr = 0.3 # arcsec
    PSF_beta = 3.0
    PSF_e1 = 0.01
    PSF_e2 = 0.06
    zenith_angle = 20 * galsim.degrees
    shear_g1 = 0.01
    shear_g2 = 0.02
    filter_wave, filter_throughput = np.genfromtxt(os.path.join(datapath, 'LSST_r.dat')).T
    wgood = (filter_wave > 500) & (filter_wave < 720) # truncate out-of-band wavelengths
    filter_wave = filter_wave[wgood][0::100]  # sparsify from 1 Ang binning to 100 Ang binning
    filter_throughput = filter_throughput[wgood][0::100]

    direct_img = direct_computation(galaxy_n, galaxy_hlr, galaxy_e1, galaxy_e2,
                                    galaxy_wave, galaxy_photons,
                                    PSF_hlr, PSF_beta, PSF_e1, PSF_e2, zenith_angle,
                                    shear_g1, shear_g2,
                                    filter_wave, filter_throughput)

    galsim_img = galsim_computation(galaxy_n, galaxy_hlr, galaxy_e1, galaxy_e2,
                                    galaxy_wave, galaxy_photons,
                                    PSF_hlr, PSF_beta, PSF_e1, PSF_e2, zenith_angle,
                                    shear_g1, shear_g2,
                                    filter_wave, filter_throughput)
    printval(direct_img, galsim_img)
    # Since peak is around 1, this tests consistency to part in 10^5 level.
    np.testing.assert_array_almost_equal(
            direct_img.array, galsim_img.array, 5,
            err_msg="Directly computed chromatic image disagrees with image created using "
                    "galsim.chromatic")

def test_chromatic_add():
    """Test the `+` operator on ChromaticObjects"""

    #bulge component parameters
    bulge_n = 4.0
    bulge_hlr = 1.0 # arcsec
    bulge_e1 = 0.4
    bulge_e2 = 0.2
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "../examples/data/"))
    bulge_wave, bulge_flambda = np.genfromtxt(os.path.join(datapath, 'CWW_E_ext.sed')).T
    bulge_photons = bulge_flambda * bulge_wave # ergs -> N_photons
    bulge_photons *= 2.e-7 # Manually adjusting to have peak of ~1 count
    bulge_wave /= 10 # Angstrom -> nm
    bulge = galsim.ChromaticBaseObject(galsim.Sersic, bulge_wave, bulge_photons,
                                     n=bulge_n, half_light_radius=bulge_hlr)
    bulge.applyShear(e1=bulge_e1, e2=bulge_e2)

    #disk component parameters
    disk_n = 1.0
    disk_hlr = 1.0 # arcsec
    disk_e1 = 0.4
    disk_e2 = 0.2
    disk_wave, disk_flambda = np.genfromtxt(os.path.join(datapath, 'CWW_Sbc_ext.sed')).T
    disk_photons = disk_flambda * disk_wave # ergs -> N_photons
    disk_photons *= 2.e-7 # Manually adjusting to have peak of ~1 count
    disk_wave /= 10 # Angstrom -> nm
    disk = galsim.ChromaticBaseObject(galsim.Sersic, disk_wave, disk_photons,
                                     n=disk_n, half_light_radius=disk_hlr)
    disk.applyShear(e1=disk_e1, e2=disk_e2)

    shear_g1 = 0.01
    shear_g2 = 0.02

    zenith_angle = 20 * galsim.degrees

    #line that actually tests the `+` operator
    obj = bulge + disk
    obj.applyShear(g1=shear_g1, g2=shear_g2)

    #now work out the PSF
    PSF_hlr = 0.3 # arcsec
    PSF_beta = 3.0
    PSF_e1 = 0.01
    PSF_e2 = 0.06

    pixel_scale = 0.2 * galsim.arcsec
    stamp_size = 31
    r610 = refraction_in_pixels(610, zenith_angle, pixel_scale)
    dilate_fn = lambda wave: (0, refraction_in_pixels(wave, zenith_angle, pixel_scale) - r610)

    PSF = galsim.ChromaticShiftAndDilate(galsim.Moffat,
                                        shift_fn=dilate_fn,
                                        dilate_fn=relative_seeing,
                                        beta=PSF_beta,
                                        half_light_radius=PSF_hlr)
    PSF.applyShear(e1=PSF_e1, e2=PSF_e2)

    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "../examples/data/"))
    filter_wave, filter_throughput = np.genfromtxt(os.path.join(datapath, 'LSST_r.dat')).T
    wgood = (filter_wave > 500) & (filter_wave < 720) # truncate out-of-band wavelengths
    filter_wave = filter_wave[wgood][0::100]  # sparsify from 1 Ang binning to 100 Ang binning
    filter_throughput = filter_throughput[wgood][0::100]

    pixel = galsim.Pixel(pixel_scale / galsim.arcsec)
    gal = galsim.ChromaticConvolve([obj, PSF, pixel])
    image = galsim.ImageD(stamp_size, stamp_size, pixel_scale / galsim.arcsec)
    gal.draw(filter_wave, filter_throughput, image=image)

    # Compare this image to one drawn directly from the individual profiles
    bulge_img = galsim_computation(bulge_n, bulge_hlr, bulge_e1, bulge_e2,
                                   bulge_wave, bulge_photons,
                                   PSF_hlr, PSF_beta, PSF_e1, PSF_e2, zenith_angle,
                                   shear_g1, shear_g2,
                                   filter_wave, filter_throughput)
    disk_img = galsim_computation(disk_n, disk_hlr, disk_e1, disk_e2,
                                   disk_wave, disk_photons,
                                   PSF_hlr, PSF_beta, PSF_e1, PSF_e2, zenith_angle,
                                   shear_g1, shear_g2,
                                   filter_wave, filter_throughput)

    printval(image, bulge_img+disk_img)
    np.testing.assert_array_almost_equal(
            image.array, bulge_img.array+disk_img.array, 5,
            err_msg="`+` operator doesn't match manual image addition")

    bulge_img += disk_img
    printval(image, bulge_img)
    np.testing.assert_array_almost_equal(
            image.array, bulge_img.array, 5,
            err_msg="`+` operator doesn't match manual image addition")


if __name__ == "__main__":
    test_direct_vs_galsim()
    test_chromatic_add()
