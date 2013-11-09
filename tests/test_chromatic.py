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

def air_refractive_index(wave):
    """Compute the refractive index of air at standard temperature and pressure

    @param wave  Wavelength in nanometers (Array)
    @returns refractive index
    """
    sigma_squared = 1.0 / (wave * 1.e-3)**2
    n = (64.328 + (29498.1 / (146.0 - sigma_squared))
         + (255.4 / (41.0 - sigma_squared))) * 1.e-6 + 1.0
    return n

def atmosphere_refraction_angle(wave, zenith_angle):
    """Compute the angle of refraction for photon entering the atmosphere.

    Photons refract when transitioning from space, where the refractive index n = 1.0 exactly, to
    air, where the refractive index is slightly greater than 1.0.  This function computes the
    change in zenith angle for a photon with a given wavelength.  Output is a positive number of
    radians, even though the apparent zenith angle technically decreases due to this effect.

    @param wave    Wavelength in nanometers (Array)
    @param zenith_angle  Zenith angle in radians
    @returns       Absolute value of change in zenith angle in radians
    """
    n_squared = air_refractive_index(wave)**2
    r0 = (n_squared - 1.0) / (2.0 * n_squared)
    return r0 * np.tan(zenith_angle)

def refraction_in_pixels(wave, zenith_angle, plate_scale):
    """Compute the shift in PSF centroid due to refraction.

    @param wave   Wavelength in nanometers (Array)
    @param zenith_angle  Zenith angle in radians
    @param plate_scale   Arcseconds per pixel
    """

    shift = atmosphere_refraction_angle(wave, zenith_angle) # radians
    shift /= plate_scale/3600*np.pi/180 # -> pixels
    return shift

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
                       plate_scale=0.2, stamp_size=31):
    """Compute a test image of a galaxy using a chromatic PSF directly, i.e. without using the
    galsim.chromatic module.

    @param galaxy_wave        Wavelength array for galaxy spectrum in nanometers
    @param galaxy_photons     Flux(?) array for galaxy spectral density in units of photons per
                              nanometer.
    @param zenith_angle       in radians.
    @param filter_wave        Wavelength array describing filter throughput in nanometers
    @param filter_throughput  Dimensionless throughput array.  I.e., probability to detect photon.
    @param plate_scale        Arcseconds per pixel.  Default 0.2 corresponds to LSST camera.
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
    r610 = refraction_in_pixels(610, zenith_angle, plate_scale)

    # pre-compute shifts and radii in parallel
    shifts = refraction_in_pixels(wave, zenith_angle, plate_scale) - r610
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
    pixel = galsim.Pixel(plate_scale)
    gal = galsim.Convolve([obj, PSF, pixel])
    image = galsim.ImageD(stamp_size, stamp_size, plate_scale)
    gal.draw(image=image)
    return image

def galsim_computation(galaxy_n, galaxy_hlr, galaxy_e1, galaxy_e2, galaxy_wave, galaxy_photons,
                       PSF_hlr, PSF_beta, PSF_e1, PSF_e2, zenith_angle,
                       shear_g1, shear_g2,
                       filter_wave, filter_throughput,
                       plate_scale=0.2, stamp_size=31):
    """Compute a test image of a galaxy using the galsim.chromatic module.

    @param galaxy_wave        Wavelength array for galaxy spectrum in nanometers
    @param galaxy_photons     Flux(?) array for galaxy spectral density in units of photons per
                              nanometer.
    @param zenith_angle       in radians.
    @param filter_wave        Wavelength array describing filter throughput in nanometers
    @param filter_throughput  Dimensionless throughput array.  I.e., probability to detect photon.
    @param plate_scale        Arcseconds per pixel.  Default 0.2 corresponds to LSST camera.
    @returns                  GalSim ImageD
    """

    obj = galsim.ChromaticBaseObject(galsim.Sersic, galaxy_wave, galaxy_photons,
                                     n=galaxy_n, half_light_radius=galaxy_hlr)
    obj.applyShear(e1=galaxy_e1, e2=galaxy_e2)
    obj.applyShear(g1=shear_g1, g2=shear_g2)

    r610 = refraction_in_pixels(610, zenith_angle, plate_scale)
    dilate_fn = lambda wave: (0, refraction_in_pixels(wave, zenith_angle, plate_scale) - r610)

    PSF = galsim.ChromaticShiftAndDilate(galsim.Moffat,
                                        shift_fn=dilate_fn,
                                        dilate_fn=relative_seeing,
                                        beta=PSF_beta,
                                        half_light_radius=PSF_hlr)
    PSF.applyShear(e1=PSF_e1, e2=PSF_e2)

    pixel = galsim.Pixel(plate_scale)
    gal = galsim.ChromaticConvolve([obj, PSF, pixel])
    image = galsim.ImageD(stamp_size, stamp_size, plate_scale)
    gal.draw(filter_wave, filter_throughput, image=image)
    return image

def test_direct_vs_galsim():
    """Compare two images, one created directly but without galsim.chromatic module, and one using
    the galsim.chromatic module.
    """
    galaxy_n = 4.0
    galaxy_hlr = 1.0
    galaxy_e1 = 0.4
    galaxy_e2 = 0.2
    galaxy_wave, galaxy_flambda = np.genfromtxt('test_spectra/CWW_Im_ext.sed').T
    galaxy_photons = galaxy_flambda * galaxy_wave # ergs -> N_photons
    galaxy_photons *= 2.e-7 # Manually adjusting to have peak of ~1 count
    galaxy_wave /= 10 # Angstrom -> nm
    PSF_hlr = 0.3
    PSF_beta = 3.0
    PSF_e1 = 0.01
    PSF_e2 = 0.06
    zenith_angle = 20 * np.pi/180 # radians
    shear_g1 = 0.01
    shear_g2 = 0.02
    filter_wave, filter_throughput = np.genfromtxt('test_filters/LSST_r.dat').T
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
    # Since peak is around 1, this tests consistency to part in 10^4 level.
    np.testing.assert_array_almost_equal(
            direct_img.array, galsim_img.array, 4,
            err_msg="Directly computed chromatic image disagrees with image created using "
                    "galsim.chromatic")

    # total photons in infinite aperture
    # galaxy_photons_interp = np.interp(filter_wave, galaxy_wave, galaxy_photons)
    # dwave = filter_wave[1] - filter_wave[0]
    # sum_photons = (galaxy_photons_interp * filter_throughput * dwave).sum()
    # print sum_photons

if __name__ == "__main__":
    test_direct_vs_galsim()
