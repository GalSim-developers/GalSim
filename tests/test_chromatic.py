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

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

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
                       pixel_scale=0.2*galsim.arcsec, stamp_size=32):
    """Create a test image of a galaxy by directly simulating a chromatic PSF
    wavelength-by-wavelength, i.e., without using the galsim.chromatic module.

    @param galaxy_wave        Wavelength array for galaxy spectrum in nanometers
    @param galaxy_photons     Galaxy photon spectral energy distribution in units of photons per
                              nanometer.
    @param zenith_angle       as a galsim.Angle
    @param filter_wave        Wavelength array describing filter throughput in nanometers
    @param filter_throughput  Dimensionless throughput array.  I.e., probability to detect photon.
    @param pixel_scale        Arcseconds per pixel as a galsim.Angle.  Default 0.2 corresponds to
                              LSST camera.
    @returns                  GalSim ImageD
    """
    # do intermediate calculations using filter wavelength array, interpolating when necessary
    wave = filter_wave
    throughput = filter_throughput
    detected_photons = galsim.LookupTable(galaxy_wave, galaxy_photons)(wave) * throughput

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
                       pixel_scale=0.2*galsim.arcsec, stamp_size=32):
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
    shift_fn = lambda wave: (0, refraction_in_pixels(wave, zenith_angle, pixel_scale) - r610)

    PSF = galsim.ChromaticShiftAndDilate(galsim.Moffat,
                                        shift_fn=shift_fn,
                                        dilate_fn=relative_seeing,
                                        beta=PSF_beta,
                                        half_light_radius=PSF_hlr)
    PSF.applyShear(e1=PSF_e1, e2=PSF_e2)

    pixel = galsim.Pixel(pixel_scale / galsim.arcsec)
    gal = galsim.ChromaticConvolve([obj, PSF, pixel])
    image = galsim.ImageD(stamp_size, stamp_size, pixel_scale / galsim.arcsec)
    gal.draw(filter_wave, filter_throughput, image=image)
    return image

def test_chromatic_direct_vs_galsim():
    """Compare two images, one created directly but without galsim.chromatic module, and one using
    the galsim.chromatic module.
    """
    import time
    t1 = time.time()

    galaxy_n = 4.0
    galaxy_hlr = 1.0 # arcsec
    galaxy_e1 = 0.4
    galaxy_e2 = 0.2
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
    wgood = (filter_wave >= 500) & (filter_wave <= 720) # truncate out-of-band wavelengths
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
    # Since peak is around 1, this tests consistency to part in 10^3 level.
    np.testing.assert_array_almost_equal(
            direct_img.array, galsim_img.array, 3,
            err_msg="Directly computed chromatic image disagrees with image created using "
                    "galsim.chromatic")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def sample_bulge_gal():
    bulge_n = 4.0
    bulge_hlr = 1.0 # arcsec
    bulge_e1 = 0.4
    bulge_e2 = 0.2
    bulge_wave, bulge_flambda = np.genfromtxt(os.path.join(datapath, 'CWW_E_ext.sed')).T
    bulge_photons = bulge_flambda * bulge_wave # ergs -> N_photons
    bulge_photons *= 2.e-7 # Manually adjusting to have peak of ~1 count
    bulge_wave /= 10 # Angstrom -> nm
    bulge = galsim.ChromaticBaseObject(galsim.Sersic, bulge_wave, bulge_photons,
                                     n=bulge_n, half_light_radius=bulge_hlr)
    bulge.applyShear(e1=bulge_e1, e2=bulge_e2)
    return bulge

def sample_disk_gal():
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
    return disk

def sample_PSF(pixel_scale=0.2*galsim.arcsec):
    zenith_angle = 20 * galsim.degrees
    PSF_hlr = 0.3 # arcsec
    PSF_beta = 3.0
    PSF_e1 = 0.01
    PSF_e2 = 0.06
    r610 = refraction_in_pixels(610, zenith_angle, pixel_scale)
    shift_fn = lambda wave: (0, refraction_in_pixels(wave, zenith_angle, pixel_scale) - r610)

    PSF = galsim.ChromaticShiftAndDilate(galsim.Moffat,
                                        shift_fn=shift_fn,
                                        dilate_fn=relative_seeing,
                                        beta=PSF_beta,
                                        half_light_radius=PSF_hlr)
    PSF.applyShear(e1=PSF_e1, e2=PSF_e2)
    return PSF

def sample_filter():
    filter_wave, filter_throughput = np.genfromtxt(os.path.join(datapath, 'LSST_r.dat')).T
    wgood = (filter_wave >= 500) & (filter_wave <= 720) # truncate out-of-band wavelengths
    filter_wave = filter_wave[wgood][0::100]  # sparsify from 1 Ang binning to 100 Ang binning
    filter_throughput = filter_throughput[wgood][0::100]
    return filter_wave, filter_throughput

def test_chromatic_add():
    """Test the `+` operator on ChromaticObjects"""
    import time
    t1 = time.time()

    pixel_scale = 0.2 * galsim.arcsec
    stamp_size = 32

    bulge = sample_bulge_gal()
    disk = sample_disk_gal()
    PSF = sample_PSF(pixel_scale=pixel_scale)
    filter_wave, filter_throughput = sample_filter()
    shear_g1 = 0.01
    shear_g2 = 0.02

    #line that actually tests the `+` operator
    obj = bulge + disk
    obj.applyShear(g1=shear_g1, g2=shear_g2)

    pixel = galsim.Pixel(pixel_scale / galsim.arcsec)
    gal = galsim.ChromaticConvolve([obj, PSF, pixel])
    image = galsim.ImageD(stamp_size, stamp_size, pixel_scale / galsim.arcsec)
    gal.draw(filter_wave, filter_throughput, image=image)

    bulge_img = galsim.ImageD(stamp_size, stamp_size, pixel_scale / galsim.arcsec)
    bulge_part = galsim.ChromaticConvolve([bulge, PSF, pixel])
    bulge_part.draw(filter_wave, filter_throughput, image=bulge_img)
    disk_img = galsim.ImageD(stamp_size, stamp_size, pixel_scale / galsim.arcsec)
    disk_part = galsim.ChromaticConvolve([disk, PSF, pixel])
    disk_part.draw(filter_wave, filter_throughput, image=disk_img)

    otherimage = bulge_img+disk_img
    printval(image, otherimage)
    np.testing.assert_array_almost_equal(
            image.array, otherimage.array, 5,
            err_msg="`+` operator doesn't match manual image addition")

    bulge_img += disk_img
    printval(image, bulge_img)
    np.testing.assert_array_almost_equal(
            image.array, bulge_img.array, 5,
            err_msg="`+=` operator doesn't match manual image addition")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_chromatic_add_draw():
    """Test ChromaticAdd.draw()"""
    import time
    t1 = time.time()

    #bulge component parameters
    bulge_n = 4.0
    bulge_hlr = 1.0 # arcsec
    bulge_e1 = 0.4
    bulge_e2 = 0.2
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

    #shear!
    bulge.applyShear(g1=shear_g1, g2=shear_g2)
    disk.applyShear(g1=shear_g1, g2=shear_g2)

    #now work out the PSF
    PSF_hlr = 0.3 # arcsec
    PSF_beta = 3.0
    PSF_e1 = 0.01
    PSF_e2 = 0.06

    pixel_scale = 0.2 * galsim.arcsec
    stamp_size = 32
    r610 = refraction_in_pixels(610, zenith_angle, pixel_scale)
    shift_fn = lambda wave: (0, refraction_in_pixels(wave, zenith_angle, pixel_scale) - r610)

    PSF = galsim.ChromaticShiftAndDilate(galsim.Moffat,
                                        shift_fn=shift_fn,
                                        dilate_fn=relative_seeing,
                                        beta=PSF_beta,
                                        half_light_radius=PSF_hlr)
    PSF.applyShear(e1=PSF_e1, e2=PSF_e2)

    filter_wave, filter_throughput = np.genfromtxt(os.path.join(datapath, 'LSST_r.dat')).T
    wgood = (filter_wave >= 500) & (filter_wave <= 720) # truncate out-of-band wavelengths
    filter_wave = filter_wave[wgood][0::100]  # sparsify from 1 Ang binning to 100 Ang binning
    filter_throughput = filter_throughput[wgood][0::100]

    pixel = galsim.Pixel(pixel_scale / galsim.arcsec)
    gal = galsim.ChromaticAdd([galsim.ChromaticConvolve([bulge, PSF, pixel]),
                               galsim.ChromaticConvolve([disk, PSF, pixel])])
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
            err_msg="ChromaticAdd.draw() doesn't match manual image addition")

    bulge_img += disk_img
    printval(image, bulge_img)
    np.testing.assert_array_almost_equal(
            image.array, bulge_img.array, 5,
            err_msg="ChromaticAdd.draw() doesn't match manual image addition")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_dcr_moments():
    """Check that zenith-direction surface brightness distribution first and second moments obey
    expected behavior for differential chromatic refraction when comparing stars drawn with
    different SEDs."""

    import time
    t1 = time.time()

    zenith_angle = 45.0 * galsim.degrees
    pixel_scale = 0.2 * galsim.arcsec
    stamp_size = 64

    wave1, flambda1 = np.genfromtxt(os.path.join(datapath, 'CWW_Sbc_ext.sed')).T
    wave2, flambda2 = np.genfromtxt(os.path.join(datapath, 'CWW_E_ext.sed')).T
    photons1 = flambda1 * wave1
    photons2 = flambda2 * wave2

    gal1 = galsim.ChromaticBaseObject(galsim.Gaussian, wave1, photons1, fwhm=1.0)
    gal2 = galsim.ChromaticBaseObject(galsim.Gaussian, wave2, photons2, fwhm=1.0)

    r610 = refraction_in_pixels(610, zenith_angle, pixel_scale)
    shift_fn = lambda wave: (0, refraction_in_pixels(wave, zenith_angle, pixel_scale) - r610)
    PSF = galsim.ChromaticShiftAndDilate(galsim.Gaussian,
                                         shift_fn=shift_fn,
                                         fwhm=0.6)
    pix = galsim.Pixel(pixel_scale / galsim.arcsec)

    scn1 = galsim.ChromaticConvolve([gal1, PSF, pix])
    scn2 = galsim.ChromaticConvolve([gal2, PSF, pix])

    img1 = galsim.ImageD(stamp_size, stamp_size, 0.2)
    img2 = galsim.ImageD(stamp_size, stamp_size, 0.2)

    filter_wave, filter_throughput = np.genfromtxt(os.path.join(datapath, 'LSST_r.dat')).T
    wgood = (filter_wave >= 500) & (filter_wave <= 720) # truncate out-of-band wavelengths
    filter_wave = filter_wave[wgood][0::100]  # sparsify from 1 Ang binning to 100 Ang binning
    filter_throughput = filter_throughput[wgood][0::100]

    scn1.draw(filter_wave, filter_throughput, image=img1)
    scn2.draw(filter_wave, filter_throughput, image=img2)

    m1 = getmoments(img1)
    m2 = getmoments(img2)
    dR_image = (m1[1] - m2[1]) * pixel_scale / galsim.arcsec
    dV_image = (m1[3] - m2[3]) * (pixel_scale / galsim.arcsec)**2

    sed1 = galsim.LookupTable(wave1, photons1)
    sed2 = galsim.LookupTable(wave2, photons2)
    filt = galsim.LookupTable(filter_wave, filter_throughput)

    #analytic first moment differences
    numR1 = galsim.integ.int1d((lambda w:(refraction_in_pixels(w, zenith_angle, pixel_scale)
                                          * filt(w) * sed1(w))),
                               500, 720)
    numR2 = galsim.integ.int1d((lambda w:(refraction_in_pixels(w, zenith_angle, pixel_scale)
                                          * filt(w) * sed2(w))),
                               500, 720)
    den1 = galsim.integ.int1d((lambda w:(filt(w) * sed1(w))), 500, 720)
    den2 = galsim.integ.int1d((lambda w:(filt(w) * sed2(w))), 500, 720)

    R1 = numR1/den1
    R2 = numR2/den2
    dR_analytic = R1 - R2

    #analytic second moment differences
    numV1 = galsim.integ.int1d((lambda w:((refraction_in_pixels(w, zenith_angle, pixel_scale)-R1)**2
                                          * filt(w) * sed1(w))),
                               500, 720)
    numV2 = galsim.integ.int1d((lambda w:((refraction_in_pixels(w, zenith_angle, pixel_scale)-R2)**2
                                          * filt(w) * sed2(w))),
                               500, 720)
    V1 = numV1/den1
    V2 = numV2/den2
    dV_analytic = V1 - V2

    np.testing.assert_almost_equal(dR_image, dR_analytic, 4,
                                   err_msg="Moment Shift from DCR doesn't match analytic formula")
    np.testing.assert_almost_equal(dV_image, dV_analytic, 4,
                                   err_msg="Moment Shift from DCR doesn't match analytic formula")

    print 'image delta R:    {}'.format(dR_image)
    print 'analytic delta R: {}'.format(dR_analytic)
    print 'image delta V:    {}'.format(dV_image)
    print 'analytic delta V: {}'.format(dV_analytic)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_chromatic_seeing_moments():
    """Check that surface brightness distribution second moments obey expected behavior
    for chromatic seeing when comparing stars drawn with different SEDs."""

    import time
    t1 = time.time()

    pixel_scale = 0.01 * galsim.arcsec
    stamp_size = 512

    wave1, flambda1 = np.genfromtxt(os.path.join(datapath, 'CWW_Sbc_ext.sed')).T
    wave2, flambda2 = np.genfromtxt(os.path.join(datapath, 'CWW_E_ext.sed')).T
    photons1 = flambda1 * wave1
    photons2 = flambda2 * wave2

    filter_wave, filter_throughput = np.genfromtxt(os.path.join(datapath, 'LSST_r.dat')).T
    wgood = (filter_wave >= 500) & (filter_wave <= 720) # truncate out-of-band wavelengths
    filter_wave = filter_wave[wgood][0::100]  # sparsify from 1 Ang binning to 100 Ang binning
    filter_throughput = filter_throughput[wgood][0::100]

    gal1 = galsim.ChromaticBaseObject(galsim.Gaussian, wave1, photons1, fwhm=1e-6)
    gal2 = galsim.ChromaticBaseObject(galsim.Gaussian, wave2, photons2, fwhm=1e-6)

    pix = galsim.Pixel(pixel_scale / galsim.arcsec)

    indices = [-0.2, 0.6, 1.0]
    for index in indices:

        PSF = galsim.ChromaticShiftAndDilate(galsim.Gaussian,
                                             dilate_fn=lambda w: (w/500.0)**index,
                                             fwhm=0.5)

        scn1 = galsim.ChromaticConvolve([gal1, PSF, pix])
        scn2 = galsim.ChromaticConvolve([gal2, PSF, pix])

        img1 = galsim.ImageD(stamp_size, stamp_size, 0.2)
        img2 = galsim.ImageD(stamp_size, stamp_size, 0.2)

        scn1.draw(filter_wave, filter_throughput, image=img1)
        scn2.draw(filter_wave, filter_throughput, image=img2)

        m1 = getmoments(img1)
        m2 = getmoments(img2)
        dr2byr2_image = ((m1[2]+m1[3]) - (m2[2]+m2[3])) / (m1[2]+m1[3])

        sed1 = galsim.LookupTable(wave1, photons1)
        sed2 = galsim.LookupTable(wave2, photons2)
        filt = galsim.LookupTable(filter_wave, filter_throughput)

        #analytic moment differences
        num1 = galsim.integ.int1d(lambda w:(w/500.0)**(2*index) * filt(w) * sed1(w), 500, 720)
        num2 = galsim.integ.int1d(lambda w:(w/500.0)**(2*index) * filt(w) * sed2(w), 500, 720)
        den1 = galsim.integ.int1d(lambda w:filt(w) * sed1(w), 500, 720)
        den2 = galsim.integ.int1d(lambda w:filt(w) * sed2(w), 500, 720)

        r2_1 = num1/den1
        r2_2 = num2/den2

        dr2byr2_analytic = (r2_1 - r2_2) / r2_1

        np.testing.assert_almost_equal(dr2byr2_image, dr2byr2_analytic, 4,
                                       err_msg="Moment Shift from chromatic seeing doesn't"+
                                               " match analytic formula")

        print 'image delta(r^2) / r^2:    {}'.format(dr2byr2_image)
        print 'analytic delta(r^2) / r^2: {}'.format(dr2byr2_analytic)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_monochromatic_filter():
    """Check that ChromaticObject drawn through a very narrow band filter matches analogous GSObject.
    """

    import time
    t1 = time.time()

    zenith_angle = 45.0 * galsim.degrees
    pixel_scale = 0.2 * galsim.arcsec
    stamp_size = 256

    wave, flambda = np.genfromtxt(os.path.join(datapath, 'CWW_Sbc_ext.sed')).T
    photons = flambda * wave
    gal = galsim.ChromaticBaseObject(galsim.Gaussian, wave, photons, fwhm=1.0)

    r610 = refraction_in_pixels(610, zenith_angle, pixel_scale)
    shift_fn = lambda wave: (0, refraction_in_pixels(wave, zenith_angle, pixel_scale) - r610)
    dilate_fn = lambda wave: (wave/500.0)**(-0.2)
    PSF = galsim.ChromaticShiftAndDilate(galsim.Gaussian,
                                         shift_fn=shift_fn,
                                         dilate_fn=dilate_fn,
                                         fwhm=0.6)
    pix = galsim.Pixel(pixel_scale / galsim.arcsec)
    scn = galsim.ChromaticConvolve([gal, PSF, pix])

    gal2 = galsim.Gaussian(fwhm=1.0)

    filter_waves = [350, 475, 625, 750, 875, 975] # approx ugrizy central wavelengths

    for filter_wave in filter_waves:
        img = galsim.ImageD(stamp_size, stamp_size, 0.2)
        scn.draw([filter_wave - 0.1, filter_wave + 0.1], [1.0]*2, image=img)

        PSF2 = galsim.Gaussian(fwhm=0.6)
        PSF2.applyDilation(dilate_fn(filter_wave))
        PSF2.applyShift(shift_fn(filter_wave))
        scn2 = galsim.Convolve([gal2, PSF2, pix])
        img2 = galsim.ImageD(stamp_size, stamp_size, 0.2)
        scn2.draw(image=img2)

        img /= img.array.max()
        img2 /= img2.array.max()

        np.testing.assert_array_almost_equal(img.array, img2.array, 3,
                err_msg="ChromaticObject.draw() with monochromatic filter doesn't match"+
                        "GSObject.draw()")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_chromatic_direct_vs_galsim()
    test_chromatic_add()
    test_chromatic_add_draw()
    test_dcr_moments()
    test_chromatic_seeing_moments()
    test_monochromatic_filter()
