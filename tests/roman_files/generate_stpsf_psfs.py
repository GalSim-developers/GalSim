"""Generate Cycle 9 reference images through the normal STPSF WFI interface.

Run from the repository root with STPSF and its reference data installed::

    STPSF_PATH=/path/to/stpsf-data python tests/roman_files/generate_stpsf_psfs.py \
        /path/to/wim_zernikes_cycle9.csv.gz

The original Cycle 9 CSV is archived at:
https://github.com/Roman-HLIS-Cosmology-PIT/PSFSim/blob/main/src/psfsim/data/wim_zernikes_cycle9.csv.gz
Its numeric columns match the WIM sheet of the bundled 20210204 spreadsheet,
with repeated detector/wavelength/field-point rows removed.

Current STPSF requires a different CSV schema. The adapter below changes only column
names, position units/conventions, and waves to meters; it retains every detector,
field point, wavelength, and original coefficient sign. The position axis signs follow
WebbPSF's original Cycle 9 loader, with the origin placed at the zero-based center
of the cropped science image (2043.5, 2043.5):
https://github.com/spacetelescope/webbpsf/blob/v1.2.1/webbpsf/roman.py

STPSF performs field/wavelength interpolation, omits piston/tip/tilt, constructs the
phase, and propagates the PSF. No GalSim code or mocked optical methods are used.
Requires the WFI optical ordering fixed in STPSF PR 196. Generated with STPSF 2.3.1.
Reference images retain STPSF's native output orientation; no reflection is applied
to make them agree with GalSim.

The eight reference cases also use STPSF's SIAF distortion helper and POPPY's
pixel binning. Current WFI.calc_psf assumes post-Cycle-9 pupils include distortion,
so it no longer calls that helper itself. Use the bundled January 2022 Roman SIAF,
not a newer calibration, for this legacy comparison. PySIAF also supplies the sky
positions and telescope attitude, independently of GalSim's TAN-SIP implementation.

The cases mix SCAs, bands, chip positions, and wavelengths, including all four
corners of the camera without taking the Cartesian product of those choices.
POPPY's Zernike phase is temporarily patched to continue outside its normalization
circle. A direct pure-piston probe verifies this behavior during generation.

One separate, finely sampled SCA 8 image resolves its split diffraction spike.
It is used only to compare the spike directions, retaining normal STPSF phase handling.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
import poppy
import pysiaf
from pysiaf.utils import rotations
import stpsf


# Monkey patch to fix a bug in Poppy. See https://github.com/spacetelescope/poppy/pull/694
def poppy_clips_phase():
    """Determine whether POPPY zeros Zernike OPD outside the normalization circle."""
    # A pure piston is constant everywhere, so an exterior zero unambiguously
    # identifies clipping without comparing PSFs or duplicating a Zernike evaluator.
    piston = 1.e-9
    optic = poppy.ZernikeWFE(coefficients=[piston]*u.m, radius=1*u.m)
    wave = poppy.Wavefront(npix=8, diam=4.)
    y, x = wave.coordinates()
    outside = np.hypot(x, y) > 1
    opd = optic.get_opd(wave)
    np.testing.assert_allclose(opd[~outside], piston, rtol=1.e-12, atol=0)
    if np.all(opd[outside] == 0):
        return True
    else:
        np.testing.assert_allclose(opd[outside], piston, rtol=1.e-12, atol=0)
        return False

# This will be patched in as poppy.ZernikeWFE.get_opd
def unclipped_zernike_opd(self, wave):
    """Temporarily continue POPPY's Zernike phase beyond its normalization circle."""
    # Use POPPY's own radial polynomials and Noll ordering, with no GalSim calculation.
    # The normalization circle sets the coefficient scale, not the illuminated pupil.
    y, x = self.get_coordinates(wave)
    rho = np.hypot(x, y)/self.radius.to_value(u.m)
    theta = np.arctan2(y, x)
    opd = np.zeros(wave.shape)
    for j, coefficient in enumerate(self.coefficients, 1):
        coefficient = coefficient.to_value(u.m)
        if coefficient == 0:
            continue
        n, m = poppy.zernike.noll_indices(j)
        radial = poppy.zernike.R(n, m, rho)
        if m == 0:
            term = np.sqrt(n+1)*radial
        elif m > 0:
            term = np.sqrt(2*(n+1))*radial*np.cos(m*theta)
        else:
            term = np.sqrt(2*(n+1))*radial*np.sin(abs(m)*theta)
        opd += coefficient*term
    if self.aperture_stop:
        opd[rho > 1] = 0
    return opd


def convert_cycle9_table(input_path, output_path):
    """Translate the complete original Cycle 9 CSV into current STPSF's input schema."""
    original = Table.read(input_path, format='ascii.csv')
    converted = Table()
    converted['sca'] = original['sca']
    converted['fov'] = original['field_point']
    converted['wave'] = original['wavelength'] * 1000  # microns -> nm
    # Use zero-based positions on the 4088-pixel science image. The FPA-to-science
    # axis signs follow the original loader; the geometric center is now 2043.5.
    converted['SCI_X'] = 2043.5 - original['local_x']*100
    converted['SCI_Y'] = 2043.5 + original['local_y']*100
    for j in range(1, 23):
        # Convert coefficients in waves to OPD in meters; the original wavelength is in microns.
        converted['Z%d' % j] = original['Z%d' % j] * original['wavelength'] * 1.e-6
    converted.write(output_path, format='ascii.csv')


def make_reference(sca, band, position, wavelength, aberration_path, work_dir, aperture, size):
    """Load Cycle 9 data with STPSF's public input methods and call WFI.calc_psf."""
    data_dir = Path(__file__).resolve().parents[2] / 'share' / 'roman'
    pupil_type = 'F184' if band in ['F184', 'K213'] else 'skinny'
    pupil_path = data_dir / ('RST_WIM_Filter_%s_SCA_%d.fits.gz' % (pupil_type, sca))
    pupil, pupil_header = fits.getdata(pupil_path, header=True)

    # The original mask encodes transmission as 0/255. STPSF expects 0/1; preserve
    # the full-resolution geometry and the original physical pixel scale.
    normalized_pupil = work_dir / ('pupil_%d.fits' % sca)
    fits.writeto(normalized_pupil, pupil/255., header=pupil_header, overwrite=True)

    image_scale = wavelength * 1.e-9 / (
        2*pupil.shape[0]*pupil_header['PIXELSCL']) * (180/np.pi) * 3600
    if aperture is not None:
        # distort_image defines its output sampling from this mean SIAF scale.
        # Eight samples per detector pixel permit square-pixel integration below.
        image_scale = (aperture.XSciScale + aperture.YSciScale)/2/8
    wfi = stpsf.WFI()
    wfi.lock_aberrations(str(aberration_path))
    wfi.detector = 'WFI%02d' % sca
    # STPSF uses F062/F087/F158 rather than GalSim's historical R062/Z087/H158 names.
    wfi.filter = 'F'+band[1:]
    wfi.detector_position = position
    wfi.lock_pupil(str(normalized_pupil))

    # STPSF adds a detector-specific wavefront-error map to the Zernike aberrations.
    # GalSim's Cycle 9 model uses only the Zernike table, so disable that extra map.
    # The public pupilopd setter rejects assignments after a detector is selected, so
    # update the private attribute directly.
    wfi._pupilopd = None
    wfi.options['jitter'] = None
    wfi.options['add_ipc'] = False
    wfi.pixelscale = image_scale
    result = wfi.calc_psf(monochromatic=wavelength * 1.e-9, fov_pixels=size,
                          detector_oversample=1, fft_oversample=3)

    # The primary image is optical only; detector effects are in separate extensions.
    # Keep STPSF's native output orientation, including its WFI pupil/phase inversion.
    header = fits.Header()
    header['WAVELEN'] = wavelength*1.e-9
    header['DETXPIXL'], header['DETYPIXL'] = position
    header['PIXELSCL'] = image_scale
    if aperture is None:
        image = result[0].data
    else:
        # Integrate the oversampled image over the pixels.
        # STPSF records zero-based pixels, whereas PySIAF's SCI coordinates are
        # one-based. This affects only the distortion helper's input metadata.
        result[0].header['DET_X'] = position[0]+1
        result[0].header['DET_Y'] = position[1]+1
        distorted = stpsf.distortion.distort_image(result, aper=aperture)
        image = poppy.utils.rebin_array(distorted, rc=(8, 8))
        header['PIXELSCL'] = image_scale*8
        header['SIAFVER'] = pysiaf.__version__
    image /= image.sum()

    header['SCA'] = sca
    header['BANDPASS'] = band
    header['STPSFVER'] = stpsf.__version__
    header['POPPYVER'] = poppy.__version__

    return image, header


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cycle9_csv', type=Path, help='Original wim_zernikes_cycle9.csv[.gz]')
    args = parser.parse_args()
    if poppy_clips_phase():
        poppy.ZernikeWFE.get_opd = unclipped_zernike_opd
    assert not poppy_clips_phase()
    references = []
    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        aberration_path = work_dir / 'cycle9.csv'
        convert_cycle9_table(args.cycle9_csv, aberration_path)
        # Test at 8 points in the focal plane across different SCAs, filters, position
        # on the sensor, and wavelength relative to the band fiducial.
        # Positions are zero-based science pixels.
        cases = [
            (7, 'F184', (2043, 2043), 1842.),  # Chip center, fiducial wavelength.
            (9, 'J129', (8, 8), 1131.),        # Lower left, blue side of band.
            (16, 'H158', (4079, 4079), 1774.), # Upper right, red side of band.
            (18, 'R062', (4079, 8), 620.),     # Lower right, fiducial wavelength.
            (1, 'Z087', (8, 4079), 977.),      # Upper left, red side of band.
            (3, 'Y106', (2043, 2043), 927.),   # Chip center, blue side of band.
            (11, 'W146', (1031, 3027), 1464.), # Off center, near band fiducial.
            (14, 'K213', (4079, 4079), 2300.), # Upper right, red side of band.
        ]

        # Define one nonzero roll independently with PySIAF. getWCS takes the FPA
        # center and its local V3 position angle; getPSF takes the V1 boresight.
        siaf = pysiaf.Siaf('Roman')
        center = siaf['WFI_CEN']
        fpa_ra, fpa_dec, fpa_pa = 90., 0., 27.
        attitude = rotations.attitude(center.V2Ref, center.V3Ref, fpa_ra, fpa_dec, fpa_pa)
        v1_ra, v1_dec = rotations.pointing(attitude, 0, 0)
        v1_pa = rotations.posangle(attitude, 0, 0)
        for sca, band, position, wavelength in cases:
            aperture = siaf['WFI%02d_FULL' % sca]
            aperture.set_attitude_matrix(attitude)
            image, header = make_reference(sca, band, position, wavelength, aberration_path,
                                           work_dir, aperture, 41*8)
            x, y = position[0]+1, position[1]+1
            ra, dec = aperture.sci_to_sky(x, y)
            header['FPA_RA'], header['FPA_DEC'], header['FPA_PA'] = fpa_ra, fpa_dec, fpa_pa
            header['V1_RA'], header['V1_DEC'], header['V3_PA'] = v1_ra, v1_dec, v1_pa
            header['STAR_RA'], header['STAR_DEC'] = ra, dec
            # Finite differences in a local west/north tangent plane give the
            # independent sky Jacobian. Longitude increments include cos(dec).
            for axis, dx, dy in [('X', 1, 0), ('Y', 0, 1)]:
                ra1, dec1 = aperture.sci_to_sky(x+dx, y+dy)
                header['DUD'+axis] = -(ra1-ra)*np.cos(np.deg2rad(dec))*3600
                header['DVD'+axis] = (dec1-dec)*3600
            hdu_type = fits.PrimaryHDU if not references else fits.ImageHDU
            references.append(hdu_type(image.astype(np.float32), header=header))

        # In addition to the images drawn on the science images including the pixel integration,
        # we draw the SCA 8 profile at finer resolution. This one is particularly interesting,
        # since one of the diffraction spikes visibly splits into two lines, so the position
        # of this split is a strong validation of the overall orienation of the profile.
        sca, band, position, wavelength = 8, 'F184', (2043, 2043), 1842.
        image, header = make_reference(sca, band, position, wavelength, aberration_path,
                                       work_dir, None, 2049)
        # Keep the 64-76 degree sector at 350-400 lambda/D used by the spike test,
        # with at least 16 pixels of margin for cubic interpolation. Preserve pixel
        # coordinates relative to the PSF center, which is outside this cutout.
        center = (image.shape[0]-1)//2
        header['GS_XMIN'] = 1180-center
        header['GS_YMIN'] = 1650-center
        image = image[1650:1840, 1180:1410]
        spike_filename = Path(__file__).with_name('stpsf_cycle9_spikes.fits')
        fits.PrimaryHDU(image.astype(np.float32), header=header).writeto(
            spike_filename, overwrite=True)
        print('Wrote', spike_filename)

    filename = Path(__file__).with_name('stpsf_cycle9_psfs.fits')
    fits.HDUList(references).writeto(filename, overwrite=True)
    print('Wrote', filename)
