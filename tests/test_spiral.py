import numpy as np
import galsim
from galsim_test_helpers import (
    check_basic,
    do_shoot,
    do_kvalue,
    check_pickle,
)
from galsim.errors import (
    GalSimValueError,
    GalSimRangeError,
    # GalSimIncompatibleValuesError,
)
import pytest


def _get_example_kw():
    return {
        'npoints': 100,
        'half_light_radius': 1.0,
        'flux': 3.5,
        'narms': 3,
        'angle_fuzz': 0.07,
        'xy_fuzz': 0.09,
        'rel_height': 0.13,
        'inclination': 45 * galsim.degrees,
        'rotation': -20 * galsim.degrees,
    }


def test_spiral_defaults():
    """
    Create a spiral galaxy and test that the getters work for
    default inputs
    """

    seed = 1234
    npoints = 100
    hlr = 1.0
    rng = galsim.BaseDeviate(seed)
    sp = galsim.Spiral(npoints, half_light_radius=hlr, rng=rng)

    assert sp.npoints == npoints, (
        'expected npoints==%d, got %d' % (npoints, sp.npoints)
    )
    assert sp.input_half_light_radius == hlr, (
        'expected hlr==%g, got %g' % (hlr, sp.input_half_light_radius)
    )

    nobj = sp.points.shape[0]
    assert nobj == npoints, (
        'expected %d objects, got %d' % (npoints, nobj)
    )

    pts = sp.points
    assert pts.shape == (npoints, 3), (
        'expected (%d,2) shape for points, got %s' % (npoints, pts.shape)
    )
    np.testing.assert_almost_equal(sp.centroid.x, pts[:, 0].mean())
    np.testing.assert_almost_equal(sp.centroid.y, pts[:, 1].mean())

    sp.calculateHLR()
    sp.calculate_e1e2()

    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    rng2 = galsim.BaseDeviate(seed)
    sp2 = galsim.Spiral(npoints, half_light_radius=hlr, rng=rng2, gsparams=gsp)

    assert sp2 != sp
    assert sp2 == sp.withGSParams(gsp)
    assert sp2 == sp.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

    # Check that they produce identical images.
    psf = galsim.Gaussian(sigma=0.8)
    conv1 = galsim.Convolve(sp.withGSParams(gsp), psf)
    conv2 = galsim.Convolve(sp2, psf)
    im1 = conv1.drawImage()
    im2 = conv2.drawImage()
    assert im1 == im2

    # Check that image is not sensitive to use of rng by other objects.
    rng3 = galsim.BaseDeviate(seed)
    sp3 = galsim.Spiral(npoints, half_light_radius=hlr, rng=rng3)
    rng3.discard(523)
    conv1 = galsim.Convolve(sp, psf)
    conv3 = galsim.Convolve(sp3, psf)
    im1 = conv1.drawImage()
    im3 = conv3.drawImage()
    assert im1 == im3

    # Run some basic tests of correctness
    check_basic(conv1, 'Spiral')

    im = galsim.ImageD(256, 256, scale=0.2)

    do_shoot(conv1, im, 'Spiral')
    do_kvalue(conv1, im, 'Spiral')

    check_pickle(sp, irreprable=True)
    check_pickle(conv1, irreprable=True)
    check_pickle(conv1, lambda x: x.drawImage(scale=1), irreprable=True)

    # Check negative flux
    sp3 = sp.withFlux(-2.3)
    assert sp3 == galsim.Spiral(
        npoints,
        half_light_radius=hlr,
        rng=galsim.BaseDeviate(seed),
        flux=-2.3,
    )
    conv = galsim.Convolve(sp3, psf)
    check_basic(conv, 'Spiral with negative flux')


@pytest.mark.parametrize('use_config', [False, True])
def test_spiral_valid_input(use_config):
    """
    Create a Spiral galaxy and test that the getters work for
    valid non-default inputs
    """

    seed = 35
    rng = galsim.BaseDeviate(seed)

    kw = _get_example_kw()

    if use_config:
        gal_config = {'type': 'Spiral'}
        gal_config.update(kw)
        config = {
            'gal': gal_config,
            'rng': rng,
        }
        sp = galsim.config.BuildGSObject(config, 'gal')[0]
    else:
        kw['rng'] = rng
        sp = galsim.Spiral(**kw)

    pts = sp.points
    nobj = pts.shape[0]
    assert nobj == kw['npoints'], (
        'expected %d objects, got %d' % (kw['npoints'], nobj)
    )

    assert pts.shape == (kw['npoints'], 3), (
        'expected (%d, 3) shape for '
        'points, got %s' % (kw['npoints'], pts.shape)
    )

    for key, val in kw.items():
        if key == 'rng':
            continue

        if key == 'half_light_radius':
            attrname = 'input_half_light_radius'
        else:
            attrname = key

        spval = getattr(sp, attrname)

        assert spval == val, (
            'expected %s==%s, got %s' % (attrname, val, spval)
        )


def test_spiral_points_input():
    """
    Check when sending points
    """

    seed = 35
    rng = galsim.BaseDeviate(seed)

    kw = dict(
        npoints=100,
        half_light_radius=1.0,
        flux=3.5,
        narms=3,
        angle_fuzz=0.07,
        xy_fuzz=0.09,
        rel_height=0.13,
        inclination=45 * galsim.degrees,
        rotation=-20 * galsim.degrees,
        rng=rng,
    )

    sp0 = galsim.Spiral(**kw)
    sp = galsim.Spiral(points=sp0.points, flux=kw['flux'])

    assert sp.flux == sp0.flux, (
        'expected flux==%g, got %g' % (sp0.flux, sp.flux)
    )
    assert np.array_equal(sp.points, sp0.points), (
        'points not equal',
    )


def test_spiral_invalid_inputs():
    kw = {
        'npoints': 100,
        'half_light_radius': 1.0,
    }

    # integers > 0
    int_names = ['npoints', 'narms']
    for name in int_names:
        this_kw = kw.copy()
        this_kw[name] = 'blah'
        with pytest.raises(GalSimValueError):
            galsim.Spiral(**this_kw)

        for val in [0, -1]:
            this_kw = kw.copy()
            this_kw[name] = val
            with pytest.raises(GalSimRangeError):
                galsim.Spiral(**this_kw)

    # floats >= 0
    float_names = ['half_light_radius', 'angle_fuzz', 'xy_fuzz', 'rel_height']
    for name in float_names:
        this_kw = kw.copy()
        this_kw[name] = 'blah'
        with pytest.raises(GalSimValueError):
            galsim.Spiral(**this_kw)

        this_kw = kw.copy()
        this_kw[name] = -0.2
        with pytest.raises(GalSimRangeError):
            galsim.Spiral(**this_kw)


def test_transforms():

    seed = 35
    rng = galsim.BaseDeviate(seed)
    kw = _get_example_kw()
    kw['rng'] = rng

    sp = galsim.Spiral(**kw)
    x, y = sp.points[:, 0], sp.points[:, 1]

    # flux
    flux = 999
    tsp = sp.withFlux(flux)
    assert tsp.flux == flux

    # scaled flux
    fac = 1.5
    tsp = sp.withScaledFlux(fac)
    assert tsp.flux == kw['flux'] * fac

    # dilate
    dilate = 1.1
    tsp = sp.dilate(dilate)
    assert tsp.input_half_light_radius == kw['half_light_radius'] * dilate

    # expand and magnify
    tsp = sp.expand(dilate)
    assert tsp.input_half_light_radius == kw['half_light_radius'] * dilate
    assert tsp.flux == kw['flux'] / dilate**2

    tsp = sp.magnify(dilate)
    assert tsp.input_half_light_radius == kw['half_light_radius'] * dilate
    assert tsp.flux == kw['flux'] / dilate**2

    # shifts
    shift = (1.5, 2.5)
    tsp = sp.shift(shift)
    assert np.all(
        (tsp.points[:, 0] == (shift[0] + sp.points[:, 0]))
        &
        (tsp.points[:, 1] == (shift[1] + sp.points[:, 1]))
    )

    # transform
    dudx, dudy, dvdx, dvdy = 1.1, -0.02, 0.03, 0.95
    tsp = sp.transform(dudx, dudy, dvdx, dvdy)
    xp = dudx * x + dudy * y
    yp = dvdx * x + dvdy * y
    assert np.all(
        (tsp.points[:, 0] == xp)
        &
        (tsp.points[:, 1] == yp)
    )

    # rotate
    rot = 20 * galsim.degrees
    tsp = sp.rotate(rot)

    radrot = rot / galsim.radians
    sinrot = np.sin(radrot)
    cosrot = np.cos(radrot)
    xp = x * cosrot - y * sinrot
    yp = x * sinrot + y * cosrot

    assert np.all(
        (tsp.points[:, 0] == xp)
        &
        (tsp.points[:, 1] == yp)
    )

    # shear
    sh = galsim.Shear(g1=0.1, g2=-0.05)
    tsp = sp.shear(sh)

    # we already tested transform above, use it again
    mat = sh.getMatrix()
    tsp2 = sp.transform(
        dudx=mat[0, 0],
        dudy=mat[0, 1],
        dvdx=mat[1, 0],
        dvdy=mat[1, 1],
    )
    assert np.all(
        (tsp.points[:, 0] == tsp2.points[:, 0])
        &
        (tsp.points[:, 1] == tsp2.points[:, 1])
    )


def test_spiral_hlr():
    """
    Create a spiral galaxy and test that the half light radius
    is consistent with the requested value
    """

    rng = galsim.BaseDeviate(8899)

    # should be within 5 sigma
    nsig = 5

    # need to look at it face on
    inc = 0 * galsim.degrees

    hlr = 1.0
    ntrial = 100
    hlr_vals = np.zeros(ntrial)

    for i in range(ntrial):
        sp = galsim.Spiral(
            npoints=10000,
            half_light_radius=hlr,
            inclination=inc,
            rng=rng,
        )
        hlr_vals[i] = sp.calculateHLR()

    hlr_mean = hlr_vals.mean()
    hlr_err = hlr_vals.std() / np.sqrt(ntrial)
    hrng = [hlr - nsig * hlr_err, hlr + nsig * hlr_err]
    hstr = f'[{hrng[0]:g}, {hrng[1]:g}]'

    mess = f'hlr {hlr_mean} outside of expected range {hstr}'
    assert abs(hlr_mean - hlr) < nsig * hlr_err, mess


def test_spiral_ellip():
    """
    Create a spiral galaxy and test that the ellipticity is about right
    """

    npoints = 10_000
    hlr = 1.0
    narms = 4
    rng = galsim.BaseDeviate(1298)

    # face on, should have low ellip
    sp = galsim.Spiral(
        npoints=npoints,
        half_light_radius=hlr,
        inclination=0 * galsim.degrees,
        narms=narms,
        rng=rng,
    )
    e1, e2 = sp.calculate_e1e2()
    etot = np.sqrt(e1**2 + e2**2)
    assert abs(etot) < 0.05

    # edge on along x axis
    sp = galsim.Spiral(
        npoints=npoints,
        half_light_radius=hlr,
        inclination=90 * galsim.degrees,
        rotation=0 * galsim.degrees,
        narms=narms,
        rng=rng,
    )
    e1, e2 = sp.calculate_e1e2()
    assert abs(e1) > 0.90

    # edge on at 45 degrees
    sp = galsim.Spiral(
        npoints=npoints,
        half_light_radius=hlr,
        inclination=90 * galsim.degrees,
        rotation=45 * galsim.degrees,
        narms=narms,
        rng=rng,
    )
    e1, e2 = sp.calculate_e1e2()
    assert abs(e2) > 0.90


def test_spiral_sed():
    """
    Test Spiral with an SED

    Based on a similar test in test_knots.py
    """
    sed = galsim.SED('CWW_E_ext.sed', 'A', 'flambda')
    sp = galsim.Spiral(
        npoints=100,
        half_light_radius=1.3,
        flux=100,
    )

    gal1 = galsim.ChromaticObject(sp) * sed
    gal2 = sp * sed
    check_pickle(gal1, irreprable=True)
    check_pickle(gal2, irreprable=True)

    # They don't test as ==, since they are formed differently.  But they are
    # functionally equal:

    bandpass = galsim.Bandpass('LSST_r.dat', 'nm')
    psf = galsim.Gaussian(fwhm=0.7)
    final1 = galsim.Convolve(gal1, psf)
    final2 = galsim.Convolve(gal2, psf)
    im1 = final1.drawImage(bandpass, scale=0.4)
    im2 = final2.drawImage(bandpass, scale=0.4)
    np.testing.assert_array_equal(im1.array, im2.array)


if __name__ == '__main__':
    test_spiral_valid_input(True)
