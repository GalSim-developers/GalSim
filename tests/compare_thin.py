import os

import numpy as np

import galsim

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

def dDCR_moments(SED1, SED2, bandpass):
    zenith_angle = np.pi/4.0 * galsim.radians
    R500 = galsim.dcr.get_refraction(500, zenith_angle)

    # analytic first moment differences
    R = lambda w:(galsim.dcr.get_refraction(w, zenith_angle) - R500) / galsim.arcsec
    numR1 = galsim.integ.int1d(lambda w: R(w) * bandpass(w) * SED1(w),
                               bandpass.blue_limit, bandpass.red_limit)
    numR2 = galsim.integ.int1d(lambda w: R(w) * bandpass(w) * SED2(w),
                               bandpass.blue_limit, bandpass.red_limit)
    den1 = galsim.integ.int1d(lambda w:bandpass(w) * SED1(w),
                              bandpass.blue_limit, bandpass.red_limit)
    den2 = galsim.integ.int1d(lambda w:bandpass(w) * SED2(w),
                              bandpass.blue_limit, bandpass.red_limit)
    R1 = numR1/den1
    R2 = numR2/den2
    dR_analytic = R1 - R2

    # analytic second moment differences
    V1_kernel = lambda w:(R(w) - R1)**2
    V2_kernel = lambda w:(R(w) - R2)**2
    numV1 = galsim.integ.int1d(lambda w:V1_kernel(w) * bandpass(w) * SED1(w),
                               bandpass.blue_limit, bandpass.red_limit)
    numV2 = galsim.integ.int1d(lambda w:V2_kernel(w) * bandpass(w) * SED2(w),
                               bandpass.blue_limit, bandpass.red_limit)
    V1 = numV1/den1
    V2 = numV2/den2
    dV_analytic = V1 - V2

    return dR_analytic, dV_analytic

def dseeing_moments(SED1, SED2, bandpass):
    index = -0.2
    # analytic moment differences
    num1 = galsim.integ.int1d(lambda w:(w/500.0)**(2*index) * bandpass(w) * SED1(w),
                              bandpass.blue_limit, bandpass.red_limit)
    num2 = galsim.integ.int1d(lambda w:(w/500.0)**(2*index) * bandpass(w) * SED2(w),
                              bandpass.blue_limit, bandpass.red_limit)
    den1 = galsim.integ.int1d(lambda w:bandpass(w) * SED1(w),
                              bandpass.blue_limit, bandpass.red_limit)
    den2 = galsim.integ.int1d(lambda w:bandpass(w) * SED2(w),
                              bandpass.blue_limit, bandpass.red_limit)

    r2_1 = num1/den1
    r2_2 = num2/den2

    dr2byr2_analytic = (r2_1 - r2_2) / r2_1
    return dr2byr2_analytic

def compare_thin():
    # compare the differences in chromatic moment shifts between two SEDs as a function of
    # Bandpass thinning.  Goals should be to keep the error below:
    # sigma(dRbar) < 0.01 arcsec
    # sigma(dV) < 0.0001 arcsec^2
    # sigma(dseeing) < 0.0001
    import glob
    SED_files = glob.glob(os.path.join(datapath, '*.sed'))
    bp_files = glob.glob(os.path.join(datapath, '*.dat'))
    SED1 = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed')).withFluxDensity(0.01, 500.0)
    SEDs = dict([(os.path.basename(SED_file), galsim.SED(SED_file)) for SED_file in SED_files])
    del SEDs['CWW_E_ext.sed']
    bands = dict([(os.path.basename(bp_file), galsim.Bandpass(bp_file)) for bp_file in bp_files])
    redshifts = [0.0, 0.5, 1.0]
    rel_errs = [1.e-6, 1.e-5, 1.e-4, 1.e-3]
    for SED_name, SED0 in SEDs.iteritems():
        for redshift in redshifts:
            SED = SED0.atRedshift(redshift).withFluxDensity(0.01, 500.0)
            for bandname, band in bands.iteritems():
                print '{} SED at z={} through {} filter'.format(
                    SED_name, redshift, bandname)
                dDCR = dDCR_moments(SED1, SED, band)
                dseeing = dseeing_moments(SED1, SED, band)
                dflux = SED1.calculateFlux(band) - SED.calculateFlux(band)
                hdr = '{0:8s} {1:>8s} {2:>8s} {3:>8s} {4:>8s} {5:>8s} {6:>8s} {7:>8s} {8:>8s}'
                print hdr.format(
                    'rel_err', 'dRbar', 'dV', 'dseeing', 'dflux',
                    'd(dRbar)', 'd(dV)', 'd(dseeing)', 'd(dflux)')
                out = '{0:8} {1:8.5f} {2:8.5f} {3:8.5f} {4:8.5f}'
                print out.format('full', dDCR[0], dDCR[1], dseeing, dflux)
                for rel_err in rel_errs:
                    band1 = band.thin(rel_err=rel_err)
                    dDCR_thinned = dDCR_moments(SED1, SED, band1)
                    dseeing_thinned = dseeing_moments(SED1, SED, band1)
                    dflux_thinned = SED1.calculateFlux(band1) - SED.calculateFlux(band1)
                    out = ('{0:8s} {1:8.5f} {2:8.5f} {3:8.5f} {4:8.5f}'
                           +' {5:8.5f} {6:8.5f} {7:8.5f} {8:8.5f}')
                    print out.format(
                        str(rel_err), dDCR_thinned[0], dDCR_thinned[1],
                        dseeing_thinned, dflux_thinned,
                        dDCR[0] - dDCR_thinned[0], dDCR[1] - dDCR_thinned[1],
                        dseeing - dseeing_thinned, dflux - dflux_thinned)
                print

if __name__ == '__main__':
    compare_thin()
