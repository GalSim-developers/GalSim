# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from __future__ import print_function
import os
import time
import numpy as np
import galsim

path, filename = os.path.split(__file__)
sedpath = os.path.abspath(os.path.join(path, "../share/"))
bppath = os.path.abspath(os.path.join(path, "../examples/data/"))

def dDCR_moments(SED1, SED2, bandpass):
    zenith_angle = np.pi/4.0 * galsim.radians
    R500 = galsim.dcr.get_refraction(500, zenith_angle) * galsim.radians

    # analytic first moment differences
    R = lambda w:(galsim.dcr.get_refraction(w, zenith_angle)*galsim.radians - R500) / galsim.arcsec
    x1 = np.union1d(bandpass.wave_list, SED1.wave_list)
    x1 = x1[(x1 >= bandpass.blue_limit) & (x1 <= bandpass.red_limit)]
    x2 = np.union1d(bandpass.wave_list, SED2.wave_list)
    x2 = x2[(x2 >= bandpass.blue_limit) & (x2 <= bandpass.red_limit)]
    numR1 = np.trapz(R(x1) * bandpass(x1) * SED1(x1), x1)
    numR2 = np.trapz(R(x2) * bandpass(x2) * SED2(x2), x2)
    den1 = SED1.calculateFlux(bandpass)
    den2 = SED2.calculateFlux(bandpass)

    R1 = numR1/den1
    R2 = numR2/den2
    dR_analytic = R1 - R2

    # analytic second moment differences
    V1_kernel = lambda w:(R(w) - R1)**2
    V2_kernel = lambda w:(R(w) - R2)**2
    numV1 = np.trapz(V1_kernel(x1) * bandpass(x1) * SED1(x1), x1)
    numV2 = np.trapz(V2_kernel(x2) * bandpass(x2) * SED2(x2), x2)
    V1 = numV1/den1
    V2 = numV2/den2
    dV_analytic = V1 - V2

    return dR_analytic, dV_analytic, len(x2)

def dseeing_moments(SED1, SED2, bandpass):
    index = -0.2
    # analytic moment differences
    x1 = np.union1d(bandpass.wave_list, SED1.wave_list)
    x1 = x1[(x1 <= bandpass.red_limit) & (x1 >= bandpass.blue_limit)]
    x2 = np.union1d(bandpass.wave_list, SED2.wave_list)
    x2 = x2[(x2 <= bandpass.red_limit) & (x2 >= bandpass.blue_limit)]
    num1 = np.trapz((x1/500)**(2*index) * bandpass(x1) * SED1(x1), x1)
    num2 = np.trapz((x2/500)**(2*index) * bandpass(x2) * SED2(x2), x2)
    den1 = SED1.calculateFlux(bandpass)
    den2 = SED2.calculateFlux(bandpass)

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
    SED_files = glob.glob(os.path.join(sedpath, '*.sed'))
    bp_files = glob.glob(os.path.join(bppath, '*.dat'))
    SED1 = galsim.SED(os.path.join(sedpath, 'CWW_E_ext.sed'),
                      wave_type='nm', flux_type='flambda').withFluxDensity(0.01, 500.0)
    SEDs = dict([(os.path.basename(SED_file),
                  galsim.SED(SED_file, wave_type='nm', flux_type='flambda')) for SED_file in SED_files])
    del SEDs['CWW_E_ext.sed']
    bands = dict([(os.path.basename(bp_file),
                   galsim.Bandpass(bp_file, wave_type='nm')) for bp_file in bp_files])
    redshifts = [0.0, 0.5, 1.0]
    rel_errs = [1.e-4, 1.e-3]
    for SED_name, SED0 in SEDs.items():
        for redshift in redshifts:
            SED = SED0.atRedshift(redshift).withFluxDensity(0.01, 500.0)
            for bandname, band in bands.items():
                print('{0} SED at z={1} through {2} filter'.format(
                    SED_name, redshift, bandname))
                dDCR = dDCR_moments(SED1, SED, band)
                dseeing = dseeing_moments(SED1, SED, band)
                flux = SED.calculateFlux(band)
                hdr = '{0:8s} {1:>8s} {2:>8s} {3:>8s} {4:>8s} {5:>8s} {6:>8s} {7:>8s} {8:>8s}'
                print(hdr.format(
                    'rel_err', 'dRbar', 'dV', 'dseeing', 'flux',
                    'd(dRbar)', 'd(dV)', 'd(dseeing)', 'd(flux)/flux'))
                out = '{0:8} {1:8.5f} {2:8.5f} {3:8.5f} {4:8.5f}'
                print(out.format('full', dDCR[0], dDCR[1], dseeing, flux))
                for rel_err in rel_errs:
                    band1 = band.thin(rel_err=rel_err)
                    dDCR_thinned = dDCR_moments(SED1, SED, band1)
                    dseeing_thinned = dseeing_moments(SED1, SED, band1)
                    flux_thinned = SED.calculateFlux(band1)
                    out = ('{0:8s} {1:8.5f} {2:8.5f} {3:8.5f} {4:8.5f}'
                           +' {5:8.5f} {6:8.5f} {7:8.5f} {8:8.5f}')
                    print(out.format(
                        str(rel_err), dDCR_thinned[0], dDCR_thinned[1],
                        dseeing_thinned, flux_thinned,
                        dDCR_thinned[0] - dDCR[0], dDCR_thinned[1] - dDCR[1],
                        dseeing_thinned - dseeing, (flux_thinned - flux)/flux))
                print()

    print('{0:8s} {1:>8s} {2:>8s}'.format('rel_err', 'time', 'Neval'))
    t0 = time.time()
    for i in range(20):
        dDCR_thinned = dDCR_moments(SED1, SED, band)
        dseeing_thinned = dseeing_moments(SED1, SED, band)
        flux_thinned = SED.calculateFlux(band)
    t1 = time.time()
    print('{0:8s} {1:8.5f} {2:8d}'.format('full', t1-t0, dDCR_thinned[2]))

    for rel_err in rel_errs:
        band1 = band.thin(rel_err=rel_err)
        t0 = time.time()
        for i in range(20):
            dDCR_thinned = dDCR_moments(SED1, SED, band1)
            dseeing_thinned = dseeing_moments(SED1, SED, band1)
            flux_thinned = SED.calculateFlux(band1)
        t1 = time.time()
        print('{0:8s} {1:8.5f} {2:8d}'.format(str(rel_err), t1-t0, dDCR_thinned[2]))

if __name__ == '__main__':
    compare_thin()
