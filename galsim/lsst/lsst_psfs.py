# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
"""@file lsst_psfs.py
Utilities for LSST-specific PSF modeling.
"""
import os.path
import galsim
from phase_psf import parse_zemax_coefs_file

### Ratio of LSST primary mirror diameter to tertiary diameter
k_m1_m3_ratio = 8.36 / 5.12

#: Fixed parameters for the LSST telescope configuration used in the aberrated optics model.
#: Pass this dictionary as the `telescope_model` argument to the `TelescopePSF` class.
#: Copy this dictionary to implement other telescope models.
lsst_telescope_model = {
    "telescope_diameter_meters": 8.36,
    "f-ratio": 1.25,
    ### Coefficients for mapping bending modes to the exit pupil wavefront.
    ### Reference: Table I.5 from Ref. 6
    ###     These values from an old LSST design with M3 offset from M1
    "coefs_bending_modes": {
        "primary": {"a": 1.0, "c": 0.0},
        "secondary": {"a": 0.91134, "c": 0.08781},
        "tertiary": {"a": 0.74379, "c": 0.33296}
    },
    ### Surfaces that can have body motion misalignments or figure errors (i.e., 'bending modes')
    "surfaces": ['primary', 'secondary', 'tertiary', 'L1', 'L2', 'L3'],
    ### Wavefront power-series expansion coefficients for modeling body motions
    "W_coefs": parse_zemax_coefs_file(os.path.join(os.path.dirname(__file__),
        "data/lsst_aberration_coefs_rband_markedup.txt")),
    ### Which rows in the input matrix of W coefficients correspond to which surfaces?
    "zemax_coefs_rownums": {
        'primary': 3,
        'secondary': 7,
        'tertiary': 11,
        'L1': [13, 14],
        'L2': [16, 17],
        'L3': [22, 23]
    },
    ### Surfaces with parameterized bending mode perturbations.
    ### The tertiary bending modes are mapped from those on the primary.
    "bending_modes_parameters": {"primary": 8, "secondary": 8},
    ### Optics with parameterized body motions.
    ### The primary mirror body motions are used to define the telescope pointing.
    ### The tertiary is monolithic with the primary.
    "body_motions_parameters": {"secondary": 2, "L1": 2, "L2": 2, "L3": 2},
    ### Exit pupil parameters for input into GalSim
    "pupil_geometry": {
        "circular": True,
        "obscuration": 0.612,
        "nstruts": 4,
        "strut_thick": 0.05
        # "strut_thick": 0.0
    }
}

class TelescopePSFLSST(galsim.TelescopePSF):
    """
    Telescope model for the LSST, including special handling of the M1/M3 monolithic mirror.
    """
    def __init__(self, verbose=False, npad=2):
        super(TelescopePSFLSST, self).__init__(lsst_telescope_model,
            verbose=verbose, npad=npad)
        self.n_surfs_bend_modes = 2

    def set_perturbation_params(self, optics_params):
        """
        Assign perturbation parameters from `optics_params`, but overwrite the M3 figure errors
        with those mapped from M1.
        """
        super(TelescopePSFLSST, self).set_perturbation_params(optics_params)
        self.surf_coefs['tertiary'] = self.map_M1_bend_modes_to_M3(self.surf_coefs['primary'])
        if self.verbose:
            print "surf_coefs:", self.surf_coefs
            print "misalignments:", self.misalignments
        return None

    def map_M1_bend_modes_to_M3(self, m1_surf_coefs=None):
        """
        Given bending modes on M1, calculate a new Zernike expansion of those
        modes on the smaller circle defined by M3.

        M1 and M3 are monolithic, so the modes on M3 should be derived from the
        modes on M1. But because they make concentric circles, the Zernike modes
        get mixed between the two mirrors.
        """
        tsqt = 2 * np.sqrt(2.)
        n = 1. / k_m1_m3_ratio
        nsq = n * n
        ncb = nsq * n
        # m3_surf_coefs = copy.copy(self.surf_coefs['primary'])
        m3_surf_coefs = np.zeros_like(m1_surf_coefs, dtype=np.float64)
        ### Z2, Z3
        m3_surf_coefs[0] = n * (m1_surf_coefs[0] + tsqt * (nsq - 1) * m1_surf_coefs[6])
        m3_surf_coefs[1] = n * (m1_surf_coefs[1] + tsqt * (nsq - 1) * m1_surf_coefs[5])
        ### Z4
        m3_surf_coefs[2] = nsq * (np.sqrt(15.) * (nsq - 1) * m1_surf_coefs[9] + m1_surf_coefs[2])
        ### Z5, Z6
        m3_surf_coefs[3] = nsq * m1_surf_coefs[3]
        m3_surf_coefs[4] = nsq * m1_surf_coefs[4]
        ### Z7, Z8
        m3_surf_coefs[5] = ncb * m1_surf_coefs[5]
        m3_surf_coefs[6] = ncb * m1_surf_coefs[6]
        ### Z9, Z10
        m3_surf_coefs[7] = ncb * m1_surf_coefs[7]
        m3_surf_coefs[8] = ncb * m1_surf_coefs[8]
        ### Z11
        m3_surf_coefs[9] = nsq * nsq * m1_surf_coefs[9]
        return m3_surf_coefs
