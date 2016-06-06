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
import numpy as np
import galsim
# from phase_psf import parse_zemax_coefs_file


class LSSTOpticsAberrationsInterp(object):
    """
    A class to calculate Zernike aberration coeficients as a function of field position

    Interpolates sensitivity matrices over the field that were calculated at 35 field 
    locations using the LSST Zemax model by Bo Xin of the LSST Systems Engineering team.
    """
    def __init__(self):
        ### Load the sensitivity matrices from the Zemax analysis
        ###   - Dimensions in the file: (35 field locations x 19 zernikes) x 50 DOFs
        ###   - Units for annular Zernikes: micron
        ###   - Units for rigid body motions: micron and arcsec
        ###   - Annular Zernikes have an obscuration ration of 0.61
        sens_mat_file = os.path.join(galsim.meta_data.share_dir, "lsst_senM_35_19_50.txt")
        dat = np.loadtxt(sens_mat_file)
        self.n_zernikes = 19 # Number of Zernike modes
        self.n_field_locs = 35 # Number of field locations
        self.ndofs = 50 # Number of degrees of freedom
        ### convert the inputs into a list of sensitivity matrices at each field location
        self.sens = [dat[(i * self.n_zernikes):((i+1) * self.n_zernikes)]
                     for i in xrange(self.n_field_locs)]
        self.sens_fc = self._init_ref_field_locations()

        x = np.array([self.sens_fc[i,0]*np.cos(self.sens_fc[i,1] * np.pi / 180.)
                      for i in xrange(self.n_field_locs)])
        y = np.array([self.sens_fc[i,0]*np.sin(self.sens_fc[i,1] * np.pi / 180.)
                      for i in xrange(self.n_field_locs)])
        self.ref_locs = np.column_stack((x, y))

        ### Initialize the list of physical optics perturbation parameters
        self.perturbations = np.zeros(self.ndofs, dtype=np.float64)
        pass

    def _init_ref_field_locations(self):
        """
        Initialize an array of the field locations for the input sensitivity matrices
        """
        ### Both r and phi are in degrees
        r = [0., 0.379, 0.841, 1.237, 1.535, 1.708]
        phi = [0., 60., 120., 180., 240., 300.]

        ### The ordering of field locations 1-35 in the sensitivity matrix list is defined by an 
        ### image provided by Bo Xin. These increase around rings with increasing radius.
        ifc = 1
        fc = np.zeros((35, 2), dtype=float)
        for ir in xrange(1, 6):
            for iphi in xrange(6):
                fc[ifc, :] = [r[ir], phi[iphi]]
                ifc += 1
        ### Add the WFS coordinates
        ### FIXME: What are th +/-1.185 shifts Bo Xin described on his slide about this analysis?
        fc[31, :] = [r[5], 45.]
        fc[32, :] = [r[5], 135.]
        fc[33, :] = [r[5], 225.]
        fc[34, :] = [r[5], 315.]
        return fc

    def set_optics_perturbations(self, perts):
        """
        Set the parameters for the physical optics perturbations from input array
        """
        assert len(perts) == self.ndofs
        self.perturbations = perts

    def _get_sens_mat_at_field_loc(self, field_coord):
        """
        Interpolate the stored sensitivity matrices to the specified field location

        @param field_coord  A list of length 2 with the (x,y) field locations in degrees
        """
        rho_gp = np.array([0.9999, 0.9999])

        n_d = self.n_field_locs
        n_d_new = field_coord.shape[0]
        locs = np.row_stack((self.ref_locs, field_coord))

        dist = get_theta_dist(locs)
        Smat = corrmat(rho_gp, dist)

        ### Interpolate each component of the sensitivity matrix independently.
        ### Use the mean GP interpolation, rather than a draw from the GP conditional distribution.
        sens_mat = np.zeros((self.n_zernikes, self.ndofs), dtype=np.float64)
        for i in xrange(self.n_zernikes):
            for j in xrange(self.ndofs):
                y = np.array([self.sens[fcndx][i, j] for fcndx in xrange(self.n_field_locs)])
                yvar = np.var(y)
                # if yvar < 1.e-18:
                    # val = self.sens[0][i, j]
                # else:
                lambda_gp = 10. / yvar
                val, var = emulator_gp_mean(n_d, n_d_new, Smat / lambda_gp, y, lmu=1.e14,
                                            return_cov=True)
                sens_mat[i, j] = val
        return sens_mat

    def get_aberrations(self, field_coord):
        """
        Evalute the model for the Zernike aberrations at the specified field location

        @param field_coord  A list of length 2 with the (x,y) field locations in degrees
        """
        aberrations = np.zeros(2 + self.n_zernikes, dtype=np.float64)
        sens_mat = self._get_sens_mat_at_field_loc(field_coord)
        zernike_coefs = np.dot(sens_mat, self.perturbations)
        ### TODO: Convert zernike coefficients to units of waves
        aberrations[2:(self.n_zernikes+2)] = zernike_coefs
        return aberrations


# =============================================================================
# Stuff for Gaussian Process interpolation over field locations
# =============================================================================
def get_theta_dist(d):
    """
    Precompute the squared distances between design points
    
    @param d    matrix of design points (nd x ptheta)
    """
    n = d.shape[0]
    indi, indj = np.triu_indices(n, k=1)
    indm = indj + n * indi
    dist = np.square(d[indi,:] - d[indj,:])
    return {"n":n, "indi":indi, "indj":indj, "indm":indm, "d":dist}

def corrmat(rho, d):
    """
    Evaulate the GP correlation matrix

    @param rho  matrix of dimensions 1 x ptheta with correlation parameters
    @param d    dict output from get_theta_dist
    """
    N = d["n"]
    R = np.zeros((N, N))
    beta = -4. * np.log(rho)
    R[d["indi"], d["indj"]] = np.array(np.exp(-np.dot(beta, d["d"].transpose()))).ravel()
    R = R + R.transpose()
    np.fill_diagonal(R, 1.)
    return R

def emulator_gp_mean(n_d, n_d_new, Sw, y_des, lmu=1.e10, return_cov=False):
    """
    Evaluate the mean emulator prediction at new locations
    """
    N1 = n_d
    N2 = n_d_new
    V11 = Sw[0:N1, 0:N1] + np.diag(np.ones(N1) / lmu)
    V22 = Sw[N1:(N1+N2), N1:(N1+N2)]
    V12 = Sw[0:N1, N1:(N1+N2)]
    V21 = Sw[N1:(N1+N2), 0:N1]
    m = np.dot(V21, np.linalg.solve(V11, y_des))
    if return_cov:
        Schur_comp = V22 - np.dot(V21, np.linalg.solve(V11, V12))
        return m, Schur_comp
    else:
        return m


# # =============================================================================
# # Forward model of the aberrated optics PSF
# # =============================================================================
# ### Ratio of LSST primary mirror diameter to tertiary diameter
# k_m1_m3_ratio = 8.36 / 5.12

# #: Fixed parameters for the LSST telescope configuration used in the aberrated optics model.
# #: Pass this dictionary as the `telescope_model` argument to the `TelescopePSF` class.
# #: Copy this dictionary to implement other telescope models.
# lsst_telescope_model = {
#     "telescope_diameter_meters": 8.36,
#     "f-ratio": 1.25,
#     ### Coefficients for mapping bending modes to the exit pupil wavefront.
#     ### Reference: Table I.5 from Ref. 6
#     ###     These values from an old LSST design with M3 offset from M1
#     "coefs_bending_modes": {
#         "primary": {"a": 1.0, "c": 0.0},
#         "secondary": {"a": 0.91134, "c": 0.08781},
#         "tertiary": {"a": 0.74379, "c": 0.33296}
#     },
#     ### Surfaces that can have body motion misalignments or figure errors (i.e., 'bending modes')
#     "surfaces": ['primary', 'secondary', 'tertiary', 'L1', 'L2', 'L3'],
#     ### Wavefront power-series expansion coefficients for modeling body motions
#     "W_coefs": parse_zemax_coefs_file(os.path.join(os.path.dirname(__file__),
#         "data/lsst_aberration_coefs_rband_markedup.txt")),
#     ### Which rows in the input matrix of W coefficients correspond to which surfaces?
#     "zemax_coefs_rownums": {
#         'primary': 3,
#         'secondary': 7,
#         'tertiary': 11,
#         'L1': [13, 14],
#         'L2': [16, 17],
#         'L3': [22, 23]
#     },
#     ### Surfaces with parameterized bending mode perturbations.
#     ### The tertiary bending modes are mapped from those on the primary.
#     "bending_modes_parameters": {"primary": 8, "secondary": 8},
#     ### Optics with parameterized body motions.
#     ### The primary mirror body motions are used to define the telescope pointing.
#     ### The tertiary is monolithic with the primary.
#     "body_motions_parameters": {"secondary": 2, "L1": 2, "L2": 2, "L3": 2},
#     ### Exit pupil parameters for input into GalSim
#     "pupil_geometry": {
#         "circular": True,
#         "obscuration": 0.612,
#         "nstruts": 4,
#         "strut_thick": 0.05
#         # "strut_thick": 0.0
#     }
# }

# class TelescopePSFLSST(galsim.TelescopePSF):
#     """
#     Telescope model for the LSST, including special handling of the M1/M3 monolithic mirror.
#     """
#     def __init__(self, verbose=False, npad=2):
#         super(TelescopePSFLSST, self).__init__(lsst_telescope_model,
#             verbose=verbose, npad=npad)
#         self.n_surfs_bend_modes = 2

#     def set_perturbation_params(self, optics_params):
#         """
#         Assign perturbation parameters from `optics_params`, but overwrite the M3 figure errors
#         with those mapped from M1.
#         """
#         super(TelescopePSFLSST, self).set_perturbation_params(optics_params)
#         self.surf_coefs['tertiary'] = self.map_M1_bend_modes_to_M3(self.surf_coefs['primary'])
#         if self.verbose:
#             print "surf_coefs:", self.surf_coefs
#             print "misalignments:", self.misalignments
#         return None

#     def map_M1_bend_modes_to_M3(self, m1_surf_coefs=None):
#         """
#         Given bending modes on M1, calculate a new Zernike expansion of those
#         modes on the smaller circle defined by M3.

#         M1 and M3 are monolithic, so the modes on M3 should be derived from the
#         modes on M1. But because they make concentric circles, the Zernike modes
#         get mixed between the two mirrors.
#         """
#         tsqt = 2 * np.sqrt(2.)
#         n = 1. / k_m1_m3_ratio
#         nsq = n * n
#         ncb = nsq * n
#         # m3_surf_coefs = copy.copy(self.surf_coefs['primary'])
#         m3_surf_coefs = np.zeros_like(m1_surf_coefs, dtype=np.float64)
#         ### Z2, Z3
#         m3_surf_coefs[0] = n * (m1_surf_coefs[0] + tsqt * (nsq - 1) * m1_surf_coefs[6])
#         m3_surf_coefs[1] = n * (m1_surf_coefs[1] + tsqt * (nsq - 1) * m1_surf_coefs[5])
#         ### Z4
#         m3_surf_coefs[2] = nsq * (np.sqrt(15.) * (nsq - 1) * m1_surf_coefs[9] + m1_surf_coefs[2])
#         ### Z5, Z6
#         m3_surf_coefs[3] = nsq * m1_surf_coefs[3]
#         m3_surf_coefs[4] = nsq * m1_surf_coefs[4]
#         ### Z7, Z8
#         m3_surf_coefs[5] = ncb * m1_surf_coefs[5]
#         m3_surf_coefs[6] = ncb * m1_surf_coefs[6]
#         ### Z9, Z10
#         m3_surf_coefs[7] = ncb * m1_surf_coefs[7]
#         m3_surf_coefs[8] = ncb * m1_surf_coefs[8]
#         ### Z11
#         m3_surf_coefs[9] = nsq * nsq * m1_surf_coefs[9]
#         return m3_surf_coefs
