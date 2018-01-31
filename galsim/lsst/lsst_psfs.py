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
import galsim.lsst
# from phase_psf import parse_zemax_coefs_file

# LSST optics aberrations characterized by Double Zernike expansion
dz_filepref = "LSST_Double_Zernike_coeffs"
dz_filesuff = ".csv"
dz_wavelength = 500. # nm


def getOpticsPSF():
    """
    Get the optics PSF for LSST observations.
    """
    pass


def _read_aberrations():
    """
    Helper routine to read LSST optics aberrations as double Zernike coefficients
    """
    infile = os.path.join(galsim.meta_data.share_dir,
                          dz_filepref + dz_filesuff)

    dat = np.genfromtxt(infile, delimiter=",")[1:40,:]
    return dat


# class LSSTOpticsAberrationsInterp(object):
#     """
#     A class to calculate Zernike aberration coeficients as a function of field position

#     Interpolates sensitivity matrices over the field that were calculated at 35 field 
#     locations using the LSST Zemax model by Bo Xin of the LSST Systems Engineering team.
#     """
#     def __init__(self):
#         ### Load the sensitivity matrices from the Zemax analysis
#         ###   - Dimensions in the file: (35 field locations x 19 zernikes) x 50 DOFs
#         ###   - Units for annular Zernikes: micron
#         ###   - Units for rigid body motions: micron and arcsec
#         ###   - Annular Zernikes have an obscuration ration of 0.61
#         sens_mat_file = os.path.join(galsim.meta_data.share_dir, "lsst_senM_35_19_50.txt")
#         dat = np.loadtxt(sens_mat_file)
#         self.n_zernikes = 19 # Number of Zernike modes
#         self.n_field_locs = 35 # Number of field locations
#         self.ndofs = 50 # Number of degrees of freedom
#         ### convert the inputs into a list of sensitivity matrices at each field location
#         self.sens = [dat[(i * self.n_zernikes):((i+1) * self.n_zernikes)]
#                      for i in xrange(self.n_field_locs)]
#         self.sens_fc = self._init_ref_field_locations()

#         x = np.array([self.sens_fc[i,0]*np.cos(self.sens_fc[i,1] * np.pi / 180.)
#                       for i in xrange(self.n_field_locs)])
#         y = np.array([self.sens_fc[i,0]*np.sin(self.sens_fc[i,1] * np.pi / 180.)
#                       for i in xrange(self.n_field_locs)])
#         self.ref_locs = np.column_stack((x, y))

#         ### Initialize the list of physical optics perturbation parameters
#         self.perturbations = np.zeros(self.ndofs, dtype=np.float64)
#         pass

#     def _init_ref_field_locations(self):
#         """
#         Initialize an array of the field locations for the input sensitivity matrices
#         """
#         ### Both r and phi are in degrees
#         r = [0., 0.379, 0.841, 1.237, 1.535, 1.708]
#         phi = [0., 60., 120., 180., 240., 300.]

#         ### The ordering of field locations 1-35 in the sensitivity matrix list is defined by an 
#         ### image provided by Bo Xin. These increase around rings with increasing radius.
#         ifc = 1
#         fc = np.zeros((35, 2), dtype=float)
#         for ir in xrange(1, 6):
#             for iphi in xrange(6):
#                 fc[ifc, :] = [r[ir], phi[iphi]]
#                 ifc += 1
#         ### Add the WFS coordinates
#         ### FIXME: What are th +/-1.185 shifts Bo Xin described on his slide about this analysis?
#         fc[31, :] = [r[5], 45.]
#         fc[32, :] = [r[5], 135.]
#         fc[33, :] = [r[5], 225.]
#         fc[34, :] = [r[5], 315.]
#         return fc

#     def set_optics_perturbations(self, perts):
#         """
#         Set the parameters for the physical optics perturbations from input array
#         """
#         assert len(perts) == self.ndofs
#         self.perturbations = perts

#     def _get_sens_mat_at_field_loc(self, field_coord):
#         """
#         Interpolate the stored sensitivity matrices to the specified field location

#         @param field_coord  A list of length 2 with the (x,y) field locations in degrees
#         """
#         rho_gp = np.array([0.9999, 0.9999])

#         n_d = self.n_field_locs
#         n_d_new = field_coord.shape[0]
#         locs = np.row_stack((self.ref_locs, field_coord))

#         dist = get_theta_dist(locs)
#         Smat = corrmat(rho_gp, dist)

#         ### Interpolate each component of the sensitivity matrix independently.
#         ### Use the mean GP interpolation, rather than a draw from the GP conditional distribution.
#         sens_mat = np.zeros((self.n_zernikes, self.ndofs), dtype=np.float64)
#         for i in xrange(self.n_zernikes):
#             for j in xrange(self.ndofs):
#                 y = np.array([self.sens[fcndx][i, j] for fcndx in xrange(self.n_field_locs)])
#                 yvar = np.var(y)
#                 # if yvar < 1.e-18:
#                     # val = self.sens[0][i, j]
#                 # else:
#                 lambda_gp = 10. / yvar
#                 val, var = emulator_gp_mean(n_d, n_d_new, Smat / lambda_gp, y, lmu=1.e14,
#                                             return_cov=True)
#                 sens_mat[i, j] = val
#         return sens_mat

#     def get_aberrations(self, field_coord):
#         """
#         Evalute the model for the Zernike aberrations at the specified field location

#         @param field_coord  A list of length 2 with the (x,y) field locations in degrees
#         """
#         aberrations = np.zeros(2 + self.n_zernikes, dtype=np.float64)
#         sens_mat = self._get_sens_mat_at_field_loc(field_coord)
#         zernike_coefs = np.dot(sens_mat, self.perturbations)
#         ### TODO: Convert zernike coefficients to units of waves
#         aberrations[2:(self.n_zernikes+2)] = zernike_coefs
#         return aberrations
