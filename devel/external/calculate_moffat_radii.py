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

import numpy as np
import galsim


def hlr_root_fwhm(fwhm, beta=2., truncationFWHM=2., flux=1., half_light_radius=1.):
    m = galsim.Moffat(beta=beta, fwhm=fwhm, flux=flux, trunc=truncationFWHM*fwhm)
    return (m.getHalfLightRadius() - half_light_radius)

def bisect_moffat_fwhm(beta=2., truncationFWHM=2., half_light_radius=1., fwhm_lower=0.1, 
                       fwhm_upper=10., tol=1.2e-16):
    """Find the Moffat FWHM providing the desired half_light_radius in the old Moffat parameter
    spec schema.

    Uses interval bisection.
    """
    y0 = hlr_root_fwhm(fwhm_lower, beta=beta, truncationFWHM=truncationFWHM, 
                       half_light_radius=half_light_radius)
    y1 = hlr_root_fwhm(fwhm_upper, beta=beta, truncationFWHM=truncationFWHM, 
                       half_light_radius=half_light_radius)
    dfwhm = fwhm_upper - fwhm_lower     
    while dfwhm >= tol:
        fwhm_mid = fwhm_lower + .5 * dfwhm
        ymid = hlr_root_fwhm(fwhm_mid, beta=beta, truncationFWHM=truncationFWHM, 
                             half_light_radius=half_light_radius)
        if y0 * ymid > 0.:  # Root not in LHS
            fwhm_lower = fwhm_mid
            y0 = hlr_root_fwhm(fwhm_lower, beta=beta, truncationFWHM=truncationFWHM, 
                               half_light_radius=half_light_radius)
        elif y1 * ymid > 0.:  # Root not in RHS
            fwhm_upper = fwhm_mid
            y1 = hlr_root_fwhm(fwhm_upper, beta=beta, truncationFWHM=truncationFWHM, 
                       half_light_radius=half_light_radius)
        elif ymid == 0.:
            break
        # Bisect interval
        dfwhm *= .5
    return fwhm_mid


