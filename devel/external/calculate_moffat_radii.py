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


