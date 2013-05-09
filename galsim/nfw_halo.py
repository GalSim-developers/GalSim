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
"""@file nfw_halo.py The "lensing engine" for drawing shears from an NFW halo.
"""

import galsim
import numpy as np


class Cosmology(object):
    """Basic cosmology calculations.

    Cosmology calculates expansion function E(a) and angular diameter distances Da(z) for a
    LambdaCDM universe.  Radiation is assumed to be zero and Dark Energy constant with w = -1 (no
    quintessence), but curvature is arbitrary.

    Based on Matthias Bartelmann's libastro.

    @param omega_m    Present day energy density of matter relative to critical density.
    @param omega_lam  Present day density of Dark Energy relative to critical density.
    """
    def __init__(self, omega_m=0.3, omega_lam=0.7):
        # no quintessence, no radiation in this universe!
        self.omega_m = omega_m
        self.omega_lam = omega_lam
        self.omega_c = (1. - omega_m - omega_lam)
        self.omega_r = 0
    
    def a(self, z):
        """Compute scale factor.

        @param z Redshift
        """
        return 1./(1+z)

    def E(self, a):
        """Evaluates expansion function.

        @param a Scale factor.
        """
        return (self.omega_r*a**(-4) + self.omega_m*a**(-3) + self.omega_c*a**(-2) + \
                self.omega_lam)**0.5

    def __angKernel(self, x):
        """Integration kernel for angular diameter distance computation.
        """
        return self.E(x**-1)**-1

    def Da(self, z, z_ref=0):
        """Compute angular diameter distance between two redshifts in units of c/H0.

        In order to get the distance in Mpc/h, multiply by ~3000.

        @param z     Redshift.
        @param z_ref Reference redshift, with z_ref <= z.
        """
        if isinstance(z, np.ndarray):
            da = np.zeros_like(z)
            for i in range(len(da)):
                da[i] = self.Da(z[i], z_ref)
            return da
        else:
            if z < 0:
                raise ValueError("Redshift z must not be negative")
            if z < z_ref:
                raise ValueError("Redshift z must not be smaller than the reference redshift")

            d = galsim.integ.int1d(self.__angKernel, z_ref+1, z+1)
            # check for curvature
            rk = (abs(self.omega_c))**0.5
            if (rk*d > 0.01):
                if self.omega_c > 0:
                    d = sinh(rk*d)/rk
                if self.omega_c < 0:
                    d = sin(rk*d)/rk
            return d/(1+z)

class NFWHalo(object):
    """Class for NFW halos.

    Compute the lensing fields shear and convergence of a NFW halo of given mass, concentration, 
    redshift, assuming Cosmology. No mass-concentration relation is employed.

    Based on Matthias Bartelmann's libastro.

    The cosmology to use can be set either by providing a Cosmology instance as cosmo,
    or by providing omega_m and/or omega_lam.  
    If only one of the latter is provided, the other is taken to be one minus that.
    If no cosmology parameters are set, a default Cosmology() is constructed.

    @param mass       Mass defined using a spherical overdensity of 200 times the critical density
                      of the universe, in units of M_solar/h.
    @param conc       Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
    @param redshift   Redshift of the halo.
    @param halo_pos   Position of halo center (in arcsec). [default=PositionD(0,0)]
    @param omega_m    Omega_matter to pass to Cosmology constructor. [default=None]
    @param omega_lam  Omega_lambda to pass to Cosmology constructor. [default=None]
    @param cosmo      A Cosmology instance. [default=None]
    """
    _req_params = { 'mass' : float , 'conc' : float , 'redshift' : float }
    _opt_params = { 'halo_pos' : galsim.PositionD , 'omega_m' : float , 'omega_lam' : float }
    _single_params = []
    _takes_rng = False

    def __init__(self, mass, conc, redshift, halo_pos=galsim.PositionD(0,0), 
                 omega_m=None, omega_lam=None, cosmo=None):
        if omega_m or omega_lam:
            if cosmo:
                raise TypeError("NFWHalo constructor received both cosmo and omega parameters")
            if not omega_m: omega_m = 1.-omega_lam
            if not omega_lam: omega_lam = 1.-omega_m
            cosmo = Cosmology(omega_m=omega_m, omega_lam=omega_lam)
        elif not cosmo:
            cosmo = Cosmology()
        elif not isinstance(cosmo,Cosmology):
            raise TypeError("Invalid cosmo parameter in NFWHalo constructor")

        # Check if halo_pos is a Position
        if isinstance(halo_pos,galsim.PositionD):
            pass  # This is what it should be
        elif isinstance(halo_pos,galsim.PositionI):
            # Convert to a PositionD
            halo_pos = galsim.PositionD(halo_pos.x, halo_pos.y)
        elif isinstance(halo_pos, tuple) and len(halo_pos) == 2:
            # Convert (x,y) tuple to PositionD
            halo_pos = galsim.PositionD(halo_pos[0], halo_pos[1])
        else:
            raise TypeError("Unable to parse the input halo_pos argument for NFWHalo")

        self.M = float(mass)
        self.c = float(conc)
        self.z = float(redshift)
        self.halo_pos = halo_pos
        self.cosmo = cosmo

        # calculate scale radius
        a = self.cosmo.a(self.z)
        # First we get the virial radius, which is defined for some spherical overdensity as
        # 3 M / [4 pi (r_vir)^3] = overdensity
        # Here we have overdensity = 200 * rhocrit, to determine R200. The factor of 1.63e-5 comes
        # from the following set of prefactors: (3 / (4 pi * 200 * rhocrit))^(1/3)
        # where rhocrit = 2.8e11 h^2 M_solar / Mpc^3.  The mass in the equation below is in
        # M_solar/h, which is how the final units are Mpc/h.
        R200 = 1.63e-5/(1+self.z) * (self.M * self.__omega(a)/self.__omega(1))**0.3333 # in Mpc/h
        self.rs = R200/self.c

        # convert scale radius in arcsec
        dl = self.cosmo.Da(self.z)*3000.; # in Mpc/h
        scale = self.rs / dl
        arcsec2rad = 1./206265;
        self.rs_arcsec = scale/arcsec2rad;

    def __omega(self, a):
        """Matter density at scale factor a.
        """
        return self.cosmo.omega_m/(self.cosmo.E(a)**2 * a**3)

    def __farcth (self, x, out=None):
        """Numerical implementation of integral functions of a spherical NFW profile.

        All expressions are a function of x, which is the radius r in units of the NFW scale radius,
        r_s.  For the derivation of these functions, see for example Wright & Brainerd (2000, ApJ,
        534, 34).
        """
        if out is None:
            out = np.zeros_like(x)

        # 3 cases: x > 1, x < 1, and |x-1| < 0.001
        mask = (x < 0.999)
        if mask.any():
            a = ((1.-x[mask])/(x[mask]+1.))**0.5
            out[mask] = 0.5*np.log((1.+a)/(1.-a))/(1-x[mask]**2)**0.5

        mask = (x > 1.001)
        if mask.any():
            a = ((x[mask]-1.)/(x[mask]+1.))**0.5
            out[mask] = np.arctan(a)/(x[mask]**2 - 1)**0.5

        # the approximation below has a maximum fractional error of 2.3e-7
        mask = (x >= 0.999) & (x <= 1.001)
        if mask.any():
            out[mask] = 5./6. - x[mask]/3.

        return out

    def __kappa(self, x, ks, out=None):
        """Calculate convergence of halo.

        @param x   Radial coordinate in units of rs (scale radius of halo), i.e., x=r/rs.
        @param ks  Lensing strength prefactor.
        @param out Numpy array into which results should be placed.
        """
        # convenience: call with single number
        if isinstance(x, np.ndarray) == False:
            return self.__kappa(np.array([x], dtype='float'), np.array([ks], dtype='float'))[0]

        if out is None:
            out = np.zeros_like(x)

        # 3 cases: x > 1, x < 1, and |x-1| < 0.001
        mask = (x < 0.999)
        if mask.any():
            a = ((1 - x[mask])/(x[mask] + 1))**0.5
            out[mask] = 2*ks[mask]/(x[mask]**2 - 1) * \
                (1 - np.log((1 + a)/(1 - a))/(1 - x[mask]**2)**0.5)

        mask = (x > 1.001)
        if mask.any():
            a = ((x[mask] - 1)/(x[mask] + 1))**0.5
            out[mask] = 2*ks[mask]/(x[mask]**2 - 1) * \
                (1 - 2*np.arctan(a)/(x[mask]**2 - 1)**0.5)

        # the approximation below has a maximum fractional error of 7.4e-7
        mask = (x >= 0.999) & (x <= 1.001)
        if mask.any():
            out[mask] = ks[mask]*(22./15. - 0.8*x[mask])

        return out

    def __gamma(self, x, ks, out=None):
        """Calculate tangential shear of halo.

        @param x   Radial coordinate in units of rs (scale radius of halo), i.e., x=r/rs.
        @param ks  Lensing strength prefactor.
        @param out Numpy array into which results should be placed
        """
        # convenience: call with single number
        if isinstance(x, np.ndarray) == False:
            return self.__gamma(np.array([x], dtype='float'), np.array([ks], dtype='float'))[0]
        if out is None:
            out = np.zeros_like(x)

        mask = (x > 0.01)
        if mask.any():
            out[mask] = 4*ks[mask]*(np.log(x[mask]/2) + 2*self.__farcth(x[mask])) * \
                x[mask]**(-2) - self.__kappa(x[mask], ks[mask])

        # the approximation below has a maximum fractional error of 1.1e-7
        mask = (x <= 0.01)
        if mask.any():
            out[mask] = 4*ks[mask]*(0.25 + 0.125 * x[mask]**2 * (3.25 + 3.0*np.log(x[mask]/2)))

        return out

    def __ks(self, z_s):
        """Lensing strength of halo as function of source redshift.
        """
        # critical density and surface density
        rho_c = 2.7722e11
        Sigma_c = 5.5444e14
        # density contrast of halo at redshift z
        a = self.cosmo.a(self.z)
        ez = self.cosmo.E(a)
        d0 = 200./3 * self.c**3/(np.log(1+self.c) - (1.*self.c)/(1+self.c))
        rho_s = rho_c * ez**2 *d0

        # lensing weights: the only thing that depends on z_s
        # this does takes some time...
        dl = self.cosmo.Da(z_s, self.z) * self.cosmo.Da(self.z) / self.cosmo.Da(z_s)
        k_s = dl * self.rs * rho_s / Sigma_c
        return k_s

    def getShear(self, pos, z_s, units=galsim.arcsec, reduced=True):
        """Calculate (reduced) shear of halo at specified positions.

        @param pos       Position(s) of the source(s), assumed to be post-lensing!
                         Valid ways to input this:
                           - Single galsim.PositionD (or PositionI) instance
                           - tuple of floats: (x,y)
                           - list of galsim.PositionD (or PositionI) instances
                           - tuple of lists: ( xlist, ylist )
                           - NumPy array of galsim.PositionD (or PositionI) instances
                           - tuple of NumPy arrays: ( xarray, yarray )
                           - Multidimensional NumPy array, as long as array[0] contains
                             x-positions and array[1] contains y-positions
        @param z_s       Source redshift(s).
        @param units     Angular units of coordinates. [default = arcsec]
        @param reduced   Whether returned shear(s) should be reduced shears. [default=True]

        @return (g1,g2)   [g1 and g2 are each a list if input was a list]
        """
        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = galsim.utilities._convertPositions(pos, units, 'getShear')

        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(r)
        g = self.__gamma(r, ks)

        # convert to observable = reduced shear
        if reduced:
            kappa = self.__kappa(r, ks)
            g /= 1 - kappa

        # pure tangential shear, no cross component
        dx = pos_x - self.halo_pos.x
        dy = pos_y - self.halo_pos.y
        drsq = dx*dx+dy*dy
        drsq[drsq==0.] = 1. # Avoid division by 0
        cos2phi = (dx*dx-dy*dy)/drsq
        sin2phi = 2*dx*dy/drsq
        g1 = -g*cos2phi
        g2 = -g*sin2phi

        # Make outputs in proper format: be careful here, we want consistent inputs and outputs
        # (e.g., if given a Numpy array, return one as well).  But don't attempt to index "pos"
        # until you know that it can be indexed, i.e., that it's not just a single PositionD,
        # because then bad things will happen (TypeError).
        if isinstance(pos, galsim.PositionD):
            return g1[0], g2[0]
        if isinstance(pos[0], np.ndarray):
            return g1, g2
        elif len(g) == 1 and not isinstance(pos[0],list):
            return g1[0], g2[0]
        else:
            return g1.tolist(), g2.tolist()


    def getConvergence(self, pos, z_s, units=galsim.arcsec):
        """Calculate convergence of halo at specified positions.

        @param pos     Position(s) of the source(s), assumed to be post-lensing!
                       Valid ways to input this:
                         - Single galsim.PositionD (or PositionI) instance
                         - tuple of floats: (x,y)
                         - list of galsim.PositionD (or PositionI) instances
                         - tuple of lists: ( xlist, ylist )
                         - NumPy array of galsim.PositionD (or PositionI) instances
                         - tuple of NumPy arrays: ( xarray, yarray )
                         - Multidimensional NumPy array, as long as array[0] contains
                           x-positions and array[1] contains y-positions
        @param z_s     Source redshift(s).
        @param units   Angular units of coordinates. [default = arcsec]

        @return kappa or list of kappa values.
        """

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = galsim.utilities._convertPositions(pos, units, 'getKappa')

        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(r)
        kappa = self.__kappa(r, ks)

        # Make outputs in proper format: be careful here, we want consistent inputs and outputs
        # (e.g., if given a Numpy array, return one as well).  But don't attempt to index "pos"
        # until you know that it can be indexed, i.e., that it's not just a single PositionD,
        # because then bad things will happen (TypeError).
        if isinstance(pos, galsim.PositionD):
            return kappa[0]
        elif isinstance(pos[0], np.ndarray):
            return kappa
        elif len(kappa) == 1 and not isinstance(pos[0], list):
            return kappa[0]
        else:
            return kappa.tolist()

    def getMagnification(self, pos, z_s, units=galsim.arcsec):
        """Calculate magnification of halo at specified positions.

        @param pos     Position(s) of the source(s), assumed to be post-lensing!
                       Valid ways to input this:
                         - Single galsim.PositionD (or PositionI) instance
                         - tuple of floats: (x,y)
                         - list of galsim.PositionD (or PositionI) instances
                         - tuple of lists: ( xlist, ylist )
                         - NumPy array of galsim.PositionD (or PositionI) instances
                         - tuple of NumPy arrays: ( xarray, yarray )
                         - Multidimensional NumPy array, as long as array[0] contains
                           x-positions and array[1] contains y-positions
        @param z_s     Source redshift(s).
        @param units   Angular units of coordinates (only arcsec implemented so far).
        @return mu     Numpy array containing the magnification at the specified position(s).
        """
        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = galsim.utilities._convertPositions(pos, units, 'getMagnification')

        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(r)
        g = self.__gamma(r, ks)
        kappa = self.__kappa(r, ks)

        mu = 1. / ( (1.-kappa)**2 - g**2 )

        # Make outputs in proper format: be careful here, we want consistent inputs and outputs
        # (e.g., if given a Numpy array, return one as well).  But don't attempt to index "pos"
        # until you know that it can be indexed, i.e., that it's not just a single PositionD,
        # because then bad things will happen (TypeError).
        if isinstance(pos, galsim.PositionD):
            return mu[0]
        elif isinstance(pos[0], np.ndarray):
            return mu
        elif len(mu) == 1 and not isinstance(pos[0],list):
            return mu[0]
        else:
            return mu.tolist()

    def getLensing(self, pos, z_s, units=galsim.arcsec):
        """Calculate lensing shear and magnification of halo at specified positions.

        @param pos         Position(s) of the source(s), assumed to be post-lensing!
                           Valid ways to input this:
                             - Single galsim.PositionD (or PositionI) instance
                             - tuple of floats: (x,y)
                             - list of galsim.PositionD (or PositionI) instances
                             - tuple of lists: ( xlist, ylist )
                             - NumPy array of galsim.PositionD (or PositionI) instances
                             - tuple of NumPy arrays: ( xarray, yarray )
                             - Multidimensional NumPy array, as long as array[0] contains
                               x-positions and array[1] contains y-positions
        @param z_s         Source redshift(s).
        @param units       Angular units of coordinates (only arcsec implemented so far).
        @return g1,g2,mu   Reduced shears and magnifications.
        """
        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = galsim.utilities._convertPositions(pos, units, 'getLensing')

        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(r)
        g = self.__gamma(r, ks)
        kappa = self.__kappa(r, ks)

        g /= 1 - kappa
        mu = 1. / ( (1.-kappa)**2 - g**2 )
        # Get the tangential shear (no x component)
        dx = pos_x - self.halo_pos.x
        dy = pos_y - self.halo_pos.y
        drsq = dx*dx+dy*dy
        drsq[drsq==0.] = 1. # Avoid division by 0
        cos2phi = (dx*dx-dy*dy)/drsq
        sin2phi = 2*dx*dy/drsq
        g1 = -g*cos2phi
        g2 = -g*sin2phi

        # Make outputs in proper format: be careful here, we want consistent inputs and outputs
        # (e.g., if given a Numpy array, return one as well).  But don't attempt to index "pos"
        # until you know that it can be indexed, i.e., that it's not just a single PositionD,
        # because then bad things will happen (TypeError).
        if isinstance(pos, galsim.PositionD):
            return g1[0], g2[0], mu[0]
        elif isinstance(pos[0], np.ndarray):
            return g1, g2, mu
        elif len(mu) == 1 and not isinstance(pos[0],list):
            return g1[0], g2[0], mu[0]
        else:
            return g1.tolist(), g2.tolist(), m.tolist()


