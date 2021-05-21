# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

from .position import PositionD
from .angle import arcsec
from . import integ
from . import utilities
from .errors import GalSimRangeError, GalSimIncompatibleValuesError

class Cosmology(object):
    """Basic cosmology calculations.

    Cosmology calculates expansion function E(a) and angular diameter distances Da(z) for a
    LambdaCDM universe.  Radiation is assumed to be zero and Dark Energy constant with w = -1 (no
    quintessence), but curvature is arbitrary.

    Based on Matthias Bartelmann's libastro.

    Parameters:
        omega_m:    Present day energy density of matter relative to critical density.
                    [default: 0.3]
        omega_lam:  Present day density of Dark Energy relative to critical density.
                    [default: 0.7]
    """
    def __init__(self, omega_m=0.3, omega_lam=0.7):
        # no quintessence, no radiation in this universe!
        self.omega_m = omega_m
        self.omega_lam = omega_lam
        self.omega_c = (1. - omega_m - omega_lam)
        #self.omega_r = 0

    def __repr__(self):
        return "galsim.Cosmology(omega_m=%r, omega_lam=%r)"%(self.omega_m, self.omega_lam)
    def __str__(self):
        return "galsim.Cosmology(%s,%s)"%(self.omega_m, self.omega_lam)
    def __eq__(self, other): return self is other or repr(self) == repr(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(repr(self))

    def a(self, z):
        """Compute scale factor.

        Parameters:
            z:  Redshift
        """
        return 1./(1+z)

    def E(self, a):
        """Evaluates expansion function.

        Parameters:
            a:  Scale factor.
        """
        #return (self.omega_r*a**(-4) + self.omega_m*a**(-3) + self.omega_c*a**(-2) + \
        #        self.omega_lam)**0.5
        return (self.omega_m*a**(-3) + self.omega_c*a**(-2) + self.omega_lam)**0.5

    def __angKernel(self, x):
        """Integration kernel for angular diameter distance computation.
        """
        return self.E(x**-1)**-1

    def Da(self, z, z_ref=0):
        """Compute angular diameter distance between two redshifts in units of c/H0.

        In order to get the distance in Mpc/h, multiply by c/H0~3000.

        Parameters:
            z:      Redshift.
            z_ref:  Reference redshift, with z_ref <= z. [default: 0]
        """
        if isinstance(z, np.ndarray):
            da = np.zeros_like(z, dtype=float)
            for i in range(len(da)):
                da[i] = self.Da(z[i], z_ref)
            return da
        else:
            if z < 0:
                raise GalSimRangeError("Redshift z must be >= 0", z, 0.)
            if z < z_ref:
                raise GalSimRangeError("Redshift z must be >= the reference redshift", z, z_ref)

            d = integ.int1d(self.__angKernel, z_ref+1, z+1)
            # check for curvature
            rk = (abs(self.omega_c))**0.5
            if (rk*d > 0.01):
                if self.omega_c > 0:
                    d = np.sinh(rk*d)/rk
                if self.omega_c < 0:
                    d = np.sin(rk*d)/rk
            return d/(1+z)

class NFWHalo(object):
    """Class describing NFW halos.

    This class computes the lensing fields shear and convergence of a spherically symmetric NFW
    halo of given mass, concentration, redshift, assuming a particular cosmology.
    No mass-concentration relation is employed.

    Based on Matthias Bartelmann's libastro.

    The cosmology to use can be set either by providing a `Cosmology` instance as cosmo,
    or by providing omega_m and/or omega_lam.
    If only one of the latter is provided, the other is taken to be one minus that.
    If no cosmology parameters are set, a default `Cosmology` is constructed.

    Parameters:
        mass:       Mass defined using a spherical overdensity of 200 times the critical density
                    of the universe, in units of M_solar/h.
        conc:       Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
        redshift:   Redshift of the halo.
        halo_pos:   `Position` of halo center (in arcsec). [default: PositionD(0,0)]
        omega_m:    Omega_matter to pass to `Cosmology` constructor. [default: None]
        omega_lam:  Omega_lambda to pass to `Cosmology` constructor. [default: None]
        cosmo:      A `Cosmology` instance. [default: None]
    """
    _req_params = { 'mass' : float , 'conc' : float , 'redshift' : float }
    _opt_params = { 'halo_pos' : PositionD , 'omega_m' : float , 'omega_lam' : float }

    def __init__(self, mass, conc, redshift, halo_pos=PositionD(0,0),
                 omega_m=None, omega_lam=None, cosmo=None):
        if omega_m is not None or omega_lam is not None:
            if cosmo is not None:
                raise GalSimIncompatibleValuesError(
                    "NFWHalo constructor received both cosmo and omega parameters",
                    cosmo=cosmo, omega_m=omega_m, omega_lam=omega_lam)
            if omega_m is None: omega_m = 1.-omega_lam
            if omega_lam is None: omega_lam = 1.-omega_m
            cosmo = Cosmology(omega_m=omega_m, omega_lam=omega_lam)
        elif cosmo is None:
            cosmo = Cosmology()
        elif not isinstance(cosmo,Cosmology):
            raise TypeError("Invalid cosmo parameter in NFWHalo constructor")

        # Make sure things are the right types.
        self.M = float(mass)
        self.c = float(conc)
        self.z = float(redshift)
        self.halo_pos = PositionD(halo_pos)
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
        dl = self.cosmo.Da(self.z)*3000. # in Mpc/h
        scale = self.rs / dl
        arcsec2rad = 1./206265
        self.rs_arcsec = scale/arcsec2rad

    def __repr__(self):
        s = "galsim.NFWHalo(mass=%r, conc=%r, redshift=%r"%(self.M, self.c, self.z)
        if self.halo_pos != PositionD(0,0):
            s += ", halo_pos=%r"%self.halo_pos
        if self.cosmo != Cosmology():
            s += ", cosmo=%r"%self.cosmo
        s += ")"
        return s
    def __str__(self):
        return "galsim.NFWHalo(mass=%s, conc=%s, redshift=%s)"%(self.M, self.c, self.z)
    def __eq__(self, other): return self is other or repr(self) == repr(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(repr(self))

    def __omega(self, a):
        """Matter density at scale factor a.
        """
        return self.cosmo.omega_m/(self.cosmo.E(a)**2 * a**3)

    def __farcth (self, x):
        """Numerical implementation of integral functions of a spherical NFW profile.

        All expressions are a function of ``x``, which is the radius r in units of the NFW scale
        radius, r_s.  For the derivation of these functions, see for example Wright & Brainerd
        (2000, ApJ, 534, 34).
        """
        out = np.zeros_like(x, dtype=float)

        # 3 cases: x > 1, x < 1, and |x-1| < 0.001
        mask = np.where(x < 0.999)[0]  # Equivalent but usually faster than `mask = (x < 0.999)`
        a = ((1.-x[mask])/(x[mask]+1.))**0.5
        out[mask] = 0.5*np.log((1.+a)/(1.-a))/(1-x[mask]**2)**0.5

        mask = np.where(x > 1.001)[0]
        a = ((x[mask]-1.)/(x[mask]+1.))**0.5
        out[mask] = np.arctan(a)/(x[mask]**2 - 1)**0.5

        # the approximation below has a maximum fractional error of 2.3e-7
        mask = np.where((x >= 0.999) & (x <= 1.001))[0]
        out[mask] = 5./6. - x[mask]/3.
        return out

    def __kappa(self, x, ks):
        """Calculate convergence of halo.

        Parameters:
            x:      Radial coordinate in units of rs (scale radius of halo), i.e., ``x=r/rs``.
            ks:     Lensing strength prefactor.
        """
        # convenience: call with single number
        if not isinstance(x, np.ndarray):
            return self.__kappa(np.array([x], dtype=float), np.array([ks], dtype=float))[0]
        out = np.zeros_like(x, dtype=float)

        # 3 cases: x > 1, x < 1, and |x-1| < 0.001
        mask = np.where(x < 0.999)[0]
        a = ((1 - x[mask])/(x[mask] + 1))**0.5
        out[mask] = 2*ks[mask]/(x[mask]**2 - 1) * (1 - np.log((1+a)/(1-a)) / (1-x[mask]**2)**0.5)

        mask = np.where(x > 1.001)[0]
        a = ((x[mask] - 1)/(x[mask] + 1))**0.5
        out[mask] = 2*ks[mask]/(x[mask]**2 - 1) * (1 - 2*np.arctan(a)/(x[mask]**2 - 1)**0.5)

        # the approximation below has a maximum fractional error of 7.4e-7
        mask = np.where((x >= 0.999) & (x <= 1.001))[0]
        out[mask] = ks[mask]*(22./15. - 0.8*x[mask])
        return out

    def __gamma(self, x, ks):
        """Calculate tangential shear of halo.

        Parameters:
            x:      Radial coordinate in units of rs (scale radius of halo), i.e., ``x=r/rs``.
            ks:     Lensing strength prefactor.
        """
        # convenience: call with single number
        if not isinstance(x, np.ndarray):
            return self.__gamma(np.array([x], dtype=float), np.array([ks], dtype=float))[0]
        out = np.zeros_like(x, dtype=float)

        mask = np.where(x > 0.01)[0]
        out[mask] = 4*ks[mask]*(np.log(x[mask]/2) + 2*self.__farcth(x[mask])) * \
            x[mask]**(-2) - self.__kappa(x[mask], ks[mask])

        # the approximation below has a maximum fractional error of 1.1e-7
        mask = np.where(x <= 0.01)[0]
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

    def getShear(self, pos, z_s, units=arcsec, reduced=True):
        """Calculate (reduced) shear of halo at specified positions.

        Parameters:
            po:         Position(s) of the source(s), assumed to be post-lensing!
                        Valid ways to input this:

                        - single `Position` instance
                        - tuple of floats: (x,y)
                        - list/array of `Position` instances
                        - tuple of lists/arrays: ( xlist, ylist )

            z_s:        Source redshift(s).
            units:      Angular units of coordinates. [default: galsim.arcsec]
            reduced:    Whether returned shear(s) should be reduced shears. [default: True]

        Returns:
            the (possibly reduced) shears as a tuple (g1,g2)

        If the input ``pos`` is given a single position, (g1,g2) are the two shear components.
        If the input ``pos`` is given a list/array of positions, they are NumPy arrays.
        """
        pos_x, pos_y = utilities._convertPositions(pos, units, 'getShear')
        return self._getShear(pos_x, pos_y, z_s, reduced)

    def _getShear(self, pos_x, pos_y, z_s, reduced=True):
        """Equivalent to `getShear`, but without some sanity checks and the positions must be
        given as ``pos_x``, ``pos_y`` in arcsec.

        Parameters:
            pos_x:      x position in arcsec (either a scalar or a numpy array)
            pos_y:      y position in arcsec (either a scalar or a numpy array)
            z_s:        Source redshift(s).
            reduced:    Whether returned shear(s) should be reduced shears. [default: True]

        Returns:
            the (possibly reduced) shears as a tuple (g1,g2) (either scalars or numpy arrays)
        """
        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if not isinstance(z_s, np.ndarray):
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
        # Avoid division by 0
        cos2phi = np.divide(dx*dx-dy*dy, drsq, where=(drsq != 0.))
        sin2phi = np.divide(2*dx*dy, drsq, where=(drsq != 0.))
        g1 = -g*cos2phi
        g2 = -g*sin2phi
        return g1, g2


    def getConvergence(self, pos, z_s, units=arcsec):
        """Calculate convergence of halo at specified positions.

        Parameters:
            pos:        Position(s) of the source(s), assumed to be post-lensing!
                        Valid ways to input this:

                        - single `Position` instance
                        - tuple of floats: (x,y)
                        - list/array of `Position` instances
                        - tuple of lists/arrays: ( xlist, ylist )

            z_s:        Source redshift(s).
            unit:       Angular units of coordinates. [default: galsim.arcsec]

        Returns:
            the convergence, kappa

        If the input ``pos`` is given a single position, kappa is the convergence value.
        If the input ``pos`` is given a list/array of positions, kappa is a NumPy array.
        """

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = utilities._convertPositions(pos, units, 'getKappa')
        return self._getConvergence(pos_x, pos_y, z_s)

    def _getConvergence(self, pos_x, pos_y, z_s):
        """Equivalent to `getConvergence`, but without some sanity checks and the positions must be
        given as ``pos_x``, ``pos_y`` in arcsec.

        Parameters:
            pos_x:      x position in arcsec (either a scalar or a numpy array)
            pos_y:      y position in arcsec (either a scalar or a numpy array)
            z_s:        Source redshift(s).

        Returns:
            the convergence as either a scalar or a numpy array
        """
        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if not isinstance(z_s, np.ndarray):
            ks = ks*np.ones_like(r)
        kappa = self.__kappa(r, ks)
        return kappa

    def getMagnification(self, pos, z_s, units=arcsec):
        """Calculate magnification of halo at specified positions.

        Parameters:
            pos:        Position(s) of the source(s), assumed to be post-lensing!
                        Valid ways to input this:

                        - single `Position` instance
                        - tuple of floats: (x,y)
                        - list/array of `Position` instances
                        - tuple of lists/arrays: ( xlist, ylist )

            z_s:        Source redshift(s).
            units:      Angular units of coordinates. [default: galsim.arcsec]

        Returns:
            the magnification mu

        If the input ``pos`` is given a single position, mu is the magnification value.
        If the input ``pos`` is given a list/array of positions, mu is a NumPy array.
        """
        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = utilities._convertPositions(pos, units, 'getMagnification')
        return self._getMagnification(pos_x, pos_y, z_s)

    def _getMagnification(self, pos_x, pos_y, z_s):
        """Equivalent to `getMagnification`, but without some sanity checks and the positions must
        be given as ``pos_x``, ``pos_y`` in arcsec.

        Parameters:
            pos_x:      x position in arcsec (either a scalar or a numpy array)
            pos_y:      y position in arcsec (either a scalar or a numpy array)
            z_s:        Source redshift(s).

        Returns:
            the magnification as either a scalar or a numpy array
        """
        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if not isinstance(z_s, np.ndarray):
            ks = ks*np.ones_like(r)
        g = self.__gamma(r, ks)
        kappa = self.__kappa(r, ks)

        mu = 1. / ( (1.-kappa)**2 - g**2 )
        return mu

    def getLensing(self, pos, z_s, units=arcsec):
        """Calculate lensing shear and magnification of halo at specified positions.

        Parameters:
            pos:        Position(s) of the source(s), assumed to be post-lensing!
                        Valid ways to input this:

                        - single `Position` instance
                        - tuple of floats: (x,y)
                        - list/array of `Position` instances
                        - tuple of lists/arrays: ( xlist, ylist )

            z_s:        Source redshift(s).
            units:      Angular units of coordinates. [default: galsim.arcsec]

        Returns:
            the reduced shears and magnifications as a tuple (g1,g2,mu)

        If the input ``pos`` is given a single position, the return values are the shear and
        magnification values at that position.
        If the input ``pos`` is given a list/array of positions, they are NumPy arrays.
        """
        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = utilities._convertPositions(pos, units, 'getLensing')
        return self._getLensing(pos_x, pos_y, z_s)

    def _getLensing(self, pos_x, pos_y, z_s):
        """Equivalent to `getLensing`, but without some sanity checks and the positions must
        be given as ``pos_x``, ``pos_y`` in arcsec.

        Parameters:
            pos_x:      x position in arcsec (either a scalar or a numpy array)
            pos_y:      y position in arcsec (either a scalar or a numpy array)
            z_s:        Source redshift(s).

        Returns:
            the reduced shears and magnifications as a tuple (g1,g2,mu) (each being
            either a scalar or a numpy array)
        """
        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if not isinstance(z_s, np.ndarray):
            ks = ks*np.ones_like(r)
        g = self.__gamma(r, ks)
        kappa = self.__kappa(r, ks)

        mu = 1. / ( (1.-kappa)**2 - g**2 )
        g /= 1 - kappa
        # Get the tangential shear (no x component)
        dx = pos_x - self.halo_pos.x
        dy = pos_y - self.halo_pos.y
        drsq = dx*dx+dy*dy
        # Avoid division by 0
        cos2phi = np.divide(dx*dx-dy*dy, drsq, where=(drsq != 0.))
        sin2phi = np.divide(2*dx*dy, drsq, where=(drsq != 0.))
        g1 = -g*cos2phi
        g2 = -g*sin2phi
        return g1, g2, mu
