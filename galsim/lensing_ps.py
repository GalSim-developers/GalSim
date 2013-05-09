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
"""@file lensing_ps.py The "lensing engine" for drawing shears from some power spectrum.
"""

import galsim
import numpy as np

def theoryToObserved(gamma1, gamma2, kappa):
    """Helper function to convert theoretical lensing quantities to observed ones.

    This helper function is used internally by PowerSpectrum.getShear, getMagnification, and
    getLensing to convert from theoretical quantities (shear and convergence) to observable ones
    (reduced shear and magnification).  Users of PowerSpectrum.buildGrid outputs can also apply this
    method directly to the outputs in order to get the values of reduced shear and magnification on
    the output grid.

    @param gamma1        The first shear component, which must be the NON-reduced shear.  This and
                         all other inputs should be supplied either as individual floating point
                         numbers, tuples, lists, or Numpy arrays.
    @param gamma2        The second (x) shear component, which must be the NON-reduced shear.
    @param kappa         The convergence.

    @return g1, g2, mu   The reduced shear and magnification, in the same form as the input gamma1,
                         gamma2, and kappa.
    """
    # check nature of inputs to make sure they are appropriate
    if type(gamma1) != type(gamma2):
        raise ValueError("Input shear components must be of the same type!")
    if type(kappa) != type(gamma1):
        raise ValueError("Input shear and convergence must be of the same type!")
    gamma1_tmp = np.array(gamma1)
    gamma2_tmp = np.array(gamma2)
    kappa_tmp = np.array(kappa)
    if gamma1_tmp.shape != gamma2_tmp.shape:
        raise ValueError("Shear arrays passed to theoryToObserved() do not have the same shape!")
    if kappa_tmp.shape != gamma1_tmp.shape:
        raise ValueError(
           "Convergence and shear arrays passed to theoryToObserved() do not have the same shape!")

    # Now convert to reduced shear and magnification
    g1 = gamma1_tmp/(1.-kappa_tmp)
    g2 = gamma2_tmp/(1.-kappa_tmp)
    mu = 1./((1.-kappa_tmp)**2 - (gamma1_tmp**2 + gamma2_tmp**2))

    # Put back into same format as inputs
    if isinstance(gamma1, float):
        return float(g1), float(g2), float(mu)
    elif isinstance(gamma1, list):
        return list(g1), list(g2), list(mu)
    elif isinstance(gamma1, tuple):
        return tuple(g1), tuple(g2), tuple(mu)
    elif isinstance(gamma1, np.ndarray):
        return g1, g2, mu
    else:
        raise ValueError("Unknown input type for shears, convergences: %s",type(gamma1))

class PowerSpectrum(object):
    """Class to represent a lensing shear field according to some power spectrum P(k)

    A PowerSpectrum represents some (flat-sky) shear power spectrum, either for gridded points or at
    arbitary positions.  This class is originally initialized with a power spectrum from which we
    would like to generate g1 and g2 (and, optionally, convergence kappa) values.  It generates
    shears on a grid, and if necessary, when getShear is called, it will interpolate to the
    requested positions.  For detail on how these processes are carried out, please see the document
    in the GalSim repository, devel/modules/lensing_engine.pdf.

    When creating a PowerSpectrum instance, you need to specify at least one of the E or B mode 
    power spectra, which is normally given as a function P(k).  The typical thing is to just 
    use a lambda function in Python (i.e., a function that is not associated with a name); 
    for example, to define P(k)=k^2, one would use `lambda k : k**2`.  But they can also be more 
    complicated user-defined functions that take a single argument k and return the power at that 
    k value, or they can be instances of the LookupTable class for power spectra that are known 
    at particular k values but for which there is not a simple analytic form.
    
    Cosmologists often express the power spectra in terms of an expansion in spherical harmonics
    (ell), i.e., the C_ell values.  In the flat-sky limit, we can replace ell with k and C_ell with
    P(k).  Thus, k and P(k) have dimensions of inverse angle and angle^2, respectively.  It is quite
    common for people to plot ell(ell+1)C_ell/2pi, a dimensionless quantity; the analogous flat-sky
    quantity is Delta^2 = k^2 P(k)/2pi.  By default, the PowerSpectrum object assumes it is getting
    P(k), but it is possible to instead give it Delta^2 by setting the optional keyword `delta2 =
    True` in the constructor.

    Also note that we generate the shears according to the input power spectrum using a DFT
    approach, which means that we implicitly assume our discrete representation of P(k) on a grid is
    one complete cell in an infinite periodic series.  We are making assumptions about what P(k) is
    doing outside of our minimum and maximum k range, and those must be kept in mind when comparing
    with theoretical expectations.

    Specifically, since the power spectrum is realized on only a finite grid it has been been
    effectively bandpass filtered between a minimum and maximum k value in each of the k1, k2
    directions.  This filter is hard: beyond the minimum and maximum k range the P(k) is set to
    zero.  See the buildGrid method for more information.

    Therefore, the shear generation currently does not include sample variance due to coverage of a
    finite patch.  We explicitly enforce `P(k=0)=0`, which is true for the full sky in a reasonable
    cosmological model, but it ignores the fact that our little patch of sky might reasonably live
    in some special region with respect to shear correlations.  Our `P(k=0)=0` is essentially 
    setting the integrated power below our minimum k value to zero (i.e., it's implicitly a
    statement about power in a k range, not just at `k=0` itself).  The implications of the discrete
    representation, and the `P(k=0)=0` choice, are discussed in more detail in 
    devel/modules/lensing_engine.pdf.

    Therefore, since the power spectrum is realized on a finite grid, it has been been effectively
    bandpass filtered between a minimum and maximum k value in each of the k1, k2 directions.

    The power functions must return a list/array that is the same size as what it was given, e.g.,
    in the case of no power or constant power, a function that just returns a float would not be
    permitted; it would have to return an array of floats all with the same value.

    It is important to note that the power spectra used to initialize the PowerSpectrum object
    should use the same units for k and P(k), i.e., if k is in inverse radians then P(k) should be
    in radians^2 (as is natural for outputs from a cosmological shear power spectrum calculator).
    However, when we actually draw images, there is a natural scale that defines the pitch of the
    image (dx), which is typically taken to be arcsec.  This definition of a specific length scale
    means that by default we assume all quantities to the PowerSpectrum are in arcsec, and those are
    the units used for internal calculations, but the `units` keyword can be used to specify
    different input units for P(k) (again, within the constraint that k and P(k) must be
    consistent).  If the `delta2` keyword is set to specify that the input is actually the
    dimensionless power Delta^2, then the input `units` are taken to apply only to the k values.

    @param e_power_function A function or other callable that accepts a Numpy array of |k| values,
                            and returns the E-mode power spectrum P_E(|k|) in an array of the same
                            shape.  The function should return the power spectrum desired in the E
                            (gradient) mode of the image.  Set to None (default) for there to be no
                            E-mode power.
                            It may also be a string that can be converted to a function using
                            eval('lambda k : ' + e_power_function), a LookupTable, or file_name from
                            which to read in a LookupTable.  If a file_name is given, the resulting
                            LookupTable uses the defaults for the LookupTable class, namely spline
                            interpolation in P(k).  Users who wish to deviate from those defaults
                            (for example, to interpolate in log(P) and log(k), as might be more
                            natural for power-law functions) should instead read in the file to
                            create a LookupTable using the necessary non-default settings.
    @param b_power_function A function or other callable that accepts a Numpy array of |k| values,
                            and returns the B-mode power spectrum P_B(|k|) in an array of the same
                            shape.  The function should return the power spectrum desired in the B
                            (curl) mode of the image.  Set to None (default) for there to be no
                            B-mode power.
                            It may also be a string that can be converted to a function using
                            eval('lambda k : ' + b_power_function), a LookupTable, or file_name from
                            which to read in a LookupTable.
    @param delta2           Is the power actually given as dimensionless Delta^2, which requires us
                            to multiply by 2pi / k^2 to get the shear power P(k) in units of
                            angle^2?  [default = False]
    @param units            The angular units used for the power spectrum (i.e. the units of 
                            k^-1 and sqrt(P)). This should be either a galsim.AngleUnit instance
                            (e.g. galsim.radians) or a string (e.g. 'radians'). [default = arcsec]
    """
    _req_params = {}
    _opt_params = { 'e_power_function' : str, 'b_power_function' : str,
                    'delta2' : bool, 'units' : str }
    _single_params = []
    _takes_rng = False

    def __init__(self, e_power_function=None, b_power_function=None, delta2=False,
                 units=galsim.arcsec):
        # Check that at least one power function is not None
        if e_power_function is None and b_power_function is None:
            raise AttributeError(
                "At least one of e_power_function or b_power_function must be provided.")
                
        self.e_power_function = e_power_function
        self.b_power_function = b_power_function
        self.delta2 = delta2

        # Try these conversions, but we don't actually keep the output.  This just 
        # provides a way to test if the arguments are sane.
        # Note: we redo this in buildGrid for real rather than keeping the outputs
        # (e.g. in self.e_power_function, self.b_power_function) so that PowerSpectrum is 
        # picklable.  It turns out lambda functions are not picklable.
        self._convert_power_function(self.e_power_function,'e_power_function')
        self._convert_power_function(self.b_power_function,'b_power_function')

        # Check validity of units
        if isinstance(units, basestring):
            # if the string is invalid, this raises a reasonable error message.
            units = galsim.angle.get_angle_unit(units)
        if not isinstance(units, galsim.AngleUnit):
            raise ValueError("units must be either an AngleUnit or a string")

        if units == galsim.arcsec:
            self.scale = 1
        else:
            self.scale = 1. * units / galsim.arcsec


    def buildGrid(self, grid_spacing=None, ngrid=None, rng=None, interpolant=None,
                  center=galsim.PositionD(0,0), units=galsim.arcsec, get_convergence=False):
        """Generate a realization of the current power spectrum on the specified grid.

        This function will generate a Gaussian random realization of the specified E and B mode
        shear power spectra at a grid of positions, specified by the input parameters `grid_spacing`
        (distance between grid points) and `ngrid` (number of grid points in each direction.)  Units
        for `grid_spacing` and `center` can be specified using the `units` keyword; the default is
        arcsec, which is how all values are stored internally.  It automatically computes and stores
        grids for the shears and convergence.  However, since many users are primarily concerned
        with shape distortion due to shear, the default is to return only the shear components; the
        `get_convergence` keyword can be used to also return the convergence.

        The quantities that are returned are the theoretical shears and convergences, usually
        denoted gamma and kappa, respectively.  Users who wish to obtain the more
        observationally-relevant reduced shear and magnification (that describe real lensing
        distortions) can either use the getShear(), getMagnification(), or getLensing() methods
        after buildGrid, or can use a convenience function that is part of galsim.lensing_ps to 
        convert from theoretical to observed quantities.

        Note that the shears generated using this method correspond to the PowerSpectrum multiplied
        by a sharp bandpass filter, set by the dimensions of the grid.

        The filter sets `P(k)` = 0 for

            |k1|, |k2| < kmin / 2

        and
            |k1|, |k2| > kmax + kmin / 2

        where
            kmin = 2. * pi / (ngrid * grid_spacing)
            kmax = pi / grid_spacing

        and where we have adopted the convention that grid points at a given `k` represent the
        interval between `k - Delta k` and `k + Delta k` (noting that the grid spacing `Delta k` in
        k space is equivalent to `kmin`).

        It is worth remembering that this bandpass filter will *not* look like a circular annulus
        in 2D k space, but is rather more like a thick-sided picture frame, having a small square
        central cutout of dimensions `kmin` by `kmin`.  These properties are visible in the shears
        generated by this method. 

        For more information on the effects of finite grid representation of the power spectrum 
        see `devel/modules/lensing_engine.pdf`.

        Note also that the convention for axis orientation differs from that for the GREAT10
        challenge, so when using codes that deal with GREAT10 challenge outputs, the sign of our g2
        shear component must be flipped.

        Some examples:

        1. Get shears on a grid of points separated by 1 arcsec:

               my_ps = galsim.PowerSpectrum(lambda k : k**2)
               g1, g2 = my_ps.buildGrid(grid_spacing = 1., ngrid = 100)

           The returned g1, g2 are 2-d numpy arrays of values, corresponding to the values of
           g1 and g2 at the locations of the grid points.

           For a given value of grid_spacing and ngrid, we could get the x and y values on the
           grid using

               import numpy as np
               min = (-ngrid/2 + 0.5) * grid_spacing
               max = (ngrid/2 - 0.5) * grid_spacing
               x, y = np.meshgrid(np.arange(min,max+grid_spacing,grid_spacing),
                                  np.arange(min,max+grid_spacing,grid_spacing))

           where the center of the grid is taken to be (0,0).

        2. Rebuild the grid using a particular rng and set the location of the center of the grid
           to be something other than the default (0,0)

               g1, g2 = my_ps.buildGrid(grid_spacing = 8., ngrid = 65,
                                        rng = galsim.BaseDeviate(1413231),
                                        center = (256.5, 256.5) )

        3. Make a PowerSpectrum from a tabulated P(k) that gets interpolated to find the power at
           all necessary values of k, then generate shears and convergences on a grid, and convert
           to reduced shear and magnification so they can be used to transform galaxy images.
           Assuming that k and P_k are either lists, tuples, or 1d Numpy arrays containing k and
           P(k):

               tab_pk = galsim.LookupTable(k, P_k)
               my_ps = galsim.PowerSpectrum(tab_pk)
               g1, g2, kappa = my_ps.buildGrid(grid_spacing = 1., ngrid = 100,
                                               get_convergence = True)
               g1_r, g2_r, mu = galsim.lensing_ps.theoryToObserved(g1, g2, kappa)

        @param grid_spacing     Spacing for an evenly spaced grid of points, by default in arcsec
                                for consistency with the natural length scale of images created
                                using the draw or drawShoot methods.  Other units can be specified
                                using the `units` keyword.
        @param ngrid            Number of grid points in each dimension.  If a number that is not
                                an int (e.g., a float) is supplied, then it gets converted to an int
                                automatically.
        @param rng              (Optional) A galsim.GaussianDeviate object for drawing the random
                                numbers.  (Alternatively, any BaseDeviate can be used.)
                                [default `rng = None`]
        @param interpolant      (Optional) Interpolant that will be used for interpolating the
                                gridded shears by methods like getShear(), getConvergence(), etc. if
                                they are later called. [default `interpolant = galsim.Linear()`]
        @param center           (Optional) If setting up a new grid, define what position you
                                want to consider the center of that grid.  Units must be consistent
                                with those for `grid_spacing`.  [default `center = (0,0)`]
        @param units            The angular units used for the positions.  [default = arcsec]
        @param get_convergence  Return the convergence in addition to the shear?  Regardless of the
                                value of `get_convergence`, the convergence will still be computed
                                and stored for future use. [Default: `get_convergence=False`]

        @return g1,g2[,kappa]   2-d NumPy arrays for the shear components g_1, g_2 and (if
                                `get_convergence=True`) convergence kappa.
        """
        # Check problem cases for regular grid of points
        if grid_spacing is None or ngrid is None:
            raise ValueError("Both a spacing and a size are required for buildGrid.")
        # Check for non-integer ngrid
        if not isinstance(ngrid, int):
            ngrid = int(ngrid)

        # Check if center is a Position
        if isinstance(center,galsim.PositionD):
            pass  # This is what it should be
        elif isinstance(center,galsim.PositionI):
            # Convert to a PositionD
            center = galsim.PositionD(center.x, center.y)
        elif isinstance(center, tuple) and len(center) == 2:
            # Convert (x,y) tuple to PositionD
            center = galsim.PositionD(center[0], center[1])
        else:
            raise TypeError("Unable to parse the input center argument for buildGrid")

        # Automatically convert units to arcsec at the outset, then forget about it.  This is
        # because PowerSpectrum by default wants to work in arsec, and all power functions are
        # automatically converted to do so, so we'll also do that here.
        if isinstance(units, basestring):
            # if the string is invalid, this raises a reasonable error message.
            units = galsim.angle.get_angle_unit(units)
        if not isinstance(units, galsim.AngleUnit):
            raise ValueError("units must be either an AngleUnit or a string")
        if units != galsim.arcsec:
            scale_fac = (1.*units) / galsim.arcsec
            center *= scale_fac
            grid_spacing *= scale_fac

        # Make a GaussianDeviate if necessary
        if rng is None:
            gd = galsim.GaussianDeviate()
        elif isinstance(rng, galsim.BaseDeviate):
            gd = galsim.GaussianDeviate(rng)
        else:
            raise TypeError("The rng provided to buildGrid is not a BaseDeviate")

        # Check that the interpolant is valid.  (Don't save the result though in case it is
        # a string -- we don't want to mess up picklability.)
        self.interpolant = interpolant
        if interpolant is None:
            pass
        else:
            galsim.utilities.convert_interpolant_to_2d(interpolant)

        # Convert power_functions into callables:
        e_power_function = self._convert_power_function(self.e_power_function,'e_power_function')
        b_power_function = self._convert_power_function(self.b_power_function,'b_power_function')

        # If we actually have dimensionless Delta^2, then we must convert to power
        # P(k) = 2pi Delta^2 / k^2, 
        # which has dimensions of angle^2.
        if e_power_function is None:
            p_E = None
        elif self.delta2:
            # Here we have to go from Delta^2 (dimensionless) to P = 2pi Delta^2 / k^2.  We want to
            # have P and therefore 1/k^2 in units of arcsec, so we won't rescale the k that goes in
            # the denominator.  This naturally gives P(k) in arcsec^2.
            p_E = lambda k : (2.*np.pi) * e_power_function(self.scale*k)/(k**2)
        elif self.scale != 1:
            # Here, the scale comes in two places:
            # The units of k have to be converted from 1/arcsec, which GalSim wants to use, into
            # whatever the power spectrum function was defined to use.
            # The units of power have to be converted from (input units)^2 as returned by the power
            # function, to Galsim's units of arcsec^2.
            # Recall that scale is (input units)/arcsec.
            p_E = lambda k : e_power_function(self.scale*k)*(self.scale**2)
        else: 
            p_E = e_power_function

        if b_power_function is None:
            p_B = None
        elif self.delta2:
            p_B = lambda k : (2.*np.pi) * b_power_function(self.scale*k)/(k**2)
        elif self.scale != 1:
            p_B = lambda k : b_power_function(self.scale*k)*(self.scale**2)
        else:
            p_B = b_power_function

        # Build the grid 
        psr = PowerSpectrumRealizer(ngrid, grid_spacing, p_E, p_B)
        self.grid_g1, self.grid_g2, self.grid_kappa = psr(gd)
            
        # Set up the images to be interpolated.
        # Note: We don't make the SBInterpolatedImages yet, since it's not picklable. 
        #       So we wait to create them when we are actually going to use them.
        self.im_g1 = galsim.ImageViewD(self.grid_g1)
        self.im_g1.setScale(grid_spacing)

        self.im_g2 = galsim.ImageViewD(self.grid_g2)
        self.im_g2.setScale(grid_spacing)

        self.im_kappa = galsim.ImageViewD(self.grid_kappa)
        self.im_kappa.setScale(grid_spacing)

        # Dealing with the center here is a bit confusing, especially if ngrid is even.
        # The InterpolatedImage will consider position (0,0) to correspond to 
        # self.im_g1.bounds.center() on the image.  We call this nominal_center.
        # However, if ngrid is even, this is slightly up and to the right of the 
        # true center. The true center x and y are at (1+ngrid)/2 * grid_spacing.
        # And finally, we may be passed a value to consider the center of the image.
        b = self.im_g1.bounds
        nominal_center = galsim.PositionD(b.center().x, b.center().y) * grid_spacing
        true_center = galsim.PositionD( (1.+ngrid)/2. , (1.+ngrid)/2. ) * grid_spacing
            
        # The offset to be added to any position is then such that if we are 
        # provided the target center position, the result will be the location of 
        # the true center with respect to the nominal center.  In other words:
        #   center + offset = true_center - nominal_center
        self.offset = true_center - nominal_center - center

        # Construct a bounds that we can use to check if a provided position will
        # end up falling on the interpolating image.
        self.bounds = galsim.BoundsD((b.xmin-0.5)*grid_spacing, (b.xmax+0.5)*grid_spacing,
                                     (b.ymin-0.5)*grid_spacing, (b.ymax+0.5)*grid_spacing)
        self.bounds.shift(-nominal_center - self.offset)

        if get_convergence:
            return self.grid_g1, self.grid_g2, self.grid_kappa
        else:
            return self.grid_g1, self.grid_g2

    def _convert_power_function(self, pf, pf_str):
        if pf is None: return None

        # Convert string inputs to either a lambda function or LookupTable
        if isinstance(pf,str):
            import os
            if os.path.isfile(pf):
                pf = galsim.LookupTable(file=pf)
            else:
                pf = eval('lambda k : ' + pf)

        # Check that the function is sane.
        # Note: Only try tests below if it's not a LookupTable.
        #       (If it's a LookupTable, then it could be a valid function that isn't 
        #        defined at k=1, and by definition it must return something that is the 
        #        same length as the input.)
        if not isinstance(pf, galsim.LookupTable):
            f1 = pf(np.array((0.1,1.)))
            fake_arr = np.zeros(2)
            fake_p = pf(fake_arr)
            if isinstance(fake_p, float):
                raise AttributeError(
                    "Power function MUST return a list/array same length as input")
        return pf


    def getShear(self, pos, units=galsim.arcsec, reduced=True):
        """
        This function can interpolate between grid positions to find the shear values for a given
        list of input positions (or just a single position).  Before calling this function, you must
        call buildGrid first to define the grid of shears and convergences on which to interpolate.
        By default, this method returns the reduced shear, which is defined in terms of shear and
        convergence as reduced shear `g=gamma/(1-kappa)`; the `reduced` keyword can be used to
        return the non-reduced shear.

        Note that the interpolation (carried out using the interpolant that was specified when
        building the gridded shears) modifies the effective power spectrum somewhat.  The user is
        responsible for choosing a grid size that is small enough not to significantly modify the
        power spectrum on the scales of interest.  Detailed tests of this functionality have not
        been carried out.

        Some examples of how to use getShear:

        1. Get the shear for a particular point:

               g1, g2 = my_ps.getShear(pos = galsim.PositionD(12, 412))

           This time the returned values are just floats and correspond to the shear for the
           provided position.

        2. You can also provide a position as a tuple to save the explicit PositionD construction:

               g1, g2 = my_ps.getShear(pos = (12, 412))

        3. Get the shears for a bunch of points at once:
        
               xlist = [ 141, 313,  12, 241, 342 ]
               ylist = [  75, 199, 306, 225, 489 ]
               poslist = [ galsim.PositionD(xlist[i],ylist[i]) for i in range(len(xlist)) ]
               g1, g2 = my_ps.getShear( poslist )
               g1, g2 = my_ps.getShear( (xlist, ylist) )

           Both calls do the same thing.  The returned g1, g2 this time are lists of g1, g2 values.
           The lists are the same length as the number of input positions.

        @param pos              Position(s) of the source(s), assumed to be post-lensing!
                                Valid ways to input this:
                                  - Single galsim.PositionD (or PositionI) instance
                                  - tuple of floats: (x,y)
                                  - list of galsim.PositionD (or PositionI) instances
                                  - tuple of lists: ( xlist, ylist )
                                  - NumPy array of galsim.PositionD (or PositionI) instances
                                  - tuple of NumPy arrays: ( xarray, yarray )
                                  - Multidimensional NumPy array, as long as array[0] contains
                                    x-positions and array[1] contains y-positions
        @param units            The angular units used for the positions.  [default = arcsec]
        @param reduced          Whether returned shear(s) should be reduced shears. [default=True]

        @return g1,g2           If given a single position: the two shear components g_1 and g_2.
                                If given a list of positions: each is a python list of values.
                                If given a NumPy array of positions: each is a NumPy array.
        """

        if not hasattr(self, 'im_g1'):
            raise RuntimeError("PowerSpectrum.buildGrid must be called before getShear")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = galsim.utilities._convertPositions(pos, units, 'getShear')

        # Set the interpolant:
        if self.interpolant is None:
            interpolant2d = galsim.InterpolantXY(galsim.Linear())
        else:
            interpolant2d = galsim.utilities.convert_interpolant_to_2d(self.interpolant)

        if reduced:
            # get reduced shear (just discard magnification)
            g1_r, g2_r, _ = galsim.lensing_ps.theoryToObserved(self.im_g1.array, self.im_g2.array,
                                                               self.im_kappa.array)
            g1_r = galsim.ImageViewD(g1_r)
            g1_r.setScale(self.im_g1.getScale())
            g1_r.setOrigin(self.im_g1.getXMin(), self.im_g1.getYMin())
            g2_r = galsim.ImageViewD(g2_r)
            g2_r.setScale(self.im_g2.getScale())
            g2_r.setOrigin(self.im_g2.getXMin(), self.im_g2.getYMin())
            # Make an SBInterpolatedImage, which will do the heavy lifting for the
            # interpolation.
            sbii_g1 = galsim.SBInterpolatedImage(g1_r, xInterp=interpolant2d)
            sbii_g2 = galsim.SBInterpolatedImage(g2_r, xInterp=interpolant2d)
        else:
            sbii_g1 = galsim.SBInterpolatedImage(self.im_g1, xInterp=interpolant2d)
            sbii_g2 = galsim.SBInterpolatedImage(self.im_g2, xInterp=interpolant2d)

        # interpolate if necessary
        g1,g2 = [], []
        for iter_pos in [ galsim.PositionD(pos_x[i],pos_y[i]) for i in range(len(pos_x)) ]:
            # Check that the position is in the bounds of the interpolated image
            if not self.bounds.includes(iter_pos):
                import warnings
                warnings.warn(
                    "Warning: position (%f,%f) not within the bounds "%(pos.x,pos.y) +
                    "of the gridded shear values: " + str(self.bounds) +
                    ".  Returning a shear of (0,0) for this point.")
                g1.append(0.)
                g2.append(0.)
            else:
                g1.append(sbii_g1.xValue(iter_pos+self.offset))
                g2.append(sbii_g2.xValue(iter_pos+self.offset))

        if isinstance(pos, galsim.PositionD):
            return g1[0], g2[0]
        elif isinstance(pos[0], np.ndarray):
            return np.array(g1), np.array(g2)
        elif len(pos_x) == 1 and not isinstance(pos[0],list):
            return g1[0], g2[0]
        else:
            return g1, g2

    def getConvergence(self, pos, units=galsim.arcsec):
        """
        This function can interpolate between grid positions to find the convergence values for a
        given list of input positions (or just a single position).  Before calling this function,
        you must call buildGrid first to define the grid of convergences on which to interpolate.

        Note that the interpolation (carried out using the interpolant that was specified when
        building the gridded shears) modifies the effective power spectrum somewhat.  The user is
        responsible for choosing a grid size that is small enough not to significantly modify the
        power spectrum on the scales of interest.

        The usage of getConvergence is the same as for getShear, except that it returns only a
        single number rather than a pair of numbers.  See documentation for getShear for some
        examples.

        @param pos              Position(s) of the source(s), assumed to be post-lensing!
                                Valid ways to input this:
                                  - Single galsim.PositionD (or PositionI) instance
                                  - tuple of floats: (x,y)
                                  - list of galsim.PositionD (or PositionI) instances
                                  - tuple of lists: ( xlist, ylist )
                                  - NumPy array of galsim.PositionD (or PositionI) instances
                                  - tuple of NumPy arrays: ( xarray, yarray )
                                  - Multidimensional NumPy array, as long as array[0] contains
                                    x-positions and array[1] contains y-positions
        @param units            The angular units used for the positions.  [default = arcsec]

        @return kappa           If given a single position: the convergence kappa.
                                If given a list of positions: a python list of values.
                                If given a NumPy array of positions: a NumPy array of values.
        """

        if not hasattr(self, 'im_kappa'):
            raise RuntimeError("PowerSpectrum.buildGrid must be called before getConvergence")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = galsim.utilities._convertPositions(pos, units, 'getConvergence')

        # Set the interpolant:
        if self.interpolant is None:
            interpolant2d = galsim.InterpolantXY(galsim.Linear())
        else:
            interpolant2d = galsim.utilities.convert_interpolant_to_2d(self.interpolant)

        # Make an SBInterpolatedImage, which will do the heavy lifting for the 
        # interpolation.
        sbii_kappa = galsim.SBInterpolatedImage(self.im_kappa, xInterp=interpolant2d)

        # interpolate if necessary
        kappa = []
        for iter_pos in [ galsim.PositionD(pos_x[i],pos_y[i]) for i in range(len(pos_x)) ]:
            # Check that the position is in the bounds of the interpolated image
            if not self.bounds.includes(iter_pos):
                import warnings
                warnings.warn(
                    "Warning: position (%f,%f) not within the bounds "%(pos.x,pos.y) +
                    "of the gridded convergence values: " + str(self.bounds) + 
                    ".  Returning a convergence of 0 for this point.")
                kappa.append(0.)
            else:
                kappa.append(sbii_kappa.xValue(iter_pos+self.offset))

        if isinstance(pos, galsim.PositionD):
            return kappa[0]
        elif isinstance(pos[0], np.ndarray):
            return np.array(kappa)
        elif len(pos_x) == 1 and not isinstance(pos[0],list): 
            return kappa[0]
        else:
            return kappa

    def getMagnification(self, pos, units=galsim.arcsec):
        """
        This function can interpolate between grid positions to find the lensing magnification (mu)
        values for a given list of input positions (or just a single position).  Before calling this
        function, you must call buildGrid first to define the grid of shears and convergences on
        which to interpolate.

        Note that the interpolation (carried out using the interpolant that was specified when
        building the gridded shears) modifies the effective power spectrum somewhat.  The user is
        responsible for choosing a grid size that is small enough not to significantly modify the
        power spectrum on the scales of interest.

        The usage of getMagnification is the same as for getShear, except that it returns only a
        single number rather than a pair of numbers.  See documentation for getShear for some
        examples.

        @param pos              Position(s) of the source(s), assumed to be post-lensing!
                                Valid ways to input this:
                                  - Single galsim.PositionD (or PositionI) instance
                                  - tuple of floats: (x,y)
                                  - list of galsim.PositionD (or PositionI) instances
                                  - tuple of lists: ( xlist, ylist )
                                  - NumPy array of galsim.PositionD (or PositionI) instances
                                  - tuple of NumPy arrays: ( xarray, yarray )
                                  - Multidimensional NumPy array, as long as array[0] contains
                                    x-positions and array[1] contains y-positions
        @param units            The angular units used for the positions.  [default = arcsec]

        @return mu              If given a single position: the magnification, mu.
                                If given a list of positions: a python list of values.
                                If given a NumPy array of positions: a NumPy array of values.
        """

        if not hasattr(self, 'im_kappa'):
            raise RuntimeError("PowerSpectrum.buildGrid must be called before getMagnification")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = galsim.utilities._convertPositions(pos, units, 'getMagnification')

        # Set the interpolant:
        if self.interpolant is None:
            interpolant2d = galsim.InterpolantXY(galsim.Linear())
        else:
            interpolant2d = galsim.utilities.convert_interpolant_to_2d(self.interpolant)

        # Calculate the magnification based on the convergence and shear
        _, _, mu = galsim.lensing_ps.theoryToObserved(self.im_g1.array, self.im_g2.array,
                                                      self.im_kappa.array)
        mu = galsim.ImageViewD(mu)
        mu.setScale(self.im_kappa.getScale())
        mu.setOrigin(self.im_kappa.getXMin(), self.im_kappa.getYMin())
        # Make an SBInterpolatedImage, which will do the heavy lifting for the 
        # interpolation.
        sbii_mu = galsim.SBInterpolatedImage(mu, xInterp=interpolant2d)

        # interpolate if necessary
        mu = []
        for iter_pos in [ galsim.PositionD(pos_x[i],pos_y[i]) for i in range(len(pos_x)) ]:
            # Check that the position is in the bounds of the interpolated image
            if not self.bounds.includes(iter_pos):
                import warnings
                warnings.warn(
                    "Warning: position (%f,%f) not within the bounds "%(pos.x,pos.y) +
                    "of the gridded convergence values: " + str(self.bounds) + 
                    ".  Returning a magnification of 0 for this point.")
                mu.append(0.)
            else:
                mu.append(sbii_mu.xValue(iter_pos+self.offset))

        if isinstance(pos, galsim.PositionD):
            return mu[0]
        elif isinstance(pos[0], np.ndarray):
            return np.array(mu)
        elif len(pos_x) == 1 and not isinstance(pos[0],list): 
            return mu[0]
        else:
            return mu

    def getLensing(self, pos, units=galsim.arcsec):
        """
        This function can interpolate between grid positions to find the lensing observable
        quantities (reduced shears g1 and g2, and magnification mu) for a given list of input
        positions (or just a single position).  Before calling this function, you must call
        buildGrid first to define the grid of shears and convergences on which to interpolate.

        Note that the interpolation (carried out using the interpolant that was specified when
        building the gridded shears) modifies the effective power spectrum somewhat.  The user is
        responsible for choosing a grid size that is small enough not to significantly modify the
        power spectrum on the scales of interest.

        The usage of getLensing is the same as for getShear, except that it returns only a single
        number rather than a pair of numbers.  See documentation for getShear for some examples.

        @param pos              Position(s) of the source(s), assumed to be post-lensing!
                                Valid ways to input this:
                                  - Single galsim.PositionD (or PositionI) instance
                                  - tuple of floats: (x,y)
                                  - list of galsim.PositionD (or PositionI) instances
                                  - tuple of lists: ( xlist, ylist )
                                  - NumPy array of galsim.PositionD (or PositionI) instances
                                  - tuple of NumPy arrays: ( xarray, yarray )
                                  - Multidimensional NumPy array, as long as array[0] contains
                                    x-positions and array[1] contains y-positions
        @param units            The angular units used for the positions.  [default = arcsec]

        @return g1,g2,mu        If given a single position: the reduced shears g1 and g2, and
                                magnification mu.
                                If given a list of positions: python lists of values.
                                If given a NumPy array of positions: NumPy arrays of values.
        """

        if not hasattr(self, 'im_kappa'):
            raise RuntimeError("PowerSpectrum.buildGrid must be called before getLensing")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = galsim.utilities._convertPositions(pos, units, 'getLensing')

        # Set the interpolant:
        if self.interpolant is None:
            interpolant2d = galsim.InterpolantXY(galsim.Linear())
        else:
            interpolant2d = galsim.utilities.convert_interpolant_to_2d(self.interpolant)

        # Calculate the magnification based on the convergence and shear
        g1_r, g2_r, mu = galsim.lensing_ps.theoryToObserved(self.im_g1.array, self.im_g2.array,
                                                            self.im_kappa.array)
        g1_r = galsim.ImageViewD(g1_r)
        g1_r.setScale(self.im_kappa.getScale())
        g1_r.setOrigin(self.im_kappa.getXMin(), self.im_kappa.getYMin())
        g2_r = galsim.ImageViewD(g2_r)
        g2_r.setScale(self.im_kappa.getScale())
        g2_r.setOrigin(self.im_kappa.getXMin(), self.im_kappa.getYMin())
        mu = galsim.ImageViewD(mu)
        mu.setScale(self.im_kappa.getScale())
        mu.setOrigin(self.im_kappa.getXMin(), self.im_kappa.getYMin())
        # Make an SBInterpolatedImage, which will do the heavy lifting for the 
        # interpolation.
        sbii_g1 = galsim.SBInterpolatedImage(g1_r, xInterp=interpolant2d)
        sbii_g2 = galsim.SBInterpolatedImage(g2_r, xInterp=interpolant2d)
        sbii_mu = galsim.SBInterpolatedImage(mu, xInterp=interpolant2d)

        # interpolate if necessary
        g1, g2, mu = [], [], []
        for iter_pos in [ galsim.PositionD(pos_x[i],pos_y[i]) for i in range(len(pos_x)) ]:
            # Check that the position is in the bounds of the interpolated image
            if not self.bounds.includes(iter_pos):
                import warnings
                warnings.warn(
                    "Warning: position (%f,%f) not within the bounds "%(pos.x,pos.y) +
                    "of the gridded convergence values: " + str(self.bounds) + 
                    ".  Returning 0 for lensing observables at this point.")
                g1.append(0.)
                g2.append(0.)
                mu.append(0.)
            else:
                g1.append(sbii_g1.xValue(iter_pos+self.offset))
                g2.append(sbii_g2.xValue(iter_pos+self.offset))
                mu.append(sbii_mu.xValue(iter_pos+self.offset))

        if isinstance(pos, galsim.PositionD):
            return g1[0], g2[0], mu[0]
        elif isinstance(pos[0], np.ndarray):
            return np.array(g1), np.array(g2), np.array(mu)
        elif len(pos_x) == 1 and not isinstance(pos[0],list): 
            return g1[0], g2[0], mu[0]
        else:
            return g1, g2, mu

class PowerSpectrumRealizer(object):
    """Class for generating realizations of power spectra with any area and pixel size.
    
    This class is not one that end-users should expect to interact with.  It is designed to quickly
    generate many realizations of the same shear power spectra on a square grid.  The initializer
    sets up the grids in k-space and computes the power on them.  It also computes spin weighting
    terms.  You can alter any of the setup properties later.  It currently only works for square
    grids (at least, much of the internals would be incorrect for non-square grids), so while it
    nominally contains arrays that could be allowed to be non-square, the constructor itself
    enforces squareness.

    @param ngrid            The size of the grid in one dimension.
    @param pixel_size       The size of the pixel sides, in units consistent with the units expected
                            by the power spectrum functions.
    @param e_power_function See description of this parameter in the documentation for the
                            PowerSpectrum class.
    @param b_power_function See description of this parameter in the documentation for the
                            PowerSpectrum class.
    """
    def __init__(self, ngrid, pixel_size, p_E, p_B):
        # Set up the k grids in x and y, and the instance variables
        self.set_size(ngrid, pixel_size)
        self.set_power(p_E, p_B)

    def set_size(self, ngrid, pixel_size):
        self.nx = ngrid
        self.ny = ngrid
        self.pixel_size = float(pixel_size)

        # Setup some handy slices for indexing different parts of k space
        self.ikx = slice(0,self.nx/2+1)       # positive kx values, including 0, nx/2
        self.ikxp = slice(1,(self.nx+1)/2)    # limit to only values with a negative value
        self.ikxn = slice(-1,self.nx/2,-1)    # negative kx values

        # We always call this with nx=ny, so behavior with nx != ny is not tested.
        # However, we make a basic attempt to enable such behavior in the future if needed.
        self.iky = slice(0,self.ny/2+1)
        self.ikyp = slice(1,(self.ny+1)/2)
        self.ikyn = slice(-1,self.ny/2,-1)

        # Set up the scalar k grid. Generally, for a box size of L (in one dimension), the grid
        # spacing in k_x or k_y is Delta k=2pi/L 
        self.kx, self.ky = galsim.utilities.kxky((self.ny,self.nx))
        self.kx /= self.pixel_size
        self.ky /= self.pixel_size

        # Compute the spin weightings
        self._generate_exp2ipsi()

    def set_power(self, p_E, p_B):
        self.p_E = p_E
        self.p_B = p_B
        if p_E is None:  self.amplitude_E = None
        else:            self.amplitude_E = np.sqrt(self._generate_power_array(p_E))/self.pixel_size
        if p_B is None:  self.amplitude_B = None
        else:            self.amplitude_B = np.sqrt(self._generate_power_array(p_B))/self.pixel_size

    def recompute_power(self):
        self.set_power(self.p_E, self.p_B)

    def __call__(self, gd):
        """Generate a realization of the current power spectrum.
        
        @param gd               A Gaussian deviate to use when generating the shear fields.
        @return g1,g2,kappa     NumPy arrays for the shear components g_1, g_2 and convergence
                                kappa.
        """
        ISQRT2 = np.sqrt(1.0/2.0)

        if not isinstance(gd, galsim.GaussianDeviate):
            raise TypeError(
                "The gd provided to the PowerSpectrumRealizer is not a GaussianDeviate!")

        # Generate a random complex realization for the E-mode, if there is one
        if self.amplitude_E is not None:
            r1 = galsim.utilities.rand_arr(self.amplitude_E.shape, gd)
            r2 = galsim.utilities.rand_arr(self.amplitude_E.shape, gd)
            E_k = np.empty((self.ny,self.nx)).astype(type(1.+1.j))
            E_k[:,self.ikx] = self.amplitude_E * (r1 + 1j*r2) * ISQRT2
            # E_k corresponds to real kappa, so E_k[-k] = conj(E_k[k])
            self._make_hermitian(E_k)
        else: E_k = 0

        # Generate a random complex realization for the B-mode, if there is one
        if self.amplitude_B is not None:
            r1 = galsim.utilities.rand_arr(self.amplitude_B.shape, gd)
            r2 = galsim.utilities.rand_arr(self.amplitude_B.shape, gd)
            B_k = np.empty((self.ny,self.nx)).astype(type(1.+1.j))
            B_k[:,self.ikx] = self.amplitude_B * (r1 + 1j*r2) * ISQRT2
            # B_k corresponds to imag kappa, so B_k[-k] = -conj(B_k[k])
            # However, we later multiply this by i, so that means here B_k[-k] = conj(B_k[k])
            self._make_hermitian(B_k)
        else:
            B_k = 0

        # In terms of kappa, the E mode is the real kappa, and the B mode is imaginary kappa:
        # In fourier space, both E_k and B_k are complex, but the same E + i B relation holds.
        kappa_k = E_k + 1j * B_k

        # Compute gamma_k as exp(2i psi) kappa_k
        # Equation 2.1.12 of Kaiser & Squires (1993, ApJ, 404, 441) is equivalent to:
        #   gamma_k = -self.exp2ipsi * kappa_k
        # But of course, they only considered real (E-mode) kappa.
        # However, this equation has a sign error.  There should not be a minus in front.
        # If you follow their subsequent deviation, you will see that they drop the minus sign
        # when they get to 2.1.15 (another - appears from the derivative).  2.1.15 is correct.
        # e.g. it correctly produces a positive point mass for tangential shear ~ 1/r^2.
        # So this implies that the minus sign in 2.1.12 should not be there.
        gamma_k = self.exp2ipsi * kappa_k

        # And go to real space to get the real-space shear and convergence fields
        gamma = self.nx * np.fft.ifft2(gamma_k)
        # Make them contiguous, since we need to use them in an Image, which requires it.
        g1 = np.ascontiguousarray(np.real(gamma))
        g2 = np.ascontiguousarray(np.imag(gamma))

        # Could do the same thing with kappa..
        #kappa = self.nx * np.fft.ifft2(kappa_k)
        #k = np.ascontiguousarray(np.real(kappa))
    
        # But, since we don't care about imag(kappa), this is a bit faster:
        if E_k is 0:
            k = np.zeros((self.ny,self.nx))
        else:
            k = self.nx * np.fft.irfft2(E_k[:,self.ikx], s=(self.ny,self.nx))

        return g1, g2, k

    def _make_hermitian(self, P_k):
        # Make P_k[-k] = conj(P_k[k])
        # First update the kx=0 values to be consistent with this.
        P_k[self.ikyn,0] = np.conj(P_k[self.ikyp,0])
        P_k[0,0] = np.real(P_k[0,0])  # Not reall necessary, since P_k[0,0] = 0, but 
                                      # I do it anyway for the sake of pedantry...
        # Then fill the kx<0 values appropriately
        P_k[self.ikyp,self.ikxn] = np.conj(P_k[self.ikyn,self.ikxp])
        P_k[self.ikyn,self.ikxn] = np.conj(P_k[self.ikyp,self.ikxp])
        P_k[0,self.ikxn] = np.conj(P_k[0,self.ikxp])
        # For even nx,ny, there are a few more changes needed.
        if self.ny % 2 == 0:
            # Note: this is a bit more complicated if you have to separately check whether
            # nx and/or ny are even.  I ignore this subtlety until we decide it is needed.
            P_k[self.ikyn,self.nx/2] = np.conj(P_k[self.ikyp,self.nx/2])
            P_k[self.ny/2,self.ikxn] = np.conj(P_k[self.ny/2,self.ikxp])
            P_k[self.ny/2,0] = np.real(P_k[self.ny/2,0])
            P_k[0,self.nx/2] = np.real(P_k[0,self.nx/2])
            P_k[self.ny/2,self.nx/2] = np.real(P_k[self.ny/2,self.nx/2])

    def _generate_power_array(self, power_function):
        # Internal function to generate the result of a power function evaluated on a grid,
        # taking into account the symmetries.
        power_array = np.empty((self.ny, self.nx/2+1))

        # Set up the scalar |k| grid using just the positive kx,ky
        k = np.sqrt(self.kx[self.iky,self.ikx]**2 + self.ky[self.iky,self.ikx]**2)

        # Fudge the value at k=0, so we don't have to evaluate power there
        k[0,0] = k[1,0]
        # Raise a clear exception for LookupTable that are not defined on the full k range!
        if isinstance(power_function, galsim.LookupTable):
            mink = np.min(k)
            maxk = np.max(k)
            if mink < power_function.x_min or maxk > power_function.x_max:
                raise ValueError(
                    "LookupTable P(k) is not defined for full k range on grid, %f<k<%f"%(mink,maxk))
        P_k = power_function(k)
        
        # Now fix the k=0 value of power to zero
        assert type(P_k) is np.ndarray
        P_k[0,0] = type(P_k[0,1])(0.)
        if np.any(P_k < 0):
            raise ValueError("Negative power found for some values of k!")

        power_array[self.iky, self.ikx] = P_k
        power_array[self.ikyn, self.ikx] = P_k[self.ikyp, self.ikx]
        return power_array
    
    def _generate_exp2ipsi(self):
        # exp2ipsi = (kx + iky)^2 / |kx + iky|^2 is the phase of the k vector.
        kz = self.kx + self.ky*1j
        # exp(2i psi) = kz^2 / |kz|^2
        ksq = kz*np.conj(kz)
        # Need to adjust denominator for kz=0 to avoid division by 0.
        ksq[0,0] = 1.
        self.exp2ipsi = kz*kz/ksq
        # Note: this leaves exp2ipsi[0,0] = 0, but it turns out that's ok, since we only
        # ever multiply it by something that is 0 anyway (amplitude[0,0] = 0).

def kappaKaiserSquires(g1, g2):
    """Perform a Kaiser & Squires (1993) inversion to get a convergence map from gridded shears.

    This function takes gridded shears and constructs a convergence map from them.  While this is
    complicated in reality by the non-gridded galaxy positions, it is a straightforward
    implementation using Fourier transforms for the case of gridded galaxy positions.  Note that
    there are additional complications when dealing with real observational issues like shape noise
    that are not handled by this function, and likewise there are known edge effects.

    Note that, like any process that attempts to recover information from discretely sampled data,
    the `kappa_E` and `kappa_B` maps returned by this function are subject to aliasing.  There will
    be distortions if there are non-zero frequency modes in the lensing field represented by g1 and
    g2 at more than half the frequency represented by the g1, g2 grid spacing.  To avoid this issue
    in practice you can smooth the input g1, g2 to effectively bandlimit them (the same smoothing
    kernel will be present in the output `kappa_E`, `kappa_B`).  If applying this function to shears
    drawn randomly according to some power spectrum, the power spectrum that is used should be
    modified to go to zero above the relevant maximum k value for the grid being used.

    @param g1  Square galsim.ImageF, galsim.ImageD or NumPy array containing the first component of
               shear.
    @param g2  Square galsim.ImageF, galsim.ImageD or NumPy array containing the second component of
               shear.

    @return kappa_E, kappa_B  The first element of this tuple represents the convergence field
                              underlying the input shears; the second element is the convergence
                              field generated were all shears rotated by 45 degrees prior to input.
                              Both are NumPy arrays.
    """
    # Checks on inputs
    import galsim.utilities
    if (isinstance(g1, (galsim.ImageD, galsim.ImageF)) and
        isinstance(g2, (galsim.ImageD, galsim.ImageF))):
        g1 = g1.array
        g2 = g2.array
    elif isinstance(g1, np.ndarray) and isinstance(g2, np.ndarray):
        pass
    else:
        raise TypeError("Input g1 and g2 must be galsim Image (float types) or NumPy arrays.")
    if g1.shape != g2.shape:
        raise ValueError("Input g1 and g2 must be the same shape.")
    if g1.shape[0] != g1.shape[1]:
        raise NotImplementedError("Non-square input shear grids not supported.")

    # Then setup the kx, ky grids
    kx, ky = galsim.utilities.kxky(g1.shape)
    kz = kx + ky*1j

    # exp(2i psi) = kz^2 / |kz|^2
    ksq = kz*np.conj(kz)
    # Need to adjust denominator for kz=0 to avoid division by 0.
    ksq[0,0] = 1.
    exp2ipsi = kz*kz/ksq

    # Build complex g = g1 + i g2
    gz = g1 + g2*1j

    # Go to fourier space
    gz_k = np.fft.fft2(gz)

    # Equation 2.1.12 of Kaiser & Squires (1993) is equivalent to:
    #   kz_k = -np.conj(exp2ipsi)*gz_k
    # However, this equation has a sign error.  There should not be a minus in front.
    # If you follow their subsequent deviation, you will see that they drop the minus sign
    # when they get to 2.1.15 (another - appears from the derivative).  2.1.15 is correct.
    # e.g. it correctly produces a positive point mass for tangential shear ~ 1/r^2.
    # So this implies that the minus sign in 2.1.12 should not be there.
    kz_k = np.conj(exp2ipsi)*gz_k

    # Come back to real space
    kz = np.fft.ifft2(kz_k)

    # kz = kappa_E + i kappa_B
    kappaE = np.real(kz)
    kappaB = np.imag(kz)
    return kappaE, kappaB


