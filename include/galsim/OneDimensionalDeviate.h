/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef GalSim_OneDimensionalDeviate_H
#define GalSim_OneDimensionalDeviate_H

#include <list>
#include <vector>
#include <functional>
#include "Random.h"
#include "PhotonArray.h"
#include "ProbabilityTree.h"
#include "SBProfile.h"
#include "Std.h"

namespace galsim {

    /**
     * @brief An interface class for functions giving differential flux vs x or r.
     *
     * Functions derived from this interface can be integrated by `Int.h` and
     * can be sampled by `OneDimensionalDeviate`.
     */
    class PUBLIC_API FluxDensity: public std::unary_function<double,double>
    {
    public:
        /// @brief virtual destructor for base class
        virtual ~FluxDensity() {}
        /**
         * @brief Interface requires implementation of operator()
         * @param[in] x The linear position or radius
         * @returns Flux or probability density at location of argument.
         */
        virtual double operator()(double x) const=0;
    };

    /**
     * @brief Class used to represent a linear interval or an annulus of probability function
     *
     * An `Interval` is a contiguous domain over which a `FluxDensity` function is well-behaved,
     * having no sign changes or extrema, which will make it easier to sample the FluxDensity
     * function over its domain using either rejection sampling or by weighting uniformly
     * distributed photons.
     *
     * This class could be made a subclass of `OneDimensionalDeviate` as it should only be used by
     * methods of that class.
     *
     * The `Interval` represents flux (or unnormalized probability) density in a continguous
     * interval on on the line, or, for `_isRadial=true`, represents axisymmetric density in an
     * annulus on the plane.
     *
     * The object keeps track of the integrated flux (or unnormalized probability) in its
     * interval/annulus, and the cumulative flux of all intervals up to and including this one.
     *
     * The `drawWithin()` method will select one photon (and flux) drawn from within this interval
     * or annulus, such that the expected flux distribution matches the FluxDensity function.
     *
     * See the `OneDimensionalDeviate` docstrings for more information.
     */
    class PUBLIC_API Interval
    {
    public:
        /**
         * @brief Constructor
         *
         * Note that no copy of the function is saved.  The function whose reference is passed must
         * remain in existence through useful lifetime of the `Interval`
         * @param[in] fluxDensity  The function giving flux (= unnormalized probability) density.
         * @param[in] xLower       Lower bound in x (or radius) of this interval.
         * @param[in] xUpper       Upper bound in x (or radius) of this interval.
         * @param[in] isRadial     Set true if this is an annulus on a plane, false for linear
         *                         interval.
         * @param[in] gsparams     GSParams object storing constants that control the accuracy of
         *                         operations.
         */
        Interval(const FluxDensity& fluxDensity,
                 double xLower,
                 double xUpper,
                 bool isRadial,
                 const GSParams& gsparams) :
            _fluxDensityPtr(&fluxDensity),
            _xLower(xLower),
            _xUpper(xUpper),
            _xRange(_xUpper - _xLower),
            _isRadial(isRadial),
            _gsparams(gsparams),
            _fluxIsReady(false)
        {}

        Interval(const Interval& rhs) :
            _fluxDensityPtr(rhs._fluxDensityPtr),
            _xLower(rhs._xLower),
            _xUpper(rhs._xUpper),
            _xRange(rhs._xRange),
            _isRadial(rhs._isRadial),
            _gsparams(rhs._gsparams),
            _fluxIsReady(false),
            _a(rhs._a), _b(rhs._b), _c(rhs._c), _d(rhs._d)
        {}

        Interval& operator=(const Interval& rhs)
        {
            // Everything else is constant, so no need to copy.
            _xLower = rhs._xLower;
            _xUpper = rhs._xUpper;
            _xRange = rhs._xRange;
            _isRadial = rhs._isRadial;
            _fluxIsReady = false;
            _a = rhs._a;
            _b = rhs._b;
            _c = rhs._c;
            _d = rhs._d;
            return *this;
        }

        /**
         * @brief Draw one photon position and flux from within this interval
         * @param[in] unitRandom An initial uniform deviate to select photon
         * @param[out] x (or radial) coordinate of the selected photon.
         * @param[out] flux flux of the selected photon = +-1
         */
        void drawWithin(double unitRandom, double& x, double& flux) const;

        /**
         * @brief Get integrated flux over this interval or annulus.
         *
         * Performs integral if not already cached.
         *
         * @returns Integrated flux in interval.
         */
        double getFlux() const { checkFlux(); return _flux; }

        /**
         * @brief Report interval bounds
         * @param[out] xLower Interval lower bound
         * @param[out] xUpper Interval upper bound
         */
        void getRange(double& xLower, double& xUpper) const
        {
            xLower = _xLower;
            xUpper = _xUpper;
        }

        /**
         * @brief Return a list of intervals that divide this one into acceptably small ones.
         *
         * This routine works by recursive bisection.  Intervals that are returned have all had
         * their fluxes integrated.  Intervals are split until the error from a linear
         * approximation to f(x) is less than toler;
         * @param[in] toler Tolerance on the flux error below which a sub-interval is not split.
         * @returns List of contiguous Intervals whose union is this one.
         */
        std::list<shared_ptr<Interval> > split(double toler);

    private:

        const FluxDensity* _fluxDensityPtr;  // Pointer to the parent FluxDensity function.
        double _xLower; // Interval lower bound
        double _xUpper; // Interval upper bound
        double _xRange; // _xUpper - _xLower  (used a lot)
        bool _isRadial; // True if domain is an annulus, otherwise domain is a linear interval.
        const GSParams& _gsparams;

        mutable bool _fluxIsReady; // True if flux has been integrated
        void checkFlux() const; // Calculate flux if it has not already been done.
        mutable double _flux; // Integrated flux in this interval (can be negative)

        // Finds the x or radius coord that would enclose fraction of this interval's flux
        // if flux were constant.
        double interpolateFlux(double fraction) const;

        double _a, _b, _c, _d;  // Coefficients used for solving for dx in the interval.

    };

    /**
     * @brief Class which implements random sampling of an arbitrary one-dimensional distribution,
     * for photon shooting.
     *
     * The point of this class is to take any function that is derived from `FluxDensity` and be
     * able to sample it with photons such that the expectation value of the flux density matches
     * the input function exactly.  This class is for functions which do not have convenient
     * analytic means of inverting their cumulative flux distribution.
     *
     * As explained in SBProfile::shoot(), both positive and negative-flux photons can exist, but we
     * aim that the absolute value of flux be nearly constant so that statistical errors are
     * predictable.  This code does this by first dividing the domain of the function into
     * `Interval` objects, with known integrated (absolute) flux in each.  To shoot a photon, a
     * UniformDeviate is selected and scaled to represent the cumulative flux that should exist
     * within the position of the photon.  The class first uses the binary-search feature built into
     * the Standard Library `set` container to locate the `Interval` that will contain the photon.
     * Then it asks the `Interval` to decide where within the `Interval` to place the photon.  As
     * noted in the `Interval` docstring, this can be done either by rejection sampling, or - if the
     * range of FluxDensity values within an interval is small - by simply adjusting the flux to
     * account for deviations from uniform flux density within the interval.
     *
     * On construction, the class must be provided with some information about the nature of the
     * function being sampled.  The length scale and flux scale of the function should be of order
     * unity.  The elements of the `range` array should be ordered, span the desired domain of the
     * function, and split the domain into intervals such that:
     * - There are no sign changes within an interval
     * - There is at most one extremum within the interval
     * - Any extremum can be localized by sampling the interval at `RANGE_DIVISION_FOR_EXTREMA`
         equidistant points.
     * - The function is smooth enough to be integrated over the interval with standard basic
     *   methods.
     */
    class PUBLIC_API OneDimensionalDeviate
    {
    public:
        /**
         * @brief constructor
         * @param[in] fluxDensity  The FluxDensity being sampled.  No copy is made, original must
         *                         stay in existence.
         * @param[in] range        Ordered argument vector specifying the domain for sampling as
         *                         described in class docstring.
         * @param[in] isRadial     Set true for an axisymmetric function on the plane; false
         *                         for linear domain.
         * @param[in] nominal_flux The expected true integral of the input fluxDensity function.
         * @param[in] gsparams     GSParams object storing constants that control the accuracy of
         *                         operations, if different from the default.
         */
        OneDimensionalDeviate(
            const FluxDensity& fluxDensity, std::vector<double>& range, bool isRadial,
            double nominal_flux, const GSParams& gsparams);

        /// @brief Return total flux in positive regions of FluxDensity
        double getPositiveFlux() const {return _positiveFlux;}

        /// @brief Return absolute value of total flux in regions of negative FluxDensity
        double getNegativeFlux() const {return _negativeFlux;}

        /**
         * @brief Draw photons from the distribution.
         *
         * If `_isRadial=true`, photons will populate the plane.  Otherwise only the x coordinate
         * of photons will be generated, for 1d distribution.
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @param[in] xandy Whether to populate both x and y values (true) or just x (false)
         */
        void shoot(PhotonArray& photons, UniformDeviate ud, bool xandy=false) const;

    private:

        const FluxDensity& _fluxDensity; // Function being sampled
        ProbabilityTree<Interval> _pt; // Binary tree of intervals for photon shooting
        double _positiveFlux; // Stored total positive flux
        double _negativeFlux; // Stored total negative flux
        const bool _isRadial; // True for 2d axisymmetric function, false for 1d function
        GSParams _gsparams;
    };

} // namespace galsim

#endif
