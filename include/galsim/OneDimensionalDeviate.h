// -*- c++ -*-
#ifndef ONE_DIMENSIONAL_DEVIATE_H
#define ONE_DIMENSIONAL_DEVIATE_H

#include <set>
#include <list>
#include <vector>
#include <functional>
#include "Random.h"
#include "PhotonArray.h"

namespace galsim {

    /**
     * @brief An interface class for functions giving differential flux vs x or r.
     *
     * Functions derived from this interface can be integrated by `Int.h` and
     * can be sampled by `OneDimensionalDeviate`.
     */
    class FluxDensity: public std::unary_function<double,double> {
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
     * having no sign changes or extrema, which will makes it easier to sample the FluxDensity
     * function over its domain using either rejection sampling or by weighting uniformly distributed
     * photons.
     *
     * This class could be made a subclass of `OneDimensionalDeviate` as it should only be used by
     * methods of that class.
     *
     * The `Interval` represents flux (or unnormalized probability) density in a continguous interval on 
     * on the line, or, for `_isRadial=true`, represents axisymmetric density in an annulus on the plane.
     *
     * The object keeps track of the integrated flux (or unnormalized probability) in its interval/annulus,
     * and the cumulative flux of all intervals up to and including this one.
     *
     * The `drawWithin()` method will select one photon (and flux) drawn from within this interval or annulus,
     * such that the expected flux distribution matches the FluxDensity function.  This can be done one of two ways: 
     * If `_useRejectionMethod=true`, then an x (or r) position is chosen that would match the chosen cumulative flux
     * were the FluxDensity uniform over the interval.  Then the FluxDensity is evaluated, compared to the maximum
     * within the interval (which is always an endpoint), and a UniformDeviate is drawn to decide whether to keep or
     * reject this x photon.  This repeats until a position is kept.
     *
     * If `_useRejectionMethod=false`, then no rejection is done.  The photon is kept, but is given a flux value
     * equal to the FluxDensity at x relative to the mean over the interval.  This is faster but makes the statistics
     * of the photons harder to interpret.
     *
     * See the `OneDimensionalDeviate` docstrings for more information.
     */
    class Interval {
    public:
        /**
         * @brief Constructor
         *
         * Note that no copy of the function is saved.  The function whose reference is passed must remain
         * existence through useful lifetime of the `Interval`
         * @param[in] fluxDensity The function giving flux (= unnormalized probability) density.
         * @param[in] xLower Lower bound in x (or radius) of this interval.
         * @param[in] xUpper Upper bound in x (or radius) of this interval.
         * @param[in] isRadial Set true if this is an annulus on a plane, false for linear interval.
         */
        Interval(const FluxDensity& fluxDensity,
                 double xLower,
                 double xUpper,
                 bool isRadial=false): _fluxDensityPtr(&fluxDensity),
                                       _xLower(xLower),
                                       _xUpper(xUpper),
                                       _isRadial(isRadial),
                                       _fluxIsReady(false) {}
        /**
         * @brief Draw one photon position and flux from within this interval
         * @param[in] absFlux The cumulative absolute flux at which the photon should be selected.
         * @param[out] x x (or radial) coordinate of the selected photon.
         * @param[out] flux flux of the selected photon, nominally +-1, but can differ if not using rejection.
         * @param[out] ud UniformDeviate used for rejection sampling, if needed.
         */
        void drawWithin(double absFlux, double& x, double& flux,
                        UniformDeviate& ud) const;
        /**
         * @brief Get integrated flux over this interval or annulus.  Performs integral if not already cached.
         * @returns Integrated flux in interval.
         */
        double getDifferentialFlux() const {checkFlux(); return _differentialFlux;}
        /**
         * @brief Return cumulative integral of absolute value of flux out to upper limit of this interval.
         *
         * Note this cannot be set internally: requires an outside agent to set this with knownledge of other intervals.
         * @returns cumulative flux to end of this interval.
         */
        double getCumulativeFlux() const {return _cumulativeFlux;}
        /**
         * @brief Set cumulative flux to end of this interval
         * @param[in] f Cumulative flux.
         */
        void setCumulativeFlux(double f)  {_cumulativeFlux = f;}
        /**
         * @brief Report interval bounds
         * @param[out] xLower Interval lower bound
         * @param[out] xUpper Interval upper bound
         */
        void getRange(double& xLower, double& xUpper) const {
            xLower = _xLower;
            xUpper = _xUpper;
        }

        /**
         * @brief Return a list of intervals that divide this one into acceptably small ones.
         *
         * This routine works by recursive bisection.  Intervals that are returned have all had their
         * fluxes integrated.  Intervals are split until the FluxDensity does not vary too much within 
         * an interval, or when their flux is below `smallFlux`.
         * @param[in] smallFlux Flux below which a sub-interval is not further split.
         * @returns List contiguous Intervals whose union is this one.
         */
        std::list<Interval> split(double smallFlux);
        /// @brief An ordering operation to use in sorting Intervals by cumulative flux.
        bool operator<(const Interval& rhs) const {
            return _cumulativeFlux<rhs._cumulativeFlux;
        }
    private:
        const FluxDensity* _fluxDensityPtr;  ///< Pointer to the parent FluxDensity function.
        double _xLower; ///< Interval lower bound
        double _xUpper; ///< Interval upper bound
        bool _isRadial; ///< True if domain is an annulus, otherwise domain is a linear interval.
        mutable bool _fluxIsReady; ///< True if flux has been integrated
        void checkFlux() const; ///< Calculate flux if it has not already been done.
        mutable double _differentialFlux; ///< Integrated flux in this interval (can be negative)
        double _cumulativeFlux; ///< Cumulative absolute-value flux including this interval
        /// @brief Finds the x or radius coord that would enclose fraction of this intervals flux if flux were constant.
        double interpolateFlux(double fraction) const; 
        /**
         * Set this variable true if returning equal fluxes with rejection method, vs returning 
         *non-unity flux weight.
         */
        bool _useRejectionMethod;
        double _maxAbsDensity; ///< Maximum absolute flux density in the interval (assumed to be at an endpoint)
        double _meanAbsDensity; ///< Mean absolute flux density in the interval.
    };

    /**
     * @brief Class which implements random sampling of an arbitrary one-dimensional distribution, for photon shooting.
     *
     * The point of this class is to take any function that is derived from `FluxDensity` and be able to sample
     * it with photons such that the expectation value of the flux density matches the input function exactly.  This
     * class is for functions which do not have convenient analytic means of inverting their cumulative flux distribution.
     *
     * As explained in SBProfile::shoot(), both positive and negative-flux photons can exist, but we aim that the absolute
     * value of flux be nearly constant so that statistical errors are predictable.  This code does this by first
     * dividing the domain of the function into `Interval` objects, with known integrated (absolute) flux in each.  To
     * shoot a photon, a UniformDeviate is selected and scaled to represent the cumulative flux that should exist within
     * the position of the photon.  The class first uses the binary-search feature built into the Standard Library `set`
     * container to locate the `Interval` that will contain the photon.  Then it asks the `Interval` to decide where
     * within the `Interval` to place the photon.  As noted in the `Interval` docstring, this can be done either by
     * rejection sampling, or - if the range of FluxDensity values within an interval is small - by simply adjusting
     * the flux to account for deviations from uniform flux density within the interval.
     *
     * On construction, the class must be provided with some information about the nature of the function being
     * sampled.  The length scale and flux scale of the function should be of order unity.  The elements of the 
     * `range` array should be ordered, span the desired domain of the function, and split the domain into intervals
     * such that:
     * * There are no sign changes within an interval
     * * There is at most one extremum within the interval
     * * Any extremum can be localized by sampling the interval at `RANGE_DIVISION_FOR_EXTREMA` equidistant points.
     * * The function is smooth enough to be integrated over the interval with standard basic methods.
     */
    class OneDimensionalDeviate {
    public:
        /**
         * @brief constructor
         * @param[in] fluxDensity The FluxDensity being sampled.  No copy is made, original must stay in existence.
         * @param[in] range Ordered argument vector specifying the domain for sampling as described in class docstring.
         * @param[in] isRadial Set true for an axisymmetric function on the plane; false (default) for linear domain.
         */
        OneDimensionalDeviate(const FluxDensity& fluxDensity, std::vector<double>& range,
                              bool isRadial=false);
        /// @brief Return total flux in positive regions of FluxDensity
        double getPositiveFlux() const {return _positiveFlux;}
        /// @brief Return absolute value of total flux in regions of negative FluxDensity
        double getNegativeFlux() const {return _negativeFlux;}
        /**
         * @brief Draw photons from the distribution.
         *
         * If `_isRadial=true`, photons will populate the plane.  Otherwise only the x coordinate of photons
         * will be generated, for 1d distribution.
         * @param[in] N number of photons to draw
         * @param[in] ud UniformDeviate used to produce random selections.
         */
        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        const FluxDensity& _fluxDensity; ///< Function being sampled
        std::set<Interval> _intervalSet; ///< Container for Intervals generated at construction
        double _positiveFlux; ///< Stored total positive flux
        double _negativeFlux; ///< Stored total negative flux
        const bool _isRadial; ///< True for 2d axisymmetric function, false for 1d function
    };

} // namespace galsim

#endif
