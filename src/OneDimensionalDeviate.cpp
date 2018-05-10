/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

//#define DEBUGLOGGING

#include "OneDimensionalDeviate.h"
#include "integ/Int.h"
#include "SBProfile.h"
#include "math/Angle.h"

// Define this variable to find azimuth (and sometimes radius within a unit disc) of 2d photons by
// drawing a uniform deviate for theta, instead of drawing 2 deviates for a point on the unit
// circle and rejecting corner photons.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using sin/cos was faster for icpc, but not g++ or clang++.
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

namespace galsim {

    // Wrapper class for doing integrals over annuli
    template <class F>
    class RTimesF: public std::unary_function<double,double> {
    public:
        RTimesF(const F& function): _function(function) {}
        double operator()(double r) const { return 2.*M_PI*r*_function(r); }
    private:
        const F& _function;
    };

    // Function to isolate an extremum of function in an interval.
    bool findExtremum( const FluxDensity& function,
                       double xmin,
                       double xmax,
                       double& extremum,
                       int divisionSteps,
                       double xFractionalTolerance = 1e-4)
    {
        if (xmax < xmin) std::swap(xmax,xmin);
        const double xTolerance = xFractionalTolerance*(xmax-xmin);
        // First bracket extremum by division into fixed number of steps
        double xStep = (xmax - xmin) / divisionSteps;
        double x1 = xmin;
        double x2 = xmin + xStep;
        double f1 = function(x1);
        double f2 = function(x2);
        double df1 = f2 - f1;
        double x3 = xmin + 2*xStep;
        double f3 = function(x3);
        double df2 = f3 - f2;

        while (df1 * df2 >= 0.) {
            xdbg<<"df1, df2 = "<<df1<<','<<df2<<std::endl;
            if (x3 >= xmax)
                return false;  // no extremum bracketed.
            x1 = x2;
            f1 = f2;
            x2 = x3;
            f2 = f3;
            df1 = df2;
            x3 += xStep;
            f3 = function(x3);
            df2 = f3 - f2;
        }
        xdbg<<"df1, df2 = "<<df1<<','<<df2<<std::endl;
        xdbg<<"f("<<x1<<") = "<<f1<<" = "<<function(x1)<<std::endl;
        xdbg<<"f("<<x2<<") = "<<f2<<" = "<<function(x2)<<std::endl;
        xdbg<<"f("<<x3<<") = "<<f3<<" = "<<function(x3)<<std::endl;

        // Fat left tells which side is the fatter one.  Keep splitting the fatter side.
        bool fatLeft = (x2-x1) > (x3-x2);

        // Then use golden sections to localize - could use Brent's method for speed.
        // Based on Numerical Recipes 10.1.
        const double GOLDEN = 2./(1+sqrt(5.));
        while (std::abs(x3-x1) > xTolerance) {
            xdbg<<"x1,x2,x3 = "<<x1<<','<<x2<<','<<x3<<std::endl;
            xdbg<<"f1,f2,f3 = "<<f1<<','<<f2<<','<<f3<<std::endl;
            xdbg<<"df1,df2 = "<<df1<<','<<df2<<std::endl;
            xdbg<<"fatleft = "<<fatLeft<<"  "<<x2-x1<<" >? "<<x3-x2<<std::endl;
            // Loop invariants:
            xassert(x1 < x2);
            xassert(x2 < x3);
            xassert(df1 == f2 - f1);
            xassert(df2 == f3 - f2);
            xassert(df1 * df2 < 0.);
            xassert(fatLeft == (x2-x1) > (x3-x2));

            if (fatLeft) {
                xdbg<<"fat left\n";
                // Split left-hand interval
                double xTrial = x1 + GOLDEN*(x2-x1);
                double fTrial = function(xTrial);
                double dfTrial = f2 - fTrial;
                xdbg<<"Trial = "<<xTrial<<","<<fTrial<<","<<dfTrial<<std::endl;
                if (dfTrial * df2 < 0.) {
                    xdbg<<"trial/2/3\n";
                    // Extremum is in trial / 2 / 3
                    x1 = xTrial;
                    f1 = fTrial;
                    df1 = dfTrial;
                    fatLeft = (x2-x1) > (x3-x2);
                } else {
                    xdbg<<"1/trial/2\n";
                    // Now bracketed in 1 / trial / 2
                    x3 = x2;
                    f3 = f2;
                    x2 = xTrial;
                    f2 = fTrial;
                    df1 = f2 - f1;
                    df2 = dfTrial;
                    fatLeft = true;
                }
            } else {
                xdbg<<"fat right\n";
                // Split right-hand interval (2 / trial / 3)
                double xTrial = x3 - GOLDEN*(x3-x2);
                double fTrial = function(xTrial);
                double dfTrial = fTrial - f2;
                xdbg<<"Trial = "<<xTrial<<","<<fTrial<<","<<dfTrial<<std::endl;
                if (dfTrial * df1 < 0.) {
                    xdbg<<"1/2/trial\n";
                    // Extremum is in 1 / 2 / trial
                    x3 = xTrial;
                    f3 = fTrial;
                    df2 = dfTrial;
                    fatLeft = (x2-x1) > (x3-x2);
                } else {
                    xdbg<<"2/trial/3\n";
                    // Now bracketed in 2 / trial / 3
                    x1 = x2;
                    f1 = f2;
                    x2 = xTrial;
                    f2 = fTrial;
                    df1 = dfTrial;
                    df2 = f3 - f2;
                    fatLeft = false;
                }
            }
        }

        // Finish with a single quadratic step to tighten up the accuracy.
        double dx1 = x2-x1;
        double dx2 = x3-x2;
        xassert(dx1 > 0);
        xassert(dx2 > 0);
        xassert(df1 * df2 < 0);
        extremum = x2 + 0.5 * (df1*dx2*dx2 + df2*dx1*dx1) / (df1*dx2 - df2*dx1);

        xdbg<<"Found extrumum at "<<extremum<<std::endl;
        return true;
    }

    double Interval::interpolateFlux(double fraction) const
    {
        // Find the x (or radius) value that encloses fraction
        // of the flux in this Interval if the function were constant
        // over the interval.
        if (_isRadial) {
            double rsq = _xLower*_xLower*(1.-fraction) + _xUpper*_xUpper*fraction;
            return sqrt(rsq);
        } else {
            return _xLower + (_xUpper - _xLower)*fraction;
        }
        return 0.;      // Will never get here.
    }


    // Select a photon from within the interval.  unitRandom
    // as an initial random value, more from ud if needed for rejections.
    void Interval::drawWithin(double unitRandom, double& x, double& flux,
                              UniformDeviate ud) const
    {
        xdbg<<"drawWithin interval\n";
        xdbg<<"_flux = "<<_flux<<std::endl;
        double fractionOfInterval = std::min(unitRandom, 1.);
        xdbg<<"fractionOfInterval = "<<fractionOfInterval<<std::endl;
        fractionOfInterval = std::max(0., fractionOfInterval);
        xdbg<<"fractionOfInterval => "<<fractionOfInterval<<std::endl;
        x = interpolateFlux(fractionOfInterval);
        xdbg<<"x = "<<x<<std::endl;
        flux = 1.;
        if (_useRejectionMethod) {
            xdbg<<"use rejection\n";
            while ( ud() > std::abs((*_fluxDensityPtr)(x)) * _invMaxAbsDensity) {
                x = interpolateFlux(ud());
            }
            xdbg<<"x => "<<x<<std::endl;
            if (_flux < 0) flux = -1.;
        } else {
            flux = (*_fluxDensityPtr)(x) * _invMeanAbsDensity;
        }
        xdbg<<"flux = "<<flux<<std::endl;
    }

    void Interval::checkFlux() const
    {
        if (_fluxIsReady) return;
        if (_isRadial) {
            // Integrate r*F
            RTimesF<FluxDensity> integrand(*_fluxDensityPtr);
            _flux = integ::int1d(integrand,
                                 _xLower, _xUpper,
                                 _gsparams.integration_relerr,
                                 _gsparams.integration_abserr);
        } else {
            // Integrate the input function
            _flux = integ::int1d(*_fluxDensityPtr,
                                 _xLower, _xUpper,
                                 _gsparams.integration_relerr,
                                 _gsparams.integration_abserr);
        }
        _fluxIsReady = true;
    }

    // Divide an interval into ones that are sufficiently small.  It's small enough if either
    // (a) The max/min FluxDensity ratio in the interval is small enough, i.e. close to constant, or
    // (b) The total flux in the interval is below smallFlux.
    // In the former case, photons will be selected by drawing from a uniform distribution and then
    // adjusting weights by flux.  In latter case, rejection sampling will be used to select
    // within interval.
    std::list<Interval> Interval::split(double smallFlux)
    {
        // Get the flux in this interval
        checkFlux();
        if (_isRadial) {
            _invMeanAbsDensity = std::abs( (M_PI*(_xUpper*_xUpper - _xLower*_xLower)) / _flux );
        } else {
            _invMeanAbsDensity = std::abs( (_xUpper - _xLower) / _flux );
        }
        double densityLower = (*_fluxDensityPtr)(_xLower);
        double densityUpper = (*_fluxDensityPtr)(_xUpper);
        _invMaxAbsDensity = 1. / std::max(std::abs(densityLower), std::abs(densityUpper));

        std::list<Interval> result;
        double densityVariation = 0.;
        if (std::abs(densityLower) > 0. && std::abs(densityUpper) > 0.)
            densityVariation = densityLower / densityUpper;
        if (densityVariation > 1.) densityVariation = 1. / densityVariation;
        if (densityVariation > _gsparams.allowed_flux_variation) {
            // Don't split if flux range is small
            _useRejectionMethod = false;
            result.push_back(*this);
        } else if (std::abs(_flux) < smallFlux) {
            // Don't split further, as it will be rare to be in this interval
            // and rejection is ok.
            _useRejectionMethod = true;
            result.push_back(*this);
        } else {
            // Split the interval.  Call (recursively) split() for left & right
            double midpoint = 0.5*(_xLower + _xUpper);
            Interval left(*_fluxDensityPtr, _xLower, midpoint, _isRadial, _gsparams);
            Interval right(*_fluxDensityPtr, midpoint, _xUpper, _isRadial, _gsparams);
            std::list<Interval> add = left.split(smallFlux);
            result.splice(result.end(), add);
            add = right.split(smallFlux);
            result.splice(result.end(), add);
        }
        return result;
    }

    OneDimensionalDeviate::OneDimensionalDeviate(const FluxDensity& fluxDensity,
                                                 std::vector<double>& range,
                                                 bool isRadial,
                                                 const GSParams& gsparams) :
        _fluxDensity(fluxDensity),
        _positiveFlux(0.),
        _negativeFlux(0.),
        _isRadial(isRadial),
        _gsparams(gsparams)
    {
        dbg<<"Start ODD constructor\n";
        dbg<<"Input range has "<<range.size()<<" entries\n";
        dbg<<"radial? = "<<isRadial<<std::endl;

        // Typedef for indices of standard containers, which don't like int values
        typedef std::vector<double>::size_type Index;

        // First calculate total flux so we know when an interval is a small amt of flux
        for (Index iRange = 0; iRange < range.size()-1; iRange++) {
            xdbg<<"range "<<iRange<<" = "<<range[iRange]<<" ... "<<range[iRange+1]<<std::endl;
            // Integrate total flux (and sign) in each range
            Interval segment(fluxDensity, range[iRange], range[iRange+1], _isRadial, _gsparams);
            double rangeFlux = segment.getFlux();
            if (rangeFlux >= 0.) _positiveFlux += rangeFlux;
            else _negativeFlux += std::abs(rangeFlux);
        }
        dbg<<"posFlux = "<<_positiveFlux<<std::endl;
        dbg<<"negFlux = "<<_negativeFlux<<std::endl;
        double totalAbsoluteFlux = _positiveFlux + _negativeFlux;
        dbg<<"totFlux = "<<totalAbsoluteFlux<<std::endl;

        if (totalAbsoluteFlux == 0.) {
            // The below calculation will crash, so do something trivial that works.
            Interval segment(fluxDensity, range[0], range[1], _isRadial, _gsparams);
            _pt.push_back(segment);
            _pt.buildTree();
            return;
        }

        // Now break each range into Intervals
        for (Index iRange = 0; iRange < range.size()-1; iRange++) {
            // See if there is an extremum to split this range:
            double extremum;
            if (findExtremum(_fluxDensity,
                             range[iRange],
                             range[iRange+1],
                             extremum,
                             _gsparams.range_division_for_extrema)) {
                xdbg<<"range "<<iRange<<" = "<<range[iRange]<<" ... "<<range[iRange+1]<<
                    "  has an extremum at "<<extremum<<std::endl;
                // Do 2 ranges
                {
                    Interval splitit(_fluxDensity, range[iRange], extremum, _isRadial, _gsparams);
                    std::list<Interval> leftList = splitit.split(
                        _gsparams.small_fraction_of_flux * totalAbsoluteFlux);
                    xdbg<<"Add "<<leftList.size()<<" intervals on left of extremem\n";
                    _pt.insert(_pt.end(), leftList.begin(), leftList.end());
                }
                {
                    Interval splitit(_fluxDensity, extremum, range[iRange+1], _isRadial, _gsparams);
                    std::list<Interval> rightList = splitit.split(
                        _gsparams.small_fraction_of_flux * totalAbsoluteFlux);
                    xdbg<<"Add "<<rightList.size()<<" intervals on right of extremem\n";
                    _pt.insert(_pt.end(), rightList.begin(), rightList.end());
                }
            } else {
                // Just single Interval in this range, no extremum:
                xdbg<<"single interval\n";
                Interval splitit(
                    _fluxDensity, range[iRange], range[iRange+1], _isRadial, _gsparams);
                std::list<Interval> leftList = splitit.split(
                    _gsparams.small_fraction_of_flux * totalAbsoluteFlux);
                xdbg<<"Add "<<leftList.size()<<" intervals\n";
                _pt.insert(_pt.end(), leftList.begin(), leftList.end());
            }
        }
        dbg<<"Total of "<<_pt.size()<<" intervals\n";
        // Build the ProbabilityTree
        double thresh = std::numeric_limits<double>::epsilon() * totalAbsoluteFlux;
        dbg<<"thresh = "<<thresh<<std::endl;
        _pt.buildTree(thresh);
    }

    void OneDimensionalDeviate::shoot(PhotonArray& photons, UniformDeviate ud, bool xandy) const
    {
        const int N = photons.size();
        dbg<<"OneDimentionalDeviate shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        dbg<<"isradial? "<<_isRadial<<std::endl;
        dbg<<"xandy = "<<xandy<<std::endl;
        dbg<<"N = "<<N<<std::endl;
        assert(N>=0);
        if (N==0) return;
        double totalAbsoluteFlux = getPositiveFlux() + getNegativeFlux();
        dbg<<"totalAbsFlux = "<<totalAbsoluteFlux<<std::endl;
        double fluxPerPhoton = totalAbsoluteFlux / N;
        if (xandy) fluxPerPhoton *= totalAbsoluteFlux;
        dbg<<"fluxPerPhoton = "<<fluxPerPhoton<<std::endl;

        // For each photon, first decide which Interval it's in, then drawWithin the interval.
        for (int i=0; i<N; i++) {
            if (_isRadial) {
#ifdef USE_COS_SIN
                double unitRandom = ud();
                Interval* chosen = _pt.find(unitRandom);
                // Now draw a radius from within selected interval
                double radius, flux;
                chosen->drawWithin(unitRandom, radius, flux, ud);
                // Draw second ud to get azimuth
                double theta = 2.*M_PI*ud();
                double sintheta, costheta;
                math::sincos(theta, sintheta, costheta);
                photons.setPhoton(i, radius*costheta, radius*sintheta, flux*fluxPerPhoton);
#else
                // Alternate method: doesn't need sin & cos but needs sqrt
                // First get a point uniformly distributed in unit circle
                double xu, yu, rsq;
                do {
                    xu = 2.*ud()-1.;
                    yu = 2.*ud()-1.;
                    rsq = xu*xu+yu*yu;
                } while (rsq>=1. || rsq==0.);
                // Now rsq is unit deviate from 0 to 1
                double unitRandom = rsq;
                const Interval* chosen = _pt.find(unitRandom);
                // Now draw a radius from within selected interval
                double radius, flux;
                chosen->drawWithin(unitRandom, radius, flux, ud);
                // Rescale x & y:
                double rScale = radius / std::sqrt(rsq);
                photons.setPhoton(i, xu*rScale, yu*rScale, flux*fluxPerPhoton);
#endif
            } else {
                // Simple 1d interpolation
                double unitRandom = ud();
                const Interval* chosen = _pt.find(unitRandom);
                // Now draw an x from within selected interval
                double x, flux;
                chosen->drawWithin(unitRandom, x, flux, ud);
                if (xandy) {
                    double y, flux2;
                    unitRandom = ud();
                    chosen = _pt.find(unitRandom);
                    chosen->drawWithin(unitRandom, y, flux2, ud);
                    photons.setPhoton(i, x, y, flux*flux2*fluxPerPhoton);
                } else {
                    photons.setPhoton(i, x, 0., flux*fluxPerPhoton);
                }
            }
        }
        dbg<<"OneDimentionalDeviate Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

} // namespace galsim
