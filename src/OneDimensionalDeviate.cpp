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
        double operator()(double r) const { return r*_function(r); }
    private:
        const F& _function;
    };

    // Function to isolate an extremum of function in an interval.
    bool findExtremum( const FluxDensity& function,
                       double xmin,
                       double xmax,
                       double& extremum,
                       int divisionSteps = 32,
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
        // of the flux in this Interval if the function were linear
        // over the interval.
        if (_isRadial) {
            // The model is pdf(r) = f0 r + (f1-f0)/(r1-r0) * (r-r0) r
            //    (where we ignore the 2pi, since it will fall out in the end)
            // Let dr = the relative fraction from rL to rU
            // Then cdf(dr) = int(pdf(r), r=r0..r0+dr(r1-r0))
            //              = (r1-r0) dr (f0 r0 + 1/2 dr (f0 (r1-2r0) + f1 r0)
            //                             + 1/3 dr^2 (f1-f0) (r1-r0))
            //              = fraction * 1/6 (r1-r0) (f0 (2r0+r1) + f1 (2r1+r0))
            // Solve for dr
            //   1/3(f1-f0)(r1-r0) dr^3 + (f0 (r1-r0) + (f1-f0) r0) dr^2  + 2 f0 r0 dr
            //          = 1/3 fraction ( f0 (2r0+r1) + f1 (2r1+r0) )
            // Solve this iteratively, ignoring the dr^3 term at first, and then adding it
            // back in as a correction.
            double d = _d * fraction;
            double dr = 2.*d / (std::sqrt(4.*_b*d + _c*_c) + _c);
            double delta;
            do {
                // Do a Newton step on the whole thing.
                // f(x) = a x^3 + b x^2 + c x = d
                // df/dx = 3 a x^2 + 2 b x + c
                double df = dr*(_c + dr*(_b + _a*dr)) - d;
                double dfddr = _c + dr*(2.*_b + 3.*_a*dr);
                delta = df / dfddr;
                dr -= delta;
            } while (std::abs(delta) > _gsparams.shoot_accuracy);
            return _xLower + _xRange * dr;
        } else {
            // The model is pdf(x) = f0 + (f1-f0)/(x1-x0) * (x-x0)
            // Let dx = the relative fraction from xL to xU
            // Then cdf(dx) = int(pdf(x), x=x0..x0+dx(x1-x0))
            //              = 1/2 (x1-x0) ( (f1-f0) dx^2 + 2f0 dx )
            //              = fraction * 1/2 (x1-x0) (f1+f0)
            // Solve for dx
            //   (f1-f0) dx^2 + 2f0 dx = fraction (f1+f0)
            double c = fraction * _c;
            // Note: Use this rather than (sqrt(ac+b^2) - b)/a, since ac << b^2 typically,
            //       so this form is less susceptible to rounding errors.
            // Also: This choice of sqrt assumes all coefficients are positive.  So when flux
            //       is negative, we need to make sure coefficients are flipped.  This is done
            //       in split() when we initially set these values.
            double dx = c / (std::sqrt(_a*c + _b*_b) + _b);
            return _xLower + _xRange * dx;
        }
    }


    // Select a photon from within the interval.
    // unitRandom is a random value to use.
    void Interval::drawWithin(double unitRandom, double& x, double& flux) const
    {
        xdbg<<"drawWithin interval\n";
        xdbg<<"_flux = "<<_flux<<std::endl;
        x = interpolateFlux(unitRandom);
        xdbg<<"x = "<<x<<std::endl;
        flux = _flux < 0 ? -1. : 1.;
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
            _flux *= 2. * M_PI;
        } else {
            // Integrate the input function
            _flux = integ::int1d(*_fluxDensityPtr,
                                 _xLower, _xUpper,
                                 _gsparams.integration_relerr,
                                 _gsparams.integration_abserr);
        }
        _fluxIsReady = true;
    }

    // Divide an interval into ones that are sufficiently small that a linear approximation
    // to the flux density across the interval is accurate to the required tolerance.
    // The tolerance is given as an input here, which should be shoot_accuracy * totalFlux.
    std::list<shared_ptr<Interval> > Interval::split(double toler)
    {
        // Get the flux in this interval
        checkFlux();

        // Check if the linear model is good enough.
        // Specifically, we know that there should not be any extrema in the interval at this
        // point, so we can just check if the flux that would result from the linear model
        // is within tolerance of the actual flux calculated by the integral.
        double fLower = (*_fluxDensityPtr)(_xLower);
        double fUpper = (*_fluxDensityPtr)(_xUpper);

        std::list<shared_ptr<Interval> > result;
        xdbg<<"  flux = "<<_flux<<std::endl;
        xdbg<<"  min, max density = "<<fLower<<"  "<<fUpper<<std::endl;
        xdbg<<"  x0, x1 = "<<_xLower<<"  "<<_xUpper<<std::endl;
        double linear_flux;
        if (_isRadial) {
            // linear flux would be 2pi int_r0..r1 f0 r + (f1-f0)/(r1-r0) (r-r0) r
            // = pi/3 (r1-r0) (f0*(2r0 + r1) + f1*(2r1 + r0))
            // All but pi (r1-r0) is what we will want to call _d.  So do it now.
            _d = (fLower*(2.*_xLower+_xUpper) + fUpper*(2.*_xUpper+_xLower)) / 3.;
            linear_flux = M_PI * _xRange * _d;
        } else {
            // linear flux would be int_x0..x1 f0 + (f1-f0)/(x1-x0) (x-x0)
            // = 1/2 (x1-x0) (f1 + f0)
            _c = fUpper + fLower;
            linear_flux = 0.5 * _xRange * _c;
        }
        xdbg<<"  If linear, flux = "<<linear_flux<<"  error = "<<linear_flux - _flux<<std::endl;
        if (std::abs(linear_flux - _flux) < toler) {
            // Store a few other combinations that will be used when drawing within interval.
            if (_isRadial) {
                double fRange = fUpper - fLower;
                _a = fRange * _xRange / 3.;
                _b = fLower * _xRange + fRange * _xLower;
                _c = 2. * fLower * _xLower;
            } else {
                _a = fUpper - fLower;
                _b = fLower;
                _d = 0.;  // Not used, but set it to 0 anyway.
            }
            if (_flux < 0) {
                // The solution we choose assumes flux is positive.  If not, all coefficients
                // should be flipped.
                _a = -_a;
                _b = -_b;
                _c = -_c;
                _d = -_d;
            }
            result.push_back(shared_ptr<Interval>(new Interval(*this)));
        } else {
            // Split the interval.  Call (recursively) split() for left & right
            double midpoint = 0.5*(_xLower + _xUpper);
            Interval left(*_fluxDensityPtr, _xLower, midpoint, _isRadial, _gsparams);
            Interval right(*_fluxDensityPtr, midpoint, _xUpper, _isRadial, _gsparams);
            std::list<shared_ptr<Interval> > add = left.split(toler);
            result.splice(result.end(), add);
            add = right.split(toler);
            result.splice(result.end(), add);
        }
        return result;
    }

    OneDimensionalDeviate::OneDimensionalDeviate(const FluxDensity& fluxDensity,
                                                 std::vector<double>& range,
                                                 bool isRadial, double nominal_flux,
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
        double empirical_flux = _positiveFlux - _negativeFlux;
        if (empirical_flux > 0) {
            // There is an edge case to deal with -- SecondKick may give us a function with
            // no flux, which is fine, but don't divide by zero in this case.
            dbg<<"empirical_flux = "<<empirical_flux<<", should be "<<nominal_flux<<std::endl;
            // Rescale the fluxes according to the expected values of the flux, input as
            // nominal_flux.  This should be the analytic integral of the input function.
            // The empirical integral is usually a bit smaller, since the upper limit is finite,
            // so this corrections means the resulting total photon flux is not too small by
            // ~gsparams.shoot_accuracy.  cf. Issue #1036.
            double factor = nominal_flux / empirical_flux;
            _positiveFlux *= factor;
            _negativeFlux *= factor;
            dbg<<"posFlux => "<<_positiveFlux<<std::endl;
            dbg<<"negFlux => "<<_negativeFlux<<std::endl;
        }
        double totalAbsoluteFlux = _positiveFlux + _negativeFlux;
        dbg<<"totAbsFlux = "<<totalAbsoluteFlux<<std::endl;

        if (totalAbsoluteFlux == 0.) {
            // The below calculation will crash, so do something trivial that works.
            shared_ptr<Interval> segment(new Interval(fluxDensity, range[0], range[1], _isRadial,
                                                      _gsparams));
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
                             extremum)) {
                xdbg<<"range "<<iRange<<" = "<<range[iRange]<<" ... "<<range[iRange+1]<<
                    "  has an extremum at "<<extremum<<std::endl;
                // Do 2 ranges
                {
                    Interval splitit(_fluxDensity, range[iRange], extremum, _isRadial, _gsparams);
                    std::list<shared_ptr<Interval> > leftList = splitit.split(
                        _gsparams.shoot_accuracy * totalAbsoluteFlux);
                    xdbg<<"Add "<<leftList.size()<<" intervals on left of extremem\n";
                    _pt.insert(_pt.end(), leftList.begin(), leftList.end());
                }
                {
                    Interval splitit(_fluxDensity, extremum, range[iRange+1], _isRadial, _gsparams);
                    std::list<shared_ptr<Interval> > rightList = splitit.split(
                        _gsparams.shoot_accuracy * totalAbsoluteFlux);
                    xdbg<<"Add "<<rightList.size()<<" intervals on right of extremem\n";
                    _pt.insert(_pt.end(), rightList.begin(), rightList.end());
                }
            } else {
                // Just single Interval in this range, no extremum:
                xdbg<<"single interval\n";
                Interval splitit(
                    _fluxDensity, range[iRange], range[iRange+1], _isRadial, _gsparams);
                std::list<shared_ptr<Interval> > leftList = splitit.split(
                    _gsparams.shoot_accuracy * totalAbsoluteFlux);
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
        xassert(N>=0);
        if (N==0) return;
        double totalAbsoluteFlux = getPositiveFlux() + getNegativeFlux();
        dbg<<"totalAbsFlux = "<<totalAbsoluteFlux<<std::endl;
        double fluxPerPhoton = totalAbsoluteFlux / N;
        if (xandy) fluxPerPhoton *= totalAbsoluteFlux;
        dbg<<"fluxPerPhoton = "<<fluxPerPhoton<<std::endl;

        // For each photon, first decide which Interval it's in, then drawWithin the interval.
        if (_isRadial) {
            for (int i=0; i<N; i++) {
#ifdef USE_COS_SIN
                double unitRandom = ud();
                const shared_ptr<Interval> chosen = _pt.find(unitRandom);
                // Now draw a radius from within selected interval
                double radius, flux;
                chosen->drawWithin(unitRandom, radius, flux);
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
                const shared_ptr<Interval> chosen = _pt.find(unitRandom);
                // Now draw a radius from within selected interval
                double radius, flux;
                chosen->drawWithin(unitRandom, radius, flux);
                // Rescale x & y:
                double rScale = radius / std::sqrt(rsq);
                photons.setPhoton(i, xu*rScale, yu*rScale, flux*fluxPerPhoton);
#endif
            }
        } else {
            for (int i=0; i<N; i++) {
                // Simple 1d interpolation
                double unitRandom = ud();
                shared_ptr<Interval> chosen = _pt.find(unitRandom);
                // Now draw an x from within selected interval
                double x, flux;
                chosen->drawWithin(unitRandom, x, flux);
                if (xandy) {
                    double y, flux2;
                    unitRandom = ud();
                    chosen = _pt.find(unitRandom);
                    chosen->drawWithin(unitRandom, y, flux2);
                    photons.setPhoton(i, x, y, flux*flux2*fluxPerPhoton);
                } else {
                    photons.setPhoton(i, x, 0., flux*fluxPerPhoton);
                }
            }
        }
        dbg<<"OneDimentionalDeviate Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

} // namespace galsim
