
//#define DEBUGLOGGING

#include "SBExponential.h"
#include "SBExponentialImpl.h"

// Define this variable to find azimuth (and sometimes radius within a unit disc) of 2d photons by 
// drawing a uniform deviate for theta, instead of drawing 2 deviates for a point on the unit 
// circle and rejecting corner photons.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using sin/cos was faster for icpc, but not g++ or clang++.
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

// Define this use the Newton-Raphson method for solving the radial value in SBExponential::shoot
// rather than using OneDimensionalDeviate.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using OneDimensionalDeviate was universally quite a bit faster.
// However, we leave this option here in case someone has an idea for massively speeding up
// the solution that might be faster than the table lookup.
//#define USE_NEWTON_RAPHSON

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    SBExponential::SBExponential(double r0, double flux) :
        SBProfile(new SBExponentialImpl(r0, flux)) {}

    SBExponential::SBExponential(const SBExponential& rhs) : SBProfile(rhs) {}

    SBExponential::~SBExponential() {}

    double SBExponential::getScaleRadius() const 
    { 
        assert(dynamic_cast<const SBExponentialImpl*>(_pimpl.get()));
        return dynamic_cast<const SBExponentialImpl&>(*_pimpl).getScaleRadius(); 
    }

    SBExponential::SBExponentialImpl::SBExponentialImpl(double r0, double flux) :
        _flux(flux), _r0(r0), _r0_sq(r0*r0)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // (1+k^2 r0^2)^-1.5 = kvalue_accuracy
        _ksq_max = (std::pow(sbp::kvalue_accuracy,-1./1.5)-1.) / _r0_sq;

        // For small k, we can use up to quartic in the taylor expansion to avoid the sqrt.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 35/16 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(sbp::kvalue_accuracy * 16./35., 1./3.) / _r0_sq;

        _flux_over_2pi = _flux / (2. * M_PI);
        _norm = _flux_over_2pi / _r0_sq;

        dbg<<"Exponential:\n";
        dbg<<"_flux = "<<_flux<<std::endl;
        dbg<<"_r0 = "<<_r0<<std::endl;
        dbg<<"_r0_sq = "<<_r0_sq<<std::endl;
        dbg<<"_ksq_max = "<<_ksq_max<<std::endl;
        dbg<<"_ksq_min = "<<_ksq_min<<std::endl;
        dbg<<"_norm = "<<_norm<<std::endl;
        dbg<<"maxK() = "<<maxK()<<std::endl;
        dbg<<"stepK() = "<<stepK()<<std::endl;
    }

    double SBExponential::SBExponentialImpl::maxK() const 
    { return SBExponential::_info.maxK() / _r0; }
    double SBExponential::SBExponentialImpl::stepK() const 
    { return SBExponential::_info.stepK() / _r0; }

    double SBExponential::SBExponentialImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x*p.x + p.y*p.y);
        return _norm * std::exp(-r/_r0);
    }

    std::complex<double> SBExponential::SBExponentialImpl::kValue(const Position<double>& k) const 
    {
        double ksq = k.x*k.x+k.y*k.y;

        if (ksq > _ksq_max) {
            return 0.;
        } else if (ksq < _ksq_min) {
            ksq *= _r0_sq;
            return _flux*(1. - 1.5*ksq*(1. - 1.25*ksq));
        } else {
            double temp = 1. + ksq*_r0_sq;
            return _flux/(temp*sqrt(temp));
            // NB: flux*std::pow(temp,-1.5) is slower.
        }
    }

    // Constructor to initialize Exponential functions for 1D deviate photon shooting
    SBExponential::ExponentialInfo::ExponentialInfo()
    {
#ifndef USE_NEWTON_RAPHSON
        // Next, set up the classes for photon shooting
        _radial.reset(new ExponentialRadialFunction());
        std::vector<double> range(2,0.);
        range[1] = -std::log(sbp::shoot_flux_accuracy);
        _sampler.reset(new OneDimensionalDeviate( *_radial, range, true));
#endif

        // Calculate maxk:
        _maxk = std::pow(sbp::maxk_threshold, -1./3.);

        // Calculate stepk:
        // int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        // Fraction excluded is thus (1+R) exp(-R)
        // A fast solution to (1+R)exp(-R) = x:
        // log(1+R) - R = log(x)
        // R = log(1+R) - log(x)
        double logx = std::log(sbp::alias_threshold);
        double R = -logx;
        for (int i=0; i<3; i++) R = std::log(1.+R) - logx;
        // Make sure it is at least 6 scale radii.
        R = std::max(6., R);
        _stepk = M_PI / R;
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBExponential::ExponentialInfo::maxK() const 
    { return _maxk; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBExponential::ExponentialInfo::stepK() const
    { return _stepk; }

    boost::shared_ptr<PhotonArray> SBExponential::ExponentialInfo::shoot(
        int N, UniformDeviate ud) const
    {
        dbg<<"ExponentialInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";
        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        dbg<<"ExponentialInfo Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    SBExponential::ExponentialInfo SBExponential::_info;

    boost::shared_ptr<PhotonArray> SBExponential::SBExponentialImpl::shoot(
        int N, UniformDeviate u) const
    {
        dbg<<"Exponential shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
#ifdef USE_NEWTON_RAPHSON
        // The cumulative distribution of flux is 1-(1+r)exp(-r).
        // Here is a way to solve for r by an initial guess followed
        // by Newton-Raphson iterations.  Probably not
        // the most efficient thing since there are logs in the iteration.

        // Accuracy to which to solve for (log of) cumulative flux distribution:
        const double Y_TOLERANCE=sbp::shoot_flux_accuracy;

        double fluxPerPhoton = _flux / N;
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));

        for (int i=0; i<N; i++) {
            double y = u();
            if (y==0.) {
                // In case of infinite radius - just set to origin:
                result->setPhoton(i,0.,0.,fluxPerPhoton);
                continue;
            }
            // Initial guess
            y = -std::log(y);
            double r = y>2. ? y : sqrt(2.*y);
            double dy = y - r + std::log(1.+r);
            while ( std::abs(dy) > Y_TOLERANCE) {
                r = r + (1.+r)*dy/r;
                dy = y - r + std::log(1.+r);
            }
            // Draw another (or multiple) randoms for azimuthal angle 
#ifdef USE_COS_SIN
            double theta = 2. * M_PI * u();
#ifdef _GLIBCXX_HAVE_SINCOS
            // Most optimizing compilers will do this automatically, but just in case...
            double sint,cost;
            sincos(theta,&sint,&cost);
#else
            double cost = std::cos(theta);
            double sint = std::sin(theta);
#endif
            double rFactor = r * _r0;
            result->setPhoton(i, rFactor * cost, rFactor * sint, fluxPerPhoton);
#else
            double xu, yu, rsq;
            do {
                xu = 2. * u() - 1.;
                yu = 2. * u() - 1.;
                rsq = xu*xu+yu*yu;
             } while (rsq >= 1. || rsq == 0.);
            double rFactor = r * _r0 / std::sqrt(rsq);
            result->setPhoton(i, rFactor * xu, rFactor * yu, fluxPerPhoton);
#endif
        }
#else
        // Get photons from the ExponentialInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = SBExponential::_info.shoot(N,u);
        result->scaleFlux(_flux_over_2pi);
        result->scaleXY(_r0);
#endif
        dbg<<"Exponential Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
