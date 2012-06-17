// -*- c++ -*-

//#define DEBUGLOGGING

#include "OneDimensionalDeviate.h"
#include "integ/Int.h"

// Define this variable to find azimuth of 2d photons by drawing a uniform deviate for theta,
// instead of drawing 2 deviates for a point on the unit circle.
//#define USE_COS_SIN

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    // Wrapper class for doing integrals over annuli
    template <class F>
    class RTimesF: public std::unary_function<double,double> {
    public:
        RTimesF(const F& function): _function(function) {}
        double operator()(double r) const {return 2.*M_PI*r*_function(r);}
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
            
        // First guess is that minimum is in half of bracket with lowest gradient
        bool fatLeft = std::abs(df1) < std::abs(df2);

        // Then use golden sections to localize - could use Brent's method for speed.
        const double GOLDEN = 2./(1+sqrt(5.));
        while (std::abs(x3-x1) > xTolerance) {
            if (fatLeft) {
                // Split left-hand interval
                double xTrial = x1 + GOLDEN*(x2-x1);
                double fTrial = function(xTrial);
                double dfTrial = f2 - fTrial;
                if (df1 * dfTrial <= 0.) {
                    // Now bracketed in 1 / trial / 2
                    x3 = x2;
                    f3 = f2;
                    x2 = xTrial;
                    f2 = fTrial;
                    df1 = f2 - f1;
                    df2 = dfTrial;
                    fatLeft = true;
                } else {
                    // Extremum is in trial / 2 / 3
                    x1 = xTrial;
                    f1 = fTrial;
                    df1 = dfTrial;
                    fatLeft = false;
                }
            } else {
                // Split right-hand interval (2 / trial / 3)
                double xTrial = x2 - GOLDEN*(x3-x2);
                double fTrial = function(xTrial);
                double dfTrial = fTrial - f2;
                if (dfTrial * df2 <= 0.) {
                    // Now bracketed in 2 / trial / 3
                    x1 = x2;
                    f1 = f2;
                    x2 = xTrial;
                    f2 = fTrial;
                    df1 = dfTrial;
                    df2 = f3 - f2;
                    fatLeft = false;
                } else {
                    // Extremum is in 1 / 2 / trial
                    x3 = xTrial;
                    f3 = fTrial;
                    df2 = dfTrial;
                    fatLeft = true;
                }
            }
        }
        extremum = x2;
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
                              UniformDeviate& ud) const 
    {
        //dbg<<"drawWithin interval\n";
        //dbg<<"_flux = "<<_flux<<std::endl;
        double fractionOfInterval = std::min(unitRandom, 1.);
        //dbg<<"fractionOfInterval = "<<fractionOfInterval<<std::endl;
        fractionOfInterval = std::max(0., fractionOfInterval);
        //dbg<<"fractionOfInterval => "<<fractionOfInterval<<std::endl;
        x = interpolateFlux(fractionOfInterval);
        //dbg<<"x = "<<x<<std::endl;
        flux = 1.;
        if (_useRejectionMethod) {
            //dbg<<"use rejection\n";
            while ( ud() > std::abs((*_fluxDensityPtr)(x)) / _maxAbsDensity) {
                x = interpolateFlux(ud());
            }
            //dbg<<"x => "<<x<<std::endl;
            if (_flux < 0) flux = -1.;
        } else {
            flux = (*_fluxDensityPtr)(x) / _meanAbsDensity;
        }
        //dbg<<"flux = "<<flux<<std::endl;
    }

    void Interval::checkFlux() const 
    {
        if (_fluxIsReady) return;
        if (_isRadial) {
            // Integrate r*F
            RTimesF<FluxDensity> integrand(*_fluxDensityPtr);
            _flux = integ::int1d(integrand, 
                                 _xLower, _xUpper,
                                 odd::RELATIVE_ERROR,
                                 odd::ABSOLUTE_ERROR);
        } else {
            // Integrate the input function
            _flux = integ::int1d(*_fluxDensityPtr, 
                                 _xLower, _xUpper,
                                 odd::RELATIVE_ERROR,
                                 odd::ABSOLUTE_ERROR);
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
            _meanAbsDensity = std::abs(_flux / 
                                       ( M_PI*(_xUpper*_xUpper - _xLower*_xLower)));
        } else {
            _meanAbsDensity = std::abs(_flux / (_xUpper - _xLower));
        }
        double densityLower = (*_fluxDensityPtr)(_xLower);
        double densityUpper = (*_fluxDensityPtr)(_xUpper);
        _maxAbsDensity = std::max(std::abs(densityLower),
                                  std::abs(densityUpper));

        std::list<Interval> result;
        double densityVariation = 0.;
        if (std::abs(densityLower) > 0. && std::abs(densityUpper) > 0.)
            densityVariation = densityLower / densityUpper;
        if (densityVariation > 1.) densityVariation = 1. / densityVariation;
        if (densityVariation > odd::ALLOWED_FLUX_VARIATION) {
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
            Interval left(*_fluxDensityPtr,_xLower, midpoint, _isRadial);
            Interval right(*_fluxDensityPtr, midpoint, _xUpper, _isRadial);
            std::list<Interval> add = left.split(smallFlux);
            result.splice(result.end(), add);
            add = right.split(smallFlux);
            result.splice(result.end(), add);
        }
        return result;
    }

    OneDimensionalDeviate::OneDimensionalDeviate(const FluxDensity& fluxDensity, 
                                                 std::vector<double>& range,
                                                 bool isRadial):
        _fluxDensity(fluxDensity),
        _positiveFlux(0.),
        _negativeFlux(0.),
        _isRadial(isRadial)
    {

        // Typedef for indices of standard containers, which don't like int values
        typedef std::vector<double>::size_type Index;

        // First calculate total flux so we know when an interval is a small amt of flux
        for (Index iRange = 0; iRange < range.size()-1; iRange++) {
            // Integrate total flux (and sign) in each range
            Interval segment(fluxDensity,   
                             range[iRange],
                             range[iRange+1],
                             _isRadial);
            double rangeFlux = segment.getFlux();
            if (rangeFlux >= 0.) _positiveFlux += rangeFlux;
            else _negativeFlux += std::abs(rangeFlux);
        }
        double totalAbsoluteFlux = _positiveFlux + _negativeFlux;

        // Now break each range into Intervals
        for (Index iRange = 0; iRange < range.size()-1; iRange++) {
            // See if there is an extremum to split this range:
            double extremum;
            if (findExtremum(_fluxDensity, 
                             range[iRange],
                             range[iRange+1],
                             extremum,
                             odd::RANGE_DIVISION_FOR_EXTREMA)) {
                // Do 2 ranges
                {
                    Interval splitit(_fluxDensity, range[iRange], extremum, _isRadial);
                    std::list<Interval> leftList = splitit.split(
                        odd::SMALL_FRACTION_OF_FLUX * totalAbsoluteFlux);
                    _pt.splice(_pt.end(), leftList);
                }
                {
                    Interval splitit(_fluxDensity, extremum, range[iRange+1], _isRadial);
                    std::list<Interval> rightList = splitit.split(
                        odd::SMALL_FRACTION_OF_FLUX * totalAbsoluteFlux);
                    _pt.splice(_pt.end(), rightList);
                }
            } else {
                // Just single Interval in this range, no extremum:
                Interval splitit(_fluxDensity, range[iRange], range[iRange+1], _isRadial);
                std::list<Interval> leftList = splitit.split(
                    odd::SMALL_FRACTION_OF_FLUX * totalAbsoluteFlux);
                _pt.splice(_pt.end(), leftList);
            }
        }
        // Build the ProbabilityTree
        _pt.buildTree();
    }

    PhotonArray OneDimensionalDeviate::shoot(int N, UniformDeviate& ud) const 
    {
        dbg<<"OneDimentionalDeviate shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        dbg<<"isradial? "<<_isRadial<<std::endl;
        dbg<<"N = "<<N<<std::endl;
        assert(N>=0);
        PhotonArray result(N);
        if (N==0) return result;
        double totalAbsoluteFlux = getPositiveFlux() + getNegativeFlux();
        dbg<<"totalAbsFlux = "<<totalAbsoluteFlux<<std::endl;
        double fluxPerPhoton = totalAbsoluteFlux / N;
        dbg<<"fluxPerPhoton = "<<fluxPerPhoton<<std::endl;

        // For each photon, first decide which Interval it's in, the drawWithin the interval.
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
                result.setPhoton(i,
                                 radius*std::cos(theta), radius*std::sin(theta),
                                 flux*fluxPerPhoton);
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
                Interval* chosen = _pt.find(unitRandom);
                // Now draw a radius from within selected interval
                double radius, flux;
                chosen->drawWithin(unitRandom, radius, flux, ud);
                // Rescale x & y:
                double rScale = radius / std::sqrt(rsq);
                result.setPhoton(i,xu*rScale, yu*rScale, flux*fluxPerPhoton);
#endif            
            } else {
                // Simple 1d interpolation
                double unitRandom = ud();
                Interval* chosen = _pt.find(unitRandom);
                // Now draw an x from within selected interval
                double x, flux;
                chosen->drawWithin(unitRandom, x, flux, ud);
                result.setPhoton(i, x, 0., flux*fluxPerPhoton);
            }
        }
        dbg<<"OneDimentionalDeviate Realized flux = "<<result.getTotalFlux()<<std::endl;
        return result;
    }

} // namespace galsim
