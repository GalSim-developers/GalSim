// -*- c++ -*-

#include "OneDimensionalDeviate.h"
#include "integ/Int.h"

namespace galsim {

    // fractional error allowed on any flux integral:
    const double RELATIVE_ERROR = 1e-6;
    // absolute error allowed (assumes the total flux is O(1)
    const double ABSOLUTE_ERROR = 1e-8;
    // Range will be split into this many parts to bracket extrema
    const int RANGE_DIVISION_FOR_EXTREMA = 32;
    // Intervals with less than this fraction of probability are
    // ok to use dominant-sampling method.
    const double SMALL_FRACTION_OF_FLUX = 1e-4;
    // Max range of allowed (abs value of) photon fluxes
    const double ALLOWED_FLUX_VARIATION = 0.81;

    // Function to isolate an extremum of function in an interval.
    template <class F>
    double findExtremum( const F& function,
                         double xmin,
                         double xmax,
                         int divisionSteps,
                         double xFractionalTolerance = 1e-4) {
        if (xmax < xmin) std::swap(xmax,xmin);
        const double xTolerance = xFractionalTolerance*(xmax-xmin);
        // First bracket extremum by division into fixed number of steps
        double xStep = (xmax - xmin) / divisionSteps;
        double x1 = xmin;
        double x2 = xmin + xStep;
        double f1 = function(x1);
        double f2 = function(x2);
        double df1 = f2 - f1;
        double x3 = 2*xStep;
        double f3 = function(x3);
        double df2 = f3 - f2;
        
        while (df1 * df2 >= 0.) {
            if (f3 >= xmax) 
                throw std::runtime_error("Did not bracket extremum in findExtremum");
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
        return x2;
    }


    template <class F>
    void Interval<F>::drawWithin(double absFlux, double& x, double& flux,
                                 UniformDeviate& ud) const {
        double fractionOfInterval = (_cumulativeFlux -  absFlux) 
            / std::abs(_differentialFlux);
        x = _xLower + (_xUpper - _xLower)*fractionOfInterval;
        flux = 1.;
        if (_useRejectionMethod) {
            while ( ud() > std::abs((*_fluxDensityPtr)(x)) / _maxAbsDensity) {
                x = _xLower + (_xUpper - _xLower)*ud();
            }
            if (_differentialFlux < 0) flux = -1.;
        } else {
            flux = (*_fluxDensityPtr)(x) / _meanAbsDensity;
        }
    }

    template <class F>
    std::list<Interval<F> > Interval<F>::split(double smallFlux) {
        // Get the flux in this interval 
        _differentialFlux = integ::int1d(*_fluxDensityPtr, 
                                         _xLower, _xUpper,
                                         RELATIVE_ERROR,
					 ABSOLUTE_ERROR);
        _meanAbsDensity = _differentialFlux / (_xUpper - _xLower);
        double densityLower = (*_fluxDensityPtr)(_xLower);
        double densityUpper = (*_fluxDensityPtr)(_xUpper);
        _maxAbsDensity = std::max(std::abs(densityLower),
                                  std::abs(densityUpper));

        std::list<Interval<F> > result;
        double densityVariation = densityLower / densityUpper;
        if (densityVariation > 1.) densityVariation = 1. / densityVariation;
        if (densityVariation > ALLOWED_FLUX_VARIATION) {
            // Don't split if flux range is small
            _useRejectionMethod = false;
            result.push_back(*this);
        } else if (_differentialFlux < smallFlux) {
            // Don't split further, as it will be rare to be in this interval
            // and rejection is ok.
            _useRejectionMethod = true;
            result.push_back(*this);
        } else {
            // Split the interval.  Call (recursively) split() for left & right
            double midpoint = 0.5*(_xLower + _xUpper);
            Interval<F> left(*_fluxDensityPtr);
            left.setRange(_xLower, midpoint);
            Interval<F> right(*_fluxDensityPtr);
            right.setRange(midpoint, _xUpper);
            std::list<Interval<F> > add = left.split(smallFlux);
            result.splice(result.end(), add);
            add = right.split(smallFlux);
            result.splice(result.end(), add);
        }
        return result;
    }

    template <class F>
    OneDimensionalDeviate<F>::OneDimensionalDeviate(const F& fluxDensity, 
                                                 std::vector<double>& range):
        _fluxDensity(fluxDensity),
        _positiveFlux(0.),
        _negativeFlux(0.)
    {

        typedef std::vector<double>::size_type Index;
        for (Index iRange = 0; iRange < range.size()-1; iRange++) {
            // Integrate total flux (and sign) in each range
            double rangeFlux = integ::int1d(fluxDensity, 
                                            range[iRange],
                                            range[iRange+1],
                                            RELATIVE_ERROR,
                                            ABSOLUTE_ERROR);
            if (rangeFlux >= 0.) _positiveFlux += rangeFlux;
            else _negativeFlux += std::abs(rangeFlux);
        }
        double totalAbsoluteFlux = _positiveFlux + _negativeFlux;

        // Collect Intervals as an un-ordered list initially
        std::list<Interval<F> > intervalList;

        // Now break each range into Intervals
        for (Index iRange = 1; iRange < range.size()-1; iRange++) {
            Interval<F> rangeA(_fluxDensity);
            Interval<F> rangeB(_fluxDensity);
            if (iRange==0) {
                rangeA.setRange(range[iRange], range[iRange+1]);
            } else {
                // Split interval at maximum or minimum
                // First a coarse search to bracket min/max
                // Then localize to desired accuracy
                // !!!!!
                double extremum = findExtremum(_fluxDensity, 
                                               range[iRange],
                                               range[iRange+1],
                                               RANGE_DIVISION_FOR_EXTREMA);
                rangeA.setRange(range[iRange], extremum);
                rangeB.setRange(extremum, range[iRange+1]);
            }
            // Divide as needed.  Splitting process integrates flux per interval.
            std::list<Interval<F> > leftList = rangeA.split(SMALL_FRACTION_OF_FLUX
                                                            *totalAbsoluteFlux);
            intervalList.splice(intervalList.end(), leftList);
            if (iRange != 0) {
                std::list<Interval<F> > rightList = rangeB.split(SMALL_FRACTION_OF_FLUX
                                                                 *totalAbsoluteFlux);
                intervalList.splice(intervalList.end(), rightList);
            }
        }
        // Accumulate fluxes and put into set structure
        double cumulativeFlux = 0.;
        for (typename std::list<Interval<F> >::iterator i=intervalList.begin();
             i != intervalList.end();
             ++i) {
            cumulativeFlux += std::abs(i->getDifferentialFlux());
            i->setCumulativeFlux(cumulativeFlux);
            _intervalSet.insert(*i);
        }
    }

    template <class F>
    PhotonArray OneDimensionalDeviate<F>::shoot(int N, UniformDeviate& ud) const {
        assert(N>=0);
        PhotonArray result(N);
        if (N==0) return result;
        double totalAbsoluteFlux = getPositiveFlux() + getNegativeFlux();
        double fluxPerPhoton = totalAbsoluteFlux / N;
        for (int i=0; i<N; i++) {
            // Create dummy Interval with randomly drawn cumulative flux
            // to use for sorting
            Interval<F> drawn(_fluxDensity);
            drawn.setCumulativeFlux(ud()*totalAbsoluteFlux);
            typename std::set<Interval<F> >::const_iterator upper =
                _intervalSet.lower_bound(drawn);
            // use last pixel if we're past the end
            if (upper == _intervalSet.end()) --upper; 
            // Now draw a position from within selected interval
            double x, flux;
            upper->drawWithin(drawn.getCumulativeFlux(), x, flux, ud);
            result.setPhoton(i, x, 0., flux*fluxPerPhoton);
        }
        return result;
    }

    // Instantiate:
    template class  OneDimensionalDeviate<InterpolantFunction>;

} // namespace galsim
