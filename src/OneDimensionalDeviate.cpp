// -*- c++ -*-

#include "OneDimensionalDeviate.h"
#include "integ/Int.h"

namespace galsim {

    // fractional error allowed on any flux integral:
    const double RELATIVE_ERROR = 1e-6;
    // absolute error allowed (assumes the total flux is O(1)
    const double ABSOLUTE_ERROR = 1e-8;
    // Range will be split into this many parts to bracket extrema
    const int RANGE_FRACTION_FOR_EXTREMA = 32;
    // Intervals with less than this fraction of probability are
    // ok to use dominant-sampling method.
    const double SMALL_FRACTION_OF_FLUX = 1e-4;
    // Max range of allowed (abs value of) photon fluxes
    const double ALLOWED_FLUX_VARIATION = 0.81;

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
                double extremum=999.;
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
