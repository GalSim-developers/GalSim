// -*- c++ -*-
#ifndef ONE_DIMENSIONAL_DEVIATE_H
#define ONE_DIMENSIONAL_DEVIATE_H

#include <set>
#include <list>
#include <vector>
#include <functional>
#include "Random.h"
#include "PhotonArray.h"
#include "Interpolant.h"

namespace galsim {

     
    template <class F>
    class Interval {
    public:
        Interval(const F& fluxDensity): _fluxDensityPtr(&fluxDensity) {}
        // given a cumulative absolute flux within this Interval,
        // choose an x value for photon and a flux (nominally +-1)
        void drawWithin(double absFlux, double& x, double& flux,
                        UniformDeviate& ud) const;
        // Differential flux is (signed) flux in this interval
        double getDifferentialFlux() const {return _differentialFlux;}
        void setDifferentialFlux(double f) {_differentialFlux = f;}
        // Cumulative absolute flux out to upper radius of this interval
        double getCumulativeFlux() const {return _cumulativeFlux;}
        void setCumulativeFlux(double f)  {_cumulativeFlux = f;}
        // Bounds of this interval
        void getRange(double& xLower, double& xUpper) const {
            xLower = _xLower;
            xUpper = _xUpper;
        }
        void setRange(double xLower, double xUpper) {
            _xLower = xLower;
            _xUpper = xUpper;
        }

        // Return list of intervals that divide this one into acceptably small ones
        // (will be used recursively)
        std::list<Interval> split(double smallFlux);
        // Order by cumulative flux for using in std::set containers
        bool operator<(const Interval& rhs) const {
            return _cumulativeFlux<rhs._cumulativeFlux;
        }
    private:
        const F* _fluxDensityPtr;
        double _xLower;
        double _xUpper;
        double _differentialFlux;
        double _cumulativeFlux;
        // Set this variable true if returning equal fluxes with rejection method.
        // False if will just return non-unity weighted flux for selected point
        bool _useRejectionMethod;
        // Max and mean flux in the interval
        double _maxAbsDensity;
        double _meanAbsDensity;
    };

    template <class F>
    class OneDimensionalDeviate {
    public:
        OneDimensionalDeviate(const F& fluxDensity, std::vector<double>& range);
        double getPositiveFlux() const {return _positiveFlux;}
        double getNegativeFlux() const {return _negativeFlux;}
        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        const F& _fluxDensity;
        std::set<Interval<F> > _intervalSet;
        double _positiveFlux;
        double _negativeFlux;

    };

    class InterpolantFunction: public std::unary_function<double,double> {
    public:
        InterpolantFunction(const Interpolant& f): _f(f) {}
        double operator()(double x) const {return _f.xval(x);}
    private:
        const Interpolant& _f;
    };

} // namespace galsim

#endif
