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

    // A base class for functions giving differential flux vs x or r
    class FluxDensity: public std::unary_function<double,double> {
    public:
        virtual ~FluxDensity() {}
        virtual double operator()(double x) const=0;
    };

    class Interval {
    public:
        Interval(const FluxDensity& fluxDensity,
                 double xLower,
                 double xUpper,
                 bool isRadial=false): _fluxDensityPtr(&fluxDensity),
                                       _xLower(xLower),
                                       _xUpper(xUpper),
                                       _isRadial(isRadial),
                                       _fluxIsReady(false) {}
        // given a cumulative absolute flux within this Interval,
        // choose an x value for photon and a flux (nominally +-1)
        void drawWithin(double absFlux, double& x, double& flux,
                        UniformDeviate& ud) const;
        // Differential flux is (signed) flux in this interval
        double getDifferentialFlux() const {checkFlux(); return _differentialFlux;}
        // Cumulative absolute flux out to upper radius of this interval
        double getCumulativeFlux() const {return _cumulativeFlux;}
        void setCumulativeFlux(double f)  {_cumulativeFlux = f;}
        // Bounds of this interval
        void getRange(double& xLower, double& xUpper) const {
            xLower = _xLower;
            xUpper = _xUpper;
        }

        // Return list of intervals that divide this one into acceptably small ones
        // (will be used recursively)
        std::list<Interval> split(double smallFlux);
        // Order by cumulative flux for using in std::set containers
        bool operator<(const Interval& rhs) const {
            return _cumulativeFlux<rhs._cumulativeFlux;
        }
    private:
        const FluxDensity* _fluxDensityPtr;
        double _xLower;
        double _xUpper;
        bool _isRadial;
        mutable bool _fluxIsReady;
        void checkFlux() const;
        mutable double _differentialFlux;
        double _cumulativeFlux;
        double interpolateFlux(double fraction) const;
        // Set this variable true if returning equal fluxes with rejection method.
        // False if will just return non-unity weighted flux for selected point
        bool _useRejectionMethod;
        // Max and mean flux in the interval
        double _maxAbsDensity;
        double _meanAbsDensity;
    };

    class OneDimensionalDeviate {
    public:
        OneDimensionalDeviate(const FluxDensity& fluxDensity, std::vector<double>& range,
                              bool isRadial=false);
        double getPositiveFlux() const {return _positiveFlux;}
        double getNegativeFlux() const {return _negativeFlux;}
        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        const FluxDensity& _fluxDensity;
        std::set<Interval> _intervalSet;
        double _positiveFlux;
        double _negativeFlux;
        const bool _isRadial;
    };

} // namespace galsim

#endif
