
//#define DEBUGLOGGING

#include "Interpolant.h"
#include "integ/Int.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    double InterpolantFunction::operator()(double x) const  { return _interp.xval(x); }

    double InterpolantXY::getPositiveFlux() const 
    {
        return _i1d->getPositiveFlux()*_i1d->getPositiveFlux()
            + _i1d->getNegativeFlux()*_i1d->getNegativeFlux();
    }

    double InterpolantXY::getNegativeFlux() const 
    { return 2.*_i1d->getPositiveFlux()*_i1d->getNegativeFlux(); }

    boost::shared_ptr<PhotonArray> InterpolantXY::shoot(int N, UniformDeviate ud) const 
    {
        dbg<<"InterpolantXY shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        // Going to assume here that there is not a need to randomize any Interpolant
        boost::shared_ptr<PhotonArray> result = _i1d->shoot(N, ud);   // get X coordinates
        result->takeYFrom(*_i1d->shoot(N, ud));
        dbg<<"InterpolantXY Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> Delta::shoot(int N, UniformDeviate ud) const 
    {
        dbg<<"InterpolantXY shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++)  {
            result->setPhoton(i, 0., 0., fluxPerPhoton);
        }
        dbg<<"Delta Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> Nearest::shoot(int N, UniformDeviate ud) const 
    {
        dbg<<"InterpolantXY shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++)  {
            result->setPhoton(i, ud()-0.5, 0., fluxPerPhoton);
        }
        dbg<<"Nearest Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> Linear::shoot(int N, UniformDeviate ud) const 
    {
        dbg<<"InterpolantXY shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++) {
            // *** Guessing here that 2 random draws is faster than a sqrt:
            result->setPhoton(i, ud() + ud() - 1., 0., fluxPerPhoton);
        }
        dbg<<"Linear Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    double Interpolant::xvalWrapped(double x, int N) const 
    {
        // sum over all arguments x+jN that are within range.
        // Start by finding x+jN closest to zero
        double xdown = x - N*std::floor(x/N + 0.5);
        assert(std::abs(xdown) <= N);
        if (xrange() <= N) {
            // This is the usual case.
            return xval(xdown);
        } else {
            double xup = xdown+N;
            double sum = 0.;
            while (std::abs(xdown) <= xrange()) {
                sum += xval(xdown);
                xdown -= N;
            }
            while (xup <= xrange()) {
                sum += xval(xup);
                xup += N;
            }
            return sum;
        }
    }

    double Lanczos::uCalc(double u) const 
    {
        double vp=_n*(2*u+1);
        double vm=_n*(2*u-1);
        double retval = (vm-1.)*Si(M_PI*(vm-1.))
            -(vm+1.)*Si(M_PI*(vm+1.))
            -(vp-1.)*Si(M_PI*(vp-1.))
            +(vp+1.)*Si(M_PI*(vp+1.));
        return retval/(2.*M_PI);
    }

    void Lanczos::setup() 
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = _n*(1-0.1*std::sqrt(_tolerance));
        const double uStep = 0.01/_n;
        _uMax = 0.;
        double u = _tab.size()>0 ? _tab.argMax() + uStep : 0.;
        while ( u - _uMax < 1./_n || u<1.1) {
            double ft = uCalc(u);
            _tab.addEntry(u, ft);
            if (std::abs(ft) > _tolerance) _uMax = u;
            u += uStep;
        }
        _u1 = uCalc(1.);
    }

    class CubicIntegrand : public std::unary_function<double,double>
    {
    public:
        CubicIntegrand(double u, const Cubic& c): _u(u), _c(c) {}
        double operator()(double x) const { return _c.xval(x)*std::cos(2*M_PI*_u*x); }

    private:
        double _u;
        const Cubic& _c;
    };

    double Cubic::uCalc(double u) const 
    {
        CubicIntegrand ci(u, *this);
        return 2.*( integ::int1d(ci, 0., 1., 0.1*_tolerance, 0.1*_tolerance)
                    + integ::int1d(ci, 1., 2., 0.1*_tolerance, 0.1*_tolerance));
    }

    void Cubic::setup() 
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = 2.-0.1*_tolerance;
        const double uStep = 0.001;
        _uMax = 0.;
        double u = _tab.size()>0 ? _tab.argMax() + uStep : 0.;
        while ( u - _uMax < 1. || u<1.1) {
            double ft = uCalc(u);
            _tab.addEntry(u, ft);
            if (std::abs(ft) > _tolerance) _uMax = u;
            u += uStep;
        }
    }

    class QuinticIntegrand : public std::unary_function<double,double>
    {
    public:
        QuinticIntegrand(double u, const Quintic& c): _u(u), _c(c) {}
        double operator()(double x) const { return _c.xval(x)*std::cos(2*M_PI*_u*x); }
    private:
        double _u;
        const Quintic& _c;
    };

    double Quintic::uCalc(double u) const 
    {
        QuinticIntegrand qi(u, *this);
        return 2.*( integ::int1d(qi, 0., 1., 0.1*_tolerance, 0.1*_tolerance)
                    + integ::int1d(qi, 1., 2., 0.1*_tolerance, 0.1*_tolerance)
                    + integ::int1d(qi, 2., 3., 0.1*_tolerance, 0.1*_tolerance));
    }

    void Quintic::setup() 
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = 3.-0.1*_tolerance;
        const double uStep = 0.001;
        _uMax = 0.;
        double u = _tab.size()>0 ? _tab.argMax() + uStep : 0.;
        while ( u - _uMax < 1. || u<1.1) {
            double ft = uCalc(u);
            _tab.addEntry(u, ft);
            if (std::abs(ft) > _tolerance) _uMax = u;
            u += uStep;
        }
    }

}

