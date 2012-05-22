#include "Interpolant.h"
#include "integ/Int.h"

namespace galsim {

    double InterpolantFunction::operator()(double x) const  {return _interp.xval(x);}

    double InterpolantXY::getPositiveFlux() const {
        return i1d.getPositiveFlux()*i1d.getPositiveFlux()
            + i1d.getNegativeFlux()*i1d.getNegativeFlux();
    }
    double InterpolantXY::getNegativeFlux() const {
        return 2.*i1d.getPositiveFlux()*i1d.getNegativeFlux();
    }
    PhotonArray InterpolantXY::shoot(int N, UniformDeviate& ud) const {
        // Going to assume here that there is not a need to randomize any Interpolant
        PhotonArray result = i1d.shoot(N, ud);   // get X coordinates
        result.takeYFrom(i1d.shoot(N, ud));
        return result;
    }

    PhotonArray Delta::shoot(int N, UniformDeviate& ud) const {
        PhotonArray result(N);
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++)  {
            result.setPhoton(i, 0., 0., fluxPerPhoton);
        }
        return result;
    }

    PhotonArray Nearest::shoot(int N, UniformDeviate& ud) const {
        PhotonArray result(N);
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++)  {
            result.setPhoton(i, ud()-0.5, 0., fluxPerPhoton);
        }
        return result;
    }

    PhotonArray Linear::shoot(int N, UniformDeviate& ud) const {
        PhotonArray result(N);
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++) {
            // *** Guessing here that 2 random draws is faster than a sqrt:
            result.setPhoton(i, ud() + ud() - 1., 0., fluxPerPhoton);
        }
        return result;
    }

    double Interpolant::xvalWrapped(double x, int N) const 
    {
        // sum over all arguments x+jN that are within range.
        // Start by finding x+jN closest to zero
        double xdown = x - N*std::floor(x/N + 0.5);
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

    double Lanczos::uCalc(double u) const 
    {
        double vp=n*(2*u+1);
        double vm=n*(2*u-1);
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
        range = n*(1-0.1*std::sqrt(tolerance));
        const double uStep = 0.01/n;
        uMax = 0.;
        double u = tab.size()>0 ? tab.argMax() + uStep : 0.;
        while ( u - uMax < 1./n || u<1.1) {
            double ft = uCalc(u);
            tab.addEntry(u, ft);
            if (std::abs(ft) > tolerance) uMax = u;
            u += uStep;
        }
        u1 = uCalc(1.);
    }

    class CubicIntegrand : public std::unary_function<double,double>
    {
    public:
        CubicIntegrand(double u_, const Cubic& c_): u(u_), c(c_) {}
        double operator()(double x) const { return c.xval(x)*std::cos(2*M_PI*u*x); }

    private:
        double u;
        const Cubic& c;
    };

    double Cubic::uCalc(double u) const 
    {
        CubicIntegrand ci(u, *this);
        return 2.*( integ::int1d(ci, 0., 1., 0.1*tolerance, 0.1*tolerance)
                    + integ::int1d(ci, 1., 2., 0.1*tolerance, 0.1*tolerance));
    }

    void Cubic::setup() 
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        range = 2.-0.1*tolerance;
        const double uStep = 0.001;
        uMax = 0.;
        double u = tab.size()>0 ? tab.argMax() + uStep : 0.;
        while ( u - uMax < 1. || u<1.1) {
            double ft = uCalc(u);
            tab.addEntry(u, ft);
            if (std::abs(ft) > tolerance) uMax = u;
            u += uStep;
        }
    }

    class QuinticIntegrand : public std::unary_function<double,double>
    {
    public:
        QuinticIntegrand(double u_, const Quintic& c_): u(u_), c(c_) {}
        double operator()(double x) const { return c.xval(x)*std::cos(2*M_PI*u*x); }
    private:
        double u;
        const Quintic& c;
    };

    double Quintic::uCalc(double u) const 
    {
        QuinticIntegrand qi(u, *this);
        return 2.*( integ::int1d(qi, 0., 1., 0.1*tolerance, 0.1*tolerance)
                    + integ::int1d(qi, 1., 2., 0.1*tolerance, 0.1*tolerance)
                    + integ::int1d(qi, 2., 3., 0.1*tolerance, 0.1*tolerance));
    }

    void Quintic::setup() 
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        range = 3.-0.1*tolerance;
        const double uStep = 0.001;
        uMax = 0.;
        double u = tab.size()>0 ? tab.argMax() + uStep : 0.;
        while ( u - uMax < 1. || u<1.1) {
            double ft = uCalc(u);
            tab.addEntry(u, ft);
            if (std::abs(ft) > tolerance) uMax = u;
            u += uStep;
        }
    }

}

