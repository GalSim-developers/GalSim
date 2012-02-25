#include "Interpolant.h"
#include "Simpson.h"

namespace sbp {

    double Interpolant::xvalWrapped(double x, int N) const 
    {
        // sum over all arguments x+jN that are within range.
        // Start by finding x+jN closest to zero
        double xdown = x - N*floor(x/N + 0.5);
        double xup = xdown+N;
        double sum = 0.;
        while (abs(xdown) <= xrange()) {
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
        range = n*(1-0.1*sqrt(tolerance));
        const double uStep = 0.01/n;
        uMax = 0.;
        double u = tab.size()>0 ? tab.argMax() + uStep : 0.;
        while ( u - uMax < 1./n || u<1.1) {
            double ft = uCalc(u);
            tab.addEntry(u, ft);
            if (abs(ft) > tolerance) uMax = u;
            u += uStep;
        }
        u1 = uCalc(1.);
    }

    class CubicIntegrand 
    {
    public:
        CubicIntegrand(double u_, const Cubic& c_): u(u_), c(c_) {}
        double operator()(double x) const { return c.xval(x)*cos(2*M_PI*u*x); }

    private:
        double u;
        const Cubic& c;
    };

    double Cubic::uCalc(double u) const 
    {
        CubicIntegrand ci(u, *this);
        return 2.*( Simp1d(ci, 0., 1., 0.1*tolerance, 0.1*tolerance)
                    + Simp1d(ci, 1., 2., 0.1*tolerance, 0.1*tolerance));
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
            if (abs(ft) > tolerance) uMax = u;
            u += uStep;
        }
    }

    class QuinticIntegrand 
    {
    public:
        QuinticIntegrand(double u_, const Quintic& c_): u(u_), c(c_) {}
        double operator()(double x) const { return c.xval(x)*cos(2*M_PI*u*x); }
    private:
        double u;
        const Quintic& c;
    };

    double Quintic::uCalc(double u) const 
    {
        QuinticIntegrand ci(u, *this);
        return 2.*( Simp1d(ci, 0., 1., 0.1*tolerance, 0.1*tolerance)
                    + Simp1d(ci, 1., 2., 0.1*tolerance, 0.1*tolerance)
                    + Simp1d(ci, 2., 3., 0.1*tolerance, 0.1*tolerance));
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
            if (abs(ft) > tolerance) uMax = u;
            u += uStep;
        }
    }

}

