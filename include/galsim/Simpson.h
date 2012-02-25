
// Templates for 1 and 2 dimensional integrations by Simpson's rule.
// from Numerical Recipes qsimp/trapzd.
//  could use Rhombic integration perhaps.

#ifndef SIMPSON_H
#define SIMPSON_H

#include "Std.h"

namespace sbp {

    //Exception thrown when too many steps required:
    class IntegralNonConvergent : public std::runtime_error 
    {
    public:
        IntegralNonConvergent(double _s) : std::runtime_error("Non-Convergent Integral"),
        result(_s) {}
        double result; //pass back the value so far
    };

    template <class T>
    double Simp1d(
        T& I,
        double x0,
        double x1,
        double ftol=0.001, //allowable fractional error
        double abstol=1.e-20, //or absolute error
        int maxSteps = 12, //2^steps samples max
        int minSteps = 4) //and min.
    {
#ifdef SIMPDEBUG
        std::cerr << "Simp1d with range " << x0 << " " << x1 << std::endl;
#endif
        double simp=0.,oldsimp=-1.e30, oldsum=-1.e30, sum=0.;
        int it=1;
        double del=x1-x0;
        sum=0.5*del*(I(x0)+I(x1));
        for (int j=1; j<=maxSteps; j++, it<<=1, del*=0.5) {
            double x=x0+0.5*del;
            double s=0.;
            for (int k=1; k<=it; k++, x+=del) s += I(x);
            sum = 0.5*(sum+del*s);
            simp = (4.*sum - oldsum) / 3.;
#ifdef SIMPDEBUG
            std::cerr << " step " << j << " new, old: " << simp
                << " " << oldsimp << std::endl;
#endif
            if (j>minSteps && 
                (fabs(simp-oldsimp)<=ftol*fabs(oldsimp) || fabs(simp-oldsimp)<=abstol) ) {
                return simp;
            }
            oldsum=sum;
            oldsimp = simp;
        }
#ifdef SIMPDEBUG
        std::cerr << "About to throw; range " << x0 << " " << x1 << std::endl;
        std::cerr << " simp= " << simp << " maxSteps " << maxSteps << std::endl;
#endif
        throw IntegralNonConvergent(simp);
    }

    // Binder class for making 1d function from 2d:
    template <class T>
    class Bind2d 
    {
    public:
        Bind2d(const T& _p, double _kx=0.) : p(_p), kxsave(_kx) {}
        double operator()(double ky) const { return p(kxsave, ky); }
        void xSet(double _kx) { kxsave=_kx; }
    private:
        T p;
        double kxsave;
    };

    // Class representing integral over 1d slice:
    template <class T>
    class SimpSlice 
    {
    public:
        SimpSlice(const T& I, double _k, double _ft=0.001, double _at=1e-20, int _m=12) :  
            b(I,0.), kmax(_k), ftol(_ft), abstol(_at), maxSteps(_m) {}

        // return integral over y for given x:
        double operator() (double kx) 
        {
            b.xSet(kx); 
            return Simp1d(b,-kmax,kmax,ftol,abstol,maxSteps);
        }

    private:
        Bind2d<T> b;
        const double kmax;
        const double ftol;
        const double abstol;
        const int maxSteps;
    };

    //Integral over a square in 2d, +-kmax:
    template <class T>
    double Simp2d (const T& I, double kmax, double ftol=0.001, double abstol=1e-20, int maxSteps=12) 
    {
        SimpSlice<T>  s(I,kmax,ftol,abstol,maxSteps);
        return Simp1d(s,-kmax,kmax,ftol,abstol,maxSteps);
    }

    //Integral over 2 dimensions - for Hermitian case, just
    // double one of the half-planes.  Real part must have been taken in
    // integrand function I.
    template <class T>
    double SimpHermitian(
        T& I, double kmax, double ftol=0.001, double abstol=1e-20, int maxSteps=12) 
    {
        SimpSlice<T>  s(I,kmax,ftol,abstol,maxSteps);
        return 2.*Simp1d(s,0.,kmax,ftol,abstol,maxSteps);
    }

}

#endif
