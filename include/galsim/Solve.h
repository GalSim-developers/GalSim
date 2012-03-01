
// Template to find the zero of an equation
// Currently uses bisection method, no solution caching.

#ifndef SOLVE_H
#define SOLVE_H

#include <cmath>
#include <limits>

#include "Std.h"

namespace galsim {

    class SolveError : public std::runtime_error 
    {
    public:
        SolveError(const std::string m) : std::runtime_error("Solve error: "+m) {}
    };

    const double defaultTolerance=1.e-7;
    const int defaultMaxSteps=40;

    enum Method { Bisect, Brent };

    template <class F, class T=double>
    class Solve 
    {
    private:
        const F&  func;
        T lBound;
        T uBound;
        T xTolerance;
        int maxSteps;
        mutable T flower;
        mutable T fupper;
        mutable bool boundsAreEvaluated;
        Method m;

    public:
        Solve(const F& func_, T lb_=0., T ub_=1.) :
            func(func_), lBound(lb_), uBound(ub_), xTolerance(defaultTolerance),
            maxSteps(defaultMaxSteps), boundsAreEvaluated(false), m(Bisect) {}

        void setMaxSteps(int m) { maxSteps=m; }
        void setMethod(Method m_) { m=m_; }
        T getXTolerance() const { return xTolerance; }
        void setXTolerance(T tol) { xTolerance=tol; }
        void setBounds(T lb, T ub) { lBound=lb; uBound=ub; }

        // Hunt for bracket, geometrically expanding range
        void bracket() 
        {
            const double factor=2.0;
            if (uBound == lBound) 
                throw SolveError("uBound=lBound in bracket()");
            if (!boundsAreEvaluated) {
                flower = func(lBound);
                fupper = func(uBound);
                boundsAreEvaluated=true;
            }
            for (int j=1; j<maxSteps; j++) {
                if (fupper*flower < 0.0) return;
                if (std::abs(flower) < std::abs(fupper)) {
                    lBound += factor*(lBound-uBound);
                    flower = func(lBound);
                } else {
                    uBound += factor*(uBound-lBound);
                    fupper = func(uBound);
                }
            }
            throw SolveError("Too many iterations in bracket()");
        }

        T root() const 
        {
            switch (m) {
              case Bisect:
                   return bisect();
              case Brent:
                   return zbrent();
              default :
                   throw SolveError("Unknown method in root()");
            }
        }

        T bisect() const 
        {
            T dx,f,fmid,xmid,rtb;

            if (!boundsAreEvaluated) {
                flower=func(lBound);
                fupper=func(uBound);
                boundsAreEvaluated = true;
            }
            f=flower;
            fmid=fupper;

            if (f*fmid >= 0.0) 
                FormatAndThrow<SolveError> () << "Root is not bracketed: " << lBound 
                    << " " << uBound;
            rtb = f < 0.0 ? (dx=uBound-lBound,lBound) : (dx=lBound-uBound,uBound);
            for (int j=1;j<=maxSteps;j++) {
                fmid=func(xmid=rtb+(dx *= 0.5));
                if (fmid <= 0.0) rtb=xmid;
                if ( (std::abs(dx) < xTolerance) || fmid == 0.0) return rtb;
            }
            throw SolveError("Too many bisections");
            return 0.0;
        }

        T zbrent() const 
        {
            T a=lBound, b=uBound, c=uBound;
            T d=b-a, e=b-a;
            T min1,min2;
            if (!boundsAreEvaluated) {
                flower=func(a);
                fupper=func(b);
                boundsAreEvaluated = true;
            }
            T fa = flower;
            T fb = fupper;

            T p,q,r,s,tol1,xm;
            if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0)) {
                FormatAndThrow<SolveError> () << "Root is not bracketed: " 
                    << lBound << " " << uBound;
            }
            T fc=fb;
            for (int iter=0;iter<=maxSteps;iter++) {
                if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
                    c=a;
                    fc=fa;
                    e=d=b-a;
                }
                if (std::abs(fc) < std::abs(fb)) {
                    a=b;
                    b=c;
                    c=a;
                    fa=fb;
                    fb=fc;
                    fc=fa;
                }
                tol1=2.0*std::numeric_limits<T>::epsilon()*std::abs(b)
                    +0.5*xTolerance;
                xm=0.5*(c-b);
                if (std::abs(xm) <= tol1 || fb == 0.0) return b;
                if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
                    s=fb/fa;
                    if (a == c) {
                        p=2.0*xm*s;
                        q=1.0-s;
                    } else {
                        q=fa/fc;
                        r=fb/fc;
                        p=s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0));
                        q=(q-1.0)*(r-1.0)*(s-1.0);
                    }
                    if (p > 0.0) q = -q;
                    p=std::abs(p);
                    min1=3.0*xm*q-std::abs(tol1*q);
                    min2=std::abs(e*q);
                    if (2.0*p < std::min(min1,min2) ) {
                        e=d;
                        d=p/q;
                    } else {
                        d=xm;
                        e=d;
                    }
                } else {
                    d=xm;
                    e=d;
                }
                a=b;
                fa=fb;
                if (std::abs(d) > tol1) b += d;
                else b += (xm>=0. ? std::abs(tol1) : -std::abs(tol1));
                fb=func(b);
            }
            throw SolveError("Maximum number of iterations exceeded in zbrent");
        }
    };

} // namespace solve
#endif
