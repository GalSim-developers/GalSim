
// Template to find the zero of an equation
// Currently uses bisection method, no solution caching.

#ifndef SOLVE_H
#define SOLVE_H

#include <cmath>
#include <limits>

#include "Std.h"

namespace galsim {

    /// @brief Exception class thrown by Solve
    class SolveError : public std::runtime_error 
    {
    public:
        SolveError(const std::string m) : std::runtime_error("Solve error: "+m) {}
    };

    const double defaultTolerance=1.e-7;
    const int defaultMaxSteps=40;

    enum Method { Bisect, Brent };

    /**
     * @brief A class that solves a provided function for a zero.
     *
     * The first template argument, F, is the type of the function.  
     * Typically you would simple function object that defines the operator() method
     * and use that for F.
     *
     * The second template argument (optional: default = double) is the type of
     * the arguments to the function.
     *
     * The solving process is in two parts:
     *
     * 1. First the solution needs to be bracketed. 
     *
     *    This can be done in several ways:
     *
     *    a) If you know the appropriate range, you can set it in the constructor:
     *
     *    @code
     *    Solve<MyFunc> solver(func, x_min, x_max);
     *    @endcode
     *
     *    b) If you don't know the range, you can set an initial guess range and let
     *    Solve expand the range until it finds a range that brackets the solution:
     *
     *    @code
     *    Solve<MyFunc> solver(func, x_min, x_max);
     *    solver.bracket()
     *    @endcode
     *
     *    c) If you know one end of the range, but not the other you can expand only one
     *    side of the range from the initial guess:
     *
     *    @code
     *    Solve<MyFunc> solver(func, 0, r_max);
     *    solver.bracketUpper()
     *    @endcode
     *
     *    (There is also a corresponding bracketLower() method.)
     *
     * 2. The next step is to solve for the root within this range.
     *    There are currently two algorithms for doing that: Bisect and Brent.
     *    You can tell Solve which one to use with:
     *
     *    @code
     *    solver.setMethod(Bisect)
     *    solver.setMethod(Brent)
     *    @endcode
     *
     *    (In fact, the former is unnecessary, since Biset is the default.)
     *
     *    Then the method root() will solve for the root in that range.
     *
     * Typical usage:
     *
     * @code
     * struct MyFunc {
     *     double operator()(double x) { return [...] }
     * };
     *
     * MyFunc func;
     * Solve<MyFunc> solver(func, xmin, xmax);
     * solver.bracket();         // If necessary
     * solver.setMethod(Brent);  // If desired
     * double result = solver.root()
     * @endcode
     */
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
        /// Constructor taking the function to solve, and the range (if known).
        Solve(const F& func_, T lb_=0., T ub_=1.) :
            func(func_), lBound(lb_), uBound(ub_), xTolerance(defaultTolerance),
            maxSteps(defaultMaxSteps), boundsAreEvaluated(false), m(Bisect) {}

        /// Set the maximum number of steps for the solving algorithm to use.
        void setMaxSteps(int m) { maxSteps=m; }

        /// Set which method to use: Bisect or Brent
        void setMethod(Method m_) { m=m_; }

        /// Set the tolerance to define when the root is close enough to 0. (abs(x) < tol)
        void setXTolerance(T tol) { xTolerance=tol; }

        /// Get the current tolerance
        T getXTolerance() const { return xTolerance; }

        /// Set the bounds for the search to new values
        void setBounds(T lb, T ub) { lBound=lb; uBound=ub; }

        //@{
        /// Get the current search bounds
        double getLowerBound() const { return lBound; }
        double getUpperBound() const { return uBound; }
        //@}

        /**
         * @brief Hunt for bracket, geometrically expanding range
         *
         * This version assumes that the root is to the side of the  end point that is 
         * closer to 0.  This will be true if the function is monotonic.
         */
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

            double delta = uBound-lBound;
            for (int j=1; j<maxSteps; j++) {
                if (fupper*flower < 0.0) return;
                if (std::abs(flower) < std::abs(fupper)) {
                    uBound = lBound;
                    fupper = flower;
                    delta *= factor;
                    lBound -= delta;
                    flower = func(lBound);
                } else {
                    lBound = uBound;
                    flower = fupper;
                    delta *= factor;
                    uBound += delta;
                    fupper = func(uBound);
                }
            }
            throw SolveError("Too many iterations in bracket()");
        }

        /**
         * @brief Hunt for bracket, geometrically expanding range
         *
         * This one only expands to the right for when you know that the lower bound is 
         * definitely to the left of the root, but the upper bound might not bracket it.
         */
        void bracketUpper() 
        {
            const double factor=2.0;
            if (uBound == lBound) 
                throw SolveError("uBound=lBound in bracketUpper()");
            if (!boundsAreEvaluated) {
                flower = func(lBound);
                fupper = func(uBound);
                boundsAreEvaluated=true;
            }

            double delta = uBound-lBound;
            for (int j=1; j<maxSteps; j++) {
                if (fupper*flower < 0.0) return;
                lBound = uBound;
                flower = fupper;
                delta *= factor;
                uBound += delta;
                fupper = func(uBound);
            }
            throw SolveError("Too many iterations in bracketUpper()");
        }

        /**
         * @brief Hunt for bracket, geometrically expanding range
         *
         * The opposite of bracketUpper -- only expand to the left.
         */
        void bracketLower() 
        {
            const double factor = 2.0;
            if (uBound == lBound) 
                throw SolveError("uBound=lBound in bracketLower()");
            if (!boundsAreEvaluated) {
                flower = func(lBound);
                fupper = func(uBound);
                boundsAreEvaluated=true;
            }

            double delta = uBound-lBound;
            for (int j=1; j<maxSteps; j++) {
                if (fupper*flower < 0.0) return;
                uBound = lBound;
                fupper = flower;
                delta *= factor;
                lBound -= delta;
                flower = func(lBound);
            }
            throw SolveError("Too many iterations in bracketLower()");
        }

        /**
         * @brief Find the root according the the method currently set.
         */
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

        /**
         * @brief A simple bisection root-finder
         */
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

        /**
         * @brief A more sophisticated root-finder using Brent's algorithm 
         */
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
