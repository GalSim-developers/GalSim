/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

// Template to find the zero of an equation
// Currently uses bisection method, no solution caching.

#ifndef GalSim_Solve_H
#define GalSim_Solve_H

#include <cmath>
#include <limits>

#include "Std.h"

namespace galsim {

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /// @brief Exception class thrown by Solve
    class PUBLIC_API SolveError : public std::runtime_error
    {
    public:
        SolveError(const std::string m) : std::runtime_error("Solve error: "+m) {}
    };

    //! @endcond

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
     *    d) Sometimes there is a fundamental limit past which you want to make sure that
     *    the range doesn't go.  e.g. For values that must be positive, you might want to make
     *    sure the range doesn't extend past 0.
     *
     *    For this case, there are the following two functions
     *    @code
     *    solver.bracketUpperWithLimit(upper_limit)
     *    solver.bracketLowerWithLimit(lower_limit)
     *    @endcode
     *
     *    Finally, note that there is nothing in the code that enforces x_min < x_max.
     *    If the two limits bracket the code in reverse order, that's ok.
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
     *    (In fact, the former is unnecessary, since Bisect is the default.)
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
    class PUBLIC_API Solve
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
        Solve(const F& func_, T lb_, T ub_) :
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
        T getLowerBound() const { return lBound; }
        T getUpperBound() const { return uBound; }
        //@}

        /// @brief Make sure the current bounds have corresponding flower, fupper
        void evaluateBounds() const
        {
            if (!boundsAreEvaluated) {
                flower = func(lBound);
                fupper = func(uBound);
                boundsAreEvaluated=true;
            }
        }


        /**
         * @brief Hunt for bracket, geometrically expanding range
         *
         * This version assumes that the root is to the side of the end point that is
         * closer to 0.  This will be true if the function is monotonic.
         */
        void bracket()
        {
            const T factor=2.0;
            if (uBound == lBound)
                throw SolveError("uBound=lBound in bracket()");
            evaluateBounds();

            T delta = uBound-lBound;
            for (int j=1; j<maxSteps; j++) {
                if (fupper*flower <= 0.0) return;
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

        // General purpose one-sided bracket.  Used by bracketUpper and bracketLower.
        // Expand in the direction of b.
        bool bracket1(T& a, T& b, T& fa, T& fb)
        {
            const T factor=2.0;
            T delta = b-a;
            for (int j=1; j<maxSteps; j++) {
                if (fa*fb <= 0.0) return true;
                a = b;
                fa = fb;
                delta *= factor;
                b += delta;
                fb = func(uBound);
            }
            return false;
        }

        /**
         * @brief Hunt for bracket, geometrically expanding range
         *
         * This one only expands to the right for when you know that the lower bound is
         * definitely to the left of the root, but the upper bound might not bracket it.
         */
        void bracketUpper()
        {
            if (uBound == lBound)
                throw SolveError("uBound=lBound in bracketUpper()");
            evaluateBounds();
            if (!bracket1(lBound,uBound,flower,fupper))
                throw SolveError("Too many iterations in bracketUpper()");
        }

        /**
         * @brief Hunt for bracket, geometrically expanding range
         *
         * The opposite of bracketUpper -- only expand to the left.
         */
        void bracketLower()
        {
            if (uBound == lBound)
                throw SolveError("uBound=lBound in bracketLower()");
            evaluateBounds();
            if (!bracket1(uBound,lBound,fupper,flower))
                throw SolveError("Too many iterations in bracketLower()");
        }

        // General purpose one-sided bracket with limit.
        // Used by bracketUpperWithLimit and bracketLowerWithLimit.
        // Expand in the direction of b.
        bool bracket1WithLimit(T& a, T& b, T& fa, T& fb, T& c)
        {
            const T factor=2.0;
            // The principal here is to use z = (b-a)/(b-c) as our variable to double each time
            // where a is the initial lBound, c is the upper_limit, and b is the uBound we
            // are trying to find.
            //
            // z = (b-a)/(b-c)
            // b-a = zb-zc
            // b = (zc-a)/(z-1)
            //
            // So if we take z' = f*z
            // z' = f(b-a)/(b-c)
            // Then
            // b' = (z'c-a)/(z'-1)
            // b' = (fcb-fca-ab+ac)/(b-c) / (fb-fa-b+c)/(b-c)
            //    = ((fc-a)b-(f-1)ac) / ((f-1)b-fa+c)
            //    = (fc(b-a) + a(c-b)) / (f(b-a) + (c-b))
            // This is our fundamental iteration formula.
            //
            // To see the behavior of this iteration, take two limiting cases.
            // 1) If b = a+d and d << c-a, and use f=2 for simplicity:
            // b' = (2cd + ac - a^2 - ad)) / (2d+c-a-d)
            //    = (a(c-a) + d(2c-a)) / (c-a+d)
            //    = (a + d(2c-a)/(c-a)) / (1+d/(c-a))
            //    = (a + d(2c-a)/(c-a)) * (1-d/(c-a))
            //    = a + d(2c-a)/(c-a) - da/(c-a)
            //    = a + d(2c-2a)/(c-a)
            //    = a + 2d
            //
            // 2) If b = c-d and d << c-a, and use f=2 for simplicity:
            // b' = (2c^2-2cd-2ca+ad) / (2c-2d-2a+d)
            //    = (2c(c-a) - d(2c-a)) / (2(c-a)-d)
            //    = (c - d(2c-a)/(2c-2a)) / (1-d/(2c-2a))
            //    = (c - d(2c-a)/(2c-2a)) * (1+d/(2c-2a))
            //    = c - d(2c-a)/2(c-a) + cd/2(c-a)
            //    = c - d(c-a)/2(c-a)
            //    = c - d/2
            //
            // So when far from the limit (1), the iteration is just like the version without
            // the limit.  But when the variable approaches the limit (2), it just goes half the
            // remaining distance each time.

            xdbg<<"Bracket with limits:\n";
            xdbg<<"a,b,c = "<<a<<','<<b<<','<<c<<std::endl;
            xdbg<<"fa,fb = "<<fa<<','<<fb<<std::endl;
            for (int j=1; j<maxSteps; j++) {
                if (fa*fb <= 0.0) return true;
                T bma = b-a; // Do this before overwriting a!
                T cmb = c-b;
                a = b;
                fa = fb;
                // We divide the top and bottom by (b-a)(c-b) for stability in case one of these
                // is especially larger than the other.
                b = (factor*c/cmb + a/bma) / (factor/cmb + 1./bma);
                xdbg<<"b -> "<<b<<std::endl;
                fb = func(b);
                xdbg<<"fb -> "<<fb<<std::endl;
            }
            return false;
        }

        /**
         * @brief Hunt for upper bracket, with an upper limit to how far it can go.
         */
        void bracketUpperWithLimit(T upper_limit)
        {
            if (uBound == lBound)
                throw SolveError("uBound=lBound in bracketUpperWithLimit()");
            if (uBound == upper_limit)
                throw SolveError("uBound=upper_limit in bracketUpperWithLimit()");
            if ((uBound-lBound) * (upper_limit-uBound) <= 0)
                throw SolveError("uBound not between lBound and upper_limit");
            evaluateBounds();
            if (!bracket1WithLimit(lBound,uBound,flower,fupper,upper_limit))
                throw SolveError("Too many iterations in bracketUpperWithLimit()");
        }

        /**
         * @brief Hunt for lower bracket, with a lower limit to how far it can go.
         */
        void bracketLowerWithLimit(T lower_limit)
        {
            if (uBound == lBound)
                throw SolveError("uBound=lBound in bracketLowerWithLimit()");
            if (lBound == lower_limit)
                throw SolveError("lBound=lower_limit in bracketLowerWithLimit()");
            if ((uBound-lBound) * (lBound-lower_limit) <= 0)
                throw SolveError("lBound not between uBound and lower_limit");
            evaluateBounds();
            if (!bracket1WithLimit(uBound,lBound,fupper,flower,lower_limit))
                throw SolveError("Too many iterations in bracketLowerWithLimit()");
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

            evaluateBounds();
            f=flower;
            fmid=fupper;

            if (f*fmid > 0.0)
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
            evaluateBounds();
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
                tol1=2.0*std::numeric_limits<T>::epsilon()*std::abs(b) + 0.5*xTolerance;
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
