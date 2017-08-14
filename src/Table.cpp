/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

//#define DEBUGLOGGING

#include "galsim/IgnoreWarnings.h"

#include "TMV.h"
#include "TMV_SymBand.h"
#include "Table.h"
#include <cmath>
#include <vector>

#include <iostream>

namespace galsim {

    // ArgVec

    template<class A>
    void ArgVec<A>::setup() const
    {
        int N = vec.size();
        const double tolerance = 0.01;
        da = (vec.back() - vec.front()) / (N-1);
        if (da == 0.) throw TableError("First and last arguments are equal.");
        equalSpaced = true;
        for (int i=1; i<N; i++) {
            if (std::abs((vec[i] - vec.front())/da - i) > tolerance) equalSpaced = false;
            if (vec[i] <= vec[i-1])
                throw TableError("Table arguments not strictly increasing.");
        }
        lastIndex = 1;
        lower_slop = (vec[1]-vec[0]) * 1.e-6;
        upper_slop = (vec[N-1]-vec[N-2]) * 1.e-6;
        isReady = true;
    }

    // Look up an index.  Use STL binary search.
    template<class A>
    int ArgVec<A>::upperIndex(const A a) const
    {
        if (!isReady) setup();
        if (a<vec.front()-lower_slop || a>vec.back()+upper_slop)
            throw TableOutOfRange(a,vec.front(),vec.back());
        // check for slop
        if (a < vec.front()) return 1;
        if (a > vec.back()) return vec.size()-1;

        if (equalSpaced) {
            int i = int( std::ceil( (a-vec.front()) / da) );
            if (i >= int(vec.size())) --i; // in case of rounding error
            if (i == 0) ++i;
            // check if we need to move ahead or back one step due to rounding errors
            while (a > vec[i]) ++i;
            while (a < vec[i-1]) --i;
            return i;
        } else {
            xassert(lastIndex >= 1);
            xassert(lastIndex < vec.size());

            if ( a < vec[lastIndex-1] ) {
                xassert(lastIndex-2 >= 0);
                // Check to see if the previous one is it.
                if (a >= vec[lastIndex-2]) return --lastIndex;
                else {
                    // Look for the entry from 0..lastIndex-1:
                    citer p = std::upper_bound(vec.begin(), vec.begin()+lastIndex-1, a);
                    xassert(p != vec.begin());
                    xassert(p != vec.begin()+lastIndex-1);
                    lastIndex = p-vec.begin();
                    return lastIndex;
                }
            } else if (a > vec[lastIndex]) {
                xassert(lastIndex+1 < vec.size());
                // Check to see if the next one is it.
                if (a <= vec[lastIndex+1]) return ++lastIndex;
                else {
                    // Look for the entry from lastIndex..end
                    citer p = std::lower_bound(vec.begin()+lastIndex+1, vec.end(), a);
                    xassert(p != vec.begin()+lastIndex+1);
                    xassert(p != vec.end());
                    lastIndex = p-vec.begin();
                    return lastIndex;
                }
            } else {
                // Then lastIndex is correct.
                return lastIndex;
            }
        }
    }

    template<class A>
    typename std::vector<A>::iterator ArgVec<A>::insert(
            typename std::vector<A>::iterator it, const A a)
    {
        isReady = false;
        return vec.insert(it, a);
    }


    // Table

    template<class V, class A>
    void Table<V,A>::addEntry(const A a, const V v)
    {
        typename std::vector<A>::const_iterator p = std::upper_bound(args.begin(), args.end(), a);
        int i = p - args.begin();
        args.insert(args.begin()+i, a);
        vals.insert(vals.begin()+i, v);
        isReady = false;
    }

    template<class V, class A>
    void Table<V,A>::setup() const
    {
        if (isReady) return;

        if (vals.size() != args.size())
            throw TableError("args and vals lengths don't match");
        if (iType == spline && vals.size() < 3)
            throw TableError("input vectors are too short to spline interpolate");
        if (vals.size() < 2 && (iType == linear || iType == ceil || iType == floor
                              || iType == nearest))
            throw TableError("input vectors are too short for interpolation");
        switch (iType) {
          case linear:
               interpolate = &Table<V,A>::linearInterpolate;
               break;
          case floor:
               interpolate = &Table<V,A>::floorInterpolate;
               break;
          case ceil:
               interpolate = &Table<V,A>::ceilInterpolate;
               break;
          case nearest:
               interpolate = &Table<V,A>::nearestInterpolate;
               break;
          case spline:
               interpolate = &Table<V,A>::splineInterpolate;
               break;
          default:
               throw TableError("interpolation method not yet implemented");
        }
        if (iType == spline) setupSpline();
        isReady = true;
    }

    //lookup and interpolate function value.
    template<class V, class A>
    V Table<V,A>::operator()(const A a) const
    {
        setup();
        if (a<argMin() || a>argMax()) return V(0);
        else {
            int i = args.upperIndex(a);
            return (this->*interpolate)(a, i);
        }
    }

    //lookup and interpolate function value.
    template<class V, class A>
    V Table<V,A>::lookup(const A a) const
    {
        setup();
        int i = args.upperIndex(a);
        return (this->*interpolate)(a, i);
    }

    //lookup and interpolate an array of function values.
    template<class V, class A>
    void Table<V,A>::interpMany(const A* argvec, V* valvec, int N) const
    {
        setup();
        int i;
        for (int k=0; k<N; k++) {
            i = args.upperIndex(argvec[k]);
            valvec[k] = (this->*interpolate)(argvec[k], i);
        }
    }

    template<class V, class A>
    V Table<V,A>::linearInterpolate(const A a, int i) const
    {
        A ax = (args[i] - a) / (args[i] - args[i-1]);
        A bx = 1.0 - ax;
        return vals[i]*bx + vals[i-1]*ax;
    }

    template<class V, class A>
    V Table<V,A>::floorInterpolate(const A a, int i) const
    {
        // On entry, it is only guaranteed that args[i-1] <= a <= args[i].
        // Normally those ='s are ok, but for floor and ceil we make the extra
        // check to see if we should choose the opposite bound.
        if (a == args[i]) i++;
        return vals[i-1];
    }

    template<class V, class A>
    V Table<V,A>::ceilInterpolate(const A a, int i) const
    {
        if (a == args[i-1]) i--;
        return vals[i];
    }

    template<class V, class A>
    V Table<V,A>::nearestInterpolate(const A a, int i) const
    {
        if ((a - args[i-1]) < (args[i] - a)) i--;
        return vals[i];
    }

    template<class V, class A>
    V Table<V,A>::splineInterpolate(const A a, int i) const
    {
#if 0
        // Direct calculation saved for comparison:
        A h = args[i] - args[i-1];
        A aa = (args[i] - a)/h;
        A bb = 1. - aa;
        return aa*vals[i-1] +bb*vals[i] +
            ((aa*aa*aa-aa)*y2[i-1]+(bb*bb*bb-bb)*y2[i]) *
            (h*h)/6.0;
#else
        // Factor out h factors, so only need 1 division by h.
        // Also, use the fact that bb = h-aa to simplify the calculation.

        A h = args[i] - args[i-1];
        A aa = (args[i] - a);
        A bb = h-aa;
        return ( aa*vals[i-1] + bb*vals[i] -
                 (1./6.) * aa * bb * ( (aa+h)*y2[i-1] +
                                       (bb+h)*y2[i]) ) / h;
#endif
    }

    template<class V, class A>
    void Table<V,A>::setupSpline() const
    {
        /**
         * Calculate the 2nd derivatives of the natural cubic spline.
         *
         * Here we follow the broad procedure outlined in this technical note by Jim
         * Armstrong, freely available online:
         * http://www.algorithmist.net/spline.html
         *
         * The system we solve is equation [7].  In our adopted notation u_i are the diagonals
         * of the matrix M, and h_i the off-diagonals.  y'' is z_i and the rhs = v_i.
         *
         * For table sizes larger than the fully trivial (2 or 3 elements), we use the
         * symmetric tridiagonal matrix solution capabilities of MJ's TMV library.
         */
        // Set up the 2nd-derivative table for splines
        int n = vals.size();
        y2.resize(n);
        // End points 2nd-derivatives zero for natural cubic spline
        y2[0] = V(0);
        y2[n-1] = V(0);
        // For 3 points second derivative at i=1 is simple
        if (n == 3){

            y2[1] = 3.*((vals[2] - vals[1]) / (args[2] - args[1]) -
                        (vals[1] - vals[0]) / (args[1] - args[0])) / (args[2] - args[0]);

        } else {  // For 4 or more points we use the TMV symmetric tridiagonal matrix solver

            tmv::SymBandMatrix<V> M(n-2, 1);
            for (int i=1; i<=n-3; i++){
                M(i, i-1) = args[i+1] - args[i];
            }
            tmv::Vector<V> rhs(n-2);
            for (int i=1; i<=n-2; i++){
                M(i-1, i-1) = 2. * (args[i+1] - args[i-1]);
                rhs(i-1) = 6. * ( (vals[i+1] - vals[i]) / (args[i+1] - args[i]) -
                                  (vals[i] - vals[i-1]) / (args[i] - args[i-1]) );
            }
            tmv::Vector<V> solution(n-2);
            solution = rhs / M;   // solve the tridiagonal system of equations
            for (int i=1; i<=n-2; i++){
                y2[i] = solution[i-1];
            }
        }

    }

    template class Table<double,double>;


    // Table2D

    template<class V, class A>
    Table2D<V,A>::Table2D(const A* _xargs, const A* _yargs, const V* _vals, int _Nx, int _Ny,
        interpolant in) : iType(in), Nx(_Nx), Ny(_Ny),
                          xargs(_xargs, _xargs+Nx), yargs(_yargs, _yargs+Ny),
                          vals(_vals, _vals+Nx*Ny)
    {
        // Map specific interpolator to `interpolate`.
        switch (iType) {
          case linear:
               interpolate = &Table2D<V,A>::linearInterpolate;
               break;
          case floor:
               interpolate = &Table2D<V,A>::floorInterpolate;
               break;
          case ceil:
               interpolate = &Table2D<V,A>::ceilInterpolate;
               break;
          case nearest:
               interpolate = &Table2D<V,A>::nearestInterpolate;
               break;
          default:
               throw TableError("interpolation method not yet implemented");
        }
    }

    //lookup and interpolate function value.
    template<class V, class A>
    V Table2D<V,A>::lookup(const A x, const A y) const
    {
        int i = xargs.upperIndex(x);
        int j = yargs.upperIndex(y);
        return (this->*interpolate)(x, y, i, j);
    }

    //lookup and interpolate an array of function values.
    //In this case, the length of xvec is assumed to be the same as the length of yvec, and
    //will also equal the length of the result array.
    template<class V, class A>
    void Table2D<V,A>::interpMany(const A* xvec, const A* yvec, V* valvec, int N) const
    {
        int i, j;
        for (int k=0; k<N; k++) {
            i = xargs.upperIndex(xvec[k]);
            j = yargs.upperIndex(yvec[k]);
            *valvec++ = (this->*interpolate)(xvec[k], yvec[k], i, j);
        }
    }

    //lookup and interpolate along the mesh of an x-array and a y-array.
    //The result will be an array with length outNx * outNy.
    template<class V, class A>
    void Table2D<V,A>::interpManyMesh(const A* xvec, const A* yvec, V* valvec,
                                       int outNx, int outNy) const
    {
        int i, j;
        for (int outi=0; outi<outNx; outi++) {
            i = xargs.upperIndex(xvec[outi]);
            for (int outj=0; outj<outNy; outj++) {
                j = yargs.upperIndex(yvec[outj]);
                *valvec++ = (this->*interpolate)(xvec[outi], yvec[outj], i, j);
            }
        }
    }

    /// Estimate many df/dx and df/dy values
    template <class V, class A>
    void Table2D<V,A>::gradient(const A x, const A y, V& dfdx, V& dfdy) const
    {
        // Note: This is really only accurate for linear interpolation.
        // The derivative for floor, ceil, nearest interpolation doesn't really make
        // much sense, so this is probably what the user would want.  However, if we
        // eventually implement spline interpolation for Table2D, then this function will
        // need to be revisited.
        int i = xargs.upperIndex(x);
        int j = yargs.upperIndex(y);
        A dx = xargs[i] - xargs[i-1];
        A dy = yargs[j] - yargs[j-1];
        V f00 = vals[(i-1)*Ny+j-1];
        V f01 = vals[(i-1)*Ny+j];
        V f10 = vals[i*Ny+j-1];
        V f11 = vals[i*Ny+j];
        A ax = (xargs[i] - x) / (xargs[i] - xargs[i-1]);
        A bx = 1.0 - ax;
        A ay = (yargs[j] - y) / (yargs[j] - yargs[j-1]);
        A by = 1.0 - ay;
        dfdx = ( (f10-f00)*ay + (f11-f01)*by ) / dx;
        dfdy = ( (f01-f00)*ax + (f11-f10)*bx ) / dy;
    }

    /// Estimate many df/dx and df/dy values
    template <class V, class A>
    void Table2D<V,A>::gradientMany(const A* xvec, const A* yvec, V* dfdxvec, V* dfdyvec,
                                    int N) const
    {
        for (int k=0; k<N; ++k) {
            gradient(xvec[k], yvec[k], dfdxvec[k], dfdyvec[k]);
        }
    }

    template<class V, class A>
    V Table2D<V,A>::linearInterpolate(const A x, const A y, int i, int j) const
    {
        A ax = (xargs[i] - x) / (xargs[i] - xargs[i-1]);
        A bx = 1.0 - ax;
        A ay = (yargs[j] - y) / (yargs[j] - yargs[j-1]);
        A by = 1.0 - ay;

        return (vals[(i-1)*Ny+j-1] * ax * ay
                + vals[i*Ny+j-1] * bx * ay
                + vals[(i-1)*Ny+j] * ax * by
                + vals[i*Ny+j] * bx * by);
    }

    template<class V, class A>
    V Table2D<V,A>::floorInterpolate(const A x, const A y, int i, int j) const
    {
        // On entry, it is only guaranteed that xargs[i-1] <= x <= xargs[i] (and similarly y).
        // Normally those ='s are ok, but for floor and ceil we make the extra
        // check to see if we should choose the opposite bound.
        if (x == xargs[i]) i++;
        if (y == yargs[j]) j++;
        return vals[(i-1)*Ny+j-1];
    }

    template<class V, class A>
    V Table2D<V,A>::ceilInterpolate(const A x, const A y, int i, int j) const
    {
        if (x == xargs[i-1]) i--;
        if (y == yargs[j-1]) j--;
        return vals[i*Ny+j];
    }

    template<class V, class A>
    V Table2D<V,A>::nearestInterpolate(const A x, const A y, int i, int j) const
    {
        if ((x - xargs[i-1]) < (xargs[i] - x)) i--;
        if ((y - yargs[j-1]) < (yargs[j] - y)) j--;
        return vals[i*Ny+j];
    }

    template class Table2D<double,double>;
}
