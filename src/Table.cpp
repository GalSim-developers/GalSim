/* -*- c++ -*-
 * Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

// icpc pretends to be GNUC, since it thinks it's compliant, but it's not.
// It doesn't understand "pragma GCC"
#ifndef __INTEL_COMPILER

// For 32-bit machines, g++ -O2 optimization in the TMV stuff below uses an optimization
// that is technically isn't known to not overflow 32 bit integers.  In fact, it is totally
// fine to use, but we need to remove a warning about it in this file for gcc >= 4.5
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 5)
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif

#endif



#include "TMV.h"
#include "TMV_SymBand.h"
#include "Table.h"
#include <cmath>
#include <vector>

#include <iostream>

namespace galsim {

    // Look up an index.  Use STL binary search; maybe faster to use
    template<class V, class A>
    int Table<V,A>::upperIndex(const A a) const
    {
        if (a<_argMin()-lower_slop || a>_argMax()+upper_slop)
            throw TableOutOfRange(a,_argMin(),_argMax());
        // check for slop
        if (a < v[0].arg) return 1;
        if (a > v[v.size()-1].arg) return v.size()-1;

        // Go directly to index if arguments are regularly spaced.
        if (equalSpaced) {
            int index = int( std::ceil( (a-_argMin()) / dx) );
            if (index >= int(v.size())) --index; // in case of rounding error
            if (index == 0) ++index;
            // check if we need to move ahead or back one step due to rounding errors
            while (a > v[index].arg) ++index;
            while (a < v[index-1].arg) --index;
            return index;
        } else {
            xassert(lastIndex >= 1);
            xassert(lastIndex < int(v.size()));

            if ( a < v[lastIndex-1].arg ) {
                xassert(lastIndex-2 >= 0);
                // Check to see if the previous one is it.
                if (a >= v[lastIndex-2].arg) return --lastIndex;
                else {
                    // Look for the entry from 0..lastIndex-1:
                    Entry e(a,0);
                    iter p = std::upper_bound(v.begin(), v.begin()+lastIndex-1, e);
                    xassert(p != v.begin());
                    xassert(p != v.begin()+lastIndex-1);
                    lastIndex = p-v.begin();
                    return lastIndex;
                }
            } else if (a > v[lastIndex].arg) {
                xassert(lastIndex+1 < int(v.size()));
                // Check to see if the next one is it.
                if (a <= v[lastIndex+1].arg) return ++lastIndex;
                else {
                    // Look for the entry from lastIndex..end
                    Entry e(a,0);
                    iter p = std::lower_bound(v.begin()+lastIndex+1, v.end(), e);
                    xassert(p != v.begin()+lastIndex+1);
                    xassert(p != v.end());
                    lastIndex = p-v.begin();
                    return lastIndex;
                }
            } else {
                // Then lastIndex is correct.
                return lastIndex;
            }
        }
    }

    //new element for table.
    template<class V, class A>
    void Table<V,A>::addEntry(const A _arg, const V _val)
    {
        Entry e(_arg,_val);
        v.push_back(e);
        isReady = false; //re-sort array next time used
    }

    template<class V, class A>
    Table<V,A>::Table(const A* argvec, const V* valvec, int N, interpolant in) :
        iType(in), isReady(false)
    {
        v.reserve(N);
        const A* aptr;
        const V* vptr;
        int i;
        for (i=0, aptr=argvec, vptr=valvec; i<N; i++, aptr++, vptr++) {
            Entry e(*aptr,*vptr);
            v.push_back(e);
        }
    }

    template<class V, class A>
    Table<V,A>::Table(const std::vector<A>& aa, const std::vector<V>& vv, interpolant in) :
        iType(in), isReady(false)
    {
        v.reserve(aa.size());
        if (vv.size() != aa.size())
            throw TableError("input vector lengths don't match");
        if (iType == spline && vv.size() < 3)
            throw TableError("input vectors are too short to spline interpolate");
        if (vv.size() < 2 && (iType == linear || iType == ceil || iType == floor
                              || iType == nearest))
            throw TableError("input vectors are too short for interpolation");
        typename std::vector<A>::const_iterator aptr=aa.begin();
        typename std::vector<V>::const_iterator vptr=vv.begin();
        for (size_t i=0; i<aa.size(); i++, ++aptr, ++vptr) {
            Entry e(*aptr,*vptr);
            v.push_back(e);
        }
    }

    //lookup & interp. function value. - this one returns 0 out of bounds.
    template<class V, class A>
    V Table<V,A>::operator() (const A a) const
    {
        setup(); //do any necessary prep
        if (a<_argMin() || a>_argMax()) return V(0);
        else {
            int i = upperIndex(a);
            return interpolate(a,i,v,y2);
        }
    }

    //lookup & interp. function value.
    template<class V, class A>
    V Table<V,A>::lookup(const A a) const
    {
        setup();
        int i = upperIndex(a);
        return interpolate(a,i,v,y2);
    }

    template <class V, class A>
    void Table<V,A>::interpMany(const A* argvec, V* valvec, int N) const
    {
        setup();
        for (int k=0; k<N; ++k) {
            int i = upperIndex(argvec[k]);
            valvec[k] = interpolate(argvec[k],i,v,y2);
        }
    }

    template<class V, class A>
    V Table<V,A>::linearInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& )
    {
        A h = v[i].arg - v[i-1].arg;
        A aa = (v[i].arg - a) / h;
        A bb = 1. - aa;
        return aa*v[i-1].val + bb*v[i].val;
    }

    template<class V, class A>
    V Table<V,A>::splineInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& y2)
    {
#if 0
        // Direct calculation saved for comparison:
        A h = v[i].arg - v[i-1].arg;
        A aa = (v[i].arg - a)/h;
        A bb = 1. - aa;
        return aa*v[i-1].val +bb*v[i].val +
            ((aa*aa*aa-aa)*y2[i-1]+(bb*bb*bb-bb)*y2[i]) *
            (h*h)/6.0;
#else
        // Factor out h factors, so only need 1 division by h.
        // Also, use the fact that bb = h-aa to simplify the calculation.
        A h = v[i].arg - v[i-1].arg;
        A aa = (v[i].arg - a);
        A bb = h-aa;
        return ( aa*v[i-1].val + bb*v[i].val -
                 (1./6.) * aa * bb * ( (aa+h)*y2[i-1] +
                                       (bb+h)*y2[i]) ) / h;
#endif
    }

    template<class V, class A>
    V Table<V,A>::floorInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& )
    {
        // On entry, it is only guaranteed that v[i-1].arg <= a <= v[i].arg.
        // Normally those ='s are ok, but for floor and ceil we make the extra
        // check to see if we should choose the opposite bound.
        if (v[i].arg == a) return v[i].val;
        else return v[i-1].val;
    }

    template<class V, class A>
    V Table<V,A>::ceilInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& )
    {
        if (v[i-1].arg == a) return v[i-1].val;
        return v[i].val;
    }

    template<class V, class A>
    V Table<V,A>::nearestInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& )
    {
        if ((a - v[i-1].arg) < (v[i].arg - a)) i--;
        return v[i].val;
    }

    template<class V, class A>
    void Table<V,A>::read(std::istream& is)
    {
        std::string line;
        const std::string comments="#;!"; //starts comment
        V vv;
        A aa;
        while (is) {
            getline(is,line);
            // skip leading white space:
            size_t i;
            for (i=0;  isspace(line[i]) && i<line.length(); i++) ;
            // skip line if blank or just comment
            if (i==line.length()) continue;
            if (comments.find(line[i])!=std::string::npos) continue;
            // try reading arg & val from line:
            std::istringstream iss(line);
            iss >> aa >> vv;
            if (iss.fail()) throw TableReadError(line) ;
            addEntry(aa,vv);
        }
    }

    // Do any necessary setup of the table before using
    template<class V, class A>
    void Table<V,A>::setup() const
    {
        if (isReady) return;

        if (v.size() <= 1)
            throw TableError("Trying to use a null Table (need at least 2 entries)");

        sortIt();
        lastIndex = 1; // Start back at the beginning for the next search.

        // See if arguments are equally spaced
        // ...within this fractional error:
        const double tolerance = 0.01;
        dx = (v.back().arg - v.front().arg) / (v.size()-1);
        if (dx == 0.)
            throw TableError("First and last Table entry are equal.");
        equalSpaced = true;
        for (int i=1; i<int(v.size()); i++) {
            if ( std::abs( ((v[i].arg-v[0].arg)/dx - i)) > tolerance) equalSpaced = false;
            if (v[i].arg == v[i-1].arg)
                throw TableError("Table has repeated arguments.");
        }

        switch (iType) {
          case linear:
               interpolate = &Table<V,A>::linearInterpolate;
               break;
          case spline :
               setupSpline();
               interpolate = &Table<V,A>::splineInterpolate;
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
          default:
               throw TableError("interpolation method not yet implemented");
        }

        lower_slop = (v[1].arg - v[0].arg) * 1.e-6;
        upper_slop = (v[v.size()-1].arg - v[v.size()-2].arg) * 1.e-6;

        isReady = true;
    }

    template <class V, class A>
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
        int n = v.size();
        y2.resize(n);
        // End points 2nd-derivatives zero for natural cubic spline
        y2[0] = V(0);
        y2[n-1] = V(0);
        // For 3 points second derivative at i=1 is simple
        if (n == 3){

            y2[1] = 3.*((v[2].val - v[1].val) / (v[2].arg - v[1].arg) -
                        (v[1].val - v[0].val) / (v[1].arg - v[0].arg)) / (v[2].arg - v[0].arg);

        } else {  // For 4 or more points we use the TMV symmetric tridiagonal matrix solver

            tmv::SymBandMatrix<V> M(n-2, 1);
            for (int i=1; i<=n-3; i++){
                M(i, i-1) = v[i+1].arg - v[i].arg;
            }
            tmv::Vector<V> rhs(n-2);
            for (int i=1; i<=n-2; i++){
                M(i-1, i-1) = 2. * (v[i+1].arg - v[i-1].arg);
                rhs(i-1) = 6. * ( (v[i+1].val - v[i].val) / (v[i+1].arg - v[i].arg) -
                                  (v[i].val - v[i-1].val) / (v[i].arg - v[i-1].arg) );
            }
            tmv::Vector<V> solution(n-2);
            solution = rhs / M;   // solve the tridiagonal system of equations
            for (int i=1; i<=n-2; i++){
                y2[i] = solution[i-1];
            }
        }
    }

    template class Table<double,double>;


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
    }

    // Look up an index.  Use STL binary search.
    template<class A>
    int ArgVec<A>::upperIndex(const A a) const
    {
        if (a<vec.front()-lower_slop || a>vec.back()+upper_slop)
            throw TableOutOfRange(a,vec.front(),vec.back());
        // check for slop
        if (a < vec.front()) return 1;
        if (a > vec.back()) return vec.size()-1;

        if (equalSpaced) {
            int i = int( std::ceil( (a-vec.front()) / da) );
            if (i >= vec.size()) --i; // in case of rounding error
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


    // Table2D

    template<class V, class A>
    Table2D<V,A>::Table2D(const A* _xargs, const A* _yargs, const V* _vals, int _Nx, int _Ny,
        interpolant in) : iType(in), Nx(_Nx), Ny(_Ny),
                          xargs(_xargs, _xargs+Nx), yargs(_yargs, _yargs+Ny)
    {
        // Allocate and fill vals
        vals.reserve(Nx*Ny);
        for (int i=0; i<Nx*Ny; i++, _vals++)
            vals.push_back(*_vals);

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
    void Table2D<V,A>::interpManyScatter(const A* xvec, const A* yvec, V* valvec, int N) const
    {
        int i, j;
        for (int k=0; k<N; k++) {
            i = xargs.upperIndex(xvec[k]);
            j = yargs.upperIndex(yvec[k]);
            valvec[k] = (this->*interpolate)(xvec[k], yvec[k], i, j);
        }
    }

    //lookup and interpolate along the outer product of an x-array and a y-array.
    //The result will be an array with length outNx * outNy.
    template<class V, class A>
    void Table2D<V,A>::interpManyOuter(const A* xvec, const A* yvec, V* valvec,
                                       int outNx, int outNy) const
    {
        int i, j;
        for (int outj=0; outj<outNy; outj++) {
            j = yargs.upperIndex(yvec[outj]);
            for (int outi=0; outi<outNx; outi++, valvec++) {
                i = xargs.upperIndex(xvec[outi]);
                *valvec = (this->*interpolate)(xvec[outi], yvec[outj], i, j);
            }
        }
    }

    template<class V, class A>
    V Table2D<V,A>::linearInterpolate(const A x, const A y, int i, int j) const
    {
        A ax = (xargs[i] - x) / (xargs[i] - xargs[i-1]);
        A bx = 1.0 - ax;
        A ay = (yargs[j] - y) / (yargs[j] - yargs[j-1]);
        A by = 1.0 - ay;
        return (vals[(j-1)*Nx+i-1] * ax * ay
                + vals[j*Nx+i-1] * ax * by
                + vals[(j-1)*Nx+i] * bx * ay
                + vals[j*Nx+i] * bx * by);
    }

    template<class V, class A>
    V Table2D<V,A>::floorInterpolate(const A x, const A y, int i, int j) const
    {
        // On entry, it is only guaranteed that xargs[i-1] <= x <= xargs[i] (and similarly y).
        // Normally those ='s are ok, but for floor and ceil we make the extra
        // check to see if we should choose the opposite bound.
        if (x == xargs[i]) i++;
        if (y == yargs[j]) j++;
        return vals[(j-1)*Nx+i-1];
    }

    template<class V, class A>
    V Table2D<V,A>::ceilInterpolate(const A x, const A y, int i, int j) const
    {
        if (x == xargs[i-1]) i--;
        if (y == yargs[j-1]) j--;
        return vals[j*Nx+i];
    }

    template<class V, class A>
    V Table2D<V,A>::nearestInterpolate(const A x, const A y, int i, int j) const
    {
        if ((x - xargs[i-1]) < (xargs[i] - x)) i--;
        if ((y - yargs[j-1]) < (yargs[j] - y)) j--;
        return vals[j*Nx+i];
    }

    template class Table2D<double,double>;


    // Table1D

    template<class V, class A>
    Table1D<V,A>::Table1D(const A* args, const V* valarray, int _N, interpolant in) :
        iType(in), N(_N), grid(args, args+N)
    {
        // Allocate and fill vals
        vals.reserve(N);
        for (int i=0; i<N; i++, valarray++)
            vals.push_back(*valarray);

        // Map specific interpolator to `interpolate`.
        switch (iType) {
          case linear:
               interpolate = &Table1D<V,A>::linearInterpolate;
               break;
          case floor:
               interpolate = &Table1D<V,A>::floorInterpolate;
               break;
          case ceil:
               interpolate = &Table1D<V,A>::ceilInterpolate;
               break;
          case nearest:
               interpolate = &Table1D<V,A>::nearestInterpolate;
               break;
          case spline:
               interpolate = &Table1D<V,A>::splineInterpolate;
               setupSpline();
               break;
          default:
               throw TableError("interpolation method not yet implemented");
        }
    }

    template<class V, class A>
    Table1D<V,A>::Table1D(const std::vector<A>& args, const std::vector<V>& valarray, interpolant in) :
        iType(in), N(args.size()), grid(args)
    {
        // Copy valarray
        vals = valarray;
        // Map specific interpolator to `interpolate`.
        switch (iType) {
          case linear:
               interpolate = &Table1D<V,A>::linearInterpolate;
               break;
          case floor:
               interpolate = &Table1D<V,A>::floorInterpolate;
               break;
          case ceil:
               interpolate = &Table1D<V,A>::ceilInterpolate;
               break;
          case nearest:
               interpolate = &Table1D<V,A>::nearestInterpolate;
               break;
          case spline:
               interpolate = &Table1D<V,A>::splineInterpolate;
               setupSpline();
               break;
          default:
               throw TableError("interpolation method not yet implemented");
        }
    }

    //lookup and interpolate function value.
    template<class V, class A>
    V Table1D<V,A>::lookup(const A x) const
    {
        int i = grid.upperIndex(x);
        return (this->*interpolate)(x, i);
    }

    //lookup and interpolate an array of function values.
    template<class V, class A>
    void Table1D<V,A>::interpMany(const A* xvec, V* valvec, int N) const
    {
        int i;
        for (int k=0; k<N; k++) {
            i = grid.upperIndex(xvec[k]);
            valvec[k] = (this->*interpolate)(xvec[k], i);
        }
    }

    template<class V, class A>
    V Table1D<V,A>::linearInterpolate(const A x, int i) const
    {
        A ax = (grid[i] - x) / (grid[i] - grid[i-1]);
        A bx = 1.0 - ax;
        return vals[i]*bx + vals[i-1]*ax;
    }

    template<class V, class A>
    V Table1D<V,A>::floorInterpolate(const A x, int i) const
    {
        // On entry, it is only guaranteed that grid[i-1] <= x <= grid[i].
        // Normally those ='s are ok, but for floor and ceil we make the extra
        // check to see if we should choose the opposite bound.
        if (x == grid[i]) i++;
        return vals[i-1];
    }

    template<class V, class A>
    V Table1D<V,A>::ceilInterpolate(const A x, int i) const
    {
        if (x == grid[i-1]) i--;
        return vals[i];
    }

    template<class V, class A>
    V Table1D<V,A>::nearestInterpolate(const A x, int i) const
    {
        if ((x - grid[i-1]) < (grid[i] - x)) i--;
        return vals[i];
    }

    template<class V, class A>
    V Table1D<V,A>::splineInterpolate(const A x, int i) const
    {
#if 0
        // Direct calculation saved for comparison:
        A h = grid[i] - grid[i-1];
        A aa = (grid[i] - x)/h;
        A bb = 1. - aa;
        return aa*vals[i-1] +bb*vals[i] +
            ((aa*aa*aa-aa)*y2[i-1]+(bb*bb*bb-bb)*y2[i]) *
            (h*h)/6.0;
#else
        // Factor out h factors, so only need 1 division by h.
        // Also, use the fact that bb = h-aa to simplify the calculation.
        A h = grid[i] - grid[i-1];
        A aa = (grid[i] - x);
        A bb = h-aa;
        return ( aa*vals[i-1] + bb*vals[i] -
                 (1./6.) * aa * bb * ( (aa+h)*y2[i-1] +
                                       (bb+h)*y2[i]) ) / h;
#endif
    }

    template<class V, class A>
    void Table1D<V,A>::setupSpline() const
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

            y2[1] = 3.*((vals[2] - vals[1]) / (grid[2] - grid[1]) -
                        (vals[1] - vals[0]) / (grid[1] - grid[0])) / (grid[2] - grid[0]);

        } else {  // For 4 or more points we use the TMV symmetric tridiagonal matrix solver

            tmv::SymBandMatrix<V> M(n-2, 1);
            for (int i=1; i<=n-3; i++){
                M(i, i-1) = grid[i+1] - grid[i];
            }
            tmv::Vector<V> rhs(n-2);
            for (int i=1; i<=n-2; i++){
                M(i-1, i-1) = 2. * (grid[i+1] - grid[i-1]);
                rhs(i-1) = 6. * ( (vals[i+1] - vals[i]) / (grid[i+1] - grid[i]) -
                                  (vals[i] - vals[i-1]) / (grid[i] - grid[i-1]) );
            }
            tmv::Vector<V> solution(n-2);
            solution = rhs / M;   // solve the tridiagonal system of equations
            for (int i=1; i<=n-2; i++){
                y2[i] = solution[i-1];
            }
        }

    }

    template class Table1D<double,double>;
}
