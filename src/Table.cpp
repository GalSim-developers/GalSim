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

//#define DEBUGLOGGING

#include <cmath>
#include <vector>
#include <iostream>
#include <deque>

#ifdef USE_TMV
#include "TMV.h"
#include "TMV_SymBand.h"
#endif

#include "fmath/fmath.hpp"  // For SSE

#include "Table.h"
#include "Interpolant.h"


namespace galsim {

    // ArgVec
    // A class to represent an argument vector for a Table or Table2D.
    class ArgVec
    {
    public:
        ArgVec(const double* args, int n);

        int upperIndex(double a) const;
        void upperIndexMany(const double* a, int* idx, int N) const;

        // A few things to look similar to a vector<dobule>
        const double* begin() const { return _vec;}
        const double* end() const { return _vec + _n;}
        double front() const { return *_vec; }
        double back() const { return *(_vec + _n - 1); }
        double operator[](int i) const { return _vec[i]; }
        size_t size() const { return _n; }

    private:
        const double* _vec;
        int _n;
        // A few convenient additional member variables.
        double _lower_slop, _upper_slop;
        bool _equalSpaced;
        double _da;
        mutable int _lastIndex;
    };

    ArgVec::ArgVec(const double* vec, int n): _vec(vec), _n(n)
    {
        xdbg<<"Make ArgVec from vector starting with: "<<vec[0]<<std::endl;
        const double tolerance = 0.01;
        _da = (back() - front()) / (_n-1);
        _equalSpaced = true;
        for (int i=1; i<_n; i++) {
            if (std::abs((_vec[i] - _vec[0])/_da - i) > tolerance) _equalSpaced = false;
        }
        _lastIndex = 1;
        _lower_slop = (_vec[1]-_vec[0]) * 1.e-6;
        _upper_slop = (_vec[_n-1]-_vec[_n-2]) * 1.e-6;
    }

    // Look up an index.  Use STL binary search.
    int ArgVec::upperIndex(double a) const
    {
        // check for slop
        if (a < front()) return 1;
        if (a > back()) return _n-1;

        if (_equalSpaced) {
            xdbg<<"Equal spaced\n";
            xdbg<<"da = "<<_da<<std::endl;
            int i = int( std::ceil( (a-front()) / _da) );
            xdbg<<"i = "<<i<<std::endl;
            if (i >= _n) i = _n-1; // in case of rounding error or off the edge
            if (i <= 0) i = 1;
            xdbg<<"i => "<<i<<std::endl;
            return i;
        } else {
            xdbg<<"Not equal spaced\n";
            xdbg<<"lastIndex = "<<_lastIndex<<"  "<<_vec[_lastIndex-1]<<" "<<_vec[_lastIndex]<<std::endl;
            xassert(_lastIndex >= 1);
            xassert(_lastIndex < _n);

            if ( a < _vec[_lastIndex-1] ) {
                xdbg<<"Go lower\n";
                xassert(_lastIndex-2 >= 0);
                // Check to see if the previous one is it.
                if (a >= _vec[_lastIndex-2]) {
                    xdbg<<"Previous works: "<<_vec[_lastIndex-2]<<std::endl;
                    return --_lastIndex;
                } else {
                    // Look for the entry from 0.._lastIndex-1:
                    const double* p = std::upper_bound(begin(), begin()+_lastIndex-1, a);
                    xassert(p != begin());
                    xassert(p != begin()+_lastIndex-1);
                    _lastIndex = p-begin();
                    xdbg<<"Success: "<<_lastIndex<<"  "<<_vec[_lastIndex]<<std::endl;
                    return _lastIndex;
                }
            } else if (a > _vec[_lastIndex]) {
                xassert(_lastIndex+1 < _n);
                // Check to see if the next one is it.
                if (a <= _vec[_lastIndex+1]) {
                    xdbg<<"Next works: "<<_vec[_lastIndex+1]<<std::endl;
                    return ++_lastIndex;
                } else {
                    // Look for the entry from _lastIndex..end
                    const double* p = std::lower_bound(begin()+_lastIndex+1, end(), a);
                    xassert(p != begin()+_lastIndex+1);
                    xassert(p != end());
                    _lastIndex = p-begin();
                    xdbg<<"Success: "<<_lastIndex<<"  "<<_vec[_lastIndex]<<std::endl;
                    return _lastIndex;
                }
            } else {
                xdbg<<"lastindex is still good.\n";
                // Then _lastIndex is correct.
                return _lastIndex;
            }
        }
    }

    void ArgVec::upperIndexMany(const double* a, int* indices, int N) const
    {
        xdbg<<"Start upperIndexMany\n";
        if (_equalSpaced) {
            xdbg << "Equal spaced\n";
            xdbg << "da = "<<_da<<'\n';
            for (int k=0; k<N; k++) {
                xdbg<<"a[k] = "<<a[k]<<std::endl;
                int idx = int(std::ceil((a[k]-front()) / _da));
                if (idx >= _n) idx = _n-1; // in case of rounding error or off the edge
                if (idx <= 0) idx = 1;
                xdbg << "idx = "<<idx<<'\n';
                indices[k] = idx;
            }
        } else {
            xdbg << "Not equal spaced\n";
            int idx = 1;
            double lowerBound = _vec[0];
            double upperBound = _vec[1];

            for (int k=0; k<N; k++) {
                xassert(idx >= 1);
                xassert(idx < _n);
                if (a[k] < front()) {
                    indices[k] = 1;
                    continue;
                }
                if (a[k] > back()) {
                    indices[k] = _n - 1;
                    continue;
                }
                if (a[k] < lowerBound) { // Go lower
                    xdbg << "Go lower\n";
                    xassert(idx-2 >= 0);
                    if (a[k] >= _vec[idx-2]) {  // Check previous index
                        xdbg << "Previous works  "<<_vec[idx-2]<<'\n';
                        --idx;
                        indices[k] = idx;
                        upperBound = lowerBound;
                        lowerBound = _vec[idx-1];
                    } else {
                        const double* p = std::upper_bound(begin(), begin()+idx-1, a[k]);
                        xassert(p != begin());
                        xassert(p != begin()+idx-1);
                        idx = p-begin();
                        indices[k] = idx;
                        upperBound = _vec[idx];
                        lowerBound = _vec[idx-1];
                        xdbg << "Sucess: "<<idx<<" "<<upperBound<<'\n';
                    }
                } else if (a[k] > upperBound) { //Go higher
                    xdbg << "Go higher\n";
                    xassert(idx+1 < _n);
                    if (a[k] <= _vec[idx+1]) { // Check next index
                        xdbg << "Next works  "<<_vec[idx-2]<<'\n';
                        ++idx;
                        indices[k] = idx;
                        lowerBound = upperBound;
                        upperBound = _vec[idx];
                    } else {
                        const double* p = std::lower_bound(begin()+idx+1, end(), a[k]);
                        xassert(p != begin()+idx+1);
                        xassert(p != end());
                        idx = p-begin();
                        indices[k] = idx;
                        upperBound = _vec[idx];
                        lowerBound = _vec[idx-1];
                        xdbg << "Sucess: "<<idx<<" "<<upperBound<<'\n';
                    }
                } else {
                    indices[k] = idx;
                }
            }
        }
    }

    // The hierarchy for TableImpl looks like:
    // TableImpl <- ABC
    // TCRTP<T> : TableImpl <- curiously recurring template pattern
    // TLinearInterp : TCRTP<TLinearInterp>
    // ... similar, Floor, Ceil, Nearest, Spline
    // TGSInterpolant<interpolant> : TCRTP<TGSInterpolant<interpolant>> <- Use Interpolant

    class Table::TableImpl {
    public:
        TableImpl(const double* args, const double* vals, int N) :
            _args(args, N), _n(N), _vals(vals) ,
            _slop_min(_args.front() - 1.e-6 * (_args.back() - _args.front())),
            _slop_max(_args.back() + 1.e-6 * (_args.back() - _args.front()))
        {}

        virtual int find(double a) const = 0;
        virtual double lookup(double a) const = 0;
        virtual double interp(double a, int i) const = 0;
        virtual void interpMany(const double* argvec, double* valvec, int N) const = 0;
        virtual double integrate(double xmin, double max) const = 0;
        virtual double integrateProduct(const Table::TableImpl& g,
                                        double xmin, double xmax, double xfact) const = 0;

        double argMin() const { return _args.front(); }
        double argMax() const { return _args.back(); }
        int size() const { return _n; }
        inline double getArg(int i) const { return _args[i]; }
        inline double getVal(int i) const { return _vals[i]; }

        virtual ~TableImpl() {}
    protected:
        ArgVec _args;
        const int _n;
        const double* _vals;
        const double _slop_min, _slop_max;
    };


    template<class T>
    class TCRTP : public Table::TableImpl {
    public:
        using Table::TableImpl::TableImpl;

        double lookup(double a) const override {
            return interp(a, find(a));
        }

        int find(double a) const override {
            return _args.upperIndex(a);
        }

        double interp(double a, int i) const override {
            if (!(a >= _slop_min && a <= _slop_max))
                throw std::runtime_error("invalid argument to Table.interp");
            return static_cast<const T*>(this)->_interp(a, i);
        }

        void interpMany(const double* xvec, double* valvec, int N) const override {
            std::vector<int> indices(N);
            _args.upperIndexMany(xvec, indices.data(), N);

            for (int k=0; k<N; k++) {
                valvec[k] = interp(xvec[k], indices[k]);
            }
        }

        double integrate(double xmin, double xmax) const override
        {
            dbg<<"Start integrate: "<<xmin<<" "<<xmax<<std::endl;
            int i = _args.upperIndex(xmin);
            double x1 = xmin;
            double f1;  // not determined yet.
            double x2 = _args[i];
            double f2 = _vals[i];

            if (x2 > xmax) {
                // If the whole integration region is between two of the tabulated points,
                // then it requires some extra special handling.
                x2 = xmax;
                dbg<<"do special case full range x1,x2 = "<<x1<<','<<x2<<std::endl;
                f1 = interp(x1, i);
                f2 = interp(x2, i);
                double ans = static_cast<const T*>(this)->integ_step_full(x1,f1,x2,f2,i);
                dbg<<"ans = "<<ans<<std::endl;
                return ans;
            }

            double ans = 0.;
            // Do the first integration step, which may need some additional work to handle
            // left edge not being a table point.
            if (x2 > x1) {
                xdbg<<"do first step: x1,x2 = "<<x1<<','<<x2<<std::endl;
                // if x1 == x2, then skip "first" step, since it is 0.
                f1 = interp(x1, i);
                double step = static_cast<const T*>(this)->integ_step_first(x1,f1,x2,f2,i);
                ans += step;
                xdbg<<"ans = "<<ans<<std::endl;
            }
            x1 = x2;
            f1 = f2;
            x2 = _args[++i];
            f2 = _vals[i];

            // Accumulate the main part of the integral
            while (x2 <= xmax && i<size()) {
                double step = static_cast<const T*>(this)->integ_step(x1,f1,x2,f2,i);
                xdbg<<"integration step = "<<step<<std::endl;
                ans += step;
                xdbg<<"ans = "<<ans<<std::endl;
                x1 = x2;
                f1 = f2;
                x2 = _args[++i];
                f2 = _vals[i];
                xdbg<<"f("<<x2<<") = "<<f2<<std::endl;
            }

            // Last step also needs special handling.
            if (x1 < xmax) {
                x2 = xmax;
                f2 = interp(xmax, i);
                xdbg<<"last f("<<x2<<") = "<<f2<<std::endl;
                double step = static_cast<const T*>(this)->integ_step_last(x1,f1,x2,f2,i);
                xdbg<<"integration step = "<<step<<std::endl;
                ans += step;
                xdbg<<"ans = "<<ans<<std::endl;
            }
            dbg<<"final ans = "<<ans<<std::endl;
            return ans;
        }

        double integrateProduct(const Table::TableImpl& g, double xmin, double xmax,
                                double xfact) const override
        {
            // Here with two sets of abscissae, we never assume we have a full interval,
            // since most of the time we won't for either f or g.  This will only be a little
            // inefficient when f is spline interpolated and g was a function or happens to have
            // exactly the same x values.  Not our dominant use case, so it doesn't seem worth
            // trying to optimize for that.
            dbg<<"Start integrateProduct: "<<xmin<<" "<<xmax<<" "<<xfact<<std::endl;
            double x1 = xmin;
            double xx1 = x1 * xfact;
            int i = find(xx1);
            int j = g.find(x1);

            double f1 = interp(xx1, i);
            double g1 = g.interp(x1, j);
            double x2 = g.getArg(j);
            double xx2 = x2 * xfact;
            if (getArg(i) < xx2) {
                xx2 = getArg(i);
                x2 = xx2 / xfact;
            }
            double f2 = interp(xx2, i);
            double g2 = g.interp(x2, j);
            dbg<<"Start at x1 = "<<x1<<", f("<<xx1<<") = "<<f1<<", g("<<x1<<") = "<<g1<<std::endl;
            dbg<<"First x2 = "<<x2<<", f("<<xx2<<") = "<<f2<<", g("<<x2<<") = "<<g2<<std::endl;

            double ans = 0.;
            while (x2 < xmax) {
                double step = static_cast<const T*>(this)->integ_prod_step(
                    x1,f1,x2,f2,i,xfact,g1,g2);
                xdbg<<"integration step ("<<x1<<" .. "<<x2<<") = "<<step<<std::endl;
                ans += step;
                xdbg<<"ans = "<<ans<<std::endl;
                x1 = x2;
                f1 = f2;
                g1 = g2;
                // Either xx2 == _args[i] or x2 == g._args[j] (or both).
                assert((xx2 == getArg(i)) || (x2 == g.getArg(j)));
                if (xx2 == getArg(i)) ++i;
                else assert(xx2 < getArg(i));
                if (x2 == g.getArg(j)) ++j;
                else assert(x2 < g.getArg(j));
                x2 = g.getArg(j);
                xx2 = x2 * xfact;
                if (getArg(i) < xx2) {
                    xx2 = getArg(i);
                    x2 = xx2 / xfact;
                }
                f2 = interp(xx2, i);
                g2 = g.interp(x2, j);
            }

            // Last step needs special handling.
            x2 = xmax;
            xx2 = x2 * xfact;
            f2 = interp(xx2, i);
            g2 = g.interp(x2, j);
            double step = static_cast<const T*>(this)->integ_prod_step(x1,f1,x2,f2,i,xfact,g1,g2);
            xdbg<<"integration step ("<<x1<<" .. "<<x2<<") = "<<step<<std::endl;
            ans += step;
            dbg<<"final ans = "<<ans<<std::endl;
            return ans;
        }

        // Many of the cases don't need any special handling for first/last steps.
        // So for those, just use the regular step.  Only override if necessary.
        double integ_step(double x1, double f1, double x2, double f2, int i) const {
            return static_cast<const T*>(this)->integ_step(x1,f1,x2,f2,i);
        }
        double integ_step_first(double x1, double f1, double x2, double f2, int i) const {
            return integ_step(x1,f1,x2,f2,i);
        }
        double integ_step_last(double x1, double f1, double x2, double f2, int i) const {
            return integ_step(x1,f1,x2,f2,i);
        }
        double integ_step_full(double x1, double f1, double x2, double f2, int i) const {
            return integ_step(x1,f1,x2,f2,i);
        }
        double integ_prod_step(double x1, double f1, double x2, double f2, int i, double xfact,
                               double g1, double g2) const {
            return static_cast<const T*>(this)->integ_prod_step(x1,f1,x2,f2,i,xfact,g1,g2);
        }
    };


    class TFloor : public TCRTP<TFloor> {
    public:
        using TCRTP<TFloor>::TCRTP;

        double _interp(double a, int i) const {
            // On entry, it is only guaranteed that _args[i-1] <= a <= _args[i].
            // Normally those ='s are ok, but for floor and ceil we make the extra
            // check to see if we should choose the opposite bound.
            if (a == _args[i]) i++;
            return _vals[i-1];
        }
        double integ_step(double x1, double f1, double x2, double f2, int i) const {
            return f1 * (x2-x1);
        }
        // No special handling needed for first or last.

        double integ_prod_step(double x1, double f1, double x2, double f2, int i, double xfact,
                               double g1, double g2) const {
            return 0.5 * f1 * (g1+g2) * (x2-x1);
        }
    };


    class TCeil : public TCRTP<TCeil> {
    public:
        using TCRTP<TCeil>::TCRTP;

        double _interp(double a, int i) const {
            if (a == _args[i-1]) i--;
            return _vals[i];
        }
        double integ_step(double x1, double f1, double x2, double f2, int i) const {
            return f2 * (x2-x1);
        }
        // No special handling needed for first or last.

        double integ_prod_step(double x1, double f1, double x2, double f2, int i, double xfact,
                               double g1, double g2) const {
            return 0.5 * f2 * (g1+g2) * (x2-x1);
        }
    };


    class TNearest : public TCRTP<TNearest> {
    public:
        using TCRTP<TNearest>::TCRTP;

        double _interp(double a, int i) const {
            if ((a - _args[i-1]) < (_args[i] - a)) i--;
            return _vals[i];
        }
        double integ_step(double x1, double f1, double x2, double f2, int i) const {
            return 0.5 * (f1+f2) * (x2-x1);
        }
        double integ_step_first(double x1, double f1, double x2, double f2, int i) const {
            double x0 = _args[i-1];
            double xm = 0.5 * (x0+x2);
            if (x1 >= xm) {
                return f2 * (x2-x1);
            } else {
                return f1 * (xm-x1) + f2 * (x2-xm);
            }
        }
        double integ_step_last(double x1, double f1, double x2, double f2, int i) const {
            double x3 = _args[i];
            double xm = 0.5 * (x1+x3);
            if (x2 <= xm) {
                return f1 * (x2-x1);
            } else {
                return f1 * (xm-x1) + f2 * (x2-xm);
            }
        }
        double integ_step_full(double x1, double f1, double x2, double f2, int i) const {
            double x0 = _args[i-1];
            double x3 = _args[i];
            double xm = 0.5 * (x0+x3);
            if (x2 <= xm) {
                return f1 * (x2-x1);
            } else if (x1 >= xm) {
                return f2 * (x2-x1);
            } else {
                return f1 * (xm-x1) + f2 * (x2-xm);
            }
        }

        double integ_prod_step(double x1, double f1, double x2, double f2, int i, double xfact,
                               double g1, double g2) const {
            double x0 = _args[i-1] / xfact;
            double x3 = _args[i] / xfact;
            double xm = 0.5 * (x0+x3);
            if (x2 <= xm) {
                return 0.5 * f1 * (g1+g2) * (x2-x1);
            } else if (x1 >= xm) {
                return 0.5 * f2 * (g1+g2) * (x2-x1);
            } else {
                double gm = (g1*(x2-xm) + g2*(xm-x1)) / (x2-x1);
                return 0.5 * f1 * (g1+gm) * (xm-x1) + 0.5 * f2 * (gm+g2) * (x2-xm);
            }
        }
    };


    class TLinear : public TCRTP<TLinear> {
    public:
        using TCRTP<TLinear>::TCRTP;

        double _interp(double a, int i) const {
            double ax = (_args[i] - a) / (_args[i] - _args[i-1]);
            double bx = 1.0 - ax;
            return _vals[i]*bx + _vals[i-1]*ax;
        }
        double integ_step(double x1, double f1, double x2, double f2, int i) const {
            return 0.5 * (f1+f2) * (x2-x1);
        }
        // No special handling needed for first or last.

        double integ_prod_step(double x1, double f1, double x2, double f2, int i, double xfact,
                               double g1, double g2) const {
            return (1./6.) * (x2-x1) * (f1*(2.*g1 + g2) + f2*(g1 + 2.*g2));
        }
    };


    class TSpline : public TCRTP<TSpline> {
    public:
        TSpline(const double* args, const double* vals, int N) :
            TCRTP<TSpline>::TCRTP(args, vals, N)
        {
            setupSpline();
        }

        double _interp(double a, int i) const {
#if 0
            // Direct calculation saved for comparison:
            double h = _args[i] - _args[i-1];
            double aa = (_args[i] - a)/h;
            double bb = 1. - aa;
            return aa*_vals[i-1] +bb*_vals[i] +
                ((aa*aa*aa-aa)*_y2[i-1]+(bb*bb*bb-bb)*_y2[i]) *
                (h*h)/6.0;
#else
            // Factor out h factors, so only need 1 division by h.
            // Also, use the fact that bb = h-aa to simplify the calculation.

            double h = _args[i] - _args[i-1];
            double aa = (_args[i] - a);
            double bb = h-aa;
            return ( aa*_vals[i-1] + bb*_vals[i] -
                     (1./6.) * aa * bb * ( (aa+h)*_y2[i-1] +
                                           (bb+h)*_y2[i]) ) / h;
#endif
        }
        double integ_step(double x1, double f1, double x2, double f2, int i) const {
            // It turns out that the integral over the spline is close to the same as
            // the trapezoid rule.  There is a small correction bit for the cubic part.
            double h = x2-x1;
            double h3 = h*h*h;
            return 0.5 * (f1+f2) * h - (1./24.) * (_y2[i-1]+_y2[i]) * h3;
        }
        double integ_step_first(double x1, double f1, double x2, double f2, int i) const {
            double h = x2-x1;
            double h3 = h*h*h;
            double x0 = _args[i-1];
            double z2 = x1+x2-2*x0;
            return 0.5 * (f1+f2) * h - (1./24.) * (_y2[i-1]*h + _y2[i]*z2) * h3 / (x2-x0);
        }
        double integ_step_last(double x1, double f1, double x2, double f2, int i) const {
            double h = x2-x1;
            double h3 = h*h*h;
            double x3 = _args[i];
            double z1 = 2*x3-x1-x2;
            return 0.5 * (f1+f2) * h - (1./24.) * (_y2[i-1]*z1 + _y2[i]*h) * h3 / (x3-x1);
        }
        double integ_step_full(double x1, double f1, double x2, double f2, int i) const {
            // When the endpoints aren't the tabulated points, the trapozoid part stays the same,
            // but there is a slight adjustment to the correction term.
            double h = x2-x1;
            double h3 = h*h*h;
            double x0 = _args[i-1];
            double x3 = _args[i];
            double z1 = 2*x3-x1-x2;
            double z2 = x1+x2-2*x0;
            return 0.5 * (f1+f2) * h - (1./24.) * (_y2[i-1]*z1 + _y2[i]*z2) * h3 / (x3-x0);
        }

        double integ_prod_step(double x1, double f1, double x2, double f2, int i, double xfact,
                               double g1, double g2) const {
            double h = x2-x1;
            double h3 = h*h*h;
            double x0 = _args[i-1] / xfact;
            double x3 = _args[i] / xfact;
            double z1 = x1*g2 + x2*g1 + (15*x3 - 8*(x1+x2)) * (g1+g2);
            double z2 = x1*g1 + x2*g2 + (7*(x1+x2) - 15*x0) * (g1+g2);

            double step = (1./6.) * h * (f1*(2.*g1 + g2) + f2*(g1 + 2.*g2));
            step -= (1./360.) * (_y2[i-1] * z1 + _y2[i] * z2) * h3 * xfact * xfact / (x3-x0);
            return step;
        }

    private:
        std::vector<double> _y2;
        void setupSpline();
    };

    void TSpline::setupSpline()
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
        _y2.resize(_n);
        // End points 2nd-derivatives zero for natural cubic spline
        _y2[0] = 0.;
        _y2[_n-1] = 0.;
        assert(_n >= 2);
        if (_n == 2) {
            // Nothing to do if only 2 points.
        } else if (_n == 3) {
            // For 3 points second derivative at i=1 is simple

            _y2[1] = 3.*((_vals[2] - _vals[1]) / (_args[2] - _args[1]) -
                        (_vals[1] - _vals[0]) / (_args[1] - _args[0])) / (_args[2] - _args[0]);

        } else {
            // For 4 or more points we use the TMV symmetric tridiagonal matrix solver

#ifdef USE_TMV
            tmv::SymBandMatrix<double> M(_n-2, 1);
            for (int i=1; i<=_n-3; i++){
                M(i, i-1) = _args[i+1] - _args[i];
            }
            tmv::Vector<double> rhs(_n-2);
            for (int i=1; i<=_n-2; i++){
                M(i-1, i-1) = 2. * (_args[i+1] - _args[i-1]);
                rhs(i-1) = 6. * ( (_vals[i+1] - _vals[i]) / (_args[i+1] - _args[i]) -
                                  (_vals[i] - _vals[i-1]) / (_args[i] - _args[i-1]) );
            }
            tmv::Vector<double> solution(_n-2);
            solution = rhs / M;   // solve the tridiagonal system of equations
            for (int i=1; i<=_n-2; i++) {
                _y2[i] = solution[i-1];
            }
#else
            // Eigen doesn't have a BandMatrix class (at least not one that is functional)
            // But in this case, the band matrix is so simple and stable (diagonal dominant)
            // that we can just use the Thomas algorithm to solve it directly.
            // https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
            std::vector<double> c(_n-3);  // Just need a single temporary vector.
            for (int i=1; i<=_n-2; i++) {
                _y2[i] = 6. * ( (_vals[i+1] - _vals[i]) / (_args[i+1] - _args[i]) -
                                (_vals[i] - _vals[i-1]) / (_args[i] - _args[i-1]) );
            }
            double bb = 2. * (_args[2] - _args[0]);
            for (int i=1; i<=_n-2; ++i) {
                _y2[i] /= bb;
                if (i == _n-2) break;
                double a = _args[i+1] - _args[i];
                c[i-1] = a;
                c[i-1] /= bb;
                bb = 2. * (_args[i+2] - _args[i]);
                bb -= a * c[i-1];
                _y2[i+1] -= a * _y2[i];
            }
            for (int i=_n-3; i>0; --i) {
                _y2[i] -= c[i-1] * _y2[i+1];
            }
#endif
        }
    }

    class TGSInterpolant : public TCRTP<TGSInterpolant> {
    public:
        TGSInterpolant(const double* args, const double* vals, int N, const Interpolant* gsinterp) :
            TCRTP<TGSInterpolant>::TCRTP(args, vals, N), _gsinterp(gsinterp) {}

        double _interp(double a, int i) const {
            double dagrid = _args[i] - _args[i-1];
            double da = (a - _args[i-1])/dagrid;

            int iaMin, iaMax;
            if (_gsinterp->isExactAtNodes()) {
                if (std::abs(da) < 10.*std::numeric_limits<double>::epsilon()) {
                    iaMin = iaMax = i-1;
                } else if (std::abs(da-1.) < 10.*std::numeric_limits<double>::epsilon()) {
                    iaMin = iaMax = i;
                } else {
                    iaMin = i-1 + int(std::ceil(da-_gsinterp->xrange()));
                    iaMax = i-1 + int(std::floor(da+_gsinterp->xrange()));
                }
            } else {
                iaMin = i-1 + int(std::ceil(da-_gsinterp->xrange()));
                iaMax = i-1 + int(std::floor(da+_gsinterp->xrange()));
            }
            iaMin = std::max(iaMin, 0);
            iaMax = std::min(iaMax, _n-1);
            if (iaMin > iaMax) return 0.0;
            double sum = 0.0;
            for(int ia=iaMin; ia<=iaMax; ia++) {
                sum += _vals[ia] * _gsinterp->xval(i-1+da-ia);
            }
            return sum;
        }
        double integ_step(double x1, double f1, double x2, double f2, int i) const {
            throw std::runtime_error("integration not implemented for gsinterp Tables");
        }
        double integ_prod_step(double x1, double f1, double x2, double f2, int i, double xfact,
                               double g1, double g2) const {
            throw std::runtime_error("integration not implemented for gsinterp Tables");
        }

    private:
        const Interpolant* _gsinterp;
    };


    // Table

    Table::Table(const double* args, const double* vals, int N, Table::interpolant in) {
        _makeImpl(args, vals, N, in);
    }

    Table::Table(const double* args, const double* vals, int N, const Interpolant* gsinterp) {
        _makeImpl(args, vals, N, gsinterp);
    }

    void Table::_makeImpl(const double* args, const double* vals, int N,
                          Table::interpolant in) {
        switch(in) {
            case floor:
                _pimpl.reset(new TFloor(args, vals, N));
                break;
            case ceil:
                _pimpl.reset(new TCeil(args, vals, N));
                break;
            case nearest:
                _pimpl.reset(new TNearest(args, vals, N));
                break;
            case linear:
                _pimpl.reset(new TLinear(args, vals, N));
                break;
            case spline:
                _pimpl.reset(new TSpline(args, vals, N));
                break;
            default:
                throw std::runtime_error("invalid interpolation method");
        }
    }

    void Table::_makeImpl(const double* args, const double* vals, int N,
                          const Interpolant* gsinterp) {
        _pimpl.reset(new TGSInterpolant(args, vals, N, gsinterp));
    }

    double Table::argMin() const
    { return _pimpl->argMin(); }

    double Table::argMax() const
    { return _pimpl->argMax(); }

    size_t Table::size() const
    { return _pimpl->size(); }

    //lookup and interpolate function value.
    double Table::operator()(double a) const
    {
        if (a<argMin() || a>argMax()) return 0.;
        else return _pimpl->lookup(a);
    }

    //lookup and interpolate function value.
    double Table::lookup(double a) const
    { return _pimpl->lookup(a); }

    //lookup and interpolate an array of function values.
    void Table::interpMany(const double* argvec, double* valvec, int N) const
    { _pimpl->interpMany(argvec, valvec, N); }

    double Table::integrate(double xmin, double xmax) const
    { return _pimpl->integrate(xmin, xmax); }

    double Table::integrateProduct(const Table& g, double xmin, double xmax, double xfact) const
    { return _pimpl->integrateProduct(*g._pimpl, xmin, xmax, xfact); }

    void TableBuilder::finalize()
    {
        if (_in == Table::gsinterp)
            _makeImpl(&_xvec[0], &_fvec[0], _xvec.size(), _gsinterp);
        else
            _makeImpl(&_xvec[0], &_fvec[0], _xvec.size(), _in);
        _final = true;
    }

    // The hierarchy for Table2DImpl looks like:
    // Table2DImpl <- ABC
    // T2DCRTP<T> : Table2DImpl <- curiously recurring template pattern
    // T2DLinearInterp : T2DCRTP<T2DLinearInterp>
    // ... similar, Floor, Ceil, Nearest, Spline
    // T2DGSInterpolant<interpolant> : T2DCRTP<T2DGSInterpolant<interpolant>> <- Use Interpolant

    class Table2D::Table2DImpl {
    public:
        Table2DImpl(const double* xargs, const double* yargs, const double* vals, int Nx, int Ny) :
            _xargs(xargs, Nx), _yargs(yargs, Ny), _vals(vals), _nx(Nx), _ny(Ny) {}

        virtual double lookup(double x, double y) const = 0;
        virtual void interpMany(const double* xvec, const double* yvec, double* valvec,
                                int N) const = 0;
        virtual void interpGrid(const double* xvec, const double* yvec, double* valvec,
                                int Nx, int Ny) const = 0;
        virtual void gradient(double x, double y, double& dfdx, double& dfdy) const = 0;
        virtual void gradientMany(const double* xvec, const double* yvec,
                                  double* dfdxvec, double* dfdyvec, int N) const = 0;
        virtual void gradientGrid(const double* xvec, const double* yvec,
                                  double* dfdxvec, double* dfdyvec, int Nx, int Ny) const = 0;
        virtual ~Table2DImpl() {}
    protected:
        const ArgVec _xargs;
        const ArgVec _yargs;
        const double* _vals;
        const int _nx;
        const int _ny;
    };


    template<class T>
    class T2DCRTP : public Table2D::Table2DImpl {
    public:
        using Table2D::Table2DImpl::Table2DImpl;

        double lookup(double x, double y) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            return static_cast<const T*>(this)->interp(x, y, i, j);
        }

        void interpMany(const double* xvec, const double* yvec, double* valvec, int N) const {
            std::vector<int> xindices(N);
            std::vector<int> yindices(N);
            _xargs.upperIndexMany(xvec, xindices.data(), N);
            _yargs.upperIndexMany(yvec, yindices.data(), N);

            for (int k=0; k<N; k++) {
                valvec[k] = static_cast<const T*>(this)->interp(
                    xvec[k], yvec[k], xindices[k], yindices[k]
                );
            }
        }

        void interpGrid(const double* xvec, const double* yvec, double* valvec, int Nx, int Ny) const {
            std::vector<int> xindices(Nx);
            std::vector<int> yindices(Ny);
            _xargs.upperIndexMany(xvec, xindices.data(), Nx);
            _yargs.upperIndexMany(yvec, yindices.data(), Ny);

            for (int ky=0, k=0; ky<Ny; ky++) {
                for (int kx=0; kx<Nx; kx++, k++) {
                    valvec[k] = static_cast<const T*>(this)->interp(
                        xvec[kx], yvec[ky], xindices[kx], yindices[ky]
                    );
                }
            }
        }

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            static_cast<const T*>(this)->grad(x, y, i, j, dfdx, dfdy);
        }

        void gradientMany(const double* xvec, const double* yvec,
                          double* dfdxvec, double* dfdyvec, int N) const {
            std::vector<int> xindices(N);
            std::vector<int> yindices(N);
            _xargs.upperIndexMany(xvec, xindices.data(), N);
            _yargs.upperIndexMany(yvec, yindices.data(), N);

            for (int k=0; k<N; k++) {
                static_cast<const T*>(this)->grad(
                    xvec[k], yvec[k], xindices[k], yindices[k], dfdxvec[k], dfdyvec[k]
                );
            }
        }

        void gradientGrid(const double* xvec, const double* yvec,
                          double* dfdxvec, double* dfdyvec, int Nx, int Ny) const {
            std::vector<int> xindices(Nx);
            std::vector<int> yindices(Ny);
            _xargs.upperIndexMany(xvec, xindices.data(), Nx);
            _yargs.upperIndexMany(yvec, yindices.data(), Ny);

            for (int ky=0, k=0; ky<Ny; ky++) {
                for (int kx=0; kx<Nx; kx++, k++) {
                    static_cast<const T*>(this)->grad(
                        xvec[kx], yvec[ky], xindices[kx], yindices[ky], dfdxvec[k], dfdyvec[k]
                    );
                }
            }
        }

    };


    class T2DFloor : public T2DCRTP<T2DFloor> {
    public:
        using T2DCRTP::T2DCRTP;

        double interp(double x, double y, int i, int j) const {
            // From upperIndex, it is only guaranteed that _xargs[i-1] <= x <= _xargs[i] (and similarly y).
            // Normally those ='s are ok, but for floor and ceil we make the extra
            // check to see if we should choose the opposite bound.
            if (x == _xargs[i]) i++;
            if (y == _yargs[j]) j++;
            return _vals[(j-1)*_nx+i-1];
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            throw std::runtime_error("gradient not implemented for floor interp");
        }
    };


    class T2DCeil : public T2DCRTP<T2DCeil> {
    public:
        using T2DCRTP::T2DCRTP;

        double interp(double x, double y, int i, int j) const {
            if (x == _xargs[i-1]) i--;
            if (y == _yargs[j-1]) j--;
            return _vals[j*_nx+i];
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            throw std::runtime_error("gradient not implemented for ceil interp");
        }
    };


    class T2DNearest : public T2DCRTP<T2DNearest> {
    public:
        using T2DCRTP::T2DCRTP;

        double interp(double x, double y, int i, int j) const {
            if ((x - _xargs[i-1]) < (_xargs[i] - x)) i--;
            if ((y - _yargs[j-1]) < (_yargs[j] - y)) j--;
            return _vals[j*_nx+i];
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            throw std::runtime_error("gradient not implemented for nearest interp");
        }
    };


    class T2DLinear : public T2DCRTP<T2DLinear> {
    public:
        using T2DCRTP::T2DCRTP;

        double interp(double x, double y, int i, int j) const {
            double ax = (_xargs[i] - x) / (_xargs[i] - _xargs[i-1]);
            double ay = (_yargs[j] - y) / (_yargs[j] - _yargs[j-1]);
            double bx = 1.0 - ax;
            double by = 1.0 - ay;

            return (_vals[(j-1)*_nx+i-1] * ax * ay
                    + _vals[(j-1)*_nx+i] * bx * ay
                    + _vals[j*_nx+i-1] * ax * by
                    + _vals[j*_nx+i] * bx * by);
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            double dx = _xargs[i] - _xargs[i-1];
            double dy = _yargs[j] - _yargs[j-1];
            double f00 = _vals[(j-1)*_nx+i-1];
            double f01 = _vals[j*_nx+i-1];
            double f10 = _vals[(j-1)*_nx+i];
            double f11 = _vals[j*_nx+i];
            double ax = (_xargs[i] - x) / (_xargs[i] - _xargs[i-1]);
            double bx = 1.0 - ax;
            double ay = (_yargs[j] - y) / (_yargs[j] - _yargs[j-1]);
            double by = 1.0 - ay;
            dfdx = ( (f10-f00)*ay + (f11-f01)*by ) / dx;
            dfdy = ( (f01-f00)*ax + (f11-f10)*bx ) / dy;
        }
    };


    class T2DSpline : public T2DCRTP<T2DSpline> {
    public:
        T2DSpline(const double* xargs, const double* yargs, const double* vals, int Nx, int Ny,
                  const double* dfdx, const double* dfdy, const double* d2fdxdy) :
            T2DCRTP<T2DSpline>(xargs, yargs, vals, Nx, Ny), _dfdx(dfdx), _dfdy(dfdy), _d2fdxdy(d2fdxdy) {}

        double interp(double x, double y, int i, int j) const {
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            // Need to first interpolate the y-values and the y-derivatives in the x direction.
            double val0 = oneDSpline(dx, _vals[(j-1)*_nx+i-1], _vals[(j-1)*_nx+i],
                                     _dfdx[(j-1)*_nx+i-1]*dxgrid, _dfdx[(j-1)*_nx+i]*dxgrid);
            double val1 = oneDSpline(dx, _vals[j*_nx+i-1], _vals[j*_nx+i],
                                     _dfdx[j*_nx+i-1]*dxgrid, _dfdx[j*_nx+i]*dxgrid);
            double der0 = oneDSpline(dx, _dfdy[(j-1)*_nx+i-1], _dfdy[(j-1)*_nx+i],
                                     _d2fdxdy[(j-1)*_nx+i-1]*dxgrid, _d2fdxdy[(j-1)*_nx+i]*dxgrid);
            double der1 = oneDSpline(dx, _dfdy[j*_nx+i-1], _dfdy[j*_nx+i],
                                     _d2fdxdy[j*_nx+i-1]*dxgrid, _d2fdxdy[j*_nx+i]*dxgrid);

            return oneDSpline(dy, val0, val1, der0*dygrid, der1*dygrid);
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            // xgradient;
            double val0 = oneDGrad(dx, _vals[(j-1)*_nx+i-1], _vals[(j-1)*_nx+i],
                                   _dfdx[(j-1)*_nx+i-1]*dxgrid, _dfdx[(j-1)*_nx+i]*dxgrid);
            double val1 = oneDGrad(dx, _vals[j*_nx+i-1], _vals[j*_nx+i],
                                   _dfdx[j*_nx+i-1]*dxgrid, _dfdx[j*_nx+i]*dxgrid);
            double der0 = oneDGrad(dx, _dfdy[(j-1)*_nx+i-1], _dfdy[(j-1)*_nx+i],
                                   _d2fdxdy[(j-1)*_nx+i-1]*dxgrid, _d2fdxdy[(j-1)*_nx+i]*dxgrid);
            double der1 = oneDGrad(dx, _dfdy[j*_nx+i-1], _dfdy[j*_nx+i],
                                   _d2fdxdy[j*_nx+i-1]*dxgrid, _d2fdxdy[j*_nx+i]*dxgrid);
            dfdx = oneDSpline(dy, val0, val1, der0*dygrid, der1*dygrid)/dxgrid;

            // ygradient
            val0 = oneDGrad(dy, _vals[(j-1)*_nx+i-1], _vals[j*_nx+i-1],
                            _dfdy[(j-1)*_nx+i-1]*dygrid, _dfdy[j*_nx+i-1]*dygrid);
            val1 = oneDGrad(dy, _vals[(j-1)*_nx+i], _vals[j*_nx+i],
                            _dfdy[(j-1)*_nx+i]*dygrid, _dfdy[j*_nx+i]*dygrid);
            der0 = oneDGrad(dy, _dfdx[(j-1)*_nx+i-1], _dfdx[j*_nx+i-1],
                            _d2fdxdy[(j-1)*_nx+i-1]*dygrid, _d2fdxdy[j*_nx+i-1]*dygrid);
            der1 = oneDGrad(dy, _dfdx[(j-1)*_nx+i], _dfdx[j*_nx+i],
                            _d2fdxdy[(j-1)*_nx+i]*dygrid, _d2fdxdy[j*_nx+i]*dygrid);
            dfdy = oneDSpline(dx, val0, val1, der0*dxgrid, der1*dxgrid)/dygrid;
        }
    private:

        double oneDSpline(double x, double val0, double val1, double der0, double der1) const {
            // assuming that x is between 0 and 1, val0 and val1 are the values at
            // 0 and 1, and der0 and der1 are the derivatives at 0 and 1.
            // Let f(x) = ax^3 + bx^2 + cx + d
            //  => f'(x) = 3ax^2 + 2bx + c
            //  => f(0) = d
            //     f(1) = a + b + c + d
            //     f'(0) = c
            //     f'(1) = 3a + 2b + c
            //  Solve above for a,b,c,d to obtain below:
            double a = 2*(val0-val1) + der0 + der1;
            double b = 3*(val1-val0) - 2*der0 - der1;
            double c = der0;
            double d = val0;

            return d + x*(c + x*(b + x*a));
        }

        double oneDGrad(double x, double val0, double val1, double der0, double der1) const {
            double a = 2*(val0-val1) + der0 + der1;
            double b = 3*(val1-val0) - 2*der0 - der1;
            double c = der0;
            return c + x*(2*b + x*3*a);
        }

        const double* _dfdx;
        const double* _dfdy;
        const double* _d2fdxdy;
    };


    class T2DGSInterpolant : public T2DCRTP<T2DGSInterpolant> {
    public:
        T2DGSInterpolant(const double* xargs, const double* yargs, const double* vals, int Nx, int Ny,
                         const Interpolant* gsinterp) :
            T2DCRTP<T2DGSInterpolant>(xargs, yargs, vals, Nx, Ny), _gsinterp(gsinterp) {}

        double interp(double x, double y, int i, int j) const {
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            // Stealing from XTable::interpolate
            int ixMin, ixMax, iyMin, iyMax;
            if (_gsinterp->isExactAtNodes()
                && std::abs(dx) < 10.*std::numeric_limits<double>::epsilon()) {
                    ixMin = ixMax = i-1;
            } else {
                ixMin = i-1 + int(std::ceil(dx-_gsinterp->xrange()));
                ixMax = i-1 + int(std::floor(dx+_gsinterp->xrange()));
            }
            ixMin = std::max(ixMin, 0);
            ixMax = std::min(ixMax, _nx-1);
            if (ixMin > ixMax) return 0.0;
            if (_gsinterp->isExactAtNodes()
                && std::abs(dy) < 10.*std::numeric_limits<double>::epsilon()) {
                    iyMin = iyMax = j-1;
            } else {
                iyMin = j-1 + int(std::ceil(dy-_gsinterp->xrange()));
                iyMax = j-1 + int(std::floor(dy+_gsinterp->xrange()));
            }
            iyMin = std::max(iyMin, 0);
            iyMax = std::min(iyMax, _ny-1);
            if (iyMin > iyMax) return 0.0;

            // The interpolated value is the sum of
            //      I(x'-x) * I(y'-y) * vals(x',y')
            // where I is the interpolant function, summed from ixMin..ixMax and iyMin..iyMax.
            // The one trick we use to speed this up is to recognize that successive calls to this
            // function tend to have the same value for y.  (for(x=...) is typically an inner loop.)
            // So we cache the sums in the y direction, many of which we will need again for
            // the next call.

            double sum = 0.0;
            if (y != _cacheY) {
                _clearCache();
                _cacheY = y;
            } else if (ixMax == ixMin && !_cache.empty()) {
                // Special case for interpolation on a single ix value:
                // See if we already have this row in cache:
                int index = ixMin - _cacheStartX;
                // if (index < 0) index += _N;  // JM: I don't understand this line...
                if (index >= 0 && index < int(_cache.size()))
                    // We have it!
                    return _cache[index];
                else
                    // Desired row not in cache - kill cache, continue as normal.
                    // (But don't clear ywt, since that's still good.)
                    _cache.clear();
            }
            // Build y factors for interpolant
            int ny = iyMax - iyMin + 1;
            // This is also cached if possible.  It gets cleared with y!=cacheY above.
            if (_ywt.empty()) {
                _ywt.resize(ny);
                for (int ii=0; ii<ny; ii++) {
                    _ywt[ii] = _gsinterp->xval(j-1+dy-(ii+iyMin));
                }
            } else {
                assert(int(_ywt.size()) == ny);
            }
            // cache always holds sequential x values (no wrap).  Throw away
            // elements until we get to the one we need first
            std::deque<double>::iterator nextSaved = _cache.begin();
            while (nextSaved != _cache.end() && _cacheStartX != ixMin) {
                _cache.pop_front();
                ++_cacheStartX;
                nextSaved = _cache.begin();
            }
            for (int ix=ixMin; ix<=ixMax; ix++) {
                double sumx = 0.0;
                if (nextSaved != _cache.end()) {
                    // This sumx is cached
                    sumx = *nextSaved;
                    ++nextSaved;
                } else {
                    // Need to compute a new sumx
                    const double* dptr = &_vals[iyMin*_nx+ix];
                    std::vector<double>::const_iterator ywt_it = _ywt.begin();
                    int count = ny;
                    for(; count; --count, dptr+=_nx) sumx += (*ywt_it++) * (*dptr);
                    xassert(ywt_it == _ywt.end());
                    // Add to back of cache
                    if (_cache.empty()) _cacheStartX = ix;
                    _cache.push_back(sumx);
                    nextSaved = _cache.end();
                }
                sum += sumx * _gsinterp->xval(i-1+dx-ix);
            }
            return sum;
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            throw std::runtime_error("gradient not implemented for Interp interp");
        }

    private:
        const Interpolant* _gsinterp;

        void _clearCache() const {
            _cache.clear();
            _ywt.clear();
        }

        mutable std::deque<double> _cache;
        mutable std::vector<double> _ywt;
        mutable double _cacheY;
        mutable int _cacheStartX;
    };


    Table2D::Table2D(const double* xargs, const double* yargs, const double* vals,
                     int Nx, int Ny, interpolant in) :
        _pimpl(_makeImpl(xargs, yargs, vals, Nx, Ny, in)) {}


    Table2D::Table2D(const double* xargs, const double* yargs, const double* vals,
                     int Nx, int Ny,
                     const double* dfdx, const double* dfdy, const double* d2fdxdy) :
        _pimpl(_makeImpl(xargs, yargs, vals, Nx, Ny, dfdx, dfdy, d2fdxdy)) {}


    Table2D::Table2D(const double* xargs, const double* yargs, const double* vals,
                     int Nx, int Ny, const Interpolant* gsinterp) :
        _pimpl(_makeImpl(xargs, yargs, vals, Nx, Ny, gsinterp)) {}


    std::shared_ptr<Table2D::Table2DImpl> Table2D::_makeImpl(
            const double* xargs, const double* yargs, const double* vals,
            int Nx, int Ny, interpolant in)
    {
        switch(in) {
            case floor:
                return std::make_shared<T2DFloor>(xargs, yargs, vals, Nx, Ny);
            case ceil:
                return std::make_shared<T2DCeil>(xargs, yargs, vals, Nx, Ny);
            case nearest:
                return std::make_shared<T2DNearest>(xargs, yargs, vals, Nx, Ny);
            case linear:
                return std::make_shared<T2DLinear>(xargs, yargs, vals, Nx, Ny);
            default:
                throw std::runtime_error("invalid interpolation method");
        }
    }

    std::shared_ptr<Table2D::Table2DImpl> Table2D::_makeImpl(
            const double* xargs, const double* yargs, const double* vals,
            int Nx, int Ny,
            const double* dfdx, const double* dfdy, const double* d2fdxdy)
    {
            return std::make_shared<T2DSpline>(xargs, yargs, vals, Nx, Ny, dfdx, dfdy, d2fdxdy);
    }

    std::shared_ptr<Table2D::Table2DImpl> Table2D::_makeImpl(
            const double* xargs, const double* yargs, const double* vals,
            int Nx, int Ny, const Interpolant* gsinterp)
    {
            return std::make_shared<T2DGSInterpolant>(xargs, yargs, vals, Nx, Ny, gsinterp);
    }

    double Table2D::lookup(double x, double y) const {
        return _pimpl->lookup(x, y);
    }

    void Table2D::interpMany(const double* xvec, const double* yvec, double* valvec, int N) const {
        _pimpl->interpMany(xvec, yvec, valvec, N);
    }

    void Table2D::interpGrid(const double* xvec, const double* yvec, double* valvec, int Nx, int Ny) const {
        _pimpl->interpGrid(xvec, yvec, valvec, Nx, Ny);
    }

    /// Estimate df/dx, df/dy at a single location
    void Table2D::gradient(double x, double y, double& dfdx, double& dfdy) const {
        _pimpl->gradient(x, y, dfdx, dfdy);
    }

    /// Estimate many df/dx and df/dy values
    void Table2D::gradientMany(const double* xvec, const double* yvec,
                               double* dfdxvec, double* dfdyvec, int N) const {
        _pimpl->gradientMany(xvec, yvec, dfdxvec, dfdyvec, N);
    }

    void Table2D::gradientGrid(const double* xvec, const double* yvec,
                               double* dfdxvec, double* dfdyvec, int Nx, int Ny) const {
        _pimpl->gradientGrid(xvec, yvec, dfdxvec, dfdyvec, Nx, Ny);
    }

    void WrapArrayToPeriod(double* x, int n, double x0, double period)
    {
#ifdef __SSE2__
        for (; n && !IsAligned(x); --n, ++x)
            *x -= period * floor((*x-x0)/period);

        int n2 = n>>1;
        int na = n2<<1;
        n -= na;

        if (n2) {
            __m128d xx0 = _mm_set1_pd(x0);
            __m128d xperiod = _mm_set1_pd(period);
            __m128d xzero = _mm_set1_pd(0.);
            __m128d* xx = reinterpret_cast<__m128d*>(x);
            do {
                __m128d xoffset = _mm_sub_pd(*xx, xx0);
                __m128d nperiod = _mm_div_pd(xoffset,xperiod);
                __m128d floornp = _mm_cvtepi32_pd(_mm_cvttpd_epi32(nperiod));
                __m128d shift = _mm_mul_pd(xperiod, floornp);
                __m128d neg = _mm_cmpge_pd(xoffset, xzero);
                __m128d shift2 = _mm_sub_pd(shift, _mm_andnot_pd(neg, xperiod));
                *xx = _mm_sub_pd(*xx, shift2);
                ++xx;
            } while (--n2);
        }

        if (n) {
            x += na;
            *x -= period * floor((*x-x0)/period);
        }
#else
        for (; n; --n, ++x)
            *x -= period * floor((*x-x0)/period);
#endif
    }
}
