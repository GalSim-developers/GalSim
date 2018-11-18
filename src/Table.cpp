/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
            _args(args, N), _n(N), _vals(vals) {}

        virtual double lookup(double a) const = 0;
        virtual void interpMany(const double* argvec, double* valvec, int N) const = 0;

        double argMin() const { return _args.front(); }
        double argMax() const { return _args.back(); }
        size_t size() const { return _args.size(); }

        virtual ~TableImpl() {}
    protected:
        ArgVec _args;
        const int _n;
        const double* _vals;
    };


    template<class T>
    class TCRTP : public Table::TableImpl {
    public:
        using Table::TableImpl::TableImpl;

        double lookup(double a) const override {
            int i = _args.upperIndex(a);
            return static_cast<const T*>(this)->interp(a, i);
        }

        void interpMany(const double* xvec, double* valvec, int N) const override {
            std::vector<int> indices(N);
            _args.upperIndexMany(xvec, indices.data(), N);

            for (int k=0; k<N; k++) {
                valvec[k] = static_cast<const T*>(this)->interp(xvec[k], indices[k]);
            }
        }
    };


    class TFloor : public TCRTP<TFloor> {
    public:
        using TCRTP<TFloor>::TCRTP;

        double interp(double a, int i) const {
            // On entry, it is only guaranteed that _args[i-1] <= a <= _args[i].
            // Normally those ='s are ok, but for floor and ceil we make the extra
            // check to see if we should choose the opposite bound.
            if (a == _args[i]) i++;
            return _vals[i-1];
        }
    };


    class TCeil : public TCRTP<TCeil> {
    public:
        using TCRTP<TCeil>::TCRTP;

        double interp(double a, int i) const {
            if (a == _args[i-1]) i--;
            return _vals[i];
        }
    };


    class TNearest : public TCRTP<TNearest> {
    public:
        using TCRTP<TNearest>::TCRTP;

        double interp(double a, int i) const {
            if ((a - _args[i-1]) < (_args[i] - a)) i--;
            return _vals[i];
        }
    };


    class TLinear : public TCRTP<TLinear> {
    public:
        using TCRTP<TLinear>::TCRTP;

        double interp(double a, int i) const {
            double ax = (_args[i] - a) / (_args[i] - _args[i-1]);
            double bx = 1.0 - ax;
            return _vals[i]*bx + _vals[i-1]*ax;
        }
    };


    class TSpline : public TCRTP<TSpline> {
    public:
        TSpline(const double* args, const double* vals, int N) :
            TCRTP<TSpline>::TCRTP(args, vals, N)
        {
            setupSpline();
        }

        double interp(double a, int i) const {
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
        // For 3 points second derivative at i=1 is simple
        if (_n == 3){

            _y2[1] = 3.*((_vals[2] - _vals[1]) / (_args[2] - _args[1]) -
                        (_vals[1] - _vals[0]) / (_args[1] - _args[0])) / (_args[2] - _args[0]);

        } else {  // For 4 or more points we use the TMV symmetric tridiagonal matrix solver

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

        double interp(double a, int i) const {
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
    {
        _pimpl->interpMany(argvec, valvec, N);
    }

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
            _xargs(xargs, Nx), _yargs(yargs, Ny), _vals(vals), _ny(Ny) {}

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

            for (int kx=0, k=0; kx<Nx; kx++) {
                for (int ky=0; ky<Ny; ky++, k++) {
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

            for (int kx=0, k=0; kx<Nx; kx++) {
                for (int ky=0; ky<Ny; ky++, k++) {
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
            return _vals[(i-1)*_ny+j-1];
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
            return _vals[i*_ny+j];
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
            return _vals[i*_ny+j];
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

            return (_vals[(i-1)*_ny+j-1] * ax * ay
                    + _vals[i*_ny+j-1] * bx * ay
                    + _vals[(i-1)*_ny+j] * ax * by
                    + _vals[i*_ny+j] * bx * by);
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            double dx = _xargs[i] - _xargs[i-1];
            double dy = _yargs[j] - _yargs[j-1];
            double f00 = _vals[(i-1)*_ny+j-1];
            double f01 = _vals[(i-1)*_ny+j];
            double f10 = _vals[i*_ny+j-1];
            double f11 = _vals[i*_ny+j];
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
            double val0 = oneDSpline(dx, _vals[(i-1)*_ny+j-1], _vals[i*_ny+j-1],
                                     _dfdx[(i-1)*_ny+j-1]*dxgrid, _dfdx[i*_ny+j-1]*dxgrid);
            double val1 = oneDSpline(dx, _vals[(i-1)*_ny+j], _vals[i*_ny+j],
                                     _dfdx[(i-1)*_ny+j]*dxgrid, _dfdx[i*_ny+j]*dxgrid);
            double der0 = oneDSpline(dx, _dfdy[(i-1)*_ny+j-1], _dfdy[i*_ny+j-1],
                                     _d2fdxdy[(i-1)*_ny+j-1]*dxgrid, _d2fdxdy[i*_ny+j-1]*dxgrid);
            double der1 = oneDSpline(dx, _dfdy[(i-1)*_ny+j], _dfdy[i*_ny+j],
                                     _d2fdxdy[(i-1)*_ny+j]*dxgrid, _d2fdxdy[i*_ny+j]*dxgrid);

            return oneDSpline(dy, val0, val1, der0*dygrid, der1*dygrid);
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            // xgradient;
            double val0 = oneDGrad(dx, _vals[(i-1)*_ny+j-1], _vals[i*_ny+j-1],
                                   _dfdx[(i-1)*_ny+j-1]*dxgrid, _dfdx[i*_ny+j-1]*dxgrid);
            double val1 = oneDGrad(dx, _vals[(i-1)*_ny+j], _vals[i*_ny+j],
                                   _dfdx[(i-1)*_ny+j]*dxgrid, _dfdx[i*_ny+j]*dxgrid);
            double der0 = oneDGrad(dx, _dfdy[(i-1)*_ny+j-1], _dfdy[i*_ny+j-1],
                                   _d2fdxdy[(i-1)*_ny+j-1]*dxgrid, _d2fdxdy[i*_ny+j-1]*dxgrid);
            double der1 = oneDGrad(dx, _dfdy[(i-1)*_ny+j], _dfdy[i*_ny+j],
                                   _d2fdxdy[(i-1)*_ny+j]*dxgrid, _d2fdxdy[i*_ny+j]*dxgrid);
            dfdx = oneDSpline(dy, val0, val1, der0*dygrid, der1*dygrid)/dxgrid;

            // ygradient
            val0 = oneDGrad(dy, _vals[(i-1)*_ny+j-1], _vals[(i-1)*_ny+j],
                            _dfdy[(i-1)*_ny+j-1]*dygrid, _dfdy[(i-1)*_ny+j]*dygrid);
            val1 = oneDGrad(dy, _vals[i*_ny+j-1], _vals[i*_ny+j],
                            _dfdy[i*_ny+j-1]*dygrid, _dfdy[i*_ny+j]*dygrid);
            der0 = oneDGrad(dy, _dfdx[(i-1)*_ny+j-1], _dfdx[(i-1)*_ny+j],
                            _d2fdxdy[(i-1)*_ny+j-1]*dygrid, _d2fdxdy[(i-1)*_ny+j]*dygrid);
            der1 = oneDGrad(dy, _dfdx[i*_ny+j-1], _dfdx[i*_ny+j],
                            _d2fdxdy[i*_ny+j-1]*dygrid, _d2fdxdy[i*_ny+j]*dygrid);
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
            T2DCRTP<T2DGSInterpolant>(xargs, yargs, vals, Nx, Ny), _nx(Nx), _gsinterp(*gsinterp) {}

        double interp(double x, double y, int i, int j) const {
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            // Stealing from XTable::interpolate
            int ixMin, ixMax, iyMin, iyMax;
            if (_gsinterp.isExactAtNodes()
                && std::abs(dx) < 10.*std::numeric_limits<double>::epsilon()) {
                    ixMin = ixMax = i-1;
            } else {
                ixMin = i-1 + int(std::ceil(dx-_gsinterp.xrange()));
                ixMax = i-1 + int(std::floor(dx+_gsinterp.xrange()));
            }
            ixMin = std::max(ixMin, 0);
            ixMax = std::min(ixMax, _nx-1);
            if (ixMin > ixMax) return 0.0;
            if (_gsinterp.isExactAtNodes()
                && std::abs(dy) < 10.*std::numeric_limits<double>::epsilon()) {
                    iyMin = iyMax = j-1;
            } else {
                iyMin = j-1 + int(std::ceil(dy-_gsinterp.xrange()));
                iyMax = j-1 + int(std::floor(dy+_gsinterp.xrange()));
            }
            iyMin = std::max(iyMin, 0);
            iyMax = std::min(iyMax, _ny-1);
            if (iyMin > iyMax) return 0.0;

            double sum = 0.0;
            const InterpolantXY* ixy = dynamic_cast<const InterpolantXY*> (&_gsinterp);
            if (ixy) {
                // Interpolant is seperable
                // We have the opportunity to speed up the calculation by
                // re-using the sums over rows.  So we will keep a
                // cache of them.
                if (y != _cacheY || ixy != _cacheInterp) {
                    _clearCache();
                    _cacheY = y;
                    _cacheInterp = ixy;
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
                        _ywt[ii] = ixy->xval1d(j-1+dy-(ii+iyMin));
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
                        // This row is cached
                        sumx = *nextSaved;
                        ++nextSaved;
                    } else {
                        // Need to compute a new row's sum
                        const double* dptr = &_vals[ix*_ny+iyMin];
                        std::vector<double>::const_iterator ywt_it = _ywt.begin();
                        int count = ny;
                        for(; count; --count) sumx += (*ywt_it++) * (*dptr++);
                        xassert(ywt_it == _ywt.end());
                        // Add to back of cache
                        if (_cache.empty()) _cacheStartX = ix;
                        _cache.push_back(sumx);
                        nextSaved = _cache.end();
                    }
                    sum += sumx * ixy->xval1d(i-1+dx-ix);
                }
            } else {
                for(int iy=iyMin; iy<=iyMax; iy++) {
                    for(int ix=ixMin; ix<=ixMax; ix++) {
                        sum += _vals[ix*_ny+iy] * _gsinterp.xval(i-1+dx-ix, j-1+dy-iy);
                    }
                }
            }
            return sum;
        }

        void grad(double x, double y, int i, int j, double& dfdx, double& dfdy) const {
            throw std::runtime_error("gradient not implemented for Interp interp");
        }

    private:
        const int _nx;
        const InterpolantXY _gsinterp;

        void _clearCache() const {
            _cache.clear();
            _ywt.clear();
        }

        mutable std::deque<double> _cache;
        mutable std::vector<double> _ywt;
        mutable double _cacheY;
        mutable int _cacheStartX;
        mutable const InterpolantXY* _cacheInterp;
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
