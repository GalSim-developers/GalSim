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

#ifdef USE_TMV
#include "TMV.h"
#include "TMV_SymBand.h"
#endif

#include "Table.h"

namespace galsim {

    // ArgVec
    // A class to represent an argument vector for a Table or Table2D.
    class ArgVec
    {
    public:
        ArgVec(const double* args, int n);

        int upperIndex(double a) const;

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
            if (i >= _n) --i; // in case of rounding error
            if (i == 0) ++i;
            // check if we need to move ahead or back one step due to rounding errors
            while (a > _vec[i]) ++i;
            while (a < _vec[i-1]) --i;
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

    // TableImpl
    class Table::TableImpl
    {
    public:
        TableImpl(const double* args, const double* vals, int N, Table::interpolant in);

        double argMin() const { return _args.front(); }
        double argMax() const { return _args.back(); }
        size_t size() const { return _n; }

        double lookup(double a) const;

    private:
        Table::interpolant _iType;
        ArgVec _args;
        const double* _vals;
        int _n;
        std::vector<double> _y2;

        typedef double (TableImpl::*MemFn)(double x, int i) const;
        MemFn interpolate;
        double linearInterpolate(double a, int i) const;
        double floorInterpolate(double a, int i) const;
        double ceilInterpolate(double a, int i) const;
        double nearestInterpolate(double a, int i) const;
        double splineInterpolate(double a, int i) const;

        void setupSpline();
    };

    Table::TableImpl::TableImpl(const double* args, const double* vals, int N,
                                Table::interpolant in):
        _iType(in), _args(args, N), _vals(vals), _n(N)
    {
        switch (_iType) {
          case Table::linear:
               interpolate = &TableImpl::linearInterpolate;
               break;
          case Table::floor:
               interpolate = &TableImpl::floorInterpolate;
               break;
          case Table::ceil:
               interpolate = &TableImpl::ceilInterpolate;
               break;
          case Table::nearest:
               interpolate = &TableImpl::nearestInterpolate;
               break;
          case Table::spline:
               interpolate = &TableImpl::splineInterpolate;
               break;
          default:
               throw std::runtime_error("invalid interpolation method");
        }
        if (_iType == Table::spline) setupSpline();
    }

    void Table::TableImpl::setupSpline()
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

    //lookup and interpolate function value.
    double Table::TableImpl::lookup(double a) const
    {
        dbg<<"lookup "<<a<<std::endl;
        int i = _args.upperIndex(a);
        dbg<<"i = "<<i<<std::endl;
        double val = (this->*interpolate)(a, i);
        dbg<<"val = "<<val<<std::endl;
        return val;
    }

    double Table::TableImpl::linearInterpolate(double a, int i) const
    {
        double ax = (_args[i] - a) / (_args[i] - _args[i-1]);
        double bx = 1.0 - ax;
        return _vals[i]*bx + _vals[i-1]*ax;
    }

    double Table::TableImpl::floorInterpolate(double a, int i) const
    {
        // On entry, it is only guaranteed that _args[i-1] <= a <= _args[i].
        // Normally those ='s are ok, but for floor and ceil we make the extra
        // check to see if we should choose the opposite bound.
        if (a == _args[i]) i++;
        return _vals[i-1];
    }

    double Table::TableImpl::ceilInterpolate(double a, int i) const
    {
        if (a == _args[i-1]) i--;
        return _vals[i];
    }

    double Table::TableImpl::nearestInterpolate(double a, int i) const
    {
        if ((a - _args[i-1]) < (_args[i] - a)) i--;
        return _vals[i];
    }

    double Table::TableImpl::splineInterpolate(double a, int i) const
    {
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



    // Table

    Table::Table(const double* args, const double* vals, int N, Table::interpolant in)
    { _pimpl.reset(new TableImpl(args, vals, N, in)); }

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
        for (int k=0; k<N; k++) {
            valvec[k] = _pimpl->lookup(argvec[k]);
        }
    }

    void TableBuilder::finalize()
    {
        _pimpl.reset(new TableImpl(&_xvec[0], &_fvec[0], _xvec.size(), _in));
        _final = true;
    }


    class Table2D::Table2DImpl {
    public:
        virtual double lookup(double x, double y) const = 0;
        virtual void interpMany(const double* xvec, const double* yvec, double* valvec,
                                int N) const = 0;
        virtual void gradient(double x, double y, double& dfdx, double& dfdy) const = 0;
        virtual void gradientMany(const double* xvec, const double* yvec,
                                  double* dfdxvec, double* dfdyvec, int N) const = 0;
    };


    template<class T2DInterp>
    class ConcreteTable2DImpl : public Table2D::Table2DImpl {
    public:
        ConcreteTable2DImpl(
            const double* xargs, const double* yargs, const double* vals,
            int Nx, int Ny,
            const double* dfdx=nullptr, const double* dfdy=nullptr, const double* d2fdxdy=nullptr
        ) : _interp(xargs, yargs, vals, Nx, Ny, dfdx, dfdy, d2fdxdy) {}

        double lookup(double x, double y) const {
            return _interp.interp(x, y);
        }

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            _interp.gradient(x, y, dfdx, dfdy);
        }

        void interpMany(const double* xvec, const double* yvec, double* valvec, int N) const {
            for (int k=0; k<N; k++) {
                *valvec++ = _interp.interp(xvec[k], yvec[k]);
            }
        }

        void gradientMany(const double* xvec, const double* yvec,
                          double* dfdxvec, double* dfdyvec, int N) const {
            for (int k=0; k<N; k++) {
                _interp.gradient(xvec[k], yvec[k], dfdxvec[k], dfdyvec[k]);
            }
        }

    private:
        T2DInterp _interp;
    };


    class T2DInterp {
    public:
        T2DInterp(
            const double* xargs, const double* yargs, const double* vals,
            int Nx, int Ny,
            const double* dfdx=nullptr, const double* dfdy=nullptr, const double* d2fdxdy=nullptr
        ) : _xargs(xargs, Nx), _yargs(yargs, Ny), _vals(vals), _ny(Ny),
            _dfdx(dfdx), _dfdy(dfdy), _d2fdxdy(d2fdxdy) {}
    protected:
        ArgVec _xargs;
        ArgVec _yargs;
        const double* _vals;
        const int _ny;
        const double* _dfdx;
        const double* _dfdy;
        const double* _d2fdxdy;
    };


    class T2DLinearInterp : public T2DInterp {
    public:
        using T2DInterp::T2DInterp;

        double interp(double x, double y) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);

            double ax = (_xargs[i] - x) / (_xargs[i] - _xargs[i-1]);
            double ay = (_yargs[j] - y) / (_yargs[j] - _yargs[j-1]);
            double bx = 1.0 - ax;
            double by = 1.0 - ay;

            return (_vals[(i-1)*_ny+j-1] * ax * ay
                    + _vals[i*_ny+j-1] * bx * ay
                    + _vals[(i-1)*_ny+j] * ax * by
                    + _vals[i*_ny+j] * bx * by);
        }

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
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


    class T2DFloorInterp : public T2DInterp {
    public:
        using T2DInterp::T2DInterp;

        double interp(double x, double y) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            // From upperIndex, it is only guaranteed that _xargs[i-1] <= x <= _xargs[i] (and similarly y).
            // Normally those ='s are ok, but for floor and ceil we make the extra
            // check to see if we should choose the opposite bound.
            if (x == _xargs[i]) i++;
            if (y == _yargs[j]) j++;
            return _vals[(i-1)*_ny+j-1];
        }

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            throw std::runtime_error("gradient not implemented for floor interp");
        }
    };


    class T2DCeilInterp : public T2DInterp {
    public:
        using T2DInterp::T2DInterp;

        double interp(double x, double y) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            if (x == _xargs[i-1]) i--;
            if (y == _yargs[j-1]) j--;
            return _vals[i*_ny+j];
        }

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            throw std::runtime_error("gradient not implemented for ceil interp");
        }
    };


    class T2DNearestInterp : public T2DInterp {
    public:
        using T2DInterp::T2DInterp;

        double interp(double x, double y) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            if ((x - _xargs[i-1]) < (_xargs[i] - x)) i--;
            if ((y - _yargs[j-1]) < (_yargs[j] - y)) j--;
            return _vals[i*_ny+j];
        }

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            throw std::runtime_error("gradient not implemented for nearest interp");
        }
    };


    class T2DCubicInterp : public T2DInterp {
    public:
        using T2DInterp::T2DInterp;

        double oneDSpline(double x, double val0, double val1, double der0, double der1) const {
            // assuming that x is between 0 and 1, val0 and val1 are the values at
            // 0 and 1, and der0 and der1 are the derivatives at 0 and 1.

            // I'm assuming the compiler will reduce this all down...
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

        double interp(double x, double y) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
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

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
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
    };


    class T2DCubicConvolveInterp : public T2DInterp {
    public:
        using T2DInterp::T2DInterp;

        double oneDSpline(double x, double fm1, double f0, double f1, double f2) const {
            double a = -fm1 + 3*(f0-f1) + f2;
            double b = 2*fm1 - 5*f0 + 4*f1 - f2;
            double c = f1-fm1;
            double d = 2*f0;

            return 0.5*(d + x*(c + x*(b + x*a)));
        }

        double oneDGrad(double x, double fm1, double f0, double f1, double f2) const {
            double a = -fm1 + 3*(f0-f1) + f2;
            double b = 2*fm1 - 5*f0 + 4*f1 - f2;
            double c = f1-fm1;
            return 0.5*(c + x*(2*b + x*3*a));
        }

        double interp(double x, double y) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            // First interpolate in the x direction.
            double valm1 = oneDSpline(dx, _vals[(i-2)*_ny+j-2], _vals[(i-1)*_ny+j-2], _vals[(i+0)*_ny+j-2], _vals[(i+1)*_ny+j-2]);
            double val0 =  oneDSpline(dx, _vals[(i-2)*_ny+j-1], _vals[(i-1)*_ny+j-1], _vals[(i+0)*_ny+j-1], _vals[(i+1)*_ny+j-1]);
            double val1 =  oneDSpline(dx, _vals[(i-2)*_ny+j+0], _vals[(i-1)*_ny+j+0], _vals[(i+0)*_ny+j+0], _vals[(i+1)*_ny+j+0]);
            double val2 =  oneDSpline(dx, _vals[(i-2)*_ny+j+1], _vals[(i-1)*_ny+j+1], _vals[(i+0)*_ny+j+1], _vals[(i+1)*_ny+j+1]);
            return oneDSpline(dy, valm1, val0, val1, val2);
        }

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            // x-gradient
            double valm1 = oneDGrad(dx, _vals[(i-2)*_ny+j-2], _vals[(i-1)*_ny+j-2], _vals[(i+0)*_ny+j-2], _vals[(i+1)*_ny+j-2]);
            double val0 =  oneDGrad(dx, _vals[(i-2)*_ny+j-1], _vals[(i-1)*_ny+j-1], _vals[(i+0)*_ny+j-1], _vals[(i+1)*_ny+j-1]);
            double val1 =  oneDGrad(dx, _vals[(i-2)*_ny+j+0], _vals[(i-1)*_ny+j+0], _vals[(i+0)*_ny+j+0], _vals[(i+1)*_ny+j+0]);
            double val2 =  oneDGrad(dx, _vals[(i-2)*_ny+j+1], _vals[(i-1)*_ny+j+1], _vals[(i+0)*_ny+j+1], _vals[(i+1)*_ny+j+1]);
            dfdx = oneDSpline(dy, valm1, val0, val1, val2)/dxgrid;

            // y-gradient
            valm1 = oneDGrad(dy, _vals[(i-2)*_ny+j-2], _vals[(i-2)*_ny+j-1], _vals[(i-2)*_ny+j+0], _vals[(i-2)*_ny+j+1]);
            val0 =  oneDGrad(dy, _vals[(i-1)*_ny+j-2], _vals[(i-1)*_ny+j-1], _vals[(i-1)*_ny+j+0], _vals[(i-1)*_ny+j+1]);
            val1 =  oneDGrad(dy, _vals[(i+0)*_ny+j-2], _vals[(i+0)*_ny+j-1], _vals[(i+0)*_ny+j+0], _vals[(i+0)*_ny+j+1]);
            val2 =  oneDGrad(dy, _vals[(i+1)*_ny+j-2], _vals[(i+1)*_ny+j-1], _vals[(i+1)*_ny+j+0], _vals[(i+1)*_ny+j+1]);
            dfdy = oneDSpline(dx, valm1, val0, val1, val2)/dygrid;
        }
    };


    template<int a>
    class LanczosInterp : public T2DInterp {
    public:
        using T2DInterp::T2DInterp;

        double kernel(double x) const {
            if (x == 0) return 1.0;
            return a*std::sin(M_PI*x)*std::sin(M_PI*x/a) / M_PI / M_PI / x / x;
        }

        double dkernel(double x) const {
            if (x == 0) return 0.0;
            double result = M_PI*x*std::sin(M_PI*x)*std::cos(M_PI*x/a);
            result += a*std::sin(M_PI*x/a)*(M_PI*x*std::cos(M_PI*x)-2*std::sin(M_PI*x));
            result /= M_PI*M_PI*x*x*x;
            return result;
        }

        double oneDSpline(double x, const double* vals) const {
            // assume *vals corresponds to i = 1-a, and that x in [0, 1].
            double result = 0.0;
            for (int k=1-a; k<(a+1); k++) {
                result += (*vals++)*kernel(x-k);
            }
            return result;
        }

        double oneDGrad(double x, const double* vals) const {
            double result = 0.0;
            for (int k=1-a; k<(a+1); k++)
                result += (*vals++)*dkernel(x-k);
            return result;
        }

        double interp(double x, double y) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            double vals[2*a];
            double* vPtr = &vals[0];
            for (int k=1-a; k<(a+1); k++) {
                *vPtr++ = oneDSpline(dy, &_vals[(i-1+k)*_ny+j-a]);
            }

            return oneDSpline(dx, vals);
        }

        void gradient(double x, double y, double& dfdx, double& dfdy) const {
            int i = _xargs.upperIndex(x);
            int j = _yargs.upperIndex(y);
            double dxgrid = _xargs[i] - _xargs[i-1];
            double dygrid = _yargs[j] - _yargs[j-1];
            double dx = (x - _xargs[i-1])/dxgrid;
            double dy = (y - _yargs[j-1])/dygrid;

            // y-gradient
            double vals[2*a];
            double* vPtr = &vals[0];
            for (int k=1-a; k<(a+1); k++)
                *vPtr++ = oneDGrad(dy, &_vals[(i-1+k)*_ny+j-a]);
            dfdy = oneDSpline(dx, vals)/dxgrid;

            // x-gradient
            vPtr = &vals[0];
            for (int k=1-a; k<(a+1); k++) {
                double xvals[2*a];
                double* xvPtr = &xvals[0];
                for (int k2=1-a; k2<(a+1); k2++)
                    *xvPtr++ = _vals[(i-1+k2)*_ny+j+k];
                *vPtr++ = oneDGrad(dx, &xvals[0]);
            }
            dfdx = oneDSpline(dy, vals)/dygrid;
        }
    };


    Table2D::Table2D(
        const double* xargs, const double* yargs, const double* vals,
        int Nx, int Ny, interpolant in,
        const double* dfdx, const double* dfdy, const double* d2fdxdy
    ) : _pimpl(makeImpl(xargs, yargs, vals, Nx, Ny, in, dfdx, dfdy, d2fdxdy)) {}

    std::shared_ptr<Table2D::Table2DImpl> Table2D::makeImpl(
        const double* xargs, const double* yargs, const double* vals,
        int Nx, int Ny, interpolant in,
        const double* dfdx, const double* dfdy, const double* d2fdxdy
    ) {
        switch(in) {
            case floor:
                return std::make_shared<ConcreteTable2DImpl<T2DFloorInterp>>(
                    xargs, yargs, vals, Nx, Ny);
            case ceil:
                return std::make_shared<ConcreteTable2DImpl<T2DCeilInterp>>(
                    xargs, yargs, vals, Nx, Ny);
            case nearest:
                return std::make_shared<ConcreteTable2DImpl<T2DNearestInterp>>(
                    xargs, yargs, vals, Nx, Ny);
            case linear:
                return std::make_shared<ConcreteTable2DImpl<T2DLinearInterp>>(
                    xargs, yargs, vals, Nx, Ny);
            case cubic:
                return std::make_shared<ConcreteTable2DImpl<T2DCubicInterp>>(
                    xargs, yargs, vals, Nx, Ny, dfdx, dfdy, d2fdxdy);
            case cubicConvolve:
                return std::make_shared<ConcreteTable2DImpl<T2DCubicConvolveInterp>>(
                    xargs, yargs, vals, Nx, Ny);
            case lanczos1:
                return std::make_shared<ConcreteTable2DImpl<LanczosInterp<1>>>(
                    xargs, yargs, vals, Nx, Ny);
            case lanczos2:
                return std::make_shared<ConcreteTable2DImpl<LanczosInterp<2>>>(
                    xargs, yargs, vals, Nx, Ny);
            case lanczos3:
                return std::make_shared<ConcreteTable2DImpl<LanczosInterp<3>>>(
                    xargs, yargs, vals, Nx, Ny);
            case lanczos4:
                return std::make_shared<ConcreteTable2DImpl<LanczosInterp<4>>>(
                    xargs, yargs, vals, Nx, Ny);
            case lanczos5:
                return std::make_shared<ConcreteTable2DImpl<LanczosInterp<5>>>(
                    xargs, yargs, vals, Nx, Ny);
            case lanczos6:
                return std::make_shared<ConcreteTable2DImpl<LanczosInterp<6>>>(
                    xargs, yargs, vals, Nx, Ny);
            case lanczos7:
                return std::make_shared<ConcreteTable2DImpl<LanczosInterp<7>>>(
                    xargs, yargs, vals, Nx, Ny);

            default:
                throw std::runtime_error("invalid interpolation method");
        }
    }

    double Table2D::lookup(double x, double y) const {
        return _pimpl->lookup(x, y);
    }

    void Table2D::interpMany(const double* xvec, const double* yvec, double* valvec, int N) const {
        _pimpl->interpMany(xvec, yvec, valvec, N);
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

}
