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

#ifndef GalSim_Table_H
#define GalSim_Table_H

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <functional>

#include "Std.h"
#include "OneDimensionalDeviate.h"


namespace galsim {

    // Used by LookupTable2D in Python.  Defined in Table.cpp
    PUBLIC_API void WrapArrayToPeriod(double* x, int n, double x0, double period);

    class PUBLIC_API Interpolant;

    /**
     * @brief A class to represent lookup tables for a function y = f(x).
     */
    class PUBLIC_API Table :
        public FluxDensity
    {
    public:
        enum interpolant { linear, floor, ceil, nearest, spline, gsinterp };

        /// Table from args, vals
        Table(const double* args, const double* vals, int N, interpolant in);
        Table(const double* args, const double* vals, int N, const Interpolant* gsinterp);

        double argMin() const;
        double argMax() const;
        size_t size() const;

        /// interp, return double(0) if beyond bounds
        /// This is a virtual function from FluxDensity, which lets a Table be a FluxDensity.
        double operator()(double a) const;

        /// interp, but exception if beyond bounds
        double lookup(double a) const;

        /// interp many values at once
        void interpMany(const double* argvec, double* valvec, int N) const;

        // Integrate the function from xmin to xmax
        double integrate(double xmin, double xmax) const;

        // Integrate f(x) g(x) from xmin to xmax
        double integrateProduct(const Table& g, double xmin, double xmax, double xfact) const;

        class TableImpl;
    protected:
        shared_ptr<TableImpl> _pimpl;

        Table() {}  // TableBuilder needs this, since it delays making the _pimpl.

        void _makeImpl(const double* args, const double* vals, int N, interpolant in);
        void _makeImpl(const double* args, const double* vals, int N, const Interpolant* gsinterp);
    };

    // This version keeps its own storage of the arg/val arrays.
    // Use it by successively adding entries, which must be in increasing order of x.
    // Then when done, call finalize() to build up the lookup table.
    class PUBLIC_API TableBuilder:
        public Table
    {
    public:
        TableBuilder(interpolant in): _final(false), _in(in) {}
        TableBuilder(const Interpolant* gsinterp) :
            _final(false), _in(Table::gsinterp), _gsinterp(gsinterp) {}

        bool finalized() const { return _final; }

        double lookup(double a) const
        {
            assert(_final);
            return Table::lookup(a);
        }

        /// Insert an (x, y(x)) pair into the table.
        void addEntry(double x, double f)
        {
            assert(!_final);
            _xvec.push_back(x);
            _fvec.push_back(f);
        }

        void finalize();

    private:

        bool _final;
        interpolant _in;
        const Interpolant* _gsinterp;
        std::vector<double> _xvec;
        std::vector<double> _fvec;
    };

    /**
     * @brief A class to represent lookup tables for a function z = f(x, y).
     */
    class PUBLIC_API Table2D
    {
    public:
        enum interpolant { linear, floor, ceil, nearest, spline, gsinterp };

        /// Table from xargs, yargs, vals
        Table2D(const double* xargs, const double* yargs, const double* vals,
                int Nx, int Ny, interpolant in);
        Table2D(const double* xargs, const double* yargs, const double* vals,
                int Nx, int Ny, const double* dfdx, const double* dfdy, const double* d2fdxdy);
        Table2D(const double* xargs, const double* yargs, const double* vals,
                int Nx, int Ny, const Interpolant* gsinterp);

        /// interp
        double lookup(double x, double y) const;

        /// interp many values at once
        void interpMany(const double* xvec, const double* yvec, double* valvec, int N) const;

        void interpGrid(const double* xvec, const double* yvec, double* valvec, int Nx, int Ny) const;

        /// Estimate df/dx, df/dy at a single location
        void gradient(double x, double y, double& dfdx, double& dfdy) const;

        /// Estimate many df/dx and df/dy values
        void gradientMany(const double* xvec, const double* yvec,
                          double* dfdxvec, double* dfdyvec, int N) const;

        void gradientGrid(const double* xvec, const double* yvec,
                          double* dfdxvec, double* dfdyvec, int Nx, int Ny) const;

        class Table2DImpl;
    protected:
        const shared_ptr<Table2DImpl> _pimpl;

        static std::shared_ptr<Table2DImpl> _makeImpl(
            const double* xargs, const double* yargs, const double* vals,
            int Nx, int Ny, interpolant in);
        static std::shared_ptr<Table2DImpl> _makeImpl(
            const double* xargs, const double* yargs, const double* vals,
            int Nx, int Ny,
            const double* dfdx, const double* dfdy, const double* d2fdxdy);
        static std::shared_ptr<Table2DImpl> _makeImpl(
            const double* xargs, const double* yargs, const double* vals,
            int Nx, int Ny, const Interpolant* gsinterp);
    };
}

#endif
