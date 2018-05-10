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

    /**
     * @brief A class to represent lookup tables for a function y = f(x).
     */
    class Table:
        public FluxDensity
    {
    public:
        enum interpolant { linear, floor, ceil, nearest, spline };

        /// Table from args, vals
        Table(const double* args, const double* vals, int N, interpolant in);

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

    protected:
        Table() {}  // TableBuilder needs this, since it delays making the _pimpl.

        class TableImpl;
        shared_ptr<TableImpl> _pimpl;
    };

    // This version keeps its own storage of the arg/val arrays.
    // Use it by successively adding entries, which must be in increasing order of x.
    // Then when done, call finalize() to build up the lookup table.
    class TableBuilder:
        public Table
    {
    public:
        /// Table from args, vals
        TableBuilder(interpolant in): _final(false), _in(in) {}

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
        std::vector<double> _xvec;
        std::vector<double> _fvec;
    };

    /**
     * @brief A class to represent lookup tables for a function z = f(x, y).
     */
    class Table2D
    {
    public:
        enum interpolant { linear, floor, ceil, nearest };

        /// Table from xargs, yargs, vals
        Table2D(const double* xargs, const double* yargs, const double* vals,
                int Nx, int Ny, interpolant in);

        /// interp, but exception if beyond bounds
        double lookup(double x, double y) const;

        /// interp many values at once
        void interpMany(const double* xvec, const double* yvec, double* valvec, int N) const;
        void interpManyMesh(const double* xvec, const double* yvec, double* valvec,
                            int outNx, int outNy) const;

        /// Estimate df/dx, df/dy at a single location
        void gradient(double x, double y, double& dfdxvec, double& dfdyvec) const;

        /// Estimate many df/dx and df/dy values
        void gradientMany(const double* xvec, const double* yvec,
                          double* dfdxvec, double* dfdyvec, int N) const;

    protected:
        class Table2DImpl;
        shared_ptr<Table2DImpl> _pimpl;
    };
}

#endif
