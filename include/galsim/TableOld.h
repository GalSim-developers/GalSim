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

#ifndef GalSim_TableOld_H
#define GalSim_TableOld_H

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
     * @brief A class to represent lookup tables for a function z = f(x, y).
     */
    class Table2DOld
    {
    public:
        enum interpolant { linear, floor, ceil, nearest };

        /// Table from xargs, yargs, vals
        Table2DOld(const double* xargs, const double* yargs, const double* vals,
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
        class Table2DOldImpl;
        shared_ptr<Table2DOldImpl> _pimpl;
    };
}

#endif
