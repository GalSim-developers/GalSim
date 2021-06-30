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

#ifndef GalSim_Horner_H
#define GalSim_Horner_H

#include "Std.h"

namespace galsim {
namespace math {

    // Use Horner's method to evaluate polynomial
    // result[i] = coef[0] + coef[1]*x[i] + coef[2]*x[i]**2 + ...
    // The result array should already be allocated and have the same size as x.
    PUBLIC_API void Horner(
        const double* x, const int nx, const double* coef, const int nc, double* result);

    // 2D version of Horner's method
    // result[i] = coef[0,0] + coef[0,1]*y[i] + coef[0,2]*y[i]**2 + ...
    //             + (coef[1,0] + coef[1,1]*y[i] + coef[1,2]*y[i]**2 + ...) * x[i]
    //             + (coef[2,0] + coef[2,1]*y[i] + coef[2,2]*y[i]**2 + ...) * x[i]**2
    //             + ...
    // The result array should already be allocated and have the same size as x and y.
    // This also requires a temporary array of the same size as well.
    PUBLIC_API void Horner2D(
        const double* x, const double* y, const int nx,
        const double* coef, const int ncx, const int ncy,
        double* result, double* temp);

} }

#endif

