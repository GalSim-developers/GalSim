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

#include "math/Horner.h"

namespace galsim {
namespace math {

    void Horner(double* x, const int nx, double* coef, const int nc, double* result)
    {
        // Start at highest power
        double* c = coef + nc-1;
        // Ignore any trailing zeros
        while (*c == 0. && c > coef) --c;
        // Repeatedly multiply by x and add next coefficient
        for (int i=0; i<nx; ++i) result[i] = *c;
        while(--c >= coef) {
            for (int i=0; i<nx; ++i) result[i] = result[i]*x[i] + *c;
        }
        // In the last step, we will have added the constant term, and we're done.
    }

}}
