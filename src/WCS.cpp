/* -*- c++ -*-
 * Copyright (c) 2012-2020 by the GalSim developers team on GitHub
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

#include "WCS.h"

namespace galsim {

    void ApplyCD(int n, double* x, double* y, const double* cd)
    {
        // For a 2x2 matrix multiplies two vectors, it's actually best to just do it manually.
        double a = cd[0];
        double b = cd[1];
        double c = cd[2];
        double d = cd[3];

        double u,v;
        for(; n; --n) {
            u = a * *x + b * *y;
            v = c * *x + d * *y;
            *x++ = u;
            *y++ = v;
        }
    }
}
