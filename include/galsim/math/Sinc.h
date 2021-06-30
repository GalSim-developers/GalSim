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

#ifndef GalSim_Sinc_H
#define GalSim_Sinc_H
/**
 * @file math/Sinc.h
 * @brief Contains the sinc function as well as the integral of sin(t)/t
 */

#include "Std.h"

namespace galsim {
namespace math {

    // sinc(x) is defined here as sin(Pi x) / (Pi x)
    PUBLIC_API double sinc(double x);

    // Utility for calculating the integral of sin(t)/t from 0 to x.  Note the official definition
    // does not have pi multiplying t.
    PUBLIC_API double Si(double x);

}
}

#endif
