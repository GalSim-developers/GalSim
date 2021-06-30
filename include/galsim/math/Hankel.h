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

#ifndef GalSim_Hankel_H
#define GalSim_Hankel_H

#include <functional>
#include "Std.h"

namespace galsim {
namespace math {

    PUBLIC_API double hankel_trunc(
        const std::function<double(double)> f, double k, double nu, double maxr,
        double relerr=1.e-6, double abserr=1.e-12, int nzeros=10);
    PUBLIC_API double hankel_inf(
        const std::function<double(double)> f, double k, double nu,
        double relerr=1.e-6, double abserr=1.e-12, int nzeros=10);

}
}

#endif
