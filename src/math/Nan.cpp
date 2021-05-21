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

// This needs to be math.h, not cmath, since sometimes cmath puts isnan in the std
// namespace and undefines the regular isnan.  But sometimes it doesn't.
// So writing std::isnan is not portable.
#include "math.h"

namespace galsim {
namespace math {

    template <typename T>
    bool isNan(T x)
    {
#ifdef isnan
        return isnan(x);
#else
        // Depending on the IEEE conformity, at least one of these will always work to detect
        // nans, but neither one by itself is completely reliable.
        return (x != x) || !(x*x >= 0);
#endif
    }

    template bool isNan(float x);
    template bool isNan(double x);

}}
