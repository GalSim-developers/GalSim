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
#ifndef GalSim_RealGalaxy_H
#define GalSim_RealGalaxy_H

/**
 * @file RealGalaxy.h
 *
 * Helper function for the ChromaticRealGalaxy class in python.
 */

#include <complex>
#include "Std.h"

namespace galsim {

    PUBLIC_API void ComputeCRGCoefficients(
        std::complex<double>* coef, std::complex<double>* Sigma,
        const double* w, const std::complex<double>* kimgs,
        const std::complex<double>* psf_eff_kimgs,
        const int nsed, const int nband, const int nkx, const int nky);
}

#endif
