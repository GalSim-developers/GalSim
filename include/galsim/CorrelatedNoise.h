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

#ifndef GalSim_CorrelatedNoise_H
#define GalSim_CorrelatedNoise_H

/**
 * @file CorrelatedNoise.h @brief Contains functions used with the python-layer CorrFunc
 * classes for handling the correlation properties of noise in Images.
 */

#include <complex>
#include "Image.h"
#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Return, as a square Image, a noise covariance matrix between every element in an Image
     * with supplied `bounds` and pixel scale `dx` for a correlation function represented as an
     * SBProfile.
     *
     * The matrix is symmetric, and therefore only the upper triangular elements are actually
     * written into.  The rest are initialized and remain as zero.
     *
     * For an example of this function in use, see `galsim/correlatednoise.py`.
     */
    PUBLIC_API void calculateCovarianceMatrix(
        ImageView<double>& cov, const SBProfile& sbp,
        const Bounds<int>& bounds, double dx);

}
#endif
