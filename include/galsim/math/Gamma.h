/* -*- c++ -*-
 * Copyright (c) 2012-2019 by the GalSim developers team on GitHub
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

#ifndef GalSim_IncompleteGamma_H
#define GalSim_IncompleteGamma_H
/**
 * @file math/IncompleteGamma.h
 * @brief Contains an implementation of the incomplete Gamma function ported from netlib
 */

namespace galsim {
namespace math {

#if __cplusplus >= 201103L
    // Note, the regular (not incomplete) gamma function is available as std::tgamma(x) in
    // c++11, so we don't need to implement that one ourselves (nor use boost).
    using std::tgamma;
    using std::lgamma;
#else
    // But if not using c++11, then we need to implement it ourselves.
    double tgamma(double x);
    double lgamma(double x);
#endif

    // This specific function is what boost calls gamma_p:
    //
    //     P(a,x) = gamma(a, x) / Gamma(a).
    //
    // Wolfram calls it the Regularized Gamma Function:
    // cf. http://mathworld.wolfram.com/RegularizedGammaFunction.html
    double gamma_p(double a, double x);


} }

#endif

