/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

#ifndef GalSim_Bessel_H
#define GalSim_Bessel_H
/**
 * @file math/Bessel.h
 * @brief Contains implementations of some bessel functions ported from netlib.
 */

namespace galsim {
namespace math {

    // Functions defined in src/Bessel.cpp
    double BesselJ0(double x);
    double BesselJ1(double x);
    double BesselJ(double nu, double x);
    double BesselY(double nu, double x);
    double BesselK(double nu, double x);
    double BesselI(double nu, double x);

} }

#endif

