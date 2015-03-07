/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL GALSIM_ARRAY_API
// This is the only one that doesn't have NO_IMPORT_ARRAY.
#include "numpy/arrayobject.h"

namespace galsim {

    void pyExportAngle();
    void pyExportBounds();
    void pyExportImage();
    void pyExportSBProfile();
    void pyExportSBAdd();
    void pyExportSBConvolve();
    void pyExportSBDeconvolve();
    void pyExportSBTransform();
    void pyExportSBBox();
    void pyExportSBGaussian();
    void pyExportSBExponential();
    void pyExportSBSersic();
    void pyExportSBSpergel();
    void pyExportSBMoffat();
    void pyExportSBAiry();
    void pyExportSBShapelet();
    void pyExportSBInterpolatedImage();
    void pyExportSBKolmogorov();
    void pyExportRandom();
    void pyExportNoise();
    void pyExportTable();
    void pyExportInterpolant();
    void pyExportCorrelationFunction();
    void pyExportCDModel();

    namespace hsm {
        void pyExportHSM();
    } // namespace hsm

    namespace integ {
        void pyExportInteg();
    } // namespace integ

    namespace bessel {
        void pyExportBessel();
    } // namespace integ

} // namespace galsim

BOOST_PYTHON_MODULE(_galsim) {
    import_array(); // for numpy
    galsim::pyExportAngle();
    galsim::pyExportBounds();
    galsim::pyExportImage();
    galsim::pyExportSBProfile();
    galsim::pyExportSBAdd();
    galsim::pyExportSBConvolve();
    galsim::pyExportSBDeconvolve();
    galsim::pyExportSBTransform();
    galsim::pyExportSBBox();
    galsim::pyExportSBGaussian();
    galsim::pyExportSBExponential();
    galsim::pyExportSBSersic();
    galsim::pyExportSBSpergel();
    galsim::pyExportSBMoffat();
    galsim::pyExportSBAiry();
    galsim::pyExportSBShapelet();
    galsim::pyExportSBInterpolatedImage();
    galsim::pyExportSBKolmogorov();
    galsim::pyExportRandom();
    galsim::pyExportNoise();
    galsim::pyExportInterpolant();
    galsim::pyExportCorrelationFunction();
    galsim::pyExportCDModel();
    galsim::hsm::pyExportHSM();
    galsim::integ::pyExportInteg();
    galsim::pyExportTable();
    galsim::bessel::pyExportBessel();
}
