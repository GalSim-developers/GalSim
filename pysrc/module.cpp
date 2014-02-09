/* -*- c++ -*-
 * Copyright 2012-2014 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
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
    void pyExportCppShear();
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

    namespace hsm {
        void pyExportHSM();
    } // namespace hsm

    namespace integ {
        void pyExportInteg();
    } // namespace integ

} // namespace galsim

BOOST_PYTHON_MODULE(_galsim) {
    import_array(); // for numpy
    galsim::pyExportAngle();
    galsim::pyExportBounds();
    galsim::pyExportCppShear();
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
    galsim::pyExportSBMoffat();
    galsim::pyExportSBAiry();
    galsim::pyExportSBShapelet();
    galsim::pyExportSBInterpolatedImage();
    galsim::pyExportSBKolmogorov();
    galsim::pyExportRandom();
    galsim::pyExportNoise();
    galsim::pyExportInterpolant();
    galsim::pyExportCorrelationFunction();
    galsim::hsm::pyExportHSM();
    galsim::integ::pyExportInteg();
    galsim::pyExportTable();
}
