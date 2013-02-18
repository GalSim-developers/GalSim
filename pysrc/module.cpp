// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
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
#include "boost/python.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#include "numpy/arrayobject.h"

namespace galsim {

    void pyExportAngle();
    void pyExportBounds();
    void pyExportCppShear();
    void pyExportImage();
    void pyExportPhotonArray();
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
    void pyExportInterpolant();
    void pyExportCorrelationFunction();

    namespace hsm {
        void pyExportPSFCorr();
    } // namespace hsm

    namespace integ {
        void pyExportInteg();
    } // namespace hsm

} // namespace galsim

BOOST_PYTHON_MODULE(_galsim) {
    import_array(); // for numpy
    galsim::pyExportAngle();
    galsim::pyExportBounds();
    galsim::pyExportCppShear();
    galsim::pyExportImage();
    galsim::pyExportPhotonArray();
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
    galsim::pyExportInterpolant();
    galsim::pyExportCorrelationFunction();
    galsim::hsm::pyExportPSFCorr();
    galsim::integ::pyExportInteg();
}
