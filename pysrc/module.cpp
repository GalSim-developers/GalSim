/* -*- c++ -*-
 * Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

#include "Python.h"
#include "PyBind11Helper.h"

namespace galsim {
    void pyExportBounds(py::module&);
    void pyExportPhotonArray(py::module&);
    void pyExportImage(py::module&);
    void pyExportSBProfile(py::module&);
    void pyExportSBAdd(py::module&);
    void pyExportSBConvolve(py::module&);
    void pyExportSBDeconvolve(py::module&);
    void pyExportSBFourierSqrt(py::module&);
    void pyExportSBTransform(py::module&);
    void pyExportSBBox(py::module&);
    void pyExportSBGaussian(py::module&);
    void pyExportSBDeltaFunction(py::module&);
    void pyExportSBExponential(py::module&);
    void pyExportSBSersic(py::module&);
    void pyExportSBSpergel(py::module&);
    void pyExportSBMoffat(py::module&);
    void pyExportSBAiry(py::module&);
    void pyExportSBShapelet(py::module&);
    void pyExportSBInterpolatedImage(py::module&);
    void pyExportSBKolmogorov(py::module&);
    void pyExportSBInclinedExponential(py::module&);
    void pyExportSBInclinedSersic(py::module&);
    void pyExportSBVonKarman(py::module&);
    void pyExportSBSecondKick(py::module&);
    void pyExportRandom(py::module&);
    void pyExportTable(py::module&);
    void pyExportInterpolant(py::module&);
    void pyExportCDModel(py::module&);
    void pyExportSilicon(py::module&);
    void pyExportRealGalaxy(py::module&);
    void pyExportWCS(py::module&);
    void pyExportUtilities(py::module&);

    namespace hsm {
        void pyExportHSM(py::module&);
    }

    namespace integ {
        void pyExportInteg(py::module&);
    }

    namespace math {
        void pyExportBessel(py::module&);
        void pyExportHorner(py::module&);
    }

} // namespace galsim

PYBIND11_MODULE(_galsim, _galsim)
{
    galsim::pyExportBounds(_galsim);
    galsim::pyExportPhotonArray(_galsim);
    galsim::pyExportImage(_galsim);
    galsim::pyExportSBProfile(_galsim);
    galsim::pyExportSBAdd(_galsim);
    galsim::pyExportSBConvolve(_galsim);
    galsim::pyExportSBDeconvolve(_galsim);
    galsim::pyExportSBFourierSqrt(_galsim);
    galsim::pyExportSBTransform(_galsim);
    galsim::pyExportSBBox(_galsim);
    galsim::pyExportSBGaussian(_galsim);
    galsim::pyExportSBDeltaFunction(_galsim);
    galsim::pyExportSBExponential(_galsim);
    galsim::pyExportSBSersic(_galsim);
    galsim::pyExportSBSpergel(_galsim);
    galsim::pyExportSBMoffat(_galsim);
    galsim::pyExportSBAiry(_galsim);
    galsim::pyExportSBShapelet(_galsim);
    galsim::pyExportSBInterpolatedImage(_galsim);
    galsim::pyExportSBKolmogorov(_galsim);
    galsim::pyExportSBInclinedExponential(_galsim);
    galsim::pyExportSBInclinedSersic(_galsim);
    galsim::pyExportSBVonKarman(_galsim);
    galsim::pyExportSBSecondKick(_galsim);
    galsim::pyExportRandom(_galsim);
    galsim::pyExportTable(_galsim);
    galsim::pyExportInterpolant(_galsim);
    galsim::pyExportCDModel(_galsim);
    galsim::pyExportSilicon(_galsim);
    galsim::pyExportRealGalaxy(_galsim);
    galsim::pyExportWCS(_galsim);
    galsim::pyExportUtilities(_galsim);

    galsim::hsm::pyExportHSM(_galsim);
    galsim::integ::pyExportInteg(_galsim);
    galsim::math::pyExportBessel(_galsim);
    galsim::math::pyExportHorner(_galsim);
}
