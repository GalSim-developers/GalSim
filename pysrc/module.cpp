/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
    void pyExportBounds(PYBIND11_MODULE&);
    void pyExportPhotonArray(PYBIND11_MODULE&);
    void pyExportImage(PYBIND11_MODULE&);
    void pyExportSBProfile(PYBIND11_MODULE&);
    void pyExportSBAdd(PYBIND11_MODULE&);
    void pyExportSBConvolve(PYBIND11_MODULE&);
    void pyExportSBDeconvolve(PYBIND11_MODULE&);
    void pyExportSBFourierSqrt(PYBIND11_MODULE&);
    void pyExportSBTransform(PYBIND11_MODULE&);
    void pyExportSBBox(PYBIND11_MODULE&);
    void pyExportSBGaussian(PYBIND11_MODULE&);
    void pyExportSBDeltaFunction(PYBIND11_MODULE&);
    void pyExportSBExponential(PYBIND11_MODULE&);
    void pyExportSBSersic(PYBIND11_MODULE&);
    void pyExportSBSpergel(PYBIND11_MODULE&);
    void pyExportSBMoffat(PYBIND11_MODULE&);
    void pyExportSBAiry(PYBIND11_MODULE&);
    void pyExportSBShapelet(PYBIND11_MODULE&);
    void pyExportSBInterpolatedImage(PYBIND11_MODULE&);
    void pyExportSBKolmogorov(PYBIND11_MODULE&);
    void pyExportSBInclinedExponential(PYBIND11_MODULE&);
    void pyExportSBInclinedSersic(PYBIND11_MODULE&);
    void pyExportRandom(PYBIND11_MODULE&);
    void pyExportTable(PYBIND11_MODULE&);
    void pyExportInterpolant(PYBIND11_MODULE&);
    void pyExportCDModel(PYBIND11_MODULE&);
    void pyExportSilicon(PYBIND11_MODULE&);
    void pyExportRealGalaxy(PYBIND11_MODULE&);
    void pyExportWCS(PYBIND11_MODULE&);

    namespace hsm {
        void pyExportHSM(PYBIND11_MODULE&);
    }

    namespace integ {
        void pyExportInteg(PYBIND11_MODULE&);
    }

    namespace math {
        void pyExportBessel(PYBIND11_MODULE&);
    }

} // namespace galsim

PYBIND11_PLUGIN(_galsim)
{
    PYBIND11_MAKE_MODULE(_galsim);

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
    galsim::pyExportRandom(_galsim);
    galsim::pyExportTable(_galsim);
    galsim::pyExportInterpolant(_galsim);
    galsim::pyExportCDModel(_galsim);
    galsim::pyExportSilicon(_galsim);
    galsim::pyExportRealGalaxy(_galsim);
    galsim::pyExportWCS(_galsim);

    galsim::hsm::pyExportHSM(_galsim);
    galsim::integ::pyExportInteg(_galsim);
    galsim::math::pyExportBessel(_galsim);

    PYBIND11_RETURN_PTR(_galsim);
}
