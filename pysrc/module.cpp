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

#include "Python.h"
#include "PyBind11Helper.h"

namespace galsim {
    void pyExportBounds(PY_MODULE&);
    void pyExportPhotonArray(PY_MODULE&);
    void pyExportImage(PY_MODULE&);
    void pyExportSBProfile(PY_MODULE&);
    void pyExportSBAdd(PY_MODULE&);
    void pyExportSBConvolve(PY_MODULE&);
    void pyExportSBDeconvolve(PY_MODULE&);
    void pyExportSBFourierSqrt(PY_MODULE&);
    void pyExportSBTransform(PY_MODULE&);
    void pyExportSBBox(PY_MODULE&);
    void pyExportSBGaussian(PY_MODULE&);
    void pyExportSBDeltaFunction(PY_MODULE&);
    void pyExportSBExponential(PY_MODULE&);
    void pyExportSBSersic(PY_MODULE&);
    void pyExportSBSpergel(PY_MODULE&);
    void pyExportSBMoffat(PY_MODULE&);
    void pyExportSBAiry(PY_MODULE&);
    void pyExportSBShapelet(PY_MODULE&);
    void pyExportSBInterpolatedImage(PY_MODULE&);
    void pyExportSBKolmogorov(PY_MODULE&);
    void pyExportSBInclinedExponential(PY_MODULE&);
    void pyExportSBInclinedSersic(PY_MODULE&);
    void pyExportSBVonKarman(PY_MODULE&);
    void pyExportSBSecondKick(PY_MODULE&);
    void pyExportRandom(PY_MODULE&);
    void pyExportTable(PY_MODULE&);
    void pyExportInterpolant(PY_MODULE&);
    void pyExportCDModel(PY_MODULE&);
    void pyExportSilicon(PY_MODULE&);
    void pyExportRealGalaxy(PY_MODULE&);
    void pyExportWCS(PY_MODULE&);

    namespace hsm {
        void pyExportHSM(PY_MODULE&);
    }

    namespace integ {
        void pyExportInteg(PY_MODULE&);
    }

    namespace math {
        void pyExportBessel(PY_MODULE&);
        void pyExportHorner(PY_MODULE&);
    }

} // namespace galsim

PYBIND11_MODULE(_galsim, _galsim)
{
    BP_SCOPE(_galsim);

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

    galsim::hsm::pyExportHSM(_galsim);
    galsim::integ::pyExportInteg(_galsim);
    galsim::math::pyExportBessel(_galsim);
    galsim::math::pyExportHorner(_galsim);
}
