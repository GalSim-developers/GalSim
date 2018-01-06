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
    void pyExportBounds(PB11_MODULE&);
    void pyExportPhotonArray(PB11_MODULE&);
    void pyExportImage(PB11_MODULE&);
    void pyExportSBProfile(PB11_MODULE&);
    void pyExportSBAdd(PB11_MODULE&);
    void pyExportSBConvolve(PB11_MODULE&);
    void pyExportSBDeconvolve(PB11_MODULE&);
    void pyExportSBFourierSqrt(PB11_MODULE&);
    void pyExportSBTransform(PB11_MODULE&);
    void pyExportSBBox(PB11_MODULE&);
    void pyExportSBGaussian(PB11_MODULE&);
    void pyExportSBDeltaFunction(PB11_MODULE&);
    void pyExportSBExponential(PB11_MODULE&);
    void pyExportSBSersic(PB11_MODULE&);
    void pyExportSBSpergel(PB11_MODULE&);
    void pyExportSBMoffat(PB11_MODULE&);
    void pyExportSBAiry(PB11_MODULE&);
    void pyExportSBShapelet(PB11_MODULE&);
    void pyExportSBInterpolatedImage(PB11_MODULE&);
    void pyExportSBKolmogorov(PB11_MODULE&);
    void pyExportSBInclinedExponential(PB11_MODULE&);
    void pyExportSBInclinedSersic(PB11_MODULE&);
    void pyExportRandom(PB11_MODULE&);
    void pyExportTable(PB11_MODULE&);
    void pyExportInterpolant(PB11_MODULE&);
    void pyExportCDModel(PB11_MODULE&);
    void pyExportSilicon(PB11_MODULE&);
    void pyExportRealGalaxy(PB11_MODULE&);
    void pyExportWCS(PB11_MODULE&);

    namespace hsm {
        void pyExportHSM(PB11_MODULE&);
    }

    namespace integ {
        void pyExportInteg(PB11_MODULE&);
    }

    namespace math {
        void pyExportBessel(PB11_MODULE&);
    }

} // namespace galsim

PB11_MAKE_MODULE(_galsim)
{
    PB11_START_MODULE(_galsim);

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

    PB11_END_MODULE(_galsim);
}
