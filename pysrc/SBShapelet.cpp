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

#include "PyBind11Helper.h"
#include "SBShapelet.h"

namespace galsim {

    static void fit(double sigma, int order, size_t idata,
                    const BaseImage<double>& image, double scale,
                    const Position<double>& center)
    {
        LVector bvec(order);
        ShapeletFitImage(sigma, bvec, image, scale, center);

        double* data = reinterpret_cast<double*>(idata);
        int size = PQIndex::size(order);
        for (int i=0; i<size; ++i) data[i] = bvec.rVector()[i];
    }

    static SBShapelet* construct(
        double sigma, int order, size_t idata, GSParams gsparams)
    {
        double* data = reinterpret_cast<double*>(idata);
        int size = PQIndex::size(order);
        VectorXd v(size);
        for (int i=0; i<size; ++i) v[i] = data[i];
        LVector bvec(order, v);
        return new SBShapelet(sigma, bvec, gsparams);
    }

    void pyExportSBShapelet(PY_MODULE& _galsim)
    {
        py::class_<SBShapelet, BP_BASES(SBProfile)>(GALSIM_COMMA "SBShapelet" BP_NOINIT)
            .def(PY_INIT(&construct));

        GALSIM_DOT def("ShapeletFitImage", &fit);
    }

} // namespace galsim
