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
#include "PhotonArray.h"

namespace galsim {

    template <typename T, typename W>
    static void WrapTemplates(W& wrapper) {
        wrapper
            .def("addTo", (double (PhotonArray::*)(ImageView<T>) const) &PhotonArray::addTo)
            .def("setFrom",
                 (int (PhotonArray::*)(const BaseImage<T>&, double, BaseDeviate))
                 &PhotonArray::setFrom);
    }

    static PhotonArray* construct(int N, size_t ix, size_t iy, size_t iflux,
                                  size_t idxdz, size_t idydz, size_t iwave, bool is_corr)
    {
        double *x = reinterpret_cast<double*>(ix);
        double *y = reinterpret_cast<double*>(iy);
        double *flux = reinterpret_cast<double*>(iflux);
        double *dxdz = reinterpret_cast<double*>(idxdz);
        double *dydz = reinterpret_cast<double*>(idydz);
        double *wave = reinterpret_cast<double*>(iwave);
        return new PhotonArray(N, x, y, flux, dxdz, dydz, wave, is_corr);
    }

    void pyExportPhotonArray(PY_MODULE& _galsim)
    {
        py::class_<PhotonArray> pyPhotonArray(GALSIM_COMMA "PhotonArray" BP_NOINIT);
        pyPhotonArray
            .def(PY_INIT(&construct))
            .def("convolve", &PhotonArray::convolve);
        WrapTemplates<double>(pyPhotonArray);
        WrapTemplates<float>(pyPhotonArray);
    }

} // namespace galsim
