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
#include "RealGalaxy.h"

namespace galsim {

    void CallComputeCRGCoefficients(size_t coef_data, size_t Sigma_data,
                                    size_t w_data, size_t kimgs_data, size_t psf_data,
                                    const int nsed, const int nband, const int nkx, const int nky)
    {
        typedef std::complex<double> CD;
        CD* coef = reinterpret_cast<CD*>(coef_data);
        CD* Sigma = reinterpret_cast<CD*>(Sigma_data);
        const double* w = reinterpret_cast<const double*>(w_data);
        const CD* kimgs = reinterpret_cast<const CD*>(kimgs_data);
        const CD* psf = reinterpret_cast<const CD*>(psf_data);
        ComputeCRGCoefficients(coef, Sigma, w, kimgs, psf, nsed, nband, nkx, nky);
    };

    void pyExportRealGalaxy(PY_MODULE& _galsim) {
        GALSIM_DOT def("ComputeCRGCoefficients", &CallComputeCRGCoefficients);
    }

} // namespace galsim
