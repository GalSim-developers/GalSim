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

#include "galsim/IgnoreWarnings.h"
#include "boost/python.hpp" // header that includes Python.h always needs to come first

#include "Silicon.h"
#include "Random.h"

namespace bp = boost::python;

namespace galsim {

    template <typename T, typename W>
    static void WrapTemplates(W& wrapper) {
        typedef double (Silicon::*accumulate_fn)(const PhotonArray&, UniformDeviate,
                                                 ImageView<T>);
        wrapper
            .def("accumulate", (accumulate_fn)&Silicon::accumulate);
    }


    static Silicon* MakeSilicon(int NumVertices, double NumElect, int Nx, int Ny, int QDist,
                                double Nrecalc, double DiffStep, double PixelSize,
                                double SensorThickness, size_t idata)
    {
        double* data = reinterpret_cast<double*>(idata);
        int NumPolys = Nx * Ny + 2;
        int Nv = 4 * NumVertices + 4;
        return new Silicon(NumVertices, NumElect, Nx, Ny, QDist,
                           Nrecalc, DiffStep, PixelSize, SensorThickness, data);
    }

    void pyExportSilicon()
    {
        bp::class_<Silicon> pySilicon("Silicon", bp::no_init);
        pySilicon
            .def("__init__", bp::make_constructor(&MakeSilicon, bp::default_call_policies()));

        WrapTemplates<double>(pySilicon);
        WrapTemplates<float>(pySilicon);
    }

} // namespace galsim

