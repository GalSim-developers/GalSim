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

#include "PyBind11Helper.h"
#include "Silicon.h"
#include "Random.h"

namespace galsim {

    template <typename T, typename W>
    static void WrapTemplates(W& wrapper) {
        typedef double (Silicon::*accumulate_fn)(const PhotonArray&, UniformDeviate,
                                                 ImageView<T>, Position<int>);
        wrapper.def("accumulate", (accumulate_fn)&Silicon::accumulate);
    }

    static BP_CONSTRUCTOR(MakeSilicon, Silicon,
                          int NumVertices, double NumElect, int Nx, int Ny, int QDist,
                          double Nrecalc, double DiffStep, double PixelSize,
                          double SensorThickness, size_t idata,
                          const Table& treeRingTable,
                          const Position<double>& treeRingCenter,
                          const Table& abs_length_table)
    {
        double* data = reinterpret_cast<double*>(idata);
        PB11_PLACEMENT_NEW Silicon(NumVertices, NumElect, Nx, Ny, QDist,
                                       Nrecalc, DiffStep, PixelSize, SensorThickness, data,
                                       treeRingTable, treeRingCenter, abs_length_table);
    }

    void pyExportSilicon(PB11_MODULE& _galsim)
    {
        bp::class_<Silicon> pySilicon(GALSIM_COMMA "Silicon" BP_NOINIT);
        pySilicon.def("__init__", BP_MAKE_CONSTRUCTOR(&MakeSilicon));

        WrapTemplates<double>(pySilicon);
        WrapTemplates<float>(pySilicon);
    }

} // namespace galsim

