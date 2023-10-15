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

#include "PyBind11Helper.h"
#include "Silicon.h"
#include "Random.h"

namespace galsim {

    template <typename T, typename W>
    static void WrapTemplates(W& wrapper) {
        typedef void (Silicon::*subtract_fn)(ImageView<T>);
        typedef void (Silicon::*add_fn)(ImageView<T>);
        typedef void (Silicon::*init_fn)(ImageView<T>, Position<int>);
        typedef double (Silicon::*accumulate_fn)(const PhotonArray&, int, int, BaseDeviate,
                                                 ImageView<T>);
        typedef void (Silicon::*update_fn)(ImageView<T>);
        typedef void (Silicon::*area_fn)(ImageView<T>, Position<int>, bool);

        wrapper.def("subtractDelta", (subtract_fn)&Silicon::subtractDelta);
        wrapper.def("addDelta", (add_fn)&Silicon::addDelta);
        wrapper.def("initialize", (init_fn)&Silicon::initialize);
        wrapper.def("accumulate", (accumulate_fn)&Silicon::accumulate);
        wrapper.def("update", (update_fn)&Silicon::update);
        wrapper.def("fill_with_pixel_areas", (area_fn)&Silicon::fillWithPixelAreas);
    }

    static Silicon* MakeSilicon(
        int NumVertices, double NumElect, int Nx, int Ny, int QDist,
        double DiffStep, double PixelSize,
        double SensorThickness, size_t idata,
        const Table& treeRingTable, const Position<double>& treeRingCenter,
        const Table& abs_length_table, bool transpose)
    {
        double* data = reinterpret_cast<double*>(idata);
        return new Silicon(NumVertices, NumElect, Nx, Ny, QDist,
                           DiffStep, PixelSize, SensorThickness, data,
                           treeRingTable, treeRingCenter, abs_length_table, transpose);
    }

    void pyExportSilicon(py::module& _galsim)
    {
        py::class_<Silicon> pySilicon(_galsim, "Silicon");
        pySilicon.def(py::init(&MakeSilicon));

        WrapTemplates<double>(pySilicon);
        WrapTemplates<float>(pySilicon);

        _galsim.def("SetOMPThreads", &SetOMPThreads);
        _galsim.def("GetOMPThreads", &GetOMPThreads);
    }

} // namespace galsim

