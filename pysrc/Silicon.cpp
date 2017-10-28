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

#define BOOST_NO_CXX11_SMART_PTR
#include <boost/python.hpp> // header that includes Python.h always needs to come first
#include <boost/python/stl_iterator.hpp>

#include "Silicon.h"
#include "Random.h"
#include "NumpyHelper.h"

namespace bp = boost::python;

namespace galsim {
namespace {

    struct PySilicon {

        template <typename U, typename W>
        static void wrapTemplates(W & wrapper)
        {
            typedef double (Silicon::*accumulate_fn)(const PhotonArray&, UniformDeviate,
                                                     ImageView<U>, Position<int>);
            wrapper
                .def("accumulate",
                     (accumulate_fn)&Silicon::accumulate,
                     (bp::args("photons", "rng", "image", "orig_center")),
                     "Accumulate photons in image")
                ;
        }


        static Silicon* MakeSilicon(int NumVertices, double NumElect, int Nx, int Ny, int QDist,
                                    double Nrecalc, double DiffStep, double PixelSize,
                                    double SensorThickness, const bp::object& array,
                                    const Table<double, double>& treeRingTable,
                                    const Position<double>& treeRingCenter,
                                    const Table<double, double>& abs_length_table)
        {
            double* data = 0;
            boost::shared_ptr<double> owner;
            int step = 0;
            int stride = 0;
            CheckNumpyArray(array, 2, false, data, owner, step, stride);
            if (step != 1)
                throw std::runtime_error("Silicon vertex_data requires step == 1");
            if (stride != 5)
                throw std::runtime_error("Silicon vertex_data requires stride == 5");
            if (GetNumpyArrayDim(array.ptr(), 1) != 5)
                throw std::runtime_error("Silicon vertex_data requires ncol == 5");
            int NumPolys = Nx * Ny + 2;
            int Nv = 4 * NumVertices + 4;
            if (GetNumpyArrayDim(array.ptr(), 0) != Nv*(NumPolys-2))
                throw std::runtime_error("Silicon vertex_data has the wrong number of rows");
            return new Silicon(NumVertices, NumElect, Nx, Ny, QDist,
                               Nrecalc, DiffStep, PixelSize, SensorThickness, data,
                               treeRingTable, treeRingCenter, abs_length_table);
        }

        static void wrap()
        {
            bp::class_<Silicon> pySilicon("Silicon", bp::no_init);
            pySilicon
                .def("__init__", bp::make_constructor(
                        &MakeSilicon, bp::default_call_policies(),
                        (bp::args("NumVertices", "NumElect", "Nx", "Ny", "QDist",
                                  "Nrecalc", "DiffStep", "PixelSize",
                                  "SensorThickness", "vertex_data",
                                  "treeRingTable", "treeRingCenter",
                                  "abs_length_table"))))
                .enable_pickling()
                ;
            wrapTemplates<double>(pySilicon);
            wrapTemplates<float>(pySilicon);
        }

    }; // struct PySilicon

} // anonymous

void pyExportSilicon()
{
    PySilicon::wrap();
}

} // namespace galsim

