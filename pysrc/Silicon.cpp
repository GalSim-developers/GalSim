/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

namespace bp = boost::python;

namespace galsim {
namespace {

    struct PySilicon {

        template <typename U, typename W>
        static void wrapTemplates(W & wrapper) {
            wrapper
                .def("accumulate",
                     (double (Silicon::*)(const PhotonArray&, UniformDeviate, ImageView<U>) const)&Silicon::accumulate,
                     (bp::args("photons", "rng", "image")),
                     "Accumulate photons in image")
                ;
        }

        static void wrap()
        {
            bp::class_<Silicon> pySilicon("Silicon", bp::no_init);
            pySilicon
                .def(bp::init<std::string>(bp::args("config_file")))
                .enable_pickling()
                ;
            bp::register_ptr_to_python< boost::shared_ptr<Silicon> >();
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


/*

Was:

Silicon::Silicon (std::string inname)

Is:

  Silicon::Silicon (int NumVertices, int NumElec, int Nx, int Ny, int QDist, double DiffStep, double** vertex_data)

*/
