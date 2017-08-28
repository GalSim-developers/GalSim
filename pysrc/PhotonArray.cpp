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

#include "PhotonArray.h"

namespace bp = boost::python;

namespace galsim {
namespace {

    struct PyPhotonArray {
        template <typename U, typename W>
        static void wrapTemplates(W& wrapper) {
            wrapper
                .def("addTo",
                    (double (PhotonArray::*)(ImageView<U>) const)
                    &PhotonArray::addTo,
                    (bp::arg("image")))
                .def("setFrom",
                    (int (PhotonArray::*)(const BaseImage<U>&, double, UniformDeviate))
                    &PhotonArray::setFrom,
                    bp::args("image", "maxFlux", "ud"));
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

        static void wrap()
        {
            bp::class_<PhotonArray> pyPhotonArray("PhotonArray", bp::no_init);
            pyPhotonArray
                .def("__init__", bp::make_constructor(
                    &construct, bp::default_call_policies(),
                    bp::args("N", "x", "y", "flux", "dxdz", "dydz", "wavelength", "is_corr")))
                .def("convolve", &PhotonArray::convolve, (bp::args("rhs", "ud")),
                     "Convolve this PhotonArray with another")
                .enable_pickling()
                ;
            wrapTemplates<double>(pyPhotonArray);
            wrapTemplates<float>(pyPhotonArray);
        }
    }; // struct PyPhotonArray

} // anonymous

void pyExportPhotonArray()
{
    PyPhotonArray::wrap();
}

} // namespace galsim
