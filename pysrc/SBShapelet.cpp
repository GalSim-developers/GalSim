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
#include "boost/python.hpp"

#include "SBShapelet.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBShapelet
    {
        static void fit(double sigma, int order, size_t idata,
                        const BaseImage<double>& image, double scale,
                        const Position<double>& center)
        {
            LVector bvec(order);
            ShapeletFitImage(sigma, bvec, image, scale, center);

            double* data = reinterpret_cast<double*>(idata);
            int size = PQIndex::size(order);
            tmv::VectorView<double> v = tmv::VectorViewOf(data, size);
            v = bvec.rVector();
        }

        static SBShapelet* construct(double sigma, int order, size_t idata, GSParams gsparams)
        {
            double* data = reinterpret_cast<double*>(idata);
            int size = PQIndex::size(order);
            LVector bvec(order, tmv::VectorViewOf(data, size));
            return new SBShapelet(sigma, bvec, gsparams);
        }

        static void wrap() {
            bp::class_<SBShapelet,bp::bases<SBProfile> >("SBShapelet", bp::no_init)
                .def("__init__", bp::make_constructor(
                        &construct, bp::default_call_policies()));

            bp::def("ShapeletFitImage", &fit);
        }
    };

    void pyExportSBShapelet()
    {
        PySBShapelet::wrap();
    }

} // namespace galsim
