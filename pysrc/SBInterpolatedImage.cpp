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

#include "SBInterpolatedImage.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBInterpolatedImage
    {
        template <typename U, typename W>
        static void wrapTemplates(W& wrapper)
        {
            wrapper
                .def(bp::init<const BaseImage<U> &, const Bounds<int>&, const Bounds<int>&,
                     const Interpolant&, const Interpolant&,
                     double, double, GSParams>());

            typedef double (*cscf_func_type)(const BaseImage<U>&, double);
            bp::def("CalculateSizeContainingFlux", cscf_func_type(&CalculateSizeContainingFlux));
        }

        static void wrap()
        {
            bp::class_< SBInterpolatedImage, bp::bases<SBProfile> > pySBInterpolatedImage(
                "SBInterpolatedImage", bp::no_init);
            pySBInterpolatedImage
                .def("calculateMaxK", &SBInterpolatedImage::calculateMaxK);
            wrapTemplates<float>(pySBInterpolatedImage);
            wrapTemplates<double>(pySBInterpolatedImage);
        }

    };

    struct PySBInterpolatedKImage
    {
        static void wrap()
        {
            bp::class_< SBInterpolatedKImage, bp::bases<SBProfile> > pySBInterpolatedKImage(
                "SBInterpolatedKImage", bp::no_init);
            pySBInterpolatedKImage
                .def(bp::init<const BaseImage<std::complex<double> > &,
                              double, const Interpolant&, GSParams>());
        }

    };

    void pyExportSBInterpolatedImage()
    {
        PySBInterpolatedImage::wrap();
    }

    void pyExportSBInterpolatedKImage()
    {
        PySBInterpolatedKImage::wrap();
    }

} // namespace galsim
