/* -*- c++ -*-
 * Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
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
                .def(bp::init<const BaseImage<U> &,
                     boost::shared_ptr<Interpolant>,
                     boost::shared_ptr<Interpolant>,
                     double, double, double, boost::shared_ptr<GSParams> >(
                         (bp::arg("image"),
                          bp::arg("xInterp"), bp::arg("kInterp"),
                          bp::arg("pad_factor")=4.,
                          bp::arg("stepk")=0., bp::arg("maxk")=0.,
                          bp::arg("gsparams")=bp::object())
                     )
                )
                ;
        }

        static void wrap()
        {
            bp::class_< SBInterpolatedImage, bp::bases<SBProfile> > pySBInterpolatedImage(
                "SBInterpolatedImage", bp::init<const SBInterpolatedImage &>()
            );
            pySBInterpolatedImage
                .def("calculateStepK", &SBInterpolatedImage::calculateStepK,
                     bp::arg("max_stepk")=0.)
                .def("calculateMaxK", &SBInterpolatedImage::calculateMaxK, bp::arg("max_maxk")=0.)
                .def("getImage", &SBInterpolatedImage::getImage)
                .def("getXInterp", &SBInterpolatedImage::getXInterp)
                .def("getKInterp", &SBInterpolatedImage::getKInterp)
                ;
            wrapTemplates<float>(pySBInterpolatedImage);
            wrapTemplates<double>(pySBInterpolatedImage);
        }

    };

    struct PySBInterpolatedKImage
    {
        template <typename U, typename W>
        static void wrapTemplates(W& wrapper)
        {
            wrapper
                .def(bp::init<const BaseImage<U> &,
                              const BaseImage<U> &,
                              double, double,
                              boost::shared_ptr<Interpolant>,
                              boost::shared_ptr<GSParams> >(
                                  (bp::arg("real_kimage"),
                                   bp::arg("imag_kimage"),
                                   bp::arg("dk"),
                                   bp::arg("stepk"),
                                   bp::arg("kInterp"),
                                   bp::arg("gsparams")=bp::object())
                     ))
                ;
        }

        static void wrap()
        {
            bp::class_< SBInterpolatedKImage, bp::bases<SBProfile> > pySBInterpolatedKImage(
                "SBInterpolatedKImage", bp::init<const SBInterpolatedKImage &>()
            );
            pySBInterpolatedKImage
                .def(bp::init<const BaseImage<double> &,
                              double, double, double,
                              boost::shared_ptr<Interpolant>,
                              double, double, bool,
                              boost::shared_ptr<GSParams> >(
                                  (bp::arg("data"),
                                   bp::arg("dk"),
                                   bp::arg("stepk"),
                                   bp::arg("maxk"),
                                   bp::arg("kInterp"),
                                   bp::arg("xcen"),
                                   bp::arg("ycen"),
                                   bp::arg("cenIsSet"),
                                   bp::arg("gsparams")=bp::object())
                     ))
                .def("getKInterp", &SBInterpolatedKImage::getKInterp)
                .def("dK", &SBInterpolatedKImage::dK)
                .def("_cenIsSet", &SBInterpolatedKImage::cenIsSet)
                .def("_getKData", &SBInterpolatedKImage::getKData)
                ;
            wrapTemplates<float>(pySBInterpolatedKImage);
            wrapTemplates<double>(pySBInterpolatedKImage);
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
