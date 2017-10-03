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
#define BOOST_PYTHON_MAX_ARITY 20  // We have a function with 17 params here...
                                   // c.f. www.boost.org/libs/python/doc/v2/configuration.html
#include "boost/python.hpp"

#include "SBProfile.h"
#include "SBTransform.h"

namespace bp = boost::python;

namespace galsim {

    struct PyGSParams
    {
        static void wrap() {
            bp::class_<GSParams> ("GSParams", bp::no_init)
                .def(bp::init<
                    int, int, double, double, double, double, double, double, double, double,
                    double, double, double, double, int, double>((
                        bp::arg("minimum_fft_size"),
                        bp::arg("maximum_fft_size"),
                        bp::arg("folding_threshold"),
                        bp::arg("stepk_minimum_hlr"),
                        bp::arg("maxk_threshold"),
                        bp::arg("kvalue_accuracy"),
                        bp::arg("xvalue_accuracy"),
                        bp::arg("table_spacing"),
                        bp::arg("realspace_relerr"),
                        bp::arg("realspace_abserr"),
                        bp::arg("integration_relerr"),
                        bp::arg("integration_abserr"),
                        bp::arg("shoot_accuracy"),
                        bp::arg("allowed_flux_variation"),
                        bp::arg("range_division_for_extrema"),
                        bp::arg("small_fraction_of_flux"))));
        }
    };


    struct PySBProfile
    {
        template <typename U, typename W>
        static void wrapTemplates(W & wrapper) {
            // We don't need to wrap templates in a separate function, but it keeps us
            // from having to repeat each of the lines below for each type.
            // We also don't need to make 'W' a template parameter in this case,
            // but it's easier to do that than write out the full class_ type.
            wrapper
                .def("draw",
                     (double (SBProfile::*)(ImageView<U>, double) const)&SBProfile::draw,
                     (bp::arg("image"), bp::arg("dx")),
                     "Draw in-place and return the summed flux.");
            wrapper
                .def("drawK",
                     (void (SBProfile::*)(ImageView<std::complex<U> >, double) const)
                     &SBProfile::drawK,
                     (bp::arg("image"), bp::arg("dk")),
                     "Draw k-space image.");
        }

        static void wrap() {
            bp::class_<SBProfile> pySBProfile("SBProfile", bp::no_init);
            pySBProfile
                .def("xValue", &SBProfile::xValue)
                .def("kValue", &SBProfile::kValue)
                .def("maxK", &SBProfile::maxK)
                .def("stepK", &SBProfile::stepK)
                .def("isAxisymmetric", &SBProfile::isAxisymmetric)
                .def("hasHardEdges", &SBProfile::hasHardEdges)
                .def("isAnalyticX", &SBProfile::isAnalyticX)
                .def("isAnalyticK", &SBProfile::isAnalyticK)
                .def("centroid", &SBProfile::centroid)
                .def("getFlux", &SBProfile::getFlux)
                .def("getPositiveFlux", &SBProfile::getPositiveFlux)
                .def("getNegativeFlux", &SBProfile::getNegativeFlux)
                .def("maxSB", &SBProfile::maxSB)
                .def("shoot", &SBProfile::shoot, bp::args("pa", "u"))
                ;
            wrapTemplates<float>(pySBProfile);
            wrapTemplates<double>(pySBProfile);
        }
    };

    void pyExportSBProfile()
    {
        PySBProfile::wrap();
        PyGSParams::wrap();
    }

} // namespace galsim
