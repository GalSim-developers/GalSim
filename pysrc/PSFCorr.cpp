// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */
#include "boost/python.hpp"
#include "hsm/PSFCorr.h"

namespace bp = boost::python;

namespace galsim {
namespace hsm {
namespace {

struct PyHSMParams {

    static void wrap() {

        static const char* doc = 
            ""
            "";

        bp::class_<HSMParams> pyHSMParams("HSMParams", doc, bp::no_init);
        pyHSMParams
            .def(bp::init<
                 double, double, double, int, int, long, long, double, double, double, int, double>(
                     (bp::arg("nsig_rg")=3.0,
                      bp::arg("nsig_rg2")=3.6,
                      bp::arg("max_moment_nsig2")=25.0,
                      bp::arg("regauss_too_small")=1,
                      bp::arg("adapt_order")=2,
                      bp::arg("max_mom2_iter")=400,
                      bp::arg("num_iter_default")=-1,
                      bp::arg("bound_correct_wt")=0.25,
                      bp::arg("max_amoment")=8000.,
                      bp::arg("max_ashift")=15.,
                      bp::arg("ksb_moments_max")=4,
                      bp::arg("failed_moments")=-1000.))
            )
            .def_readonly("nsig_rg",&HSMParams::nsig_rg)
            .def_readonly("nsig_rg2",&HSMParams::nsig_rg2)
            .def_readonly("max_moment_nsig2",&HSMParams::max_moment_nsig2)
            .def_readonly("regauss_too_small",&HSMParams::regauss_too_small)
            .def_readonly("adapt_order",&HSMParams::adapt_order)
            .def_readonly("max_mom2_iter",&HSMParams::max_mom2_iter)
            .def_readonly("num_iter_default",&HSMParams::num_iter_default)
            .def_readonly("bound_correct_wt",&HSMParams::bound_correct_wt)
            .def_readonly("max_amoment",&HSMParams::max_amoment)
            .def_readonly("max_ashift",&HSMParams::max_ashift)
            .def_readonly("ksb_moments_max",&HSMParams::ksb_moments_max)
            .def_readonly("failed_moments",&HSMParams::failed_moments)
            ;
    }
};

struct PyCppHSMShapeData {

    template <typename U, typename V>
    static void wrapTemplates() {
        typedef CppHSMShapeData (* FAM_func)(const ImageView<U> &, const ImageView<int> &, 
                                             double, double, double, double,
                                             boost::shared_ptr<HSMParams>);
        bp::def("_FindAdaptiveMomView",
                FAM_func(&FindAdaptiveMomView),
                (bp::arg("object_image"), bp::arg("object_mask_image"), bp::arg("guess_sig")=5.0, 
                 bp::arg("precision")=1.0e-6, bp::arg("guess_x_centroid")=-1000.0, 
                 bp::arg("guess_y_centroid")=-1000.0,
                 bp::arg("hsmparams")=bp::object()),
                "Find adaptive moments of an image (with some optional args).");

        typedef CppHSMShapeData (* ESH_func)(const ImageView<U> &, const ImageView<V> &, 
                                             const ImageView<int> &, float, const char *,
                                             unsigned long, double, double, double, double, double,
                                             boost::shared_ptr<HSMParams>);
        bp::def("_EstimateShearHSMView",
                ESH_func(&EstimateShearHSMView),
                (bp::arg("gal_image"), bp::arg("PSF_image"), bp::arg("gal_mask_image"), bp::arg("sky_var")=0.0,
                 bp::arg("shear_est")="REGAUSS", bp::arg("flags")=0xe, bp::arg("guess_sig_gal")=5.0,
                 bp::arg("guess_sig_PSF")=3.0, bp::arg("precision")=1.0e-6, bp::arg("guess_x_centroid")=-1000.0,
                 bp::arg("guess_y_centroid")=-1000.0,
                 bp::arg("hsmparams")=bp::object()),
                "Estimate PSF-corrected shear for a galaxy, given a PSF (and some optional args).");
    };

    static void wrap() {
        static char const * doc = 
            "CppHSMShapeData object represents information from the HSM moments and PSF-correction\n"
            "functions.  See C++ docs for more detail.\n"
            ;

        bp::class_<CppHSMShapeData>("_CppHSMShapeData", doc, bp::init<>())
            .def_readwrite("image_bounds", &CppHSMShapeData::image_bounds)
            .def_readwrite("moments_status", &CppHSMShapeData::moments_status)
            .def_readwrite("observed_shape", &CppHSMShapeData::observed_shape)
            .def_readwrite("moments_sigma", &CppHSMShapeData::moments_sigma)
            .def_readwrite("moments_amp", &CppHSMShapeData::moments_amp)
            .def_readwrite("moments_rho4", &CppHSMShapeData::moments_rho4)
            .def_readwrite("moments_centroid", &CppHSMShapeData::moments_centroid)
            .def_readwrite("moments_n_iter", &CppHSMShapeData::moments_n_iter)
            .def_readwrite("correction_status", &CppHSMShapeData::correction_status)
            .def_readwrite("corrected_e1", &CppHSMShapeData::corrected_e1)
            .def_readwrite("corrected_e2", &CppHSMShapeData::corrected_e2)
            .def_readwrite("corrected_g1", &CppHSMShapeData::corrected_g1)
            .def_readwrite("corrected_g2", &CppHSMShapeData::corrected_g2)
            .def_readwrite("meas_type", &CppHSMShapeData::meas_type)
            .def_readwrite("corrected_shape_err", &CppHSMShapeData::corrected_shape_err)
            .def_readwrite("correction_method", &CppHSMShapeData::correction_method)
            .def_readwrite("resolution_factor", &CppHSMShapeData::resolution_factor)
            .def_readwrite("error_message", &CppHSMShapeData::error_message)
            ;

        wrapTemplates<float, float>();
        wrapTemplates<double, double>();
        wrapTemplates<double, float>();
        wrapTemplates<float, double>();
        wrapTemplates<int, int>();
    }
};

} // anonymous

void pyExportPSFCorr() {
    PyCppHSMShapeData::wrap();
    PyHSMParams::wrap();
}

} // namespace hsm
} // namespace galsim

