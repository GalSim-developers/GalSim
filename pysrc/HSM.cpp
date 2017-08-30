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

#define BOOST_PYTHON_MAX_ARITY 22  // We have a function with 21 params here...
                                   // c.f. www.boost.org/libs/python/doc/v2/configuration.html

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include "hsm/PSFCorr.h"

namespace bp = boost::python;

namespace galsim {
namespace hsm {
namespace {

struct PyHSMParams {

    static void wrap() {

        bp::class_<HSMParams> pyHSMParams("HSMParams", bp::no_init);
        pyHSMParams
            .def(bp::init<
                 double, double, double, int, int, double, long, long, double, double, double,
                 int, double, double, double>(
                     (bp::arg("nsig_rg"),
                      bp::arg("nsig_rg2"),
                      bp::arg("max_moment_nsig2"),
                      bp::arg("regauss_too_small"),
                      bp::arg("adapt_order"),
                      bp::arg("convergence_threshold"),
                      bp::arg("max_mom2_iter"),
                      bp::arg("num_iter_default"),
                      bp::arg("bound_correct_wt"),
                      bp::arg("max_amoment"),
                      bp::arg("max_ashift"),
                      bp::arg("ksb_moments_max"),
                      bp::arg("ksb_sig_weight"),
                      bp::arg("ksb_sig_factor"),
                      bp::arg("failed_moments"))))
            ;
    }
};

struct PyShapeData {

    static CppShapeData* ShapeData_init(
        const galsim::Bounds<int>& image_bounds, int moments_status,
        float observed_e1, float observed_e2,
        float moments_sigma, float moments_amp,
        const galsim::Position<double>& moments_centroid,
        double moments_rho4, int moments_n_iter,
        int correction_status, float corrected_e1, float corrected_e2,
        float corrected_g1, float corrected_g2, std::string meas_type,
        float corrected_shape_err, std::string correction_method,
        float resolution_factor, float psf_sigma,
        float psf_e1, float psf_e2, std::string error_message)
    {
        CppShapeData* data = new CppShapeData();
        data->image_bounds = image_bounds;
        data->moments_status = moments_status;
        data->observed_e1 = observed_e1;
        data->observed_e2 = observed_e2;
        data->moments_sigma = moments_sigma;
        data->moments_amp = moments_amp;
        data->moments_centroid = moments_centroid;
        data->moments_rho4 = moments_rho4;
        data->moments_n_iter = moments_n_iter;
        data->correction_status = correction_status;
        data->corrected_e1 = corrected_e1;
        data->corrected_e2 = corrected_e2;
        data->corrected_g1 = corrected_g1;
        data->corrected_g2 = corrected_g2;
        data->meas_type = meas_type;
        data->corrected_shape_err = corrected_shape_err;
        data->correction_method = correction_method;
        data->resolution_factor = resolution_factor;
        data->psf_sigma = psf_sigma;
        data->psf_e1 = psf_e1;
        data->psf_e2 = psf_e2;
        data->error_message = error_message;
        return data;
    }

    template <typename U, typename V>
    static void wrapTemplates() {
        typedef CppShapeData (*FAM_func)(const BaseImage<U>&, const BaseImage<int>&,
                                         double, double, Position<double>,
                                         const HSMParams&);
        bp::def("_FindAdaptiveMomView",
                FAM_func(&FindAdaptiveMomView),
                (bp::arg("object_image"), bp::arg("object_mask_image"), bp::arg("guess_sig")=5.0,
                 bp::arg("precision")=1.0e-6, bp::arg("guess_centroid")=Position<double>(0.,0.),
                 bp::arg("hsmparams")=bp::object()),
                "Find adaptive moments of an image (with some optional args).");

        typedef CppShapeData (*ESH_func)(const BaseImage<U>&, const BaseImage<V>&,
                                         const BaseImage<int>&, float, const char *,
                                         const std::string&, double, double, double, Position<double>,
                                         const HSMParams&);
        bp::def("_EstimateShearView",
                ESH_func(&EstimateShearView),
                (bp::arg("gal_image"), bp::arg("PSF_image"), bp::arg("gal_mask_image"),
                 bp::arg("sky_var")=0.0, bp::arg("shear_est")="REGAUSS",
                 bp::arg("recompute_flux")="FIT",
                 bp::arg("guess_sig_gal")=5.0, bp::arg("guess_sig_PSF")=3.0,
                 bp::arg("precision")=1.0e-6, bp::arg("guess_centroid")=Position<double>(0.,0.),
                 bp::arg("hsmparams")=bp::object()),
                "Estimate PSF-corrected shear for a galaxy, given a PSF (and some optional args).");
    };

    static void wrap() {
        bp::class_<CppShapeData>("CppShapeData", "", bp::no_init)
            .def(bp::init<>())
            .def(bp::init<const CppShapeData&>())
            .def("__init__",
                 bp::make_constructor(
                     &ShapeData_init, bp::default_call_policies(), (
                         bp::arg("image_bounds"), bp::arg("moments_status"),
                         bp::arg("observed_e1"), bp::arg("observed_e2"),
                         bp::arg("moments_sigma"), bp::arg("moments_amp"),
                         bp::arg("moments_centroid"),
                         bp::arg("moments_rho4"), bp::arg("moments_n_iter"),
                         bp::arg("correction_status"),
                         bp::arg("corrected_e1"), bp::arg("corrected_e2"),
                         bp::arg("corrected_g1"), bp::arg("corrected_g2"), bp::arg("meas_type"),
                         bp::arg("corrected_shape_err"), bp::arg("correction_method"),
                         bp::arg("resolution_factor"), bp::arg("psf_sigma"),
                         bp::arg("psf_e1"), bp::arg("psf_e2"), bp::arg("error_message")
                     )
                 )
            )
            .def_readonly("image_bounds", &CppShapeData::image_bounds)
            .def_readonly("moments_status", &CppShapeData::moments_status)
            .def_readonly("observed_e1", &CppShapeData::observed_e1)
            .def_readonly("observed_e2", &CppShapeData::observed_e2)
            .def_readonly("moments_sigma", &CppShapeData::moments_sigma)
            .def_readonly("moments_amp", &CppShapeData::moments_amp)
            .def_readonly("moments_centroid", &CppShapeData::moments_centroid)
            .def_readonly("moments_rho4", &CppShapeData::moments_rho4)
            .def_readonly("moments_n_iter", &CppShapeData::moments_n_iter)
            .def_readonly("correction_status", &CppShapeData::correction_status)
            .def_readonly("corrected_e1", &CppShapeData::corrected_e1)
            .def_readonly("corrected_e2", &CppShapeData::corrected_e2)
            .def_readonly("corrected_g1", &CppShapeData::corrected_g1)
            .def_readonly("corrected_g2", &CppShapeData::corrected_g2)
            .def_readonly("meas_type", &CppShapeData::meas_type)
            .def_readonly("corrected_shape_err", &CppShapeData::corrected_shape_err)
            .def_readonly("correction_method", &CppShapeData::correction_method)
            .def_readonly("resolution_factor", &CppShapeData::resolution_factor)
            .def_readonly("psf_sigma", &CppShapeData::psf_sigma)
            .def_readonly("psf_e1", &CppShapeData::psf_e1)
            .def_readonly("psf_e2", &CppShapeData::psf_e2)
            .def_readonly("error_message", &CppShapeData::error_message)
            .enable_pickling()
            ;

        wrapTemplates<float, float>();
        wrapTemplates<double, double>();
        wrapTemplates<double, float>();
        wrapTemplates<float, double>();
    }
};

} // anonymous

void pyExportHSM() {
    PyShapeData::wrap();
    PyHSMParams::wrap();
}

} // namespace hsm
} // namespace galsim
