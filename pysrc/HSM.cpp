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

        static const char* doc =
            "HSMParams stores a set of numbers that determine how the moments/shape estimation\n"
            "routines make speed/accuracy tradeoff decisions and/or store their results.\n"
            "\n"
            "The parameters, along with their default values, are as follows:\n"
            "\n"
            "nsig_rg           A parameter used to optimize convolutions by cutting off the galaxy\n"
            "                  profile.  In the first step of the re-Gaussianization method of PSF\n"
            "                  correction, a Gaussian approximation to the pre-seeing galaxy is\n"
            "                  calculated. If re-Gaussianization is called with the flag 0x4 (as\n"
            "                  is the default), then this approximation is cut off at nsig_rg\n"
            "                  sigma to save computation time in convolutions.\n"
            "nsig_rg2          A parameter used to optimize convolutions by cutting off the PSF\n"
            "                  residual profile.  In the re-Gaussianization method of PSF\n"
            "                  correction, a `PSF residual' (the difference between the true PSF\n"
            "                  and its best-fit Gaussian approximation) is constructed. If\n"
            "                  re-Gaussianization is called with the flag 0x8 (as is the default),\n"
            "                  then this PSF residual is cut off at nsig_rg2 sigma to save\n"
            "                  computation time in convolutions.\n"
            "max_moment_nsig2  A parameter for optimizing calculations of adaptive moments by\n"
            "                  cutting off profiles. This parameter is used to decide how many\n"
            "                  sigma^2 into the Gaussian adaptive moment to extend the moment\n"
            "                  calculation, with the weight being defined as 0 beyond this point.\n"
            "                  i.e., if max_moment_nsig2 is set to 25, then the Gaussian is\n"
            "                  extended to (r^2/sigma^2)=25, with proper accounting for elliptical\n"
            "                  geometry.  If this parameter is set to some very large number, then\n"
            "                  the weight is never set to zero and the exponential function is\n"
            "                  always called. Note: GalSim script devel/modules/test_mom_timing.py\n"
            "                  was used to choose a value of 25 as being optimal, in that for the\n"
            "                  cases that were tested, the speedups were typically factors of\n"
            "                  several, but the results of moments and shear estimation were\n"
            "                  changed by <10^-5.  Not all possible cases were checked, and so for\n"
            "                  use of this code for unusual cases, we recommend that users check\n"
            "                  that this value does not affect accuracy, and/or set it to some\n"
            "                  large value to completely disable this optimization.\n"
            "regauss_too_small A parameter for how strictly the re-Gaussianization code treats\n"
            "                  small galaxies. If this parameter is 1, then the re-Gaussianization\n"
            "                  code does not impose a cut on the apparent resolution before trying\n"
            "                  to measure the PSF-corrected shape of the galaxy; if 0, then it is\n"
            "                  stricter.  Using the default value of 1 prevents the\n"
            "                  re-Gaussianization PSF correction from completely failing at the\n"
            "                  beginning, before trying to do PSF correction, due to the crudest\n"
            "                  possible PSF correction (Gaussian approximation) suggesting that\n"
            "                  the galaxy is very small.  This could happen for some usable\n"
            "                  galaxies particularly when they have very non-Gaussian surface\n"
            "                  brightness profiles -- for example, if there's a prominent bulge\n"
            "                  that the adaptive moments attempt to fit, ignoring a more\n"
            "                  extended disk.  Setting a value of 1 is useful for keeping galaxies\n"
            "                  that would have failed for that reason.  If they later turn out to\n"
            "                  be too small to really use, this will be reflected in the final\n"
            "                  estimate of the resolution factor, and they can be rejected after\n"
            "                  the fact.\n"
            "adapt_order       The order to which circular adaptive moments should be calculated\n"
            "                  for KSB method. This parameter only affects calculations using the\n"
            "                  KSB method of PSF correction.  Warning: deviating from default\n"
            "                  value of 2 results in code running more slowly, and results have\n"
            "                  not been significantly tested.\n"
            "convergence_threshold  Accuracy (in x0, y0, and sigma, each as a fraction of sigma)\n"
            "                  when calculating adaptive moments.\n"
            "max_mom2_iter     Maximum number of iterations to use when calculating adaptive\n"
            "                  moments.  This should be sufficient in nearly all situations, with\n"
            "                  the possible exception being very flattened profiles.\n"
            "num_iter_default   Number of iterations to report in the output ShapeData structure\n"
            "                   when code fails to converge within max_mom2_iter iterations.\n"
            "bound_correct_wt   Maximum shift in centroids and sigma between iterations for\n"
            "                   adaptive moments.\n"
            "max_amoment        Maximum value for adaptive second moments before throwing\n"
            "                   exception.  Very large objects might require this value to be\n"
            "                   increased.\n"
            "max_ashift         Maximum allowed x / y centroid shift (units: pixels) between\n"
            "                   successive iterations for adaptive moments before throwing\n"
            "                   exception.\n"
            "ksb_moments_max    Use moments up to ksb_moments_max order for KSB method of PSF\n"
            "                   correction.\n"
            "ksb_sig_weight     The width of the weight function (in pixels) to use for the KSB\n"
            "                   method.  Normally, this is derived from the measured moments of the\n"
            "                   galaxy image; this keyword overrides this calculation.  Can be\n"
            "                   combined with ksb_sig_factor.\n"
            "ksb_sig_factor     Factor by which to multiply the weight function width for the KSB\n"
            "                   method (default: 1.0).  Can be combined with ksb_sig_weight.\n"
            "failed_moments     Value to report for ellipticities and resolution factor if shape\n"
            "                   measurement fails.\n";

        bp::class_<HSMParams> pyHSMParams("HSMParams", doc, bp::no_init);
        pyHSMParams
            .def(bp::init<
                 double, double, double, int, int, double, long, long, double, double, double,
                 int, double, double, double>(
                     (bp::arg("nsig_rg")=3.0,
                      bp::arg("nsig_rg2")=3.6,
                      bp::arg("max_moment_nsig2")=25.0,
                      bp::arg("regauss_too_small")=1,
                      bp::arg("adapt_order")=2,
                      bp::arg("convergence_threshold")=1.e-6,
                      bp::arg("max_mom2_iter")=400,
                      bp::arg("num_iter_default")=-1,
                      bp::arg("bound_correct_wt")=0.25,
                      bp::arg("max_amoment")=8000.,
                      bp::arg("max_ashift")=15.,
                      bp::arg("ksb_moments_max")=4,
                      bp::arg("ksb_sig_weight")=0.0,
                      bp::arg("ksb_sig_factor")=1.0,
                      bp::arg("failed_moments")=-1000.))
            )
            .def(bp::init<const HSMParams&>())
            .def_readonly("nsig_rg",&HSMParams::nsig_rg)
            .def_readonly("nsig_rg2",&HSMParams::nsig_rg2)
            .def_readonly("max_moment_nsig2",&HSMParams::max_moment_nsig2)
            .def_readonly("regauss_too_small",&HSMParams::regauss_too_small)
            .def_readonly("adapt_order",&HSMParams::adapt_order)
            .def_readonly("convergence_threshold",&HSMParams::convergence_threshold)
            .def_readonly("max_mom2_iter",&HSMParams::max_mom2_iter)
            .def_readonly("num_iter_default",&HSMParams::num_iter_default)
            .def_readonly("bound_correct_wt",&HSMParams::bound_correct_wt)
            .def_readonly("max_amoment",&HSMParams::max_amoment)
            .def_readonly("max_ashift",&HSMParams::max_ashift)
            .def_readonly("ksb_moments_max",&HSMParams::ksb_moments_max)
            .def_readonly("ksb_sig_weight",&HSMParams::ksb_sig_weight)
            .def_readonly("ksb_sig_factor",&HSMParams::ksb_sig_factor)
            .def_readonly("failed_moments",&HSMParams::failed_moments)
            .enable_pickling()
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
                                         boost::shared_ptr<HSMParams>);
        bp::def("_FindAdaptiveMomView",
                FAM_func(&FindAdaptiveMomView),
                (bp::arg("object_image"), bp::arg("object_mask_image"), bp::arg("guess_sig")=5.0,
                 bp::arg("precision")=1.0e-6, bp::arg("guess_centroid")=Position<double>(0.,0.),
                 bp::arg("hsmparams")=bp::object()),
                "Find adaptive moments of an image (with some optional args).");

        typedef CppShapeData (*ESH_func)(const BaseImage<U>&, const BaseImage<V>&,
                                         const BaseImage<int>&, float, const char *,
                                         const std::string&, double, double, double, Position<double>,
                                         boost::shared_ptr<HSMParams>);
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
