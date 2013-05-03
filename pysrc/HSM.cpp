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
            "failed_moments     Value to report for ellipticities and resolution factor if shape\n"
            "                   measurement fails.\n";

        bp::class_<HSMParams> pyHSMParams("_HSMParams", doc, bp::no_init);
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
            .def_readwrite("nsig_rg",&HSMParams::nsig_rg)
            .def_readwrite("nsig_rg2",&HSMParams::nsig_rg2)
            .def_readwrite("max_moment_nsig2",&HSMParams::max_moment_nsig2)
            .def_readwrite("regauss_too_small",&HSMParams::regauss_too_small)
            .def_readwrite("adapt_order",&HSMParams::adapt_order)
            .def_readwrite("max_mom2_iter",&HSMParams::max_mom2_iter)
            .def_readwrite("num_iter_default",&HSMParams::num_iter_default)
            .def_readwrite("bound_correct_wt",&HSMParams::bound_correct_wt)
            .def_readwrite("max_amoment",&HSMParams::max_amoment)
            .def_readwrite("max_ashift",&HSMParams::max_ashift)
            .def_readwrite("ksb_moments_max",&HSMParams::ksb_moments_max)
            .def_readwrite("failed_moments",&HSMParams::failed_moments)
            ;
    }
};

struct PyCppShapeData {

    template <typename U, typename V>
    static void wrapTemplates() {
        typedef CppShapeData (*FAM_func)(const ImageView<U> &, const ImageView<int> &, 
                                         double, double, double, double,
                                         boost::shared_ptr<HSMParams>);
        bp::def("_FindAdaptiveMomView",
                FAM_func(&FindAdaptiveMomView),
                (bp::arg("object_image"), bp::arg("object_mask_image"), bp::arg("guess_sig")=5.0, 
                 bp::arg("precision")=1.0e-6, bp::arg("guess_x_centroid")=-1000.0, 
                 bp::arg("guess_y_centroid")=-1000.0,
                 bp::arg("hsmparams")=bp::object()),
                "Find adaptive moments of an image (with some optional args).");

        typedef CppShapeData (*ESH_func)(const ImageView<U> &, const ImageView<V> &, 
                                         const ImageView<int> &, float, const char *,
                                         const std::string&, double, double, double, double, double,
                                         boost::shared_ptr<HSMParams>);
        bp::def("_EstimateShearView",
                ESH_func(&EstimateShearView),
                (bp::arg("gal_image"), bp::arg("PSF_image"), bp::arg("gal_mask_image"),
                 bp::arg("sky_var")=0.0, bp::arg("shear_est")="REGAUSS",
                 bp::arg("recompute_flux")="FIT",
                 bp::arg("guess_sig_gal")=5.0, bp::arg("guess_sig_PSF")=3.0,
                 bp::arg("precision")=1.0e-6, bp::arg("guess_x_centroid")=-1000.0,
                 bp::arg("guess_y_centroid")=-1000.0, bp::arg("hsmparams")=bp::object()),
                "Estimate PSF-corrected shear for a galaxy, given a PSF (and some optional args).");
    };

    static void wrap() {
        static char const * doc = 
            "CppShapeData object represents information from the HSM moments and PSF-correction\n"
            "functions.  See C++ docs for more detail.\n"
            ;

        bp::class_<CppShapeData>("_CppShapeData", doc, bp::init<>())
            .def_readwrite("image_bounds", &CppShapeData::image_bounds)
            .def_readwrite("moments_status", &CppShapeData::moments_status)
            .def_readwrite("observed_shape", &CppShapeData::observed_shape)
            .def_readwrite("moments_sigma", &CppShapeData::moments_sigma)
            .def_readwrite("moments_amp", &CppShapeData::moments_amp)
            .def_readwrite("moments_rho4", &CppShapeData::moments_rho4)
            .def_readwrite("moments_centroid", &CppShapeData::moments_centroid)
            .def_readwrite("moments_n_iter", &CppShapeData::moments_n_iter)
            .def_readwrite("correction_status", &CppShapeData::correction_status)
            .def_readwrite("corrected_e1", &CppShapeData::corrected_e1)
            .def_readwrite("corrected_e2", &CppShapeData::corrected_e2)
            .def_readwrite("corrected_g1", &CppShapeData::corrected_g1)
            .def_readwrite("corrected_g2", &CppShapeData::corrected_g2)
            .def_readwrite("meas_type", &CppShapeData::meas_type)
            .def_readwrite("corrected_shape_err", &CppShapeData::corrected_shape_err)
            .def_readwrite("correction_method", &CppShapeData::correction_method)
            .def_readwrite("resolution_factor", &CppShapeData::resolution_factor)
            .def_readwrite("error_message", &CppShapeData::error_message)
            ;

        wrapTemplates<float, float>();
        wrapTemplates<double, double>();
        wrapTemplates<double, float>();
        wrapTemplates<float, double>();
        wrapTemplates<int, int>();
    }
};

} // anonymous

void pyExportHSM() {
    PyCppShapeData::wrap();
    PyHSMParams::wrap();
}

} // namespace hsm
} // namespace galsim

