#include "boost/python.hpp"
#include "hsm/PSFCorr.h"

namespace bp = boost::python;

namespace galsim {
namespace hsm {
namespace {

struct PyCppHSMShapeData {

    template <typename U, typename V>
    static void wrapTemplates() {
        typedef CppHSMShapeData (* FAM_func)(const ImageView<U> &, const ImageView<int> &, 
                                             double, double, double, double);
        bp::def("_FindAdaptiveMomView",
                FAM_func(&FindAdaptiveMomView),
                (bp::arg("object_image"), bp::arg("object_mask_image"), bp::arg("guess_sig")=5.0, 
                 bp::arg("precision")=1.0e-6, bp::arg("guess_x_centroid")=-1000.0, 
                 bp::arg("guess_y_centroid")=-1000.0),
                "Find adaptive moments of an image (with some optional args).");

        typedef CppHSMShapeData (* ESH_func)(const ImageView<U> &, const ImageView<V> &, 
                                             const ImageView<int> &, float, const char *,
                                             unsigned long, double, double, double, double, double);
        bp::def("_EstimateShearHSMView",
                ESH_func(&EstimateShearHSMView),
                (bp::arg("gal_image"), bp::arg("PSF_image"), bp::arg("gal_mask_image"), bp::arg("sky_var")=0.0,
                 bp::arg("shear_est")="REGAUSS", bp::arg("flags")=0xe, bp::arg("guess_sig_gal")=5.0,
                 bp::arg("guess_sig_PSF")=3.0, bp::arg("precision")=1.0e-6, bp::arg("guess_x_centroid")=-1000.0,
                 bp::arg("guess_y_centroid")=-1000.0),
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
            .def_readwrite("corrected_shape", &CppHSMShapeData::corrected_shape)
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
}

} // namespace hsm
} // namespace galsim

