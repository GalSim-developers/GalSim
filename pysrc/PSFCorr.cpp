#include "boost/python.hpp"
#include "hsm/PSFCorr.h"

namespace bp = boost::python;

namespace galsim {
namespace hsm {
namespace {

struct PyHSMShapeData {

    static void wrap() {
        static char const * doc = 
            "HSMShapeData object represents information from the HSM moments and PSF-correction\n"
            "functions.  See C++ docs for more detail.\n"
            ;

        bp::class_<HSMShapeData>("HSMShapeData", doc, bp::init<>())
            .def_readwrite("image_bounds", &HSMShapeData::image_bounds)
            .def_readwrite("moments_status", &HSMShapeData::moments_status)
            .def_readwrite("observed_shape", &HSMShapeData::observed_shape)
            .def_readwrite("moments_sigma", &HSMShapeData::moments_sigma)
            .def_readwrite("moments_amp", &HSMShapeData::moments_amp)
            .def_readwrite("moments_centroid", &HSMShapeData::moments_centroid)
            .def_readwrite("moments_n_iter", &HSMShapeData::moments_n_iter)
            .def_readwrite("correction_status", &HSMShapeData::correction_status)
            .def_readwrite("corrected_shape", &HSMShapeData::corrected_shape)
            .def_readwrite("corrected_shape_err", &HSMShapeData::corrected_shape_err)
            .def_readwrite("correction_method", &HSMShapeData::correction_method)
            .def_readwrite("resolution_factor", &HSMShapeData::resolution_factor)
            ;
    }
};

} // anonymous

void pyExportPSFCorr() {
    PyHSMShapeData::wrap();
}

} // namespace hsm
} // namespace galsim

