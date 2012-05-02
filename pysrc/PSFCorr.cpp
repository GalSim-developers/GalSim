#include "boost/python.hpp"
#include "hsm/PSFCorr.h"

namespace bp = boost::python;

namespace galsim {
namespace hsm {
namespace {

struct PyHSMShapeData {

    static void wrap() {
        static char const * doc = 
            "HSMShapeData object represents information from the HSM moments and PSF-correction functions.\n" 
            "See C++ docs for more detail.\n"
            ;

        bp::class_<HSMShapeData>("HSMShapeData", doc, bp::init<HSMShapeData &>());
    }
};

} // anonymous

void pyExportPSFCorr() {
    PyHSMShapeData::wrap();
}

} // namespace hsm
} // namespace galsim

