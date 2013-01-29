#include "boost/python.hpp"
#include "Interpolant.h"
#include "CorrelatedNoise.h"

namespace bp = boost::python;

namespace galsim {

    struct PyCorrelationFunctions
    {

        static void wrap() {
        bp::def("_calculateCovarianceMatrix", 
                calculateCovarianceMatrix, 
                (bp::arg("sbprofile"), bp::arg("bounds"), bp::arg("dx"))
        );
        }

    };

    void pyExportCorrelationFunction()
    {
        PyCorrelationFunctions::wrap();
    }

} // namespace galsim
