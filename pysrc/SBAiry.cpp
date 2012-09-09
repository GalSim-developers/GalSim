#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBAiry.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBAiry 
    {
        static void wrap() {
            bp::class_<SBAiry,bp::bases<SBProfile> >("SBAiry", bp::no_init)
                .def(bp::init<double,double,double>(
                        (bp::arg("lam_over_diam"), bp::arg("obscuration")=0., bp::arg("flux")=1.))
                )
                .def(bp::init<const SBAiry &>())
                .def("getLamOverD", &SBAiry::getLamOverD)
                .def("getObscuration", &SBAiry::getObscuration)
                ;
        }
    };

    void pyExportSBAiry() 
    {
        PySBAiry::wrap();
    }

} // namespace galsim
