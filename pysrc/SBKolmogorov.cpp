#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBKolmogorov.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBKolmogorov 
    {
        static void wrap() {
            bp::class_<SBKolmogorov,bp::bases<SBProfile> >("SBKolmogorov", bp::no_init)
                .def(bp::init<double,double>(
                        (bp::arg("lam_over_r0"), bp::arg("flux")=1.))
                )
                .def(bp::init<const SBKolmogorov &>())
                .def("getLamOverR0", &SBKolmogorov::getLamOverR0)
                ;
        }
    };

    void pyExportSBKolmogorov() 
    {
        PySBKolmogorov::wrap();
    }

} // namespace galsim
