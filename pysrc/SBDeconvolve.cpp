#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBDeconvolve.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBDeconvolve 
    {

        static void wrap() {
            bp::class_< SBDeconvolve, bp::bases<SBProfile> >("SBDeconvolve", bp::no_init)
                .def(bp::init<const SBProfile &>(bp::args("adaptee")))
                .def(bp::init<const SBDeconvolve &>())
                ;
        }

    };

    void pyExportSBDeconvolve() 
    {
        PySBDeconvolve::wrap();
    }

} // namespace galsim
