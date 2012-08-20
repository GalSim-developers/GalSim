#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBTransform.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBTransform 
    {
        static void wrap() {
            static char const * doc = 
                "SBTransform is an affine transformation of another SBProfile.\n"
                "Origin of original shape will now appear at x0.\n"
                "Flux is NOT conserved in transformation - SB is preserved."
                ;

            bp::class_< SBTransform, bp::bases<SBProfile> >("SBTransform", doc, bp::no_init)
                .def(bp::init<const SBProfile &, double, double, double, double,
                     Position<double>, double >(
                        (bp::args("sbin", "mA", "mB", "mC", "mD"),
                         bp::arg("x0")=Position<double>(0.,0.),
                         bp::arg("fluxScaling")=1.)
                ))
                .def(bp::init<const SBProfile &, const CppEllipse &, double>(
                        (bp::arg("sbin"), bp::arg("e")=CppEllipse(), bp::arg("fluxScaling")=1.)
                ))
                .def(bp::init<const SBTransform &>())
                ;
        }

    };

    void pyExportSBTransform() 
    {
        PySBTransform::wrap();
    }

} // namespace galsim
