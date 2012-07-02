#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBBox.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBBox 
    {

        static SBBox * construct(
            const bp::object & xw,
            const bp::object & yw,
            double flux
        ) {
            if (xw.ptr() == Py_None || yw.ptr() == Py_None) {
                PyErr_SetString(PyExc_TypeError, "SBBox requires x and y width parameters");
                bp::throw_error_already_set();
            }
            return new SBBox(bp::extract<double>(xw), bp::extract<double>(yw), flux);
        }

        static void wrap() {
            bp::class_<SBBox,bp::bases<SBProfile> >("SBBox", bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("xw")=bp::object(), bp::arg("yw")=bp::object(), 
                          bp::arg("flux")=1.))
                )
                .def(bp::init<const SBBox &>())
                .def("getXWidth", &SBBox::getXWidth)
                .def("getYWidth", &SBBox::getYWidth)
                ;
        }
    };

    void pyExportSBBox() 
    {
        PySBBox::wrap();
    }

} // namespace galsim
