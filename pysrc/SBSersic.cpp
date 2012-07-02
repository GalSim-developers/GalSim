#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBSersic.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBSersic 
    {

        static SBSersic * construct(
            double n, 
            const bp::object & half_light_radius,
            double flux
        ) {
            if (half_light_radius.ptr() == Py_None) {
                PyErr_SetString(PyExc_TypeError, "No radius parameter given");
                bp::throw_error_already_set();
            }
            return new SBSersic(n, bp::extract<double>(half_light_radius), flux);
        }
        static void wrap() {
            bp::class_<SBSersic,bp::bases<SBProfile> >("SBSersic", bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("n"), bp::arg("half_light_radius")=bp::object(),
                          bp::arg("flux")=1.)
                     )
                )
                .def(bp::init<const SBSersic &>())
                .def("getN", &SBSersic::getN)
                .def("getHalfLightRadius", &SBSersic::getHalfLightRadius)
                ;
        }
    };

    struct PySBDeVaucouleurs 
    {
        static SBDeVaucouleurs * construct(
            const bp::object & half_light_radius,
            double flux 
        ) {
            if (half_light_radius.ptr() == Py_None) {
                PyErr_SetString(PyExc_TypeError, "No radius parameter given");
                bp::throw_error_already_set();
            }
            return new SBDeVaucouleurs(bp::extract<double>(half_light_radius), flux);
        }

        static void wrap() {
            bp::class_<SBDeVaucouleurs,bp::bases<SBSersic> >(
                "SBDeVaucouleurs",bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("half_light_radius")=bp::object(), bp::arg("flux")=1.)
                     )
                )
                .def(bp::init<const SBDeVaucouleurs &>())
                .def("getHalfLightRadius", &SBDeVaucouleurs::getHalfLightRadius)
                ;
        }
    };

    void pyExportSBSersic() 
    {
        PySBSersic::wrap();
        PySBDeVaucouleurs::wrap();
    }

} // namespace galsim
