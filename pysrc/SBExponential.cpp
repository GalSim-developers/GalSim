#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBExponential.h"

namespace bp = boost::python;

namespace galsim {

    // Used by multiple profile classes to ensure at most one radius is given.
    static void checkRadii(const bp::object & r1, const bp::object & r2, const bp::object & r3) 
    {
        int nRad = (r1.ptr() != Py_None) + (r2.ptr() != Py_None) + (r3.ptr() != Py_None);
        if (nRad > 1) {
            PyErr_SetString(PyExc_TypeError, "Multiple radius parameters given");
            bp::throw_error_already_set();
        }
        if (nRad == 0) {
            PyErr_SetString(PyExc_TypeError, "No radius parameter given");
            bp::throw_error_already_set();
        }
    }

    struct PySBExponential 
    {

        static SBExponential * construct(
            const bp::object & half_light_radius,
            const bp::object & scale_radius,
            double flux
        ) {
            double s = 1.0;
            checkRadii(half_light_radius, scale_radius, bp::object());
            if (half_light_radius.ptr() != Py_None) {
                s = bp::extract<double>(half_light_radius) / 1.6783469900166605; // not analytic
            }
            if (scale_radius.ptr() != Py_None) {
                s = bp::extract<double>(scale_radius);
            }
            return new SBExponential(s, flux);
        }

        static void wrap() {
            bp::class_<SBExponential,bp::bases<SBProfile> >(
                "SBExponential",
                "SBExponential(flux=1., half_light_radius=None, scale=None)\n\n"
                "Construct an exponential profile with the given flux and either half-light radius\n"
                "or scale length.  Exactly one radius must be provided.\n",
                bp::no_init)
                .def(
                    "__init__", bp::make_constructor(
                        &construct, bp::default_call_policies(),
                        (bp::arg("half_light_radius")=bp::object(), 
                         bp::arg("scale_radius")=bp::object(), (bp::arg("flux")=1.)))
                )
                .def(bp::init<const SBExponential &>())
                .def("getScaleRadius", &SBExponential::getScaleRadius)
                ;
        }
    };

    void pyExportSBExponential() 
    {
        PySBExponential::wrap();
    }

} // namespace galsim
