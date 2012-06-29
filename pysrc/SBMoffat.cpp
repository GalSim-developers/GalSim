#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBMoffat.h"

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

    struct PySBMoffat 
    {

        static SBMoffat * construct(
            double beta, 
            const bp::object & fwhm,
            const bp::object & scale_radius,
            const bp::object & half_light_radius,
            double trunc, 
            double flux
        ) {
            double s = 1.0;
            checkRadii(half_light_radius, scale_radius, fwhm);
            SBMoffat::RadiusType rType = SBMoffat::FWHM;
            if (fwhm.ptr() != Py_None) {
                s = bp::extract<double>(fwhm);
            }
            if (scale_radius.ptr() != Py_None) {
                s = bp::extract<double>(scale_radius);
                rType = SBMoffat::SCALE_RADIUS;
            }
            if (half_light_radius.ptr() != Py_None) {
                s = bp::extract<double>(half_light_radius);
                rType = SBMoffat::HALF_LIGHT_RADIUS;
            }
            return new SBMoffat(beta, s, rType, trunc, flux);
        }

        static void wrap() {
            bp::class_<SBMoffat,bp::bases<SBProfile> >("SBMoffat", bp::no_init)
                .def("__init__", 
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("beta"), bp::arg("fwhm")=bp::object(), 
                          bp::arg("scale_radius")=bp::object(),
                          bp::arg("half_light_radius")=bp::object(),
                          bp::arg("trunc")=0., bp::arg("flux")=1.)
                     )
                )
                .def(bp::init<const SBMoffat &>())
                .def("getBeta", &SBMoffat::getBeta)
                .def("getScaleRadius", &SBMoffat::getScaleRadius)
                .def("getFWHM", &SBMoffat::getFWHM)
                .def("getHalfLightRadius", &SBMoffat::getHalfLightRadius)
                ;
        }
    };

    void pyExportSBMoffat() 
    {
        PySBMoffat::wrap();
    }

} // namespace galsim
