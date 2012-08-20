#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBGaussian.h"

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

    struct PySBGaussian 
    {

        static SBGaussian * construct(
            const bp::object & half_light_radius,
            const bp::object & sigma,
            const bp::object & fwhm,
            double flux
        ) {
            double s = 1.0;
            checkRadii(half_light_radius, sigma, fwhm);
            if (half_light_radius.ptr() != Py_None) {
                s = bp::extract<double>(half_light_radius) * 0.84932180028801907; // (2\ln2)^(-1/2)
            }
            if (sigma.ptr() != Py_None) {
                s = bp::extract<double>(sigma);
            }
            if (fwhm.ptr() != Py_None) {
                s = bp::extract<double>(fwhm) * 0.42466090014400953; // 1 / (2(2\ln2)^(1/2))
            }
            return new SBGaussian(s, flux);
        }

        static void wrap() {
            bp::class_<SBGaussian,bp::bases<SBProfile> >(
                "SBGaussian",
                "SBGaussian(flux=1., half_light_radius=None, sigma=None, fwhm=None)\n\n"
                "Construct an exponential profile with the given flux and half-light radius,\n"
                "sigma, or FWHM.  Exactly one radius must be provided.\n",
                bp::no_init)
                .def(
                    "__init__", bp::make_constructor(
                        &construct, bp::default_call_policies(),
                        (bp::arg("half_light_radius")=bp::object(), bp::arg("sigma")=bp::object(), 
                         bp::arg("fwhm")=bp::object(), bp::arg("flux")=1.))
                )
                .def(bp::init<const SBGaussian &>())
                .def("getSigma", &SBGaussian::getSigma)
                ;
        }
    };

    void pyExportSBGaussian() 
    {
        PySBGaussian::wrap();
    }

} // namespace galsim
