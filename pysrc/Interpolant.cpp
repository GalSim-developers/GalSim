#include "boost/python.hpp"
#include <cstdlib>
#include "Interpolant.h"

namespace bp = boost::python;

namespace galsim {

    struct PyInterpolant
    {

        static Interpolant* ConstructInterpolant(std::string str)
        {
            // Make it lowercase
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);

            // Return the right Interpolant according to the given string.
            if (str == "delta") return new Delta();
            else if (str == "nearest") return new Nearest();
            else if (str == "sinc") return new SincInterpolant();
            else if (str == "linear") return new Linear();
            else if (str == "cubic") return new Cubic();
            else if (str == "quintic") return new Quintic();
            else if (str.substr(0,7) == "lanczos") {
                int n = strtol(str.substr(7).c_str(),0,0);
                if (n <= 0) {
                    PyErr_SetString(PyExc_TypeError, "Invalid Lanczos order");
                    bp::throw_error_already_set();
                }
                return new Lanczos(n);
            } else {
                PyErr_SetString(PyExc_TypeError, "Invalid interpolant string");
                bp::throw_error_already_set();
                return 0;
            }
        }

        static Interpolant2d* ConstructInterpolant2d(const std::string& str)
        {
            boost::shared_ptr<Interpolant> i1d(ConstructInterpolant(str));
            return new InterpolantXY(i1d);
        }

        static void wrap()
        {
            // We wrap Interpolant classes as opaque, construct-only objects; we just
            // need to be able to make them from Python and pass them to C++.
            bp::class_<Interpolant,boost::noncopyable>("Interpolant", bp::no_init)
                .def("__init__", bp::make_constructor(
                        &ConstructInterpolant, bp::default_call_policies(), bp::arg("str")));
            bp::class_<Interpolant2d,boost::noncopyable>("Interpolant2d", bp::no_init)
                .def("__init__", bp::make_constructor(
                        &ConstructInterpolant2d, bp::default_call_policies(), bp::arg("str")));
            bp::class_<InterpolantXY,bp::bases<Interpolant2d>,boost::noncopyable>(
                "InterpolantXY",
                bp::init<boost::shared_ptr<Interpolant> >(bp::arg("i1d"))
            );
            bp::class_<Delta,bp::bases<Interpolant>,boost::noncopyable>(
                "Delta", bp::init<double>(bp::arg("width")=1E-3)
            );
            bp::class_<Nearest,bp::bases<Interpolant>,boost::noncopyable>(
                "Nearest", bp::init<double>(bp::arg("tol")=1E-3)
            );
            bp::class_<SincInterpolant,bp::bases<Interpolant>,boost::noncopyable>(
                "SincInterpolant", bp::init<double>(bp::arg("tol")=1E-3)
            );
            bp::class_<Linear,bp::bases<Interpolant>,boost::noncopyable>(
                "Linear", bp::init<double>(bp::arg("tol")=1E-3)
            );
            bp::class_<Lanczos,bp::bases<Interpolant>,boost::noncopyable>(
                "Lanczos", bp::init<int,bool,double>(
                    (bp::arg("n"), bp::arg("conserve_flux")=true, bp::arg("tol")=1E-4)
                )
            );
            bp::class_<Cubic,bp::bases<Interpolant>,boost::noncopyable>(
                "Cubic", bp::init<double>(bp::arg("tol")=1E-4)
            );
            bp::class_<Quintic,bp::bases<Interpolant>,boost::noncopyable>(
                "Quintic", bp::init<double>(bp::arg("tol")=1E-4)
            );
        }

    };
 
    void pyExportInterpolant()
    { PyInterpolant::wrap(); }

} // namespace galsim
