#include "boost/python.hpp"
#include "integ/Int.h"

#include <iostream>

namespace bp = boost::python;

namespace galsim {
namespace integ {
namespace {

    // A C++ function object that just calls a python function.
    class PyFunc :
        public std::unary_function<double, double>
    {
    public:
        PyFunc(const bp::object& func) : _func(func) {}
        double operator()(double x) const
        { return bp::extract<double>(_func(x)); }
    private:
        const bp::object& _func;
    };

    // Integrate a python function using int1d.
    bp::tuple PyInt1d(const bp::object& func, double min, double max,
                      double rel_err=DEFRELERR, double abs_err=DEFABSERR)
    { 
        PyFunc pyfunc(func);
        bool success;
        std::string err_msg;
        double res = int1d_nothrow(pyfunc, min, max, success, err_msg, rel_err, abs_err);
        if (success) {
            return bp::make_tuple(true, res);
        } else {
            return bp::make_tuple(false, err_msg);
        }
    }

} // anonymous


void pyExportInteg() {

    bp::def("PyInt1d",
            &PyInt1d, (bp::args("func", "min", "max"),
                       bp::arg("rel_err")=DEFRELERR, bp::arg("abs_err")=DEFABSERR),
            "Calculate the integral of the given 1-d function from min to max.");

}

} // namespace integ
} // namespace galsim

