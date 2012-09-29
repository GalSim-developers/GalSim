#include "boost/python.hpp"
#include "integ/Int.h"

namespace bp = boost::python;

namespace galsim {
namespace integ {
namespace {

    // A C++ function object that just calls a python function.
    class PyFunc :
        public std::unary_function<double, double>
    {
    public:
        PyFunc(bp::object func) : _func(func) {}
        double operator()(double x) const
        {
            return bp::extract<double>(_func(x));
        }
    private:
        const bp::object& _func;
    };

    // Integrate a python function using int1d.
    double PyInt1d(bp::object func, double min, double max,
                   double rel_err=DEFRELERR, double abs_err=DEFABSERR)
    {
        PyFunc pyfunc(func);
        return int1d(pyfunc, min, max, rel_err, abs_err);
    }

} // anonymous


void pyExportInt() {

    bp::def("int1d",
            &PyInt1d, (bp::args("func", "min", "max"),
                       bp::arg("rel_err")=DEFRELERR, bp::arg("abs_err")=DEFABSERR),
            "Calculate the integral of the given 1-d function from min to max.");

}

} // namespace integ
} // namespace galsim

