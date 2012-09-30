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
        PyFunc(const bp::object& func) : _func(func) 
        {
            std::cerr<<"PyFunc Constructor"<<std::endl;
            std::cout<<"_func = "<<&_func<<std::endl;
        }
        double operator()(double x) const
        {
            std::cerr<<"PyFunc op() x = "<<x<<std::endl;
            bp::object res = _func(x);
            std::cerr<<"_func(x) = "<<&res<<std::endl;
            double dres = bp::extract<double>(res);
            std::cerr<<"extract<double> = "<<&dres<<"  "<<dres<<std::endl;
            return dres;
            //return bp::extract<double>(_func(x));
        }
    private:
        const bp::object& _func;
    };

    // Integrate a python function using int1d.
    double PyInt1d(const bp::object& func, double min, double max,
                   double rel_err=DEFRELERR, double abs_err=DEFABSERR)
    {
        std::cerr<<"Start PyInt1d"<<std::endl;
        std::cerr<<"func = "<<&func<<std::endl;
        std::cerr<<"min = "<<&min<<"  "<<min<<std::endl;
        std::cerr<<"max = "<<&max<<"  "<<max<<std::endl;
        std::cerr<<"rel_err = "<<&rel_err<<"  "<<rel_err<<std::endl;
        std::cerr<<"abs_err = "<<&rel_err<<"  "<<abs_err<<std::endl;
        PyFunc pyfunc(func);
        std::cerr<<"pyfunc = "<<&pyfunc<<std::endl;
        double res = int1d(pyfunc, min, max, rel_err, abs_err);
        std::cerr<<"res = "<<&res<<"  "<<res<<std::endl;
        return res;
        //return int1d(pyfunc, min, max, rel_err, abs_err);
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

