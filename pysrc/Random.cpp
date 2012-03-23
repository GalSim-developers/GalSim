#include "boost/python.hpp"
#include "Random.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyUniformDeviate {

    static void wrap() {
        static char const * doc = 
            "\n"
            "Pseudo-random number generator with uniform distribution in interval [0.,1.).\n"
            "\n"
            "UniformDeviate is foundation of the Random.h classes: other distributions take a\n"
            "UniformDeviate as construction argument and execute some transformation of the\n" 
            "distribution. Can be seeded with a long int, or by default will be seeded by the\n"
            "system microsecond counter.\n" 
            "Copy constructor and assignment operator are kept private since you probably do not\n"
            "want two 'random' number generators producing the same sequence of numbers in your\n"
            "code!\n"
            ;

        bp::class_<UniformDeviate,boost::noncopyable>("UniformDeviate", doc, bp::init<>())
            .def(bp::init<long>(bp::arg("lseed")))
            .def("operator", &UniformDeviate::operator(), 
                 "Draw a new random number from the distribution.")
	  .def("seed", (void (UniformDeviate::*) () )&UniformDeviate::seed, 
               "Re-seed the PRNG using current time.")
	  .def("seed", (void (UniformDeviate::*) (const long) )&UniformDeviate::seed, 
               (bp::arg("lseed")), "Re-seed the PRNG using specified seed.")
            ;
    }

};

} // anonymous

void pyExportRandom() {
    PyUniformDeviate::wrap();
}

} // namespace galsim
