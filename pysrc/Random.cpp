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
            "UniformDeviate is foundation of the Random classes: other distributions take a\n"
            "UniformDeviate as construction argument and execute some transformation of the\n" 
            "distribution. Can be seeded with a long int, or by default will be seeded by the\n"
            "system microsecond counter.\n" 
            "\n"
            "Copy constructor and assignment operator are kept private since you probably do not\n"
            "want two 'random' number generators producing the same sequence of numbers in your\n"
            "code!\n"
            "\n"
            "Inititialization\n"
            "----------------\n"
            ">>> u = UniformDeviate() : Initializes u to be a UniformDeviate instance, and seeds\n"
            "                           the PRNG using current time.\n"
            "\n"
            ">>> u = UniformDeviate(lseed) : Initializes u to be a UniformDeviate instance, and\n"
            "                                seeds the PRNG using specified long integer lseed.\n" 
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to u() then generate\n"
            "pseudo-random numbers distributed uniformly in the interval [0., 1.).\n"
            "\n"
            "Re-seeding\n"
            "----------\n"
            "The seed for the UniformDeviate can be reset at any time via (e.g.):\n"
            "\n"
            ">>> u.seed() : Re-seeds using current time.\n"
            "\n"
            ">>> u.seed(lseed) : Re-seeds using specified long integer lseed.\n"
            "\n"
            ;

        bp::class_<UniformDeviate,boost::noncopyable>("UniformDeviate", doc, bp::init<>())
            .def(bp::init<long>(bp::arg("lseed")))
            .def("__call__", &UniformDeviate::operator(),
                 "Draw a new random number from the distribution.")
            .def("seed", (void (UniformDeviate::*) () )&UniformDeviate::seed, 
                 "Re-seed the PRNG using current time.")
            .def("seed", (void (UniformDeviate::*) (const long) )&UniformDeviate::seed, 
                 (bp::arg("lseed")), "Re-seed the PRNG using specified seed.")
            ;
    }

};

struct PyGaussianDeviate {

    static void wrap() {
        static char const * doc = 
            "\n"
            "Construct a new Gaussian-distributed RNG.\n"
            "\n"
            "Constructor requires reference to a UniformDeviate that generates the randoms,\n"
            "which are then transformed to Gaussian distribution.\n"
            ;
        std::cout << doc;
    }
};


} // anonymous

void pyExportRandom() {
    PyUniformDeviate::wrap();
}

} // namespace galsim
