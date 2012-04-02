#include "boost/python.hpp"
#include "Random.h"

/*
 * Barney's note 24Mar12: currently these only generate single instances of each random deviate.
 * This is not optimized for generating large arrays of random deviates.  I spoke with Jim and we 
 * thought that it would be best to write this in C++, but Barney needs to learn a little more about
 * the Numpy C API.  For now, we'll have to fill in arrays in Python manually; hopefully only a 
 * temporary workaround.
 */

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
            "Initialization\n"
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
            "Pseudo-random number generator with Gaussian distribution.\n"
            "\n"
            "GaussianDeviate is constructed with reference to a UniformDeviate that will actually\n"
            "generate the randoms, which are then transformed to Gaussian distribution with\n"
            "chosen mean and standard deviation.\n"
            "\n"
            "As for UniformDeviate, the copy constructor and assignment operator are kept private\n"
            "since you probably do not want two random number generators producing the same\n"
            "sequence of numbers in your code!\n"
            "\n"
            "Wraps the Boost.Random normal_distribution at the C++ layer so that the parent\n"
            "UniformDeviate is given once at construction, and copy/assignment are hidden.\n"
            "\n"
            "Initialization\n"
            "----------------\n"
            ">>> g = GaussianDeviate(u, mean=0., sigma=1.) \n"
            "\n"
            "Initializes g to be a GaussianDeviate instance, and repeated calls to g() will\n"
            "return successive, psuedo-random Gaussian deviates with specified mean and sigma.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "u       a UniformDeviate instance (seed set there).\n"
            "mean    semi-optional mean for Gaussian distribution (default = 0.).\n"
            "sigma   optional sigma for Gaussian distribution (default = 1.).\n"
            "\n"
            "The mean parameter is semi-optional: an ArgumentError exception will be raised if\n"
            "sigma alone is specified without an accompanying mean. However, reversing their\n"
            "ordering is handled OK provided keyword args are named. (TODO: Fix this 'feature'\n"
            "if possible!)\n"
            "\n"
            ;
        bp::class_<GaussianDeviate,boost::noncopyable>(
            "GaussianDeviate", doc, bp::init< UniformDeviate&, bp::optional<double, double> >(
                (bp::arg("u_"), bp::arg("mean"), bp::arg("sigma"))
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as GaussianDeviate lives
            ]
            )
            .def("__call__", &GaussianDeviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Gaussian deviate with current mean and sigma\n")
            .def("getMean", &GaussianDeviate::getMean, "Get current distribution mean.")
            .def("setMean", &GaussianDeviate::setMean, "Set current distribution mean.")
            .def("getSigma", &GaussianDeviate::getSigma, "Get current distribution sigma.")
            .def("setSigma", &GaussianDeviate::setSigma, "Set current distribution sigma.")
            ;
    }

};

struct PyBinomialDeviate {

    static void wrap() {
        static char const * doc = 
            "\n"
            "Pseudo-random Binomial deviate for N trials each of probability p.\n"
            "\n"
            "BinomialDeviate is constructed with reference to a UniformDeviate that will actually\n"
            "generate the randoms, which are then transformed to Binomial distribution.  N is\n"
            "number of 'coin flips,' p is probability of 'heads,' and each call returns integer\n"
            "0 <= value <= N giving number of heads.\n"  
            "\n"
            "As for UniformDeviate, the copy constructor and assignment operator are kept private\n"
            "since you probably do not want two random number generators producing the same\n"
            "sequence of numbers in your code!\n"
            "\n"
            "Wraps the Boost.Random binomial_distribution at the C++ layer so that the parent\n"
            "UniformDeviate is given once at construction, and copy/assignment are hidden.\n"
            "\n"
            "Initialization\n"
            "----------------\n"
            ">>> b = BinomialDeviate(u, N=1., p=0.5) \n"
            "\n"
            "Initializes b to be a GaussianDeviate instance, and repeated calls to b() will\n"
            "return successive, psuedo-random Binomial deviates with specified N and p.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "u       a UniformDeviate instance (seed set there).\n"
            "N       number of 'coin flips' per trial (default `N = 1`).\n"
            "p       probability of success per coin flip (default `p = 0.5`).\n"
            "\n"
            "The N parameter is semi-optional: an ArgumentError exception will be raised if p\n"
            "alone is specified without an accompanying N. However, reversing their ordering is\n"
            "handled OK provided keyword args are named. (TODO: Fix this 'feature' if possible!)\n"
            "\n"
            ;
        bp::class_<BinomialDeviate,boost::noncopyable>(
            "BinomialDeviate", doc, bp::init< UniformDeviate&, bp::optional<double, double> >(
                (bp::arg("u_"), bp::arg("N"), bp::arg("p"))
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as BinomialDeviate lives
            ]
            )
            .def("__call__", &BinomialDeviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Binomial deviate with current N and p.\n")
            .def("getN", &BinomialDeviate::getN, "Get current distribution N.")
            .def("setN", &BinomialDeviate::setN, "Set current distribution N.")
            .def("getP", &BinomialDeviate::getP, "Get current distribution p.")
            .def("setP", &BinomialDeviate::setP, "Set current distribution p.")
            ;
    }

};

struct PyPoissonDeviate {

    static void wrap() {
        static char const * doc = 
            "\n"
            "Pseudo-random Poisson deviate with specified mean.\n"
            "\n"
            "PoissonDeviate is constructed with reference to a UniformDeviate that will actually\n"
            "generate the randoms, which are then transformed to Poisson distribution.  The input\n"
            "mean sets the mean and variance of the Poisson deviate. An integer deviate with this\n"
            "distribution is returned after each call.\n"
            "\n"
            "As for UniformDeviate, the copy constructor and assignment operator are kept private\n"
            "since you probably do not want two random number generators producing the same\n"
            "sequence of numbers in your code!\n"
            "\n"
            "Wraps the Boost.Random poisson_distribution at the C++ layer so that the parent\n"
            "UniformDeviate is given once at construction, and copy/assignment are hidden.\n"
            "\n"
            "Initialization\n"
            "----------------\n"
            ">>> p = PoissonDeviate(u, mean=1.)\n"
            "\n"
            "Initializes p to be a PoissonDeviate instance, and repeated calls to p() will\n"
            "return successive, psuedo-random Poisson deviates with specified mean.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "u       a UniformDeviate instance (seed set there).\n"
            "mean    mean of the distribution (default `mean = 1`).\n"
            "\n"
            ;
        bp::class_<PoissonDeviate,boost::noncopyable>(
            "PoissonDeviate", doc, bp::init< UniformDeviate&, bp::optional<double> >(
                (bp::arg("u_"), bp::arg("mean"))
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as PoissonDeviate lives
            ]
            )
            .def("__call__", &PoissonDeviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Poisson deviate with current mean.\n")
            .def("getMean", &PoissonDeviate::getMean, "Get current distribution mean.")
            .def("setMean", &PoissonDeviate::setMean, "Set current distribution mean.")
            ;
    }

};


} // anonymous

void pyExportRandom() {
    PyUniformDeviate::wrap();
    PyGaussianDeviate::wrap();
    PyBinomialDeviate::wrap();
    PyPoissonDeviate::wrap();
}

} // namespace galsim
