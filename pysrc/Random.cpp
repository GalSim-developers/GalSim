#include "boost/python.hpp"
#include "Random.h"
#include "CcdNoise.h"
#include "Image.h"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

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
            "--------------\n"
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
            "--------------\n"
            ">>> g = GaussianDeviate(uniform, mean=0., sigma=1.) \n"
            "\n"
            "Initializes g to be a GaussianDeviate instance, and repeated calls to g() will\n"
            "return successive, psuedo-random Gaussian deviates with specified mean and sigma.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "mean     optional mean for Gaussian distribution (default = 0.).\n"
            "sigma    optional sigma for Gaussian distribution (default = 1.).\n"
            "\n"
            ;
        bp::class_<GaussianDeviate,boost::noncopyable>(
            "GaussianDeviate", doc, bp::init< UniformDeviate&, double, double >(
                (bp::arg("uniform"), bp::arg("mean")=0., bp::arg("sigma")=1.)
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
            "--------------\n"
            ">>> b = BinomialDeviate(uniform, N=1., p=0.5) \n"
            "\n"
            "Initializes b to be a GaussianDeviate instance, and repeated calls to b() will\n"
            "return successive, psuedo-random Binomial deviates with specified N and p.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "N        optional number of 'coin flips' per trial (default `N = 1`).\n"
            "p        optional probability of success per coin flip (default `p = 0.5`).\n"
            "\n"
            ;
        bp::class_<BinomialDeviate,boost::noncopyable>(
            "BinomialDeviate", doc, bp::init< UniformDeviate&, double, double >(
                (bp::arg("uniform"), bp::arg("N")=1., bp::arg("p")=0.5)
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
            "--------------\n"
            ">>> p = PoissonDeviate(uniform, mean=1.)\n"
            "\n"
            "Initializes p to be a PoissonDeviate instance, and repeated calls to p() will\n"
            "return successive, psuedo-random Poisson deviates with specified mean.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "mean     optional mean of the distribution (default `mean = 1`).\n"
            "\n"
            ;
        bp::class_<PoissonDeviate,boost::noncopyable>(
            "PoissonDeviate", doc, bp::init< UniformDeviate&, double >(
                (bp::arg("uniform"), bp::arg("mean")=1.)
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

struct PyCcdNoise{

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        // NOTE: We wrap only the Image<const U> versions because these are sufficient;
        // Image<U> inherits from Image<const U>, so the former can be implicitly
        // converted to the latter by Boost.Python.
        wrapper
            .def("__call__", (void (CcdNoise::*) (Image<U> &) )&CcdNoise::operator(),
                 "Add noise to an input Image\n"
                 "\n"
                 "On output the Image will have been given an additional stochastic noise\n"
                 "according to the gain and read noise settings.\n",
                 (bp::arg("Image")))
            ;
    }

    static void wrap() {

        static char const * doc = 
            "\n"
            "Pseudo-random number generator with a basic CCD noise model.\n"
            "\n"
            "A CcdNoise instance is initialized given a UniformDeviate, a gain level in Electrons\n"
            "per ADU used for the Poisson noise term, and a Gaussian read noise in electrons (if\n"
            "gain > 0.) or ADU (if gain < 0.).  With these parameters set, the CcdNoise operates\n"
            "on an Image, adding noise to each pixel following this model.\n" 
            "\n"
            "The class must be given a reference to a UniformDeviate when constructed, which will\n"
            "be the source of random values for the noise implementation.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            ">>> ccd_noise = CcdNoise(uniform, gain=1., readnoise=0.)\n"
            "\n"
            "Initializes ccd_noise to be a CcdNoise instance.\n"
            "\n"
            "Subsequent calls to ccd_noise(Image) with an Image instance as the first argument \n"
            "add noise following this model to that Image.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform   a UniformDeviate instance (seed set there).\n"
            "gain      the gain for each pixel; setting gain <=0 will shut off the Poisson noise,\n"
            "          and the Gaussian RMS will take the value RMS=readNoise [default=1.].\n"
            "reanoise  the read noise on each pixel; setting readNoise=0 will shut off the\n"
            "          Gaussian noise [default=0.].\n"
            "\n"
            "Calling\n"
            "-------\n"
            ">>> ccd_noise(image)\n"
            "\n"
            "Image instance image will have CCD noise added following the instantiated model.\n"
            "\n"
            ;
        
        bp::class_<CcdNoise,boost::noncopyable>pyCcdNoise(
            "CcdNoise", doc, bp::init< UniformDeviate&, double, double >(
                (bp::arg("uniform"), bp::arg("gain")=1., bp::arg("readnoise")=0.)
            )[
                bp::with_custodian_and_ward<1,2>() // keep uniform (2) as long as CcdNoise lives
            ]
	);
        pyCcdNoise
            .def("getGain", &CcdNoise::getGain, "Get gain in current noise model.")
            .def("setGain", &CcdNoise::setGain, "Set gain in current noise model.")
            .def("getReadNoise", &CcdNoise::getReadNoise, 
                 "Get read noise in current noise model.")
            .def("setReadNoise", &CcdNoise::setReadNoise, 
                 "Set read noise in current noise model.")
            ;
        wrapTemplates<int>(pyCcdNoise);
        wrapTemplates<short>(pyCcdNoise);
        wrapTemplates<float>(pyCcdNoise);
        wrapTemplates<double>(pyCcdNoise);
    }

};


} // anonymous

void pyExportRandom() {
    PyUniformDeviate::wrap();
    PyGaussianDeviate::wrap();
    PyBinomialDeviate::wrap();
    PyPoissonDeviate::wrap();
    PyCcdNoise::wrap();
}


} // namespace galsim
