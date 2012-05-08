#include "boost/python.hpp"
#include "Random.h"
#include "CCDNoise.h"
#include "Image.h"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyUniformDeviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (UniformDeviate::*) (ImageView<U>) )&UniformDeviate::applyTo,
                 "\n"
                 "Add Uniform deviates to every element in a supplied Image.\n"
                 "\n"
                 "Calling\n"
                 "-------\n"
                 ">>> UniformDeviate.applyTo(image) \n"
                 "\n"
                 "On output each element of the input Image will have a pseudo-random\n"
                 "UniformDeviate return value added to it.\n",
                 (bp::arg("image")))
            ;
    }

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

        bp::class_<UniformDeviate,boost::noncopyable>pyUniformDeviate(
            "UniformDeviate", doc, bp::init<>()
        );
        pyUniformDeviate
            .def(bp::init<long>(bp::arg("lseed")))
            .def("__call__", &UniformDeviate::operator(),
                 "Draw a new random number from the distribution.")
            .def("seed", (void (UniformDeviate::*) () )&UniformDeviate::seed, 
                 "Re-seed the PRNG using current time.")
            .def("seed", (void (UniformDeviate::*) (const long) )&UniformDeviate::seed, 
                 (bp::arg("lseed")), "Re-seed the PRNG using specified seed.")
            ;
        wrapTemplates<int>(pyUniformDeviate);
        wrapTemplates<short>(pyUniformDeviate);
        wrapTemplates<float>(pyUniformDeviate);
        wrapTemplates<double>(pyUniformDeviate);
    }

};

struct PyGaussianDeviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (GaussianDeviate::*) (ImageView<U>) )&GaussianDeviate::applyTo,
                 "\n"
                 "Add Gaussian deviates to every element in a supplied Image.\n"
                 "\n"
                 "Calling\n"
                 "-------\n"
                 ">>> GaussianDeviate.applyTo(image) \n"
                 "\n"
                 "On output each element of the input Image will have a pseudo-random\n"
                 "GaussianDeviate return value added to it, with current values of mean and\n"
                 "sigma.\n",
                 (bp::arg("image")))
            ;
    }

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
            "return successive, pseudo-random Gaussian deviates with specified mean and sigma.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "mean     optional mean for Gaussian distribution (default = 0.).\n"
            "sigma    optional sigma for Gaussian distribution (default = 1.).\n"
            "\n"
            ;
        bp::class_<GaussianDeviate,boost::noncopyable>pyGaussianDeviate(
            "GaussianDeviate", doc, bp::init< UniformDeviate&, double, double >(
                (bp::arg("uniform"), bp::arg("mean")=0., bp::arg("sigma")=1.)
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as GaussianDeviate lives
            ]
        );
        pyGaussianDeviate
            .def("__call__", &GaussianDeviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Gaussian deviate with current mean and sigma\n")
            .def("getMean", &GaussianDeviate::getMean, "Get current distribution mean.")
            .def("setMean", &GaussianDeviate::setMean, "Set current distribution mean.")
            .def("getSigma", &GaussianDeviate::getSigma, "Get current distribution sigma.")
            .def("setSigma", &GaussianDeviate::setSigma, "Set current distribution sigma.")
            ;
        wrapTemplates<int>(pyGaussianDeviate);
        wrapTemplates<short>(pyGaussianDeviate);
        wrapTemplates<float>(pyGaussianDeviate);
        wrapTemplates<double>(pyGaussianDeviate);
    }

};

struct PyBinomialDeviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (BinomialDeviate::*) (ImageView<U>) )&BinomialDeviate::applyTo,
                 "\n"
                 "Add Binomial deviates to every element in a supplied Image.\n"
                 "\n"
                 "Calling\n"
                 "-------\n"
                 ">>> BinomialDeviate.applyTo(image) \n"
                 "\n"
                 "On output each element of the input Image will have a pseudo-random\n"
                 "BinomialDeviate return value added to it, with current values of N and p.\n",
                 (bp::arg("image")))
            ;
    }

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
            "Initializes b to be a BinomialDeviate instance, and repeated calls to b() will\n"
            "return successive, pseudo-random Binomial deviates with specified N and p, which\n"
            "must both be > 0.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "N        optional number of 'coin flips' per trial (default `N = 1`).\n"
            "p        optional probability of success per coin flip (default `p = 0.5`).\n"
            "\n"
            ;
        bp::class_<BinomialDeviate,boost::noncopyable>pyBinomialDeviate(
            "BinomialDeviate", doc, bp::init< UniformDeviate&, double, double >(
                (bp::arg("uniform"), bp::arg("N")=1., bp::arg("p")=0.5)
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as BinomialDeviate lives
            ]
        );
        pyBinomialDeviate
            .def("__call__", &BinomialDeviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Binomial deviate with current N and p.\n")
            .def("getN", &BinomialDeviate::getN, "Get current distribution N.")
            .def("setN", &BinomialDeviate::setN, "Set current distribution N.")
            .def("getP", &BinomialDeviate::getP, "Get current distribution p.")
            .def("setP", &BinomialDeviate::setP, "Set current distribution p.")
            ;
        wrapTemplates<int>(pyBinomialDeviate);
        wrapTemplates<short>(pyBinomialDeviate);
        wrapTemplates<float>(pyBinomialDeviate);
        wrapTemplates<double>(pyBinomialDeviate);
    }

};

struct PyPoissonDeviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (PoissonDeviate::*) (ImageView<U>) )&PoissonDeviate::applyTo,
                 "\n"
                 "Add Poisson deviates to every element in a supplied Image.\n"
                 "\n"
                 "Calling\n"
                 "-------\n"
                 ">>> PoissonDeviate.applyTo(image) \n"
                 "\n"
                 "On output each element of the input Image will have a pseudo-random\n"
                 "PoissonDeviate return value added to it, with current mean.\n",
                 (bp::arg("image")))
            ;
    }

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
            "return successive, pseudo-random Poisson deviates with specified mean, which must be\n"
            "> 0.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "mean     optional mean of the distribution (default `mean = 1`).\n"
            "\n"
            ;
        bp::class_<PoissonDeviate,boost::noncopyable>pyPoissonDeviate(
            "PoissonDeviate", doc, bp::init< UniformDeviate&, double >(
                (bp::arg("uniform"), bp::arg("mean")=1.)
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as PoissonDeviate lives
            ]
        );
        pyPoissonDeviate
            .def("__call__", &PoissonDeviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Poisson deviate with current mean.\n")
            .def("getMean", &PoissonDeviate::getMean, "Get current distribution mean.")
            .def("setMean", &PoissonDeviate::setMean, "Set current distribution mean.")
            ;
        wrapTemplates<int>(pyPoissonDeviate);
        wrapTemplates<short>(pyPoissonDeviate);
        wrapTemplates<float>(pyPoissonDeviate);
        wrapTemplates<double>(pyPoissonDeviate);
    }

};

struct PyCCDNoise{

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (CCDNoise::*) (ImageView<U>) )&CCDNoise::applyTo,
                 "\n"
                 "Add noise to an input Image.\n"
                 "\n"
                 "Calling\n"
                 "-------\n"
                 ">>> CCDNoise.applyTo(image) \n"
                 "\n"
                 "On output the Image instance image will have been given an additional\n"
                 "stochastic noise according to the gain and read noise settings of the CCDNoise\n"
                 "instance.\n",
                 (bp::arg("image")))
            ;
    }

    static void wrap() {

        static char const * doc = 
            "\n"
            "Pseudo-random number generator with a basic CCD noise model.\n"
            "\n"
            "A CCDNoise instance is initialized given a UniformDeviate, a gain level in Electrons\n"
            "per ADU used for the Poisson noise term, and a Gaussian read noise in electrons (if\n"
            "gain > 0.) or ADU (if gain < 0.).  With these parameters set, the CCDNoise operates\n"
            "on an Image, adding noise to each pixel following this model.\n" 
            "\n"
            "The class must be given a reference to a UniformDeviate when constructed, which will\n"
            "be the source of random values for the noise implementation.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            ">>> ccd_noise = CCDNoise(uniform, gain=1., read_noise=0.)\n"
            "\n"
            "Initializes ccd_noise to be a CCDNoise instance.\n"
            "\n"
            "Subsequent calls to ccd_noise(Image) with an Image instance as the first argument \n"
            "add noise following this model to that Image.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform     a UniformDeviate instance (seed set there).\n"
            "gain        the gain for each pixel in electrons per ADU; setting gain <=0 will shut\n"
            "            off the Poisson noise, and the Gaussian rms will take the value\n" 
            "            read_noise as being in units of ADU rather than electrons [default=1.].\n"
            "read_noise  the read noise on each pixel in electrons (gain > 0.) or ADU (gain < 0.);n"
            "            setting read_noise=0. will shut off the Gaussian noise [default=0.].\n"
            "\n"
            "Calling\n"
            "-------\n"
            ">>> ccd_noise(image)\n"
            "\n"
            "Image instance image will have CCD noise added following the instantiated model.\n"
            "\n"
            ;
        
        bp::class_<CCDNoise,boost::noncopyable>pyCCDNoise(
            "CCDNoise", doc, bp::init< UniformDeviate&, double, double >(
                (bp::arg("uniform"), bp::arg("gain")=1., bp::arg("read_noise")=0.)
            )[
                bp::with_custodian_and_ward<1,2>() // keep uniform (2) as long as CCDNoise lives
            ]
        );
        pyCCDNoise
            .def("getGain", &CCDNoise::getGain, "Get gain in current noise model.")
            .def("setGain", &CCDNoise::setGain, "Set gain in current noise model.")
            .def("getReadNoise", &CCDNoise::getReadNoise, 
                 "Get read noise in current noise model.")
            .def("setReadNoise", &CCDNoise::setReadNoise, 
                 "Set read noise in current noise model.")
            ;
        wrapTemplates<int>(pyCCDNoise);
        wrapTemplates<short>(pyCCDNoise);
        wrapTemplates<float>(pyCCDNoise);
        wrapTemplates<double>(pyCCDNoise);
    }

};

struct PyWeibullDeviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (WeibullDeviate::*) (ImageView<U>) )&WeibullDeviate::applyTo,
                 "\n"
                 "Add Weibull-distributed deviates to every element in a supplied Image.\n"
                 "\n"
                 "Calling\n"
                 "-------\n"
                 ">>> WeibullDeviate.applyTo(image) \n"
                 "\n"
                 "On output each element of the input Image will have a pseudo-random\n"
                 "WeibullDeviate return value added to it, with current values of a and b.\n",
                 (bp::arg("image")))
            ;
    }

    static void wrap() {
        static char const * doc = 
            "\n"
            "Pseudo-random Weibull-distributed deviate for shape parameter a & scale parameter b\n"
            "\n"
            "WeibulllDeviate is constructed with reference to a UniformDeviate that will actually\n"
            "generate the randoms, which are then transformed to Weibull distribution with shape\n"
            "parameter a and scale parameter b.\n"
            "\n"
            "The Weibull distribution is related to a number of other probability distributions;\n"
            "in particular, it interpolates between the exponential distribution (a=1) and the \n"
            "Rayleigh distribution (a=2). See http://en.wikipedia.org/wiki/Weibull_distribution\n"
            "(a=k and b=lambda in the notation adopted in the Wikipedia article).  The Weibull\n"
            "distribution is real valued and produces deviates >= 0.\n"
            "\n"
            "As for UniformDeviate, the copy constructor and assignment operator are kept private\n"
            "since you probably do not want two random number generators producing the same\n"
            "sequence of numbers in your code!\n"
            "\n"
            "Wraps the Boost.Random weibull_distribution at the C++ layer so that the parent\n"
            "UniformDeviate is given once at construction, and copy/assignment are hidden.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            ">>> w = WeibullDeviate(uniform, a=1., b=1.) \n"
            "\n"
            "Initializes w to be a WeibullDeviate instance, and repeated calls to w() will\n"
            "return successive, pseudo-random Weibull-distributed deviates with shape and scale\n"
            "parameters a and b, which must both be > 0.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "a        shape parameter of the distribution (default a = 1).\n"
            "b        scale parameter of the distribution (default b = 1).\n"
            "\n"
            ;
        bp::class_<WeibullDeviate,boost::noncopyable>pyWeibullDeviate(
            "WeibullDeviate", doc, bp::init< UniformDeviate&, double, double >(
                (bp::arg("uniform"), bp::arg("a")=1., bp::arg("b")=1.)
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as BinomialDeviate lives
            ]
	);
        pyWeibullDeviate
            .def("__call__", &WeibullDeviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Weibull-distributed deviate with current a and b.\n")
            .def("getA", &WeibullDeviate::getA, "Get current distribution shape parameter a.")
            .def("setA", &WeibullDeviate::setA, "Set current distribution shape parameter a.")
            .def("getB", &WeibullDeviate::getB, "Get current distribution scale parameter b.")
            .def("setB", &WeibullDeviate::setB, "Set current distribution scale parameter b.")
            ;
        wrapTemplates<int>(pyWeibullDeviate);
        wrapTemplates<short>(pyWeibullDeviate);
        wrapTemplates<float>(pyWeibullDeviate);
        wrapTemplates<double>(pyWeibullDeviate);
    }

};

struct PyGammaDeviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (GammaDeviate::*) (ImageView<U>) )&GammaDeviate::applyTo,
                 "\n"
                 "Add Gamma-distributed deviates to every element in a supplied Image.\n"
                 "\n"
                 "Calling\n"
                 "-------\n"
                 ">>> GammaDeviate.applyTo(image) \n"
                 "\n"
                 "On output each element of the input Image will have a pseudo-random\n"
                 "GammaDeviate return value added to it, with current values of alpha and beta.\n",
                 (bp::arg("image")))
            ;
    }

    static void wrap() {
        static char const * doc = 
            "\n"
            "Pseudo-random Gamma-distributed deviate for parameters alpha & beta.\n"
            "\n"
            "GammaDeviate is constructed with reference to a UniformDeviate that will actually\n"
            "generate the randoms, which are then transformed to Gamma distribution with shape\n"
            "parameter alpha and scale parameter beta.\n"
            "\n"
            "See http://en.wikipedia.org/wiki/Gamma_distribution (although note that in the Boost\n"
            "random routine this class calls the notation is alpha=k and beta=theta).  The Gamma\n"
            "distribution is a real valued distribution producing deviates >= 0.\n"
            "\n"
            "As for UniformDeviate, the copy constructor and assignment operator are kept private\n"
            "since you probably do not want two random number generators producing the same\n"
            "sequence of numbers in your code!\n"
            "\n"
            "Wraps the Boost.Random gamma_distribution at the C++ layer so that the parent\n"
            "UniformDeviate is given once at construction, and copy/assignment are hidden.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            ">>> gam = GammaDeviate(uniform, alpha=1., beta=1.) \n"
            "\n"
            "Initializes gam to be a GammaDeviate instance, and repeated calls to gam() will\n"
            "return successive, pseudo-random Gamma-distributed deviates with shape and scale\n"
            "parameters alpha and beta, which must both be > 0.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "alpha    shape parameter of the distribution (default alpha = 1).\n"
            "beta     scale parameter of the distribution (default beta = 1).\n"
            "\n"
            ;
        bp::class_<GammaDeviate,boost::noncopyable>pyGammaDeviate(
            "GammaDeviate", doc, bp::init< UniformDeviate&, double, double >(
                (bp::arg("uniform"), bp::arg("alpha")=1., bp::arg("beta")=1.)
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as BinomialDeviate lives
            ]
	);
        pyGammaDeviate
            .def("__call__", &GammaDeviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Gamma-distributed deviate with current alpha and beta.\n")
            .def("getAlpha", &GammaDeviate::getAlpha, 
                 "Get current distribution shape parameter alpha.")
            .def("setAlpha", &GammaDeviate::setAlpha, 
                 "Set current distribution shape parameter alpha.")
            .def("getBeta", &GammaDeviate::getBeta, 
                 "Get current distribution scale parameter beta.")
            .def("setBeta", &GammaDeviate::setBeta, 
                 "Set current distribution scale parameter beta.")
            ;
        wrapTemplates<int>(pyGammaDeviate);
        wrapTemplates<short>(pyGammaDeviate);
        wrapTemplates<float>(pyGammaDeviate);
        wrapTemplates<double>(pyGammaDeviate);
    }

};

struct PyChi2Deviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (Chi2Deviate::*) (ImageView<U>) )&Chi2Deviate::applyTo,
                 "\n"
                 "Add Chi^2-distributed deviates to every element in a supplied Image.\n"
                 "\n"
                 "Calling\n"
                 "-------\n"
                 ">>> Chi2Deviate.applyTo(image) \n"
                 "\n"
                 "On output each element of the input Image will have a pseudo-random\n"
                 "Chi2Deviate return value added to it, with current degrees-of-freedom.\n"
                 "parameter n.\n",
                 (bp::arg("image")))
            ;
    }

    static void wrap() {
        static char const * doc = 
            "\n"
            "Pseudo-random Chi^2-distributed deviate for degrees-of-freedom parameter n.\n"
            "\n"
            "Chi2Deviate is constructed with reference to a UniformDeviate that will actually\n"
            "generate the randoms, which are then transformed to Chi^2 distribution with dof\n"
            "parameter n.\n"
            "\n"
            "See http://en.wikipedia.org/wiki/Chi-squared_distribution (although note that in the\n"
            "Boost random routine this class calls the notation adopted interprets k=n).\n"
            "The Chi^2 distribution is a real valued distribution producing deviates >= 0.\n"
            "\n"
            "As for UniformDeviate, the copy constructor and assignment operator are kept private\n"
            "since you probably do not want two random number generators producing the same\n"
            "sequence of numbers in your code!\n"
            "\n"
            "Wraps the Boost.Random chi_squared_distribution at the C++ layer so that the parent\n"
            "UniformDeviate is given once at construction, and copy/assignment are hidden.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            ">>> chis = Chi2Deviate(uniform, n=1.) \n"
            "\n"
            "Initializes chis to be a Chi2Deviate instance, and repeated calls to chis() will\n"
            "return successive, pseudo-random Chi^2-distributed deviates with degrees-of-freedom\n"
            "parameter n, which must be > 0.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "uniform  a UniformDeviate instance (seed set there).\n"
            "n        number of degrees of freedom for the output distribution (default n = 1).\n"
            "\n"
            ;
        bp::class_<Chi2Deviate,boost::noncopyable>pyChi2Deviate(
            "Chi2Deviate", doc, bp::init< UniformDeviate&, double >(
                (bp::arg("uniform"), bp::arg("n")=1.)
            )[
                bp::with_custodian_and_ward<1,2>() // keep u_ (2) as long as BinomialDeviate lives
            ]
	);
        pyChi2Deviate
            .def("__call__", &Chi2Deviate::operator(),
                 "Draw a new random number from the distribution.\n"
                 "\n"
                 "Returns a Chi2-distributed deviate with current n degrees of freedom.\n")
            .def("getN", &Chi2Deviate::getN, 
                 "Get current distribution n degrees of freedom.")
            .def("setN", &Chi2Deviate::setN, 
                 "Set current distribution n degrees of freedom.")
            ;
        wrapTemplates<int>(pyChi2Deviate);
        wrapTemplates<short>(pyChi2Deviate);
        wrapTemplates<float>(pyChi2Deviate);
        wrapTemplates<double>(pyChi2Deviate);
    }

};

} // anonymous

void pyExportRandom() {
    PyUniformDeviate::wrap();
    PyGaussianDeviate::wrap();
    PyBinomialDeviate::wrap();
    PyPoissonDeviate::wrap();
    PyCCDNoise::wrap();
    PyWeibullDeviate::wrap();
    PyGammaDeviate::wrap();
    PyChi2Deviate::wrap();
}

} // namespace galsim
