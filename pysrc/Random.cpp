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

struct PyBaseDeviate {

    static void wrap() {
        static char const * doc = 
            " Base class for all the various random deviates.\n"
            " This holds the essential random number generator that all the other classes use.\n"
            "\n"
            " All deviates have three constructors that define different ways of setting up\n"
            " the random number generator.\n"
            "\n"
            " 1) Only the arguments particular to the derived class (e.g. mean and sigma for \n"
            "    GaussianDeviate).  In this case, a new random number generator is created and\n"
            "    it is seeded using the computer's microsecond counter.\n"
            "\n"
            " 2) Using a particular seed as the first argument to the constructor.  \n"
            "    This will also create a new random number generator, but seed it with the \n"
            "    provided value.\n"
            "\n"
            " 3) Passing another BaseDeviate as the first arguemnt to the constructor.\n"
            "    This will make the new Deviate share the same underlying random number generator\n"
            "    with the other Deviate.  So you can make one Deviate (of any type), and seed\n"
            "    it with a particular deterministic value.  Then if you pass that Deviate \n"
            "    to any other one you make, they will all be using the same rng and have a \n"
            "    particular deterministic series of values.  (It doesn't have to be the first\n"
            "    one -- any one you've made later can also be used to seed a new one.)\n"
            "\n"
            " There is not much you can do with something that is only known to be a BaseDeviate\n"
            " rather than one of the derived classes other than construct it and change the \n"
            " seed, and use it as an argument to pass to other Deviate constructors.\n"
            ;

        bp::class_<BaseDeviate> pyBaseDeviate("BaseDeviate", doc, bp::init<>());
        pyBaseDeviate
            .def(bp::init<long>(bp::arg("lseed")))
            .def(bp::init<const BaseDeviate&>(bp::arg("dev")))
            .def("seed", (void (BaseDeviate::*) () )&BaseDeviate::seed, 
                 "Re-seed the PRNG using current time.")
            .def("seed", (void (BaseDeviate::*) (long) )&BaseDeviate::seed, 
                 (bp::arg("lseed")), "Re-seed the PRNG using specified seed.")
            .def("reset", (void (BaseDeviate::*) () )&BaseDeviate::reset, 
                 "Re-seed the PRNG using current time, and sever the connection to any other "
                 "Deviate.")
            .def("reset", (void (BaseDeviate::*) (long) )&BaseDeviate::reset, 
                 (bp::arg("lseed")),
                 "Re-seed the PRNG using specified seed, and sever the connection to any other "
                 "Deviate.")
            ;
    }

};
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
            "Initialization\n"
            "--------------\n"
            ">>> u = UniformDeviate() : Initializes u to be a UniformDeviate instance, and seeds\n"
            "                           the PRNG using current time.\n"
            "\n"
            ">>> u = UniformDeviate(lseed) : Initializes u to be a UniformDeviate instance, and\n"
            "                                seeds the PRNG using specified long integer lseed.\n" 
            "\n"
            ">>> u = UniformDeviate(dev) : Initializes u to be a UniformDeviate instance,\n"
            "                              and use the same RNG as dev\n"
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to u() then generate\n"
            "pseudo-random numbers distributed uniformly in the interval [0., 1.).\n"
            "\n"
            ;

        bp::class_<UniformDeviate, bp::bases<BaseDeviate> > pyUniformDeviate(
            "UniformDeviate", doc, bp::init<>()
        );
        pyUniformDeviate
            .def(bp::init<long>(bp::arg("lseed")))
            .def(bp::init<const BaseDeviate&>(bp::arg("dev")))
            .def("__call__", &UniformDeviate::operator(),
                 "Draw a new random number from the distribution.")
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
            "Initialization\n"
            "--------------\n"
            "\n"
            ">>> g = GaussianDeviate(mean=0., sigma=1.) \n"
            "\n"
            "Initializes g to be a GaussianDeviate instance using the current time for the seed.\n"
            "\n"
            ">>> g = GaussianDeviate(lseed, mean=0., sigma=1.) \n"
            "\n"
            "Initializes g using the specified seed.\n"
            "\n"
            ">>> g = GaussianDeviate(dev, mean=0., sigma=1.) \n"
            "\n"
            "Initializes g to share the same underlying random number generator as dev.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "mean     optional mean for Gaussian distribution (default = 0.).\n"
            "sigma    optional sigma for Gaussian distribution (default = 1.).\n"
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to g() then generate\n"
            "pseudo-random numbers Gaussian-distributed with the provided mean, sigma\n"
            ;
        bp::class_<GaussianDeviate, bp::bases<BaseDeviate> > pyGaussianDeviate(
            "GaussianDeviate", doc, bp::init<double, double >(
                (bp::arg("mean")=0., bp::arg("sigma")=1.)
            )
        );
        pyGaussianDeviate
            .def(bp::init<long, double, double>(
                (bp::arg("lseed"), bp::arg("mean")=0., bp::arg("sigma")=1.)
                ))
            .def(bp::init<const BaseDeviate&, double, double>(
                (bp::arg("dev"), bp::arg("mean")=0., bp::arg("sigma")=1.)
                ))
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
            "N is number of 'coin flips,' p is probability of 'heads,' and each call returns \n"
            "integer 0 <= value <= N giving number of heads.\n"  
            "\n"
            "Initialization\n"
            "--------------\n"
            "\n"
            ">>> b = BinomialDeviate(N=1., p=0.5) \n"
            "\n"
            "Initializes b to be a BinomialDeviate instance using the current time for the seed.\n"
            "\n"
            ">>> b = BinomialDeviate(lseed, N=1., p=0.5) \n"
            "\n"
            "Initializes b using the specified seed.\n"
            "\n"
            ">>> b = BinomialDeviate(dev, N=1., p=0.5) \n"
            "\n"
            "Initializes b to share the same underlying random number generator as dev.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "N        optional number of 'coin flips' per trial (default `N = 1`).\n"
            "p        optional probability of success per coin flip (default `p = 0.5`).\n"
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to b() then generate\n"
            "pseudo-random numbers binomial-distributed with the provided N, p, which\n"
            "must both be > 0.\n"
            ;
        bp::class_<BinomialDeviate, bp::bases<BaseDeviate> > pyBinomialDeviate(
            "BinomialDeviate", doc, bp::init<double, double >(
                (bp::arg("N")=1., bp::arg("p")=0.5)
            )
        );
        pyBinomialDeviate
            .def(bp::init<long, double, double>(
                (bp::arg("lseed"), bp::arg("N")=1., bp::arg("p")=0.5)
                ))
            .def(bp::init<const BaseDeviate&, double, double>(
                (bp::arg("dev"), bp::arg("N")=1., bp::arg("p")=0.5)
                ))
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
            "The input mean sets the mean and variance of the Poisson deviate. \n"
            "An integer deviate with this distribution is returned after each call.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            "\n"
            ">>> p = PoissonDeviate(mean=1.)\n"
            "\n"
            "Initializes g to be a PoissonDeviate instance using the current time for the seed.\n"
            "\n"
            ">>> p = PoissonDeviate(lseed, mean=1.)\n"
            "\n"
            "Initializes g using the specified seed.\n"
            "\n"
            ">>> p = PoissonDeviate(dev, mean=1.)\n"
            "\n"
            "Initializes g to share the same underlying random number generator as dev.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "mean     optional mean of the distribution (default `mean = 1`).\n"
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to p() will\n"
            "return successive, pseudo-random Poisson deviates with specified mean, which must be\n"
            "> 0.\n"
            ;
        bp::class_<PoissonDeviate, bp::bases<BaseDeviate> > pyPoissonDeviate(
            "PoissonDeviate", doc, bp::init<double>(
                (bp::arg("mean")=1.)
            )
        );
        pyPoissonDeviate
            .def(bp::init<long, double>(
                (bp::arg("lseed"), bp::arg("mean")=1.)
                ))
            .def(bp::init<const BaseDeviate&, double>(
                (bp::arg("dev"), bp::arg("mean")=1.)
                ))
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
            "A CCDNoise instance is initialized given a gain level in Electrons per ADU\n"
            "used for the Poisson noise term, and a Gaussian read noise in electrons (if\n"
            "gain > 0.) or ADU (if gain <= 0.).  With these parameters set, the CCDNoise operates\n"
            "on an Image, adding noise to each pixel following this model.\n" 
            "\n"
            "Initialization\n"
            "--------------\n"
            "\n"
            ">>> ccd_noise = CCDNoise(gain=1., read_noise=0.)\n"
            "\n"
            "Initializes ccd_noise to be a CCDNoise instance using the current time for the seed.\n"
            "\n"
            ">>> ccd_noise = CCDNoise(lseed, gain=1., read_noise=0.)\n"
            "\n"
            "Initializes ccd_noise to be a CCDNoise instance using the specified seed.\n"
            "\n"
            ">>> ccd_noise = CCDNoise(dev, gain=1., read_noise=0.)\n"
            "\n"
            "Initializes ccd_noise to share the same underlying random number generator as dev.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "gain        the gain for each pixel in electrons per ADU; setting gain <=0 will shut\n"
            "            off the Poisson noise, and the Gaussian rms will take the value\n" 
            "            read_noise as being in units of ADU rather than electrons [default=1.].\n"
            "read_noise  the read noise on each pixel in electrons (gain > 0.) or ADU (gain <= 0.)\n"
            "            setting read_noise=0. will shut off the Gaussian noise [default=0.].\n"
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to ccd_noise() will\n"
            "generate noise following this model.\n"
            ;
        bp::class_<CCDNoise, bp::bases<BaseDeviate> > pyCCDNoise(
            "CCDNoise", doc, bp::init<double, double >(
                (bp::arg("gain")=1., bp::arg("read_noise")=0.)
            )
        );
        pyCCDNoise
            .def(bp::init<long, double, double>(
                (bp::arg("lseed"), bp::arg("gain")=1., bp::arg("read_noise")=0.)
                ))
            .def(bp::init<const BaseDeviate&, double, double>(
                (bp::arg("dev"), bp::arg("gain")=1., bp::arg("read_noise")=0.)
                ))
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
            "The Weibull distribution is related to a number of other probability distributions;\n"
            "in particular, it interpolates between the exponential distribution (a=1) and the \n"
            "Rayleigh distribution (a=2). See http://en.wikipedia.org/wiki/Weibull_distribution\n"
            "(a=k and b=lambda in the notation adopted in the Wikipedia article).  The Weibull\n"
            "distribution is real valued and produces deviates >= 0.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            "\n"
            ">>> w = WeibullDeviate(a=1., b=1.) \n"
            "\n"
            "Initializes w to be a WeibullDeviate instance using the current time for the seed.\n"
            "\n"
            ">>> w = WeibullDeviate(lseed, a=1., b=1.) \n"
            "\n"
            "Initializes w using the specified seed.\n"
            "\n"
            ">>> w = WeibullDeviate(dev, a=1., b=1.) \n"
            "\n"
            "Initializes w to share the same underlying random number generator as dev.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "a        shape parameter of the distribution (default a = 1).\n"
            "b        scale parameter of the distribution (default b = 1).\n"
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to w() then generate\n"
            "pseudo-random numbers Weibull-distributed with shape and scale\n"
            "parameters a and b, which must both be > 0.\n"
            ;        
        bp::class_<WeibullDeviate, bp::bases<BaseDeviate> > pyWeibullDeviate(
            "WeibullDeviate", doc, bp::init<double, double >(
                (bp::arg("a")=1., bp::arg("b")=1.)
            )
        );
        pyWeibullDeviate
            .def(bp::init<long, double, double>(
                (bp::arg("lseed"), bp::arg("a")=1., bp::arg("b")=1.)
                ))
            .def(bp::init<const BaseDeviate&, double, double>(
                (bp::arg("dev"), bp::arg("a")=1., bp::arg("b")=1.)
                ))
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
            "See http://en.wikipedia.org/wiki/Gamma_distribution (although note that in the Boost\n"
            "random routine this class calls the notation is alpha=k and beta=theta).  The Gamma\n"
            "distribution is a real valued distribution producing deviates >= 0.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            "\n"
            ">>> gam = GammaDeviate(alpha=1., beta=1.) \n"
            "\n"
            "Initializes gam to be a GammaDeviate instance using the current time for the seed.\n"
            "\n"
            ">>> gam = GammaDeviate(lseed, alpha=1., beta=1.) \n"
            "\n"
            "Initializes gam using the specified seed.\n"
            "\n"
            ">>> gam = GammaDeviate(dev alpha=1., beta=1.) \n"
            "\n"
            "Initializes gam to share the same underlying random number generator as dev.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "alpha    shape parameter of the distribution (default alpha = 1).\n"
            "beta     scale parameter of the distribution (default beta = 1).\n"
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to g() will\n"
            "return successive, pseudo-random Gamma-distributed deviates with shape and scale\n"
            "parameters alpha and beta, which must both be > 0.\n"
            ;
        bp::class_<GammaDeviate, bp::bases<BaseDeviate> > pyGammaDeviate(
            "GammaDeviate", doc, bp::init<double, double >(
                (bp::arg("alpha")=1., bp::arg("beta")=1.)
            )
        );
        pyGammaDeviate
            .def(bp::init<long, double, double>(
                (bp::arg("lseed"), bp::arg("alpha")=1., bp::arg("beta")=1.)
                ))
            .def(bp::init<const BaseDeviate&, double, double>(
                (bp::arg("dev"), bp::arg("alpha")=1., bp::arg("beta")=1.)
                ))
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
            "See http://en.wikipedia.org/wiki/Chi-squared_distribution (although note that in the\n"
            "Boost random routine this class calls the notation adopted interprets k=n).\n"
            "The Chi^2 distribution is a real valued distribution producing deviates >= 0.\n"
            "\n"
            "Initialization\n"
            "--------------\n"
            "\n"
            ">>> chis = Chi2Deviate(n=1.) \n"
            "\n"
            "Initializes chis to be a Chi2Deviate instance using the current time for the seed.\n"
            "\n"
            ">>> chis = Chi2Deviate(lseed, n=1.) \n"
            "\n"
            "Initializes chis using the specified seed.\n"
            "\n"
            ">>> chis = Chi2Deviate(dev, n=1.) \n"
            "\n"
            "Initializes chis to share the same underlying random number generator as dev.\n"
            "\n"
            "Parameters:\n"
            "\n"
            "n        number of degrees of freedom for the output distribution (default n = 1).\n"
            "\n"
            "Calling\n"
            "-------\n"
            "Taking the instance from the above examples, successive calls to g() will\n"
            "return successive, pseudo-random Chi^2-distributed deviates with degrees-of-freedom\n"
            "parameter n, which must be > 0.\n"
            ;
        bp::class_<Chi2Deviate, bp::bases<BaseDeviate> > pyChi2Deviate(
            "Chi2Deviate", doc, bp::init<double >(
                (bp::arg("n")=1.)
            )
        );
        pyChi2Deviate
            .def(bp::init<long, double>(
                (bp::arg("lseed"), bp::arg("n")=1.)
                ))
            .def(bp::init<const BaseDeviate&, double>(
                (bp::arg("dev"), bp::arg("n")=1.)
                ))
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
    PyBaseDeviate::wrap();
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
