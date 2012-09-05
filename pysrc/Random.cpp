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
        
        // Note that class docstrings are now added in galsim/random.py

        bp::class_<BaseDeviate> pyBaseDeviate("BaseDeviate", "", bp::init<>());
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
            .def("reset", (void (BaseDeviate::*) (const BaseDeviate&) )&BaseDeviate::reset, 
                 (bp::arg("dev")),
                 "Re-connect this Deviate with the rng in another one")
            ;
    }

};
struct PyUniformDeviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (UniformDeviate::*) (ImageView<U>) )&UniformDeviate::applyTo,
                 "",
                 (bp::arg("image")))
            ;
    }

    static void wrap() {

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<UniformDeviate, bp::bases<BaseDeviate> > pyUniformDeviate(
            "UniformDeviate", "", bp::init<>()
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

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<GaussianDeviate, bp::bases<BaseDeviate> > pyGaussianDeviate(
            "GaussianDeviate", "", bp::init<double, double >(
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

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<BinomialDeviate, bp::bases<BaseDeviate> > pyBinomialDeviate(
            "BinomialDeviate", "", bp::init<int, double >(
                (bp::arg("N")=1, bp::arg("p")=0.5)
            )
        );
        pyBinomialDeviate
            .def(bp::init<long, int, double>(
                (bp::arg("lseed"), bp::arg("N")=1, bp::arg("p")=0.5)
                ))
            .def(bp::init<const BaseDeviate&, int, double>(
                (bp::arg("dev"), bp::arg("N")=1, bp::arg("p")=0.5)
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

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<PoissonDeviate, bp::bases<BaseDeviate> > pyPoissonDeviate(
            "PoissonDeviate", "", bp::init<double>(
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

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<CCDNoise, bp::bases<BaseDeviate> > pyCCDNoise(
            "CCDNoise", "", bp::init<double, double >(
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

        // Note that class docstrings are now added in galsim/random.py     

        bp::class_<WeibullDeviate, bp::bases<BaseDeviate> > pyWeibullDeviate(
            "WeibullDeviate", "", bp::init<double, double >(
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

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<GammaDeviate, bp::bases<BaseDeviate> > pyGammaDeviate(
            "GammaDeviate", "", bp::init<double, double >(
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

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<Chi2Deviate, bp::bases<BaseDeviate> > pyChi2Deviate(
            "Chi2Deviate", "", bp::init<double >(
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
