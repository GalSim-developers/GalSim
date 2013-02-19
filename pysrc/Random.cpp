// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */
#include "boost/python.hpp"
#include "Random.h"
#include "CCDNoise.h"
#include "Image.h"

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
            .def("seed", (void (BaseDeviate::*) () )&BaseDeviate::seed, "")
            .def("seed", (void (BaseDeviate::*) (long) )&BaseDeviate::seed, (bp::arg("lseed")), "")
            .def("reset", (void (BaseDeviate::*) () )&BaseDeviate::reset, "")
            .def("reset", (void (BaseDeviate::*) (long) )&BaseDeviate::reset, (bp::arg("lseed")), 
                 "")
            .def("reset", (void (BaseDeviate::*) (const BaseDeviate&) )&BaseDeviate::reset, 
                 (bp::arg("dev")), "")
            ;
    }

};
struct PyUniformDeviate {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (UniformDeviate::*) (ImageView<U>) )&UniformDeviate::applyTo, "",
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
            .def("__call__", &UniformDeviate::operator(), "")
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
                 "", (bp::arg("image")))
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
            .def("__call__", &GaussianDeviate::operator(), "")
            .def("getMean", &GaussianDeviate::getMean, "")
            .def("setMean", &GaussianDeviate::setMean, "")
            .def("getSigma", &GaussianDeviate::getSigma, "")
            .def("setSigma", &GaussianDeviate::setSigma, "")
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
                 "", (bp::arg("image")))
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
            .def("__call__", &BinomialDeviate::operator(), "")
            .def("getN", &BinomialDeviate::getN, "")
            .def("setN", &BinomialDeviate::setN, "")
            .def("getP", &BinomialDeviate::getP, "")
            .def("setP", &BinomialDeviate::setP, "")
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
            .def("applyTo", (void (PoissonDeviate::*) (ImageView<U>) )&PoissonDeviate::applyTo, "",
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
            .def("__call__", &PoissonDeviate::operator(), "")
            .def("getMean", &PoissonDeviate::getMean, "")
            .def("setMean", &PoissonDeviate::setMean, "")
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
            .def("applyTo", (void (CCDNoise::*) (ImageView<U>) )&CCDNoise::applyTo, "", 
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
            .def("getGain", &CCDNoise::getGain, "")
            .def("setGain", &CCDNoise::setGain, "")
            .def("getReadNoise", &CCDNoise::getReadNoise, "")
            .def("setReadNoise", &CCDNoise::setReadNoise, "")
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
                 "", (bp::arg("image")))
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
            .def("__call__", &WeibullDeviate::operator(), "")
            .def("getA", &WeibullDeviate::getA, "")
            .def("setA", &WeibullDeviate::setA, "")
            .def("getB", &WeibullDeviate::getB, "")
            .def("setB", &WeibullDeviate::setB, "")
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
            .def("applyTo", (void (GammaDeviate::*) (ImageView<U>) )&GammaDeviate::applyTo, "", 
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
            .def("__call__", &GammaDeviate::operator(), "")
            .def("getAlpha", &GammaDeviate::getAlpha, "")
            .def("setAlpha", &GammaDeviate::setAlpha, "")
            .def("getBeta", &GammaDeviate::getBeta, "")
            .def("setBeta", &GammaDeviate::setBeta, "")
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
            .def("applyTo", (void (Chi2Deviate::*) (ImageView<U>) )&Chi2Deviate::applyTo, "",
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
            .def("__call__", &Chi2Deviate::operator(), "")
            .def("getN", &Chi2Deviate::getN, "")
            .def("setN", &Chi2Deviate::setN, "")
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
