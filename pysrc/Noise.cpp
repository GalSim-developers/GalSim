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
#include "Noise.h"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyBaseNoise {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (BaseNoise::*) (ImageView<U>) )&BaseNoise::applyTo, "", 
                 (bp::arg("image")))
            ;
    }

    static void wrap() {
        // Note that class docstrings are now added in galsim/random.py

        bp::class_<BaseNoise, boost::noncopyable> pyBaseNoise("BaseNoise", "", bp::no_init);
        wrapTemplates<float>(pyBaseNoise);
        wrapTemplates<double>(pyBaseNoise);
    }

};

#if 0
struct PyGaussianNoise {

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

struct PyPoissonNoise {

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
#endif

struct PyCCDNoise{

    static void wrap() {

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<CCDNoise, bp::bases<BaseNoise> > pyCCDNoise("CCDNoise", "", bp::no_init);
        pyCCDNoise
            .def(bp::init<const BaseDeviate&, double, double>(
                (bp::arg("rng"), bp::arg("gain")=1., bp::arg("read_noise")=0.)
                ))
            .def("getGain", &CCDNoise::getGain, "")
            .def("setGain", &CCDNoise::setGain, "")
            .def("getReadNoise", &CCDNoise::getReadNoise, "")
            .def("setReadNoise", &CCDNoise::setReadNoise, "")
            ;
    }

};

} // anonymous

void pyExportNoise() {
    PyBaseNoise::wrap();
    //PyGaussianNoise::wrap();
    //PyPoissonNoise::wrap();
    PyCCDNoise::wrap();
}

} // namespace galsim
