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
        wrapTemplates<double>(pyBaseNoise);
        wrapTemplates<float>(pyBaseNoise);
        wrapTemplates<int32_t>(pyBaseNoise);
        wrapTemplates<int16_t>(pyBaseNoise);
    }

};

#if 0
struct PyGaussianNoise {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (GaussianNoise::*) (ImageView<U>) )&GaussianNoise::applyTo,
                 "", (bp::arg("image")))
            ;
    }

    static void wrap() {

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<GaussianNoise, bp::bases<BaseNoise> > pyGaussianNoise(
            "GaussianNoise", "", bp::init<double, double >(
                (bp::arg("mean")=0., bp::arg("sigma")=1.)
            )
        );
        pyGaussianNoise
            .def(bp::init<long, double, double>(
                (bp::arg("lseed"), bp::arg("mean")=0., bp::arg("sigma")=1.)
                ))
            .def(bp::init<const BaseNoise&, double, double>(
                (bp::arg("dev"), bp::arg("mean")=0., bp::arg("sigma")=1.)
                ))
            .def("__call__", &GaussianNoise::operator(), "")
            .def("getMean", &GaussianNoise::getMean, "")
            .def("setMean", &GaussianNoise::setMean, "")
            .def("getSigma", &GaussianNoise::getSigma, "")
            .def("setSigma", &GaussianNoise::setSigma, "")
            ;
        wrapTemplates<int>(pyGaussianNoise);
        wrapTemplates<short>(pyGaussianNoise);
        wrapTemplates<float>(pyGaussianNoise);
        wrapTemplates<double>(pyGaussianNoise);
    }

};

struct PyPoissonNoise {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        wrapper
            .def("applyTo", (void (PoissonNoise::*) (ImageView<U>) )&PoissonNoise::applyTo, "",
                 (bp::arg("image")))
            ;
    }

    static void wrap() {

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<PoissonNoise, bp::bases<BaseNoise> > pyPoissonNoise(
            "PoissonNoise", "", bp::init<double>(
                (bp::arg("mean")=1.)
            )
        );
        pyPoissonNoise
            .def(bp::init<long, double>(
                (bp::arg("lseed"), bp::arg("mean")=1.)
                ))
            .def(bp::init<const BaseNoise&, double>(
                (bp::arg("dev"), bp::arg("mean")=1.)
                ))
            .def("__call__", &PoissonNoise::operator(), "")
            .def("getMean", &PoissonNoise::getMean, "")
            .def("setMean", &PoissonNoise::setMean, "")
            ;
        wrapTemplates<int>(pyPoissonNoise);
        wrapTemplates<short>(pyPoissonNoise);
        wrapTemplates<float>(pyPoissonNoise);
        wrapTemplates<double>(pyPoissonNoise);
    }

};
#endif

struct PyCCDNoise{

    static void wrap() {

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<CCDNoise, bp::bases<BaseNoise> > pyCCDNoise("CCDNoise", "", bp::no_init);
        pyCCDNoise
            .def(bp::init<BaseDeviate&, double, double, double>(
                (bp::arg("rng"),
                 bp::arg("sky_level")=0.,  bp::arg("gain")=1., bp::arg("read_noise")=0.)
                ))
            .def("getSkyLevel", &CCDNoise::getSkyLevel, "")
            .def("getGain", &CCDNoise::getGain, "")
            .def("getReadNoise", &CCDNoise::getReadNoise, "")
            .def("setSkyLevel", &CCDNoise::getSkyLevel, "")
            .def("setGain", &CCDNoise::setGain, "")
            .def("setReadNoise", &CCDNoise::setReadNoise, "")
            ;
    }

};

struct PyDeviateNoise{

    static void wrap() {

        // Note that class docstrings are now added in galsim/random.py

        bp::class_<DeviateNoise, bp::bases<BaseNoise> > pyDeviateNoise(
            "DeviateNoise", "", bp::no_init);
        pyDeviateNoise
            .def(bp::init<BaseDeviate&>(bp::arg("rng")))
            ;
    }

};

} // anonymous

void pyExportNoise() {
    PyBaseNoise::wrap();
    //PyGaussianNoise::wrap();
    //PyPoissonNoise::wrap();
    PyCCDNoise::wrap();
    PyDeviateNoise::wrap();
}

} // namespace galsim
