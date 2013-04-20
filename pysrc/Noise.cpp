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
#include "NumpyHelper.h"

namespace bp = boost::python;

namespace galsim {

    // Need this special CallBack version that inherits from bp::wrapper whenever
    // you are wrapping something that has virtual functions you want to call from
    // python and have them resolve correctly.
    class BaseNoiseCallBack : public BaseNoise,
                              public bp::wrapper<BaseNoise>
    {
    public:
        BaseNoiseCallBack(boost::shared_ptr<BaseDeviate> rng) : BaseNoise(rng) {}
        BaseNoiseCallBack(const BaseNoise& rhs) : BaseNoise(rhs) {}
        ~BaseNoiseCallBack() {}

        // Need to put every virtual function here in a way that python can understand.
        double getVariance() const
        {
            if (bp::override py_func = this->get_override("getVariance"))
                return py_func();
            else
                throw std::runtime_error("Cannot call getVariance from a pure BaseNoise instance");
        }

        void setVariance(double variance)
        {
            if (bp::override py_func = this->get_override("setVariance"))
                py_func(variance);
            else
                throw std::runtime_error("Cannot call setVariance from a pure BaseNoise instance");
        }

        void scaleVariance(double variance_ratio)
        {
            if (bp::override py_func = this->get_override("scaleVariance"))
                py_func(variance_ratio);
            else
                throw std::runtime_error("Cannot call scaleVariance from a pure BaseNoise instance");
        }

        template <typename T>
        void applyTo(ImageView<T> data)
        {
            if (bp::override py_func = this->get_override("applyTo"))
                py_func(data);
            else
                throw std::runtime_error("Cannot call applyTo from a pure BaseNoise instance");
        }

        void doApplyTo(ImageView<double>& data)
        { applyTo(data); }
        void doApplyTo(ImageView<float>& data)
        { applyTo(data); }
        void doApplyTo(ImageView<int32_t>& data)
        { applyTo(data); }
        void doApplyTo(ImageView<int16_t>& data)
        { applyTo(data); }
    };

    struct PyBaseNoise {

        template <typename U, typename W>
        static void wrapTemplates(W& wrapper) {
            typedef void (BaseNoise::* applyTo_func_type)(ImageView<U>);
            wrapper
                .def("applyTo", applyTo_func_type(&BaseNoise::applyTo), "", 
                     (bp::arg("image")))
                ;
        }

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py
            bp::class_<BaseNoiseCallBack,boost::noncopyable> pyBaseNoise(
                "BaseNoise", "", bp::no_init);
            pyBaseNoise
                .def(bp::init<boost::shared_ptr<BaseDeviate> >(bp::arg("rng")=bp::object()))
                .def("getRNG", &BaseNoise::getRNG, "")
                .def("setRNG", &BaseNoise::setRNG, "")
                .def("getVariance", &BaseNoise::getVariance, "")
                .def("setVariance", &BaseNoise::setVariance, "")
                .def("scaleVariance", &BaseNoise::scaleVariance, "")
                ;
            wrapTemplates<double>(pyBaseNoise);
            wrapTemplates<float>(pyBaseNoise);
            wrapTemplates<int32_t>(pyBaseNoise);
            wrapTemplates<int16_t>(pyBaseNoise);
        }

    };


    struct PyGaussianNoise {

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py
            bp::class_<GaussianNoise, bp::bases<BaseNoise> > pyGaussianNoise(
                "GaussianNoise", "", bp::init<boost::shared_ptr<BaseDeviate>, double>(
                    (bp::arg("rng")=bp::object(), bp::arg("sigma")=1.))
            );
            pyGaussianNoise
                .def("getSigma", &GaussianNoise::getSigma, "")
                .def("setSigma", &GaussianNoise::setSigma, "")
                ;
        }

    };

    struct PyPoissonNoise {

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py

            bp::class_<PoissonNoise, bp::bases<BaseNoise> > pyPoissonNoise(
                "PoissonNoise", "", bp::init<boost::shared_ptr<BaseDeviate>, double>(
                    (bp::arg("rng")=bp::object(), bp::arg("sky_level")=0.))
            );
            pyPoissonNoise
                .def("getSkyLevel", &PoissonNoise::getSkyLevel, "")
                .def("setSkyLevel", &PoissonNoise::setSkyLevel, "")
                ;
        }

    };

    struct PyCCDNoise{

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py

            bp::class_<CCDNoise, bp::bases<BaseNoise> > pyCCDNoise("CCDNoise", "", bp::no_init);
            pyCCDNoise
                .def(bp::init<boost::shared_ptr<BaseDeviate>, double, double, double>(
                        (bp::arg("rng")=bp::object(),
                         bp::arg("sky_level")=0.,  bp::arg("gain")=1., bp::arg("read_noise")=0.)
                ))
                .def("getSkyLevel", &CCDNoise::getSkyLevel, "")
                .def("getGain", &CCDNoise::getGain, "")
                .def("getReadNoise", &CCDNoise::getReadNoise, "")
                .def("setSkyLevel", &CCDNoise::setSkyLevel, "")
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
                .def(bp::init<boost::shared_ptr<BaseDeviate> >(bp::arg("dev")))
                ;
        }

    };

    void pyExportNoise() {
        PyBaseNoise::wrap();
        PyGaussianNoise::wrap();
        PyPoissonNoise::wrap();
        PyCCDNoise::wrap();
        PyDeviateNoise::wrap();
    }

} // namespace galsim
