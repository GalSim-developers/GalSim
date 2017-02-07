/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#include "galsim/IgnoreWarnings.h"

#define BOOST_NO_CXX11_SMART_PTR
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
            if (bp::override py_func = this->get_override("_setVariance"))
                py_func(variance);
            else
                throw std::runtime_error("Cannot call setVariance from a pure BaseNoise instance");
        }

        void scaleVariance(double variance_ratio)
        {
            if (bp::override py_func = this->get_override("_scaleVariance"))
                py_func(variance_ratio);
            else
                throw std::runtime_error("Cannot call scaleVariance from a pure BaseNoise instance");
        }

        template <typename T>
        void applyToView(ImageView<T> data)
        {
            if (bp::override py_func = this->get_override("applyToView"))
                py_func(data);
            else
                throw std::runtime_error("Cannot call applyToView from a pure BaseNoise instance");
        }

        void doApplyTo(ImageView<double>& data)
        { applyToView(data); }
        void doApplyTo(ImageView<float>& data)
        { applyToView(data); }
        void doApplyTo(ImageView<int32_t>& data)
        { applyToView(data); }
        void doApplyTo(ImageView<int16_t>& data)
        { applyToView(data); }
        void doApplyTo(ImageView<uint32_t>& data)
        { applyToView(data); }
        void doApplyTo(ImageView<uint16_t>& data)
        { applyToView(data); }
    };

    struct PyBaseNoise {

        template <typename U, typename W>
        static void wrapTemplates(W& wrapper) {
            typedef void (BaseNoise::* applyTo_func_type)(ImageView<U>);
            wrapper
                .def("applyToView", applyTo_func_type(&BaseNoise::applyToView),
                     (bp::arg("image")))
                ;
        }

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py
            bp::class_<BaseNoiseCallBack,boost::noncopyable> pyBaseNoise(
                "BaseNoise", bp::no_init);
            pyBaseNoise
                // No init defined.  Cannot create a bare BaseNoise class.
                .def("getRNG", &BaseNoise::getRNG)
                .add_property("rng", &BaseNoise::getRNG)
                .def("getVariance", &BaseNoise::getVariance)
                .def("_setRNG", &BaseNoise::setRNG)
                .def("_setVariance", &BaseNoise::setVariance)
                .def("_scaleVariance", &BaseNoise::scaleVariance)
                ;
            wrapTemplates<double>(pyBaseNoise);
            wrapTemplates<float>(pyBaseNoise);
            wrapTemplates<int32_t>(pyBaseNoise);
            wrapTemplates<int16_t>(pyBaseNoise);
            wrapTemplates<uint32_t>(pyBaseNoise);
            wrapTemplates<uint16_t>(pyBaseNoise);
        }

    };


    struct PyGaussianNoise {

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py
            bp::class_<GaussianNoise, bp::bases<BaseNoise> > pyGaussianNoise(
                "GaussianNoise", bp::init<boost::shared_ptr<BaseDeviate>, double>(
                    (bp::arg("rng")=bp::object(), bp::arg("sigma")=1.))
            );
            pyGaussianNoise
                .def("getSigma", &GaussianNoise::getSigma)
                .add_property("sigma", &GaussianNoise::getSigma)
                .def("_setSigma", &GaussianNoise::setSigma)
                .enable_pickling()
                ;
        }

    };

    struct PyPoissonNoise {

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py

            bp::class_<PoissonNoise, bp::bases<BaseNoise> > pyPoissonNoise(
                "PoissonNoise", bp::init<boost::shared_ptr<BaseDeviate>, double>(
                    (bp::arg("rng")=bp::object(), bp::arg("sky_level")=0.))
            );
            pyPoissonNoise
                .def("getSkyLevel", &PoissonNoise::getSkyLevel)
                .add_property("sky_level", &PoissonNoise::getSkyLevel)
                .def("_setSkyLevel", &PoissonNoise::setSkyLevel)
                .enable_pickling()
                ;
        }

    };

    struct PyCCDNoise{

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py

            bp::class_<CCDNoise, bp::bases<BaseNoise> > pyCCDNoise("CCDNoise", bp::no_init);
            pyCCDNoise
                .def(bp::init<boost::shared_ptr<BaseDeviate>, double, double, double>(
                        (bp::arg("rng")=bp::object(),
                         bp::arg("sky_level")=0.,  bp::arg("gain")=1., bp::arg("read_noise")=0.)
                ))
                .def("getSkyLevel", &CCDNoise::getSkyLevel)
                .def("getGain", &CCDNoise::getGain)
                .def("getReadNoise", &CCDNoise::getReadNoise)
                .add_property("sky_level", &CCDNoise::getSkyLevel)
                .add_property("gain", &CCDNoise::getGain)
                .add_property("read_noise", &CCDNoise::getReadNoise)
                .def("_setSkyLevel", &CCDNoise::setSkyLevel)
                .def("_setGain", &CCDNoise::setGain)
                .def("_setReadNoise", &CCDNoise::setReadNoise)
                .enable_pickling()
                ;
        }

    };

    struct PyDeviateNoise{

        static void wrap() {

            // Note that class docstrings are now added in galsim/random.py

            bp::class_<DeviateNoise, bp::bases<BaseNoise> > pyDeviateNoise(
                "DeviateNoise", bp::no_init);
            pyDeviateNoise
                .def(bp::init<boost::shared_ptr<BaseDeviate> >(bp::arg("dev")))
                .enable_pickling()
                ;
        }

    };

    struct PyVarGaussianNoise {

        static void wrap() {
            // Note that class docstrings are now added in galsim/random.py
            bp::class_<VarGaussianNoise, bp::bases<BaseNoise> > pyVarGaussianNoise(
                "VarGaussianNoise",
                bp::init<boost::shared_ptr<BaseDeviate>, const BaseImage<float>& >(
                    (bp::arg("rng")=bp::object(), bp::arg("var_image")))
            );
            pyVarGaussianNoise
                .def("getVarImage", &VarGaussianNoise::getVarImage)
                .add_property("var_image", &VarGaussianNoise::getVarImage)
                .enable_pickling()
                ;
        }

    };


    void pyExportNoise() {
        PyBaseNoise::wrap();
        PyGaussianNoise::wrap();
        PyPoissonNoise::wrap();
        PyCCDNoise::wrap();
        PyDeviateNoise::wrap();
        PyVarGaussianNoise::wrap();
    }

} // namespace galsim
