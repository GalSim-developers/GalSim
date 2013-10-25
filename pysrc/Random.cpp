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
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#include "boost/python.hpp"
#include "Random.h"

namespace bp = boost::python;


// Note that class docstrings for all of these are now added in galsim/random.py     

namespace galsim {

    // Need this special CallBack version that inherits from bp::wrapper whenever
    // you are wrapping something that has virtual functions you want to call from
    // python and have them resolve correctly.
    class BaseDeviateCallBack : public BaseDeviate,
                                public bp::wrapper<BaseDeviate>
    {
    public:
        BaseDeviateCallBack(long lseed=0) : BaseDeviate(lseed) {}
        BaseDeviateCallBack(const BaseDeviate& rhs) : BaseDeviate(rhs) {}
        BaseDeviateCallBack(std::string& str) : BaseDeviate(str) {}
        ~BaseDeviateCallBack() {}

    protected:
        // This is the special magic needed so the virtual function calls back to the 
        // function defined in the python layer.
        double _val()
        {
            if (bp::override py_func = this->get_override("_val")) 
                return py_func();
            else 
                return BaseDeviate::_val();
        }
    };

    struct PyBaseDeviate {

        static void wrap() {
            bp::class_<BaseDeviateCallBack>
                pyBaseDeviate("BaseDeviate", "", bp::no_init);
            pyBaseDeviate
                .def(bp::init<long>(bp::arg("lseed")=0))
                .def(bp::init<const BaseDeviate&>(bp::arg("dev")))
                .def(bp::init<std::string>(bp::arg("str")))
                .def("seed", (void (BaseDeviate::*) (long) )&BaseDeviate::seed,
                     (bp::arg("lseed")=0), "")
                .def("reset", (void (BaseDeviate::*) (long) )&BaseDeviate::reset,
                     (bp::arg("lseed")=0), "")
                .def("reset", (void (BaseDeviate::*) (const BaseDeviate&) )&BaseDeviate::reset, 
                     (bp::arg("dev")), "")
                .def("clearCache", &BaseDeviate::clearCache, "")
                .def("serialize", &BaseDeviate::serialize, "")
                .def("duplicate", &BaseDeviate::duplicate, "")
                .enable_pickling()
                ;
        }

    };

    struct PyUniformDeviate {

        static void wrap() {
            bp::class_<UniformDeviate, bp::bases<BaseDeviate> >
                pyUniformDeviate("UniformDeviate", "", bp::no_init);
            pyUniformDeviate
                .def(bp::init<long>(bp::arg("lseed")=0))
                .def(bp::init<const BaseDeviate&>(bp::arg("dev")))
                .def(bp::init<std::string>(bp::arg("str")))
                .def("duplicate", &UniformDeviate::duplicate, "")
                .def("__call__", &UniformDeviate::operator(), "")
                .enable_pickling()
                ;
        }

    };

    struct PyGaussianDeviate {

        static void wrap() {
            bp::class_<GaussianDeviate, bp::bases<BaseDeviate> >
                pyGaussianDeviate("GaussianDeviate", "", bp::no_init);
            pyGaussianDeviate
                .def(bp::init<long, double, double>(
                        (bp::arg("lseed")=0, bp::arg("mean")=0., bp::arg("sigma")=1.)
                ))
                .def(bp::init<const BaseDeviate&, double, double>(
                        (bp::arg("dev"), bp::arg("mean")=0., bp::arg("sigma")=1.)
                ))
                .def(bp::init<std::string, double, double>(
                        (bp::arg("str"), bp::arg("mean")=0., bp::arg("sigma")=1.)
                ))
                .def("duplicate", &GaussianDeviate::duplicate, "")
                .def("__call__", &GaussianDeviate::operator(), "")
                .def("getMean", &GaussianDeviate::getMean, "")
                .def("setMean", &GaussianDeviate::setMean, "")
                .def("getSigma", &GaussianDeviate::getSigma, "")
                .def("setSigma", &GaussianDeviate::setSigma, "")
                .enable_pickling()
                ;
        }

    };

    struct PyBinomialDeviate {

        static void wrap() {
            bp::class_<BinomialDeviate, bp::bases<BaseDeviate> >
                pyBinomialDeviate("BinomialDeviate", "", bp::no_init);
            pyBinomialDeviate
                .def(bp::init<long, int, double>(
                        (bp::arg("lseed")=0, bp::arg("N")=1, bp::arg("p")=0.5)
                ))
                .def(bp::init<const BaseDeviate&, int, double>(
                        (bp::arg("dev"), bp::arg("N")=1, bp::arg("p")=0.5)
                ))
                .def(bp::init<std::string, int, double>(
                        (bp::arg("str")=0, bp::arg("N")=1, bp::arg("p")=0.5)
                ))
                .def("duplicate", &BinomialDeviate::duplicate, "")
                .def("__call__", &BinomialDeviate::operator(), "")
                .def("getN", &BinomialDeviate::getN, "")
                .def("setN", &BinomialDeviate::setN, "")
                .def("getP", &BinomialDeviate::getP, "")
                .def("setP", &BinomialDeviate::setP, "")
                .enable_pickling()
                ;
        }

    };

    struct PyPoissonDeviate {

        static void wrap() {
            bp::class_<PoissonDeviate, bp::bases<BaseDeviate> >
                pyPoissonDeviate("PoissonDeviate", "", bp::no_init);
            pyPoissonDeviate
                .def(bp::init<long, double>(
                        (bp::arg("lseed")=0, bp::arg("mean")=1.)
                ))
                .def(bp::init<const BaseDeviate&, double>(
                        (bp::arg("dev"), bp::arg("mean")=1.)
                ))
                .def(bp::init<std::string, double>(
                        (bp::arg("str")=0, bp::arg("mean")=1.)
                ))
                .def("duplicate", &PoissonDeviate::duplicate, "")
                .def("__call__", &PoissonDeviate::operator(), "")
                .def("getMean", &PoissonDeviate::getMean, "")
                .def("setMean", &PoissonDeviate::setMean, "")
                .enable_pickling()
                ;
        }

    };

    struct PyWeibullDeviate {

        static void wrap() {

            bp::class_<WeibullDeviate, bp::bases<BaseDeviate> >
                pyWeibullDeviate("WeibullDeviate", "", bp::no_init);
            pyWeibullDeviate
                .def(bp::init<long, double, double>(
                        (bp::arg("lseed")=0, bp::arg("a")=1., bp::arg("b")=1.)
                ))
                .def(bp::init<const BaseDeviate&, double, double>(
                        (bp::arg("dev"), bp::arg("a")=1., bp::arg("b")=1.)
                ))
                .def(bp::init<std::string, double, double>(
                        (bp::arg("str")=0, bp::arg("a")=1., bp::arg("b")=1.)
                ))
                .def("duplicate", &WeibullDeviate::duplicate, "")
                .def("__call__", &WeibullDeviate::operator(), "")
                .def("getA", &WeibullDeviate::getA, "")
                .def("setA", &WeibullDeviate::setA, "")
                .def("getB", &WeibullDeviate::getB, "")
                .def("setB", &WeibullDeviate::setB, "")
                .enable_pickling()
                ;
        }

    };

    struct PyGammaDeviate {

        static void wrap() {
            bp::class_<GammaDeviate, bp::bases<BaseDeviate> >
                pyGammaDeviate("GammaDeviate", "", bp::no_init);
            pyGammaDeviate
                .def(bp::init<long, double, double>(
                        (bp::arg("lseed")=0, bp::arg("k")=1., bp::arg("theta")=1.)
                ))
                .def(bp::init<const BaseDeviate&, double, double>(
                        (bp::arg("dev"), bp::arg("k")=1., bp::arg("theta")=1.)
                ))
                .def(bp::init<std::string, double, double>(
                        (bp::arg("str")=0, bp::arg("k")=1., bp::arg("theta")=1.)
                ))
                .def("duplicate", &GammaDeviate::duplicate, "")
                .def("__call__", &GammaDeviate::operator(), "")
                .def("getK", &GammaDeviate::getK, "")
                .def("setK", &GammaDeviate::setK, "")
                .def("getTheta", &GammaDeviate::getTheta, "")
                .def("setTheta", &GammaDeviate::setTheta, "")
                .enable_pickling()
                ;
        }

    };

    struct PyChi2Deviate {

        static void wrap() {
            bp::class_<Chi2Deviate, bp::bases<BaseDeviate> >
                pyChi2Deviate("Chi2Deviate", "", bp::no_init);
            pyChi2Deviate
                .def(bp::init<long, double>(
                        (bp::arg("lseed")=0, bp::arg("n")=1.)
                ))
                .def(bp::init<const BaseDeviate&, double>(
                        (bp::arg("dev"), bp::arg("n")=1.)
                ))
                .def(bp::init<std::string, double>(
                        (bp::arg("str")=0, bp::arg("n")=1.)
                ))
                .def("duplicate", &Chi2Deviate::duplicate, "")
                .def("__call__", &Chi2Deviate::operator(), "")
                .def("getN", &Chi2Deviate::getN, "")
                .def("setN", &Chi2Deviate::setN, "")
                .enable_pickling()
                ;
        }

    };


    void pyExportRandom() {
        PyBaseDeviate::wrap();
        PyUniformDeviate::wrap();
        PyGaussianDeviate::wrap();
        PyBinomialDeviate::wrap();
        PyPoissonDeviate::wrap();
        PyWeibullDeviate::wrap();
        PyGammaDeviate::wrap();
        PyChi2Deviate::wrap();
    }

} // namespace galsim
