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
#include "boost/python/stl_iterator.hpp"

#include "NumpyHelper.h"
#include "SBShapelet.h"

namespace bp = boost::python;

namespace galsim {

    struct PyLVector {

        static bp::object GetArrayImpl(bp::object self, bool isConst)
        {
            const LVector& lvector = bp::extract<const LVector&>(self);
            bp::object numpy_array = MakeNumpyArray(
                lvector.rVector().cptr(), lvector.size(), lvector.rVector().step(), isConst,
                lvector.getOwner());
            return numpy_array;
        }

        static bp::object GetArray(bp::object lvector) { return GetArrayImpl(lvector, false); }
        static bp::object GetConstArray(bp::object lvector) { return GetArrayImpl(lvector, true); }

        static LVector* MakeFromArray(int order, const bp::object& array)
        {
            double* data = 0;
            boost::shared_ptr<double> owner;
            int step = 0;
            int stride = 0;
            CheckNumpyArray(array,1,true,data,owner,step,stride);
            int size = GetNumpyArrayDim(array.ptr(), 0);
            if (size != PQIndex::size(order)) {
                PyErr_SetString(PyExc_ValueError, "Array for LVector is the wrong size");
                bp::throw_error_already_set();
            }
            return new LVector(order,tmv::ConstVectorView<double>(data,size,stride,tmv::NonConj));
            // Note: after building the LVector for return, it is now safe for the owner
            // to go out of scope and possibly delete the memory for data.
        }

        static bp::tuple GetPQ(const LVector& lvector, int p, int q)
        {
            std::complex<double> val = lvector(p,q);
            return bp::make_tuple(real(val),imag(val));
        }
        static void SetPQ(LVector& lvector, int p, int q, double re, double im)
        { lvector(p,q) = std::complex<double>(re,im); }

        static void wrap()
        {
            bp::class_< LVector > pyLVector("LVector", "", bp::no_init);
            pyLVector
                .def(bp::init<int>(bp::arg("order")=0))
                .def(
                    "__init__",
                    bp::make_constructor(
                        &MakeFromArray, bp::default_call_policies(),
                        (bp::arg("order"), bp::arg("array"))
                    )
                )
                .def(bp::init<const LVector&>(bp::args("other")))
                .def("copy", &LVector::copy)
                .def("resize", &LVector::resize, bp::args("order"))
                .add_property("array", &GetConstArray)
                .add_property("order", &LVector::getOrder)
                .def("size", &LVector::size)
                .def("getPQ", &GetPQ, bp::args("p","q"))
                .def("setPQ", &SetPQ, bp::args("p","q","re","im"))
                .def(bp::self * bp::other<double>())
                .def("__div__", &LVector::operator/)
                .def("__truediv__", &LVector::operator/)
                .def(bp::self + bp::other<LVector>())
                .def(bp::self - bp::other<LVector>())
                .def("dot", &LVector::dot, bp::args("other"))
                .def("rotate", &LVector::rotate, bp::args("theta"))
                .enable_pickling()
                ;

            bp::def("ShapeletSize", &PQIndex::size, bp::arg("order"),
                    "Calculate the size of a shapelet vector for a given order");

        }
    };

    struct PySBShapelet
    {
        template <typename U>
        static void wrapImageTemplates() {
            typedef void (*ShapeletFitImage_type)(
                double sigma, LVector& bvec, const BaseImage<U>& image, double scale,
                const Position<double>& center);

            bp::def("ShapeletFitImage", ShapeletFitImage_type(&ShapeletFitImage),
                    bp::args("sigma","bvec","image","image_scale","center"),
                    "Fit a Shapelet decomposition to the provided image");
        }

        static void wrap() {
            bp::class_<SBShapelet,bp::bases<SBProfile> >("SBShapelet", bp::no_init)
                .def(bp::init<double,LVector,boost::shared_ptr<GSParams> >(
                        (bp::arg("sigma"), bp::arg("bvec"),
                         bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBShapelet &>())
                .def("getSigma", &SBShapelet::getSigma)
                .def("getBVec", &SBShapelet::getBVec,
                     bp::return_value_policy<bp::copy_const_reference>())
                .enable_pickling()
                ;
            wrapImageTemplates<float>();
            wrapImageTemplates<double>();
            wrapImageTemplates<int16_t>();
            wrapImageTemplates<int32_t>();
            wrapImageTemplates<uint16_t>();
            wrapImageTemplates<uint32_t>();
        }
    };

    void pyExportSBShapelet()
    {
        PyLVector::wrap();
        PySBShapelet::wrap();
    }

} // namespace galsim
