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
#include <boost/python.hpp> // header that includes Python.h always needs to come first
#include <boost/python/stl_iterator.hpp>

#include "PhotonArray.h"
#include "NumpyHelper.h"

namespace bp = boost::python;

namespace galsim {
namespace {

    template <typename T>
    boost::shared_ptr<PhotonArray> MakePhotonsFromImage(
        const BaseImage<T>& image, double maxFlux, UniformDeviate ud)
    {
        return boost::shared_ptr<PhotonArray>(new PhotonArray(image,maxFlux,ud));
    }


    struct PyPhotonArray {
        template <typename U, typename W>
        static void wrapTemplates(W & wrapper) {
            wrapper
                .def("addTo",
                     (double (PhotonArray::*)(ImageView<U>) const)&PhotonArray::addTo,
                     (bp::arg("image")),
                     "Add flux of photons to an image by binning into pixels.")
                ;
            bp::def("MakePhotonsFromImage",
                (boost::shared_ptr<PhotonArray> (*)(const BaseImage<U>&, double, UniformDeviate))
                &MakePhotonsFromImage<U>,
                bp::args("image", "maxFlux", "ud"));
        }

        static bp::object GetXArray(PhotonArray& phot)
        {
            return MakeNumpyArray(&phot.getXVector()[0], phot.size(), 1, false,
                                  boost::shared_ptr<double>());
        }

        static bp::object GetYArray(PhotonArray& phot)
        {
            return MakeNumpyArray(&phot.getYVector()[0], phot.size(), 1, false,
                                  boost::shared_ptr<double>());
        }

        static bp::object GetFluxArray(PhotonArray& phot)
        {
            return MakeNumpyArray(&phot.getFluxVector()[0], phot.size(), 1, false,
                                  boost::shared_ptr<double>());
        }

        static bp::object GetDXDZArray(PhotonArray& phot)
        {
            return MakeNumpyArray(&phot.getDXDZVector()[0], phot.size(), 1, false,
                                  boost::shared_ptr<double>());
        }

        static bp::object GetDYDZArray(PhotonArray& phot)
        {
            return MakeNumpyArray(&phot.getDYDZVector()[0], phot.size(), 1, false,
                                  boost::shared_ptr<double>());
        }

        static bp::object GetWavelengthArray(PhotonArray& phot)
        {
            return MakeNumpyArray(&phot.getWavelengthVector()[0], phot.size(), 1, false,
                                  boost::shared_ptr<double>());
        }

        static void wrap()
        {
            bp::class_<PhotonArray> pyPhotonArray("PhotonArray", bp::no_init);
            pyPhotonArray
                .def(bp::init<int>(bp::args("N")))
                .def("size", &PhotonArray::size,
                     "Return the number of photons")
                .def("setPhoton", &PhotonArray::setPhoton,
                     (bp::arg("i"), bp::arg("x"), bp::arg("y"), bp::arg("flux")),
                     "Set x,y,flux for photon number i")
                .def("getX", &PhotonArray::getX, (bp::arg("i")),
                     "Get x for photon number i")
                .def("getY", &PhotonArray::getY, (bp::arg("i")),
                     "Get y for photon number i")
                .def("getFlux", &PhotonArray::getFlux, (bp::arg("i")),
                     "Get flux for photon number i")
                .def("getDXDZ", &PhotonArray::getDXDZ, (bp::arg("i")),
                     "Get dxdz for photon number i")
                .def("getDYDZ", &PhotonArray::getDYDZ, (bp::arg("i")),
                     "Get dydz for photon number i")
                .def("getWavelength", &PhotonArray::getWavelength, (bp::arg("i")),
                     "Get wavelength for photon number i")
                .def("getXArray", GetXArray,
                     "Get numpy array of x positions")
                .def("getYArray", GetYArray,
                     "Get numpy array of y positions")
                .def("getFluxArray", GetFluxArray,
                     "Get numpy array of fluxes")
                .def("hasAllocatedAngles", &PhotonArray::hasAllocatedAngles,
                     "Returns whether the dxdz and dydz arrays are allocated")
                .def("hasAllocatedWavelengths", &PhotonArray::hasAllocatedWavelengths,
                     "Returns whether the wavelength arrays are allocated")
                .def("getDXDZArray", GetDXDZArray,
                     "Get numpy array of dxdz values")
                .def("getDYDZArray", GetDYDZArray,
                     "Get numpy array of dydz values")
                .def("getWavelengthArray", GetWavelengthArray,
                     "Get numpy array of wavelengths")
                .def("getTotalFlux", &PhotonArray::getTotalFlux,
                     "Return the total flux of all photons")
                .def("setTotalFlux", &PhotonArray::setTotalFlux, (bp::arg("flux")),
                     "Set the total flux to a new value")
                .def("scaleFlux", &PhotonArray::scaleFlux, (bp::arg("scale")),
                     "Scale the total flux by a given factor")
                .def("scaleXY", &PhotonArray::scaleXY, (bp::arg("scale")),
                     "Scale the photon positions (x,y) a given factor")
                .def("assignAt", &PhotonArray::assignAt, (bp::args("istart", "rhs")),
                     "Assign the contents of another PhotonArray to this one starting at istart.")
                .def("convolve", &PhotonArray::convolve, (bp::args("rhs", "ud")),
                     "Convolve this PhotonArray with another")
                .def("setCorrelated", &PhotonArray::setCorrelated, (bp::arg("new_val")),
                     "Declare that the photons in this array are correlated.")
                .enable_pickling()
                ;
            bp::register_ptr_to_python< boost::shared_ptr<PhotonArray> >();
            wrapTemplates<double>(pyPhotonArray);
            wrapTemplates<float>(pyPhotonArray);
        }
    }; // struct PyPhotonArray

} // anonymous

void pyExportPhotonArray()
{
    PyPhotonArray::wrap();
}

} // namespace galsim
