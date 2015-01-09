/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
#define BOOST_PYTHON_MAX_ARITY 20  // We have a function with 17 params here...
                                   // c.f. www.boost.org/libs/python/doc/v2/configuration.html
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBProfile.h"
#include "SBTransform.h"
#include "FFT.h"  // For goodFFTSize

namespace bp = boost::python;

namespace galsim {

    struct PyGSParams {

        static void wrap() {

            bp::class_<GSParams> pyGSParams("GSParams", "", bp::no_init);
            pyGSParams
                .def(bp::init<
                    int, int, double, double, double, double, double, double, double, double,
                    double, double, double, double, int, double>((
                        bp::arg("minimum_fft_size")=128, 
                        bp::arg("maximum_fft_size")=4096,
                        bp::arg("folding_threshold")=5.e-3,
                        bp::arg("stepk_minimum_hlr")=5.,
                        bp::arg("maxk_threshold")=1.e-3,
                        bp::arg("kvalue_accuracy")=1.e-5,
                        bp::arg("xvalue_accuracy")=1.e-5,
                        bp::arg("table_spacing")=1.,
                        bp::arg("realspace_relerr")=1.e-4,
                        bp::arg("realspace_abserr")=1.e-6,
                        bp::arg("integration_relerr")=1.e-6,
                        bp::arg("integration_abserr")=1.e-8,
                        bp::arg("shoot_accuracy")=1.e-5,
                        bp::arg("allowed_flux_variation")=0.81,
                        bp::arg("range_division_for_extrema")=32,
                        bp::arg("small_fraction_of_flux")=1.e-4)
                    )
                )
                .def_readwrite("minimum_fft_size", &GSParams::minimum_fft_size)
                .def_readwrite("maximum_fft_size", &GSParams::maximum_fft_size)
                .def_readwrite("folding_threshold", &GSParams::folding_threshold)
                .def_readwrite("stepk_minimum_hlr", &GSParams::stepk_minimum_hlr)
                .def_readwrite("maxk_threshold", &GSParams::maxk_threshold)
                .def_readwrite("kvalue_accuracy", &GSParams::kvalue_accuracy)
                .def_readwrite("xvalue_accuracy", &GSParams::xvalue_accuracy)
                .def_readwrite("table_spacing", &GSParams::table_spacing)
                .def_readwrite("realspace_relerr", &GSParams::realspace_relerr)
                .def_readwrite("realspace_abserr", &GSParams::realspace_abserr)
                .def_readwrite("integration_relerr", &GSParams::integration_relerr)
                .def_readwrite("integration_abserr", &GSParams::integration_abserr)
                .def_readwrite("shoot_accuracy", &GSParams::shoot_accuracy)
                .def_readwrite("allowed_flux_variation", &GSParams::allowed_flux_variation)
                .def_readwrite("range_division_for_extrema", &GSParams::range_division_for_extrema)
                .def_readwrite("small_fraction_of_flux", &GSParams::small_fraction_of_flux)
                ;
        }
    };


    struct PySBProfile 
    {

        template <typename U, typename W>
        static void wrapTemplates(W & wrapper) {
            // We don't need to wrap templates in a separate function, but it keeps us
            // from having to repeat each of the lines below for each type.
            // We also don't need to make 'W' a template parameter in this case,
            // but it's easier to do that than write out the full class_ type.
            wrapper
                .def("drawShoot", 
                     (double (SBProfile::*)(ImageView<U>, double, UniformDeviate,
                                            double, double, bool, bool)
                      const)&SBProfile::drawShoot,
                     (bp::arg("image"), bp::arg("N")=0., bp::arg("ud"),
                      bp::arg("gain")=1., bp::arg("max_extra_noise")=0.,
                      bp::arg("poisson_flux")=true, bp::arg("add_to_image")=false),
                     "Draw object into existing image using photon shooting.\n"
                     "\n"
                     "Setting optional integer arg poissonFlux != 0 allows profile flux to vary\n"
                     "according to Poisson statistics for N samples.\n"
                     "\n"
                     "Returns total flux of photons that landed inside image bounds.")
                .def("draw", 
                     (double (SBProfile::*)(ImageView<U>, double, double) const)&SBProfile::draw,
                     (bp::arg("image"), bp::arg("gain")=1., bp::arg("wmult")=1.),
                     "Draw in-place and return the summed flux.")
                .def("drawK", 
                     (void (SBProfile::*)(ImageView<U>, ImageView<U>, 
                                          double, double) const)&SBProfile::drawK,
                     (bp::arg("re"), bp::arg("im"), bp::arg("gain")=1., bp::arg("wmult")=1.),
                     "Draw k-space image (real and imaginary components).")
                ;
        }

        static void wrap() {
            static char const * doc = 
                "\n"
                "SBProfile is an abstract base class representing all of the 2d surface\n"
                "brightness that we know how to draw.  Every SBProfile knows how to\n"
                "draw an Image<float> of itself in real and k space.  Each also knows\n"
                "what is needed to prevent aliasing or truncation of itself when drawn.\n"
                "\n"
                "Note that when you use the SBProfile::draw() routines you will get an\n"
                "image of **surface brightness** values in each pixel, not the flux\n"
                "that fell into the pixel.  To get flux, you must multiply the image by\n"
                "(dx*dx).\n"
                "\n"
                "drawK() routines are normalized such that I(0,0) is the total flux.\n"
                "\n"
                "Currently we have the following possible implementations of SBProfile:\n"
                "Basic shapes: SBBox, SBGaussian, SBExponential, SBAiry, SBSersic,\n"
                "              SBMoffat, SBKolmogorov\n"
                "SBInterpolatedImage: a representation of some arbitrary image\n"
                "SBShapelet: an object represented as a shapelets decomposition\n"
                "SBTransform: affine transformation of another SBProfile\n"
                "SBAdd: sum of SBProfiles\n"
                "SBConvolve: convolution of other SBProfiles\n"
                "SBDeconvolve: deconvolution of an SBProfile\n"
                "\n"
                "==== Drawing routines ====\n"
                "Grid on which SBProfile is drawn has pitch dx, which is taken from the\n"
                "image's scale parameter.\n"
                "\n"
                "Note that in an FFT the image may be calculated internally on a\n"
                "larger grid than the provided image to avoid folding.\n"
                "Specifying wmult > 1 will draw an image that is wmult times larger than the\n"
                "default choice, i.e. it will have finer sampling in k space and have less\n"
                "folding.\n"
                ;

            bp::class_<SBProfile> pySBProfile("SBProfile", doc, bp::no_init);
            pySBProfile
                .def(bp::init<const SBProfile &>())
                .def("xValue", &SBProfile::xValue,
                     "Return value of SBProfile at a chosen 2d position in real space.\n"
                     "May not be implemented for derived classes (e.g. SBConvolve) that\n"
                     "require an FFT to determine real-space values.")
                .def("kValue", &SBProfile::kValue,
                     "Return value of SBProfile at a chosen 2d position in k-space.")
                .def("maxK", &SBProfile::maxK, "Value of k beyond which aliasing can be neglected")
                .def("nyquistDx", &SBProfile::nyquistDx,
                     "Image pixel spacing that does not alias maxK")
                .def("getGoodImageSize", &SBProfile::getGoodImageSize,
                     "A good image size for drawing the SBProfile")
                .def("stepK", &SBProfile::stepK,
                     "Sampling in k space necessary to avoid folding of image in x space")
                .def("isAxisymmetric", &SBProfile::isAxisymmetric)
                .def("hasHardEdges", &SBProfile::hasHardEdges)
                .def("isAnalyticX", &SBProfile::isAnalyticX,
                     "True if real-space values can be determined immediately at any position "
                     "without DFT.")
                .def("isAnalyticK", &SBProfile::isAnalyticK,
                     "True if k-space values can be determined immediately at any position "
                     "without DFT.")
                .def("centroid", &SBProfile::centroid)
                .def("getFlux", &SBProfile::getFlux)
                .def("scaleFlux", &SBProfile::scaleFlux, bp::args("fluxRatio"))
                .def("shear", &SBProfile::shear, bp::arg("shear"))
                .def("rotate", &SBProfile::rotate, bp::args("theta"))
                .def("shift", &SBProfile::shift, bp::args("delta"))
                .def("expand", &SBProfile::expand, bp::args("scale"))
                .def("transform", &SBProfile::transform, bp::args("dudx", "dudy", "dvdx", "dvdy"))
                .def("shoot", &SBProfile::shoot, bp::args("n", "u"))
                ;
            wrapTemplates<float>(pySBProfile);
            wrapTemplates<double>(pySBProfile);
        }

    };


    void pyExportSBProfile() 
    {
        PySBProfile::wrap();
        PyGSParams::wrap();

        bp::def("goodFFTSize", &goodFFTSize, (bp::arg("input_size")),
                "Round up to the next larger 2^n or 3x2^n.");
    }

} // namespace galsim
