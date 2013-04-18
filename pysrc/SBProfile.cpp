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
#include "boost/python/stl_iterator.hpp"

#include "SBProfile.h"

namespace bp = boost::python;

namespace galsim {

    struct PyGSParams {

        static void wrap() {

            static const char* doc = 
                "GSParams stores a set of numbers that govern how GSObjects make various\n"
                "speed/accuracy tradeoff decisions.\n"
                "\n"
                "The parameters, along with their default values are the following:\n"
                "\n"
                "minimum_fft_size=128      Constant giving minimum FFT size we're willing to do.\n"
                "maximum_fft_size=4096     Constant giving maximum FFT size we're willing to do.\n"
                "alias_threshold=5.e-3     A threshold parameter used for setting the stepK value\n"
                "                          for FFTs.  The FFT's stepK is set so that at most a\n"
                "                          fraction alias_threshold of the flux of any profile is\n"
                "                          aliased.\n"
                "maxk_threshold=1.e-3      A threshold parameter used for setting the maxK value\n"
                "                          for FFTs.  The FFT's maxK is set so that the k-values\n"
                "                          that are excluded off the edge of the image are less\n"
                "                          than maxk_threshold.\n"
                "kvalue_accuracy=1.e-5     Accuracy of values in k-space.\n"
                "                          If a k-value is less than kvalue_accuracy, then it may\n"
                "                          be set to zero. Similarly, if an alternate calculation\n"
                "                          has errors less than kvalue_accuracy, then it may be\n"
                "                          used instead of an exact calculation.\n"
                "                          Note: This does not necessarily imply that all kvalues\n"
                "                          are this accurate.  There may be cases where other\n"
                "                          choices we have made lead to errors greater than this.\n"
                "                          But whenever we do an explicit calculation about this,\n"
                "                          this is the value we use.\n"
                "                          This should typically be set to a lower, more stringent\n"
                "                          value than maxk_threshold.\n"
                "xvalue_accuracy=1.e-5     Accuracy of values in real space.\n"
                "                          If a value in real space is less than xvalue_accuracy,\n"
                "                          then it may be set to zero. Similarly, if an alternate\n"
                "                          calculation has errors less than xvalue_accuracy, then\n"
                "                          it may be used instead of an exact calculation.\n"
                "shoot_accuracy=1.e-5      Accuracy of total flux for photon shooting\n"
                "                          The photon shooting algorithm sometimes needs to\n"
                "                          sample the radial profile out to some value.  We\n"
                "                          choose the outer radius such that the integral\n"
                "                          encloses at least (1-shoot_accuracy) of the flux.\n"
                "realspace_relerr=1.e-3    The relative accuracy for realspace convolution.\n"
                "realspace_abserr=1.e-6    The absolute accuracy for realspace convolution.\n"
                "integration_relerr=1.e-5  The relative accuracy for integrals (other than\n"
                "                          real-space convolution).\n"
                "integration_abserr=1.e-7  The absolute accuracy for integrals (other than\n"
                "                          real-space convolution).\n";

            bp::class_<GSParams> pyGSParams("GSParams", doc, bp::no_init);
            pyGSParams
                .def(bp::init<
                     int, int, double, double, double, double, double, double, double,
                     double, double>(
                         (bp::arg("minimum_fft_size")=128, 
                          bp::arg("maximum_fft_size")=4096,
                          bp::arg("alias_threshold")=5.e-3,
                          bp::arg("maxk_threshold")=1.e-3,
                          bp::arg("kvalue_accuracy")=1.e-5,
                          bp::arg("xvalue_accuracy")=1.e-5,
                          bp::arg("shoot_accuracy")=1.e-5,
                          bp::arg("realspace_relerr")=1.e-3,
                          bp::arg("realspace_abserr")=1.e-6,
                          bp::arg("integration_relerr")=1.e-5,
                          bp::arg("integration_abserr")=1.e-7))
                )
                .def_readwrite("minimum_fft_size", &GSParams::minimum_fft_size)
                .def_readwrite("maximum_fft_size", &GSParams::maximum_fft_size)
                .def_readwrite("alias_threshold", &GSParams::alias_threshold)
                .def_readwrite("maxk_threshold", &GSParams::maxk_threshold)
                .def_readwrite("kvalue_accuracy", &GSParams::kvalue_accuracy)
                .def_readwrite("xvalue_accuracy", &GSParams::xvalue_accuracy)
                .def_readwrite("shoot_accuracy", &GSParams::shoot_accuracy)
                .def_readwrite("realspace_relerr", &GSParams::realspace_relerr)
                .def_readwrite("realspace_abserr", &GSParams::realspace_abserr)
                .def_readwrite("integration_relerr", &GSParams::integration_relerr)
                .def_readwrite("integration_abserr", &GSParams::integration_abserr)
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
                     "Setting optional integer arg possionFlux != 0 allows profile flux to vary\n"
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
                "SBProfile is an abstract base class represented all of the 2d surface\n"
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
                "Basic shapes: SBBox, SBGaussian, SBExponential, SBAiry, SBSersic\n"
                "SBLaguerre: Gauss-Laguerre expansion\n"
                "SBTransform: affine transformation of another SBProfile\n"
                "SBAdd: sum of SBProfiles\n"
                "SBConvolve: convolution of other SBProfiles\n"
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
                .def("setFlux", &SBProfile::setFlux, bp::args("flux"))
                .def("applyTransformation", &SBProfile::applyTransformation, bp::args("e"))
                .def("applyShear",
                     (void (SBProfile::*)(CppShear))&SBProfile::applyShear,
                     (bp::arg("s")))
                .def("applyRotation", &SBProfile::applyRotation, bp::args("theta"))
                .def("applyShift", &SBProfile::applyShift, bp::args("dx", "dy"))
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
    }

} // namespace galsim
