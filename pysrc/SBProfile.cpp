#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBProfile.h"

namespace bp = boost::python;

namespace galsim {

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
                                            double, double, bool)
                      const)&SBProfile::drawShoot,
                     (bp::arg("image"), bp::arg("N")=0., bp::arg("ud"),
                      bp::arg("gain")=1., bp::arg("max_extra_noise")=0.,
                      bp::arg("poisson_flux")=true),
                     "Draw object into existing image using photon shooting.\n"
                     "\n"
                     "Setting optional integer arg possionFlux != 0 allows profile flux to vary \n"
                     "according to Poisson statistics for N samples.\n"
                     "\n"
                     "Returns total flux of photons that landed inside image bounds.")
                .def("draw", 
                     (double (SBProfile::*)(ImageView<U>, double, double) const)&SBProfile::draw,
                     (bp::arg("image"), bp::arg("gain")=1., bp::arg("wmult")=1.),
                     "Draw in-place and return the summed flux.")
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
                "==== Drawing routines ==== \n"
                "Grid on which SBProfile is drawn has pitch dx, which is taken from the\n"
                "image's scale parameter.\n"
                "\n"
                "Note that in an FFT the image may be calculated internally on a\n"
                "larger grid than the provided image to avoid folding.\n"
                "Specifying wmult > 1 will draw an image that is wmult times larger than the \n"
                "default choice, i.e. it will have finer sampling in k space and have less \n"
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
    }

} // namespace galsim
