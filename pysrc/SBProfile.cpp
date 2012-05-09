#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"
#include "SBProfile.h"
#include "SBDeconvolve.h"
#include "SBParse.h"

namespace bp = boost::python;

namespace galsim {
namespace {

typedef bp::return_value_policy<bp::manage_new_object> ManageNew;

// Used by multiple profile classes to ensure at most one radius is given.
void checkRadii(const bp::object & r1, const bp::object & r2, const bp::object & r3) {
    int nRad = !r1.is_none() + !r2.is_none() + !r3.is_none();
    if (nRad > 1) {
        PyErr_SetString(PyExc_TypeError, "Multiple radius parameters given");
        bp::throw_error_already_set();
    }
    if (nRad == 0) {
        PyErr_SetString(PyExc_TypeError, "No radius parameter given");
        bp::throw_error_already_set();
    }
}

struct PySBProfile {

    template <typename U, typename W>
    static void wrapTemplates(W & wrapper) {
        // We don't need to wrap templates in a separate function, but it keeps us
        // from having to repeat each of the lines below for each type.
        // We also don't need to make 'W' a template parameter in this case,
        // but it's easier to do that than write out the full class_ type.
        wrapper
            .def("fillXImage", 
                 (double (SBProfile::*)(ImageView<U> &, double) const)&SBProfile::fillXImage, 
                 bp::args("image", "dx"),
                 "Utility for drawing into Image data structures")
            .def("draw", 
                 (double (SBProfile::*)(Image<U> &, double, int) const)&SBProfile::draw,
                 (bp::arg("image"), bp::arg("dx")=0., bp::arg("wmult")=1),
                 "Draw in-place, resizing if necessary, and return the summed flux.")
            .def("draw", 
                 (double (SBProfile::*)(ImageView<U> &, double, int) const)&SBProfile::draw,
                 (bp::arg("image"), bp::arg("dx")=0., bp::arg("wmult")=1),
                 "Draw in-place and return the summed flux.")
            .def("plainDraw",
                 (double (SBProfile::*)(ImageView<U> &, double, int) const)&SBProfile::plainDraw,
                 (bp::arg("image"), bp::arg("dx")=0., bp::arg("wmult")=1),
                 "Draw in place using only real methods")
            .def("fourierDraw",
                 (double (SBProfile::*)(ImageView<U> &, double, int) const)&SBProfile::fourierDraw,
                 (bp::arg("image"), bp::arg("dx")=0., bp::arg("wmult")=1),
                 "Draw in place using only Fourier methods")
            .def("drawK",
                 (void (SBProfile::*)(ImageView<U> &, ImageView<U> &, double, int) const)&SBProfile::drawK,
                 (bp::arg("re"), bp::arg("im"), bp::arg("dx")=0., bp::arg("wmult")=1),
                 "Draw in k-space automatically")
            .def("plainDrawK",
                 (void (SBProfile::*)(ImageView<U> &, ImageView<U> &, double, int) const)&SBProfile::plainDrawK,
                 (bp::arg("re"), bp::arg("im"), bp::arg("dx")=0., bp::arg("wmult")=1),
                 "evaluate in k-space automatically")
            .def("fourierDrawK",
                 (void (SBProfile::*)(ImageView<U> &, ImageView<U> &, double, int) const)&SBProfile::fourierDrawK,
                 (bp::arg("re"), bp::arg("im"), bp::arg("dx")=0., bp::arg("wmult")=1),
                 "FT from x-space")
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
            "SBDistort: affine transformation of another SBProfile\n"
            "SBRotate: rotated version of another SBProfile\n"
            "SBAdd: sum of SBProfiles\n"
            "SBConvolve: convolution of other SBProfiles\n"
            "\n"
            "==== Drawing routines ==== \n"
            "Grid on which SBProfile is drawn has pitch dx; given dx=0. default,\n"
            "routine will choose dx to be at least fine enough for Nyquist sampling\n"
            "at maxK().  If you specify dx, image will be drawn with this dx and\n"
            "you will receive an image with the aliased frequencies included.\n"
            "\n"
            "If input image is not specified or has null dimension, a square image\n"
            "will be drawn which is big enough to avoid folding.  If drawing is\n"
            "done using FFT, it will be scaled up to a power of 2, or 3x2^n,\n"
            "whicher fits.  If input image has finite dimensions then these will be\n"
            "used, although in an FFT the image may be calculated internally on a\n"
            "larger grid to avoid folding.  Specifying wmult>1 will draw an image\n"
            "that is wmult times larger than the default choice, i.e. it will have\n"
            "finer sampling in k space and have less folding.\n"
            ;

        bp::class_<SBProfile,boost::noncopyable> pySBProfile("SBProfile", doc, bp::no_init);
        pySBProfile
            .def("duplicate", &SBProfile::duplicate, ManageNew())
            .def("xValue", &SBProfile::xValue,
                 "Return value of SBProfile at a chosen 2d position in real space.\n"
                 "May not be implemented for derived classes (e.g. SBConvolve) that\n"
                 "require an FFT to determine real-space values.")
            .def("kValue", &SBProfile::kValue,
                 "Return value of SBProfile at a chosen 2d position in k-space.")
            .def("maxK", &SBProfile::maxK, "Value of k beyond which aliasing can be neglected")
            .def("nyquistDx", &SBProfile::nyquistDx, "Image pixel spacing that does not alias maxK")
            .def("stepK", &SBProfile::stepK,
                 "Sampling in k space necessary to avoid folding of image in x space")
            .def("isAxisymmetric", &SBProfile::isAxisymmetric)
            .def("isAnalyticX", &SBProfile::isAnalyticX,
                 "True if real-space values can be determined immediately at any position with FT")
            .def("centroid", &SBProfile::centroid)
            .def("getFlux", &SBProfile::getFlux)
            .def("setFlux", &SBProfile::setFlux)
            .def("distort", &SBProfile::distort, bp::args("e"), ManageNew())
            .def("shear", &SBProfile::shear, bp::args("e1", "e2"), ManageNew())
            .def("rotate", &SBProfile::rotate, bp::args("theta"), ManageNew())
            .def("shift", &SBProfile::shift, bp::args("dx", "dy"), ManageNew())
            .def("draw", (ImageView<float> (SBProfile::*)(double, int) const)&SBProfile::draw,
                 (bp::arg("dx")=0., bp::arg("wmult")=1), "default draw routine")
            ;
        wrapTemplates<float>(pySBProfile);
        wrapTemplates<double>(pySBProfile);
    }

};

struct PySBAdd {

    // This will be wrapped as a Python constructor; it accepts an arbitrary Python iterable.
    static SBAdd * construct(bp::object const & iterable) {
        bp::stl_input_iterator<SBProfile*> begin(iterable), end;
        std::list<SBProfile*> plist(begin, end);
        return new SBAdd(plist);
    }

    static void wrap() {
        static char const * doc = 
            "Sum of SBProfile.  Note that this class stores duplicates of its summands,\n"
            "so they cannot be changed after adding them."
            ;
            
        bp::class_< SBAdd, bp::bases<SBProfile> >("SBAdd", doc, bp::init<>())
            // bp tries the overloads in reverse order, so we wrap the most general one first
            // to ensure we try it last
            .def("__init__", bp::make_constructor(&construct, bp::default_call_policies(),
                                                  bp::args("slist")))
            .def(bp::init<const SBProfile &>(bp::args("s1")))
            .def(bp::init<const SBProfile &, const SBProfile &>(bp::args("s1", "s2")))
            .def(bp::init<const SBAdd &>())
            .def("add", &SBAdd::add, (bp::arg("rhs"), bp::arg("scale")=1.))
            ;
    }

};

struct PySBDistort {

    static void wrap() {
        static char const * doc = 
            "SBDistort is an affine transformation of another SBProfile.\n"
            "Stores a duplicate of its target.\n"
            "Origin of original shape will now appear at x0.\n"
            "Flux is NOT conserved in transformation - SB is preserved."
            ;
            
        bp::class_< SBDistort, bp::bases<SBProfile> >("SBDistort", doc, bp::no_init)
            .def(bp::init<const SBProfile &, double, double, double, double, Position<double> >(
                     (bp::args("sbin", "mA", "mB", "mC", "mD"),
                      bp::arg("x0")=Position<double>(0.,0.))
                 ))
            .def(bp::init<const SBProfile &, const Ellipse &>(
                     (bp::arg("sbin"), bp::arg("e")=Ellipse())
                 ))
            .def(bp::init<const SBDistort &>())
            ;
    }

};

struct PySBConvolve {

    // This will be wrapped as a Python constructor; it accepts an arbitrary Python iterable.
    static SBConvolve * construct(bp::object const & iterable, double f) {
        bp::stl_input_iterator<SBProfile*> begin(iterable), end;
        std::list<SBProfile*> plist(begin, end);
        return new SBConvolve(plist, f);
    }

    static void wrap() {
        bp::class_< SBConvolve, bp::bases<SBProfile> >("SBConvolve", bp::init<>())
            // bp tries the overloads in reverse order, so we wrap the most general one first
            // to ensure we try it last
            .def("__init__", 
                 bp::make_constructor(&construct, bp::default_call_policies(), 
                                      (bp::arg("slist"), bp::arg("f")=1.)
                 ))
            .def(bp::init<const SBProfile &, double>(
                     (bp::args("s1"), bp::arg("f")=1.)
                 ))
            .def(bp::init<const SBProfile &, const SBProfile &, double>(
                     (bp::args("s1", "s2"), bp::arg("f")=1.)
                 ))
            .def(bp::init<const SBProfile &, const SBProfile &, const SBProfile &, double>(
                     (bp::args("s1", "s2", "s3"), bp::arg("f")=1.)
                 ))
            .def(bp::init<const SBConvolve &>())
            .def("add", &SBConvolve::add)
            ;
    }

};

struct PySBDeconvolve {

    static void wrap() {
        bp::class_< SBDeconvolve, bp::bases<SBProfile> >("SBDeconvolve", bp::no_init)
            .def(bp::init<const SBProfile &>(bp::args("adaptee")))
            .def(bp::init<const SBDeconvolve &>())
            ;
    }

};

struct PySBGaussian {

    static SBGaussian * construct(
        double flux,
        const bp::object & half_light_radius,
        const bp::object & sigma,
        const bp::object & fwhm
    ) {
        double s = 1.0;
        checkRadii(half_light_radius, sigma, fwhm);
        if (!half_light_radius.is_none()) {
            s = bp::extract<double>(half_light_radius) * 0.84932180028801907; // (2\ln2)^(-1/2)
        }
        if (!sigma.is_none()) {
            s = bp::extract<double>(sigma);
        }
        if (!fwhm.is_none()) {
            s = bp::extract<double>(fwhm) * 0.42466090014400953; // 1 / (2(2\ln2)^(1/2))
        }
        return new SBGaussian(flux, s);
    }

    static void wrap() {
        bp::class_<SBGaussian,bp::bases<SBProfile>,boost::noncopyable>(
            "SBGaussian",
            "SBGaussian(flux=1., half_light_radius=None, sigma=None, fwhm=None)\n\n"
            "Construct an exponential profile with the given flux and half-light radius,\n"
            "sigma, or FWHM.  At most one radius may be provided, and if no radius is\n"
            "provided, the profile will have a sigma of one.\n",
            bp::no_init
        )
            .def(
                "__init__", bp::make_constructor(
                    &construct, bp::default_call_policies(),
                    (bp::arg("flux")=1., bp::arg("half_light_radius")=bp::object(), 
                     bp::arg("sigma")=bp::object(), bp::arg("fwhm")=bp::object())
                )
            );
    }
};

struct PySBSersic {

    static void wrap() {
        bp::class_<SBSersic,bp::bases<SBProfile>,boost::noncopyable>(
            "SBSersic",
            bp::init<double,double,double>((bp::arg("n"), bp::arg("flux")=1., 
                                            bp::arg("half_light_radius")=1.))
        );
    }
};

struct PySBExponential {


    static SBExponential * construct(
        double flux,
        const bp::object & half_light_radius,
        const bp::object & scale_radius
    ) {
        double s = 1.0;
        checkRadii(half_light_radius, scale_radius, bp::object());
        if (!half_light_radius.is_none()) {
            s = bp::extract<double>(half_light_radius) / 1.6783469900166605; // not analytic
        }
        if (!scale_radius.is_none()) {
            s = bp::extract<double>(scale_radius);
        }
        return new SBExponential(flux, s);
    }

    static void wrap() {
        bp::class_<SBExponential,bp::bases<SBProfile>,boost::noncopyable>(
            "SBExponential",
            "SBExponential(flux=1., half_light_radius=None, scale=None)\n\n"
            "Construct an exponential profile with the given flux and either half-light radius\n"
            "or scale length.  At most one radius may be provided, and if no radius is provided\n"
            "the profile will have a scale length of one.\n",
            bp::no_init
        )
            .def(
                "__init__", bp::make_constructor(
                    &construct, bp::default_call_policies(),
                    (bp::arg("flux")=1., bp::arg("half_light_radius")=bp::object(), 
                     bp::arg("scale_radius")=bp::object())
                )
            );
    }
};

struct PySBAiry {
    static void wrap() {
        bp::class_<SBAiry,bp::bases<SBProfile>,boost::noncopyable>(
            "SBAiry",
            bp::init<double,double,double>(
                (bp::arg("D")=1., bp::arg("obs")=0., bp::arg("flux")=1.)
            )
        );
    }
};

struct PySBBox {
    static void wrap() {
        bp::class_<SBBox,bp::bases<SBProfile>,boost::noncopyable>(
            "SBBox",
            bp::init<double,double,double>(
                (bp::arg("xw")=1., bp::arg("yw")=0., bp::arg("flux")=1.)
            )
        );
    }
};

struct PySBMoffat {

    static SBMoffat * construct(
        double beta, double truncationFWHM, double flux,
        const bp::object & half_light_radius,
        const bp::object & scale_radius,
        const bp::object & fwhm
    ) {
        double s = 1.0;
        checkRadii(half_light_radius, scale_radius, fwhm);
        SBMoffat::RadiusType rType = SBMoffat::HALF_LIGHT_RADIUS;
        if (!half_light_radius.is_none()) {
            s = bp::extract<double>(half_light_radius);
        }
        if (!scale_radius.is_none()) {
            s = bp::extract<double>(scale_radius);
            rType = SBMoffat::SCALE_RADIUS;
        }
        if (!fwhm.is_none()) {
            s = bp::extract<double>(fwhm);
            rType = SBMoffat::FWHM;
        }
        return new SBMoffat(beta, truncationFWHM, flux, s, rType);
    }

    static void wrap() {
        bp::class_<SBMoffat,bp::bases<SBProfile>,boost::noncopyable>("SBMoffat", bp::no_init)
            .def("__init__", 
                 bp::make_constructor(
                     &construct, bp::default_call_policies(),
                     (bp::arg("beta"), bp::arg("truncationFWHM")=2.,
                      bp::arg("flux")=1., bp::arg("half_light_radius")=bp::object(),
                      bp::arg("scale_radius")=bp::object(), bp::arg("fwhm")=bp::object())
                 )
            )
            ;
    }
};

struct PySBDeVaucouleurs {
    static void wrap() {
        bp::class_<SBDeVaucouleurs,bp::bases<SBProfile>,boost::noncopyable>(
            "SBDeVaucouleurs",
            bp::init<double,double>((bp::arg("flux")=1., bp::arg("half_light_radius")=1.))
        );
    }
};

} // anonymous

void pyExportSBProfile() {
    PySBProfile::wrap();
    PySBAdd::wrap();
    PySBDistort::wrap();
    PySBConvolve::wrap();
    PySBDeconvolve::wrap();
    PySBGaussian::wrap();
    PySBSersic::wrap();
    PySBExponential::wrap();
    PySBAiry::wrap();
    PySBBox::wrap();
    PySBMoffat::wrap();
    PySBDeVaucouleurs::wrap();

    bp::def("SBParse", &galsim::SBParse, galsim::ManageNew());
}

} // namespace galsim
