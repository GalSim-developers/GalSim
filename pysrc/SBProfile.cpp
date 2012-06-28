#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"
#include "SBProfile.h"
#include "SBDeconvolve.h"

namespace bp = boost::python;

namespace galsim {
namespace {

typedef bp::return_value_policy<bp::manage_new_object> ManageNew;

struct PyPhotonArray {
    
    static PhotonArray * construct(bp::object const & vx, bp::object const & vy,
                                   bp::object const & vflux) {
        Py_ssize_t size = bp::len(vx);
        if (size != bp::len(vx)) {
            PyErr_SetString(PyExc_ValueError,
                            "Length of vx array does not match  length of vy array");
            bp::throw_error_already_set();
        }
        if (size != bp::len(vflux)) {
            PyErr_SetString(PyExc_ValueError,
                            "Length of vx array does not match length of vflux array");
            bp::throw_error_already_set();
        }
        std::vector<double> vx_(size);
        std::vector<double> vy_(size);
        std::vector<double> vflux_(size);
        for (Py_ssize_t n = 0; n < size; ++n) {
            vx_[n] = bp::extract<double>(vx[n]);
            vy_[n] = bp::extract<double>(vy[n]);
            vflux_[n] = bp::extract<double>(vflux[n]);
        }
        return new PhotonArray(vx_, vy_, vflux_);
    }

    static void wrap() {
        const char * doc = 
            "\n"
            "Class to hold a list of 'photon' arrival positions\n"
            "\n"
            "Class holds a vector of information about photon arrivals: x\n"
            "and y positions, and a flux carried by each photon.  It is the\n"
            "intention that fluxes of photons be nearly equal in absolute\n"
            "value so that noise statistics can be estimated by counting\n"
            "number of positive and negative photons.  This class holds the\n"
            "code that allows its flux to be added to a surface-brightness\n"
            "Image.\n"
            ;
        bp::class_<PhotonArray> pyPhotonArray("PhotonArray", doc, bp::no_init);
        pyPhotonArray
            .def(
                "__init__",
                bp::make_constructor(&construct, bp::default_call_policies(),
                                     bp::args("vx", "vy", "vflux"))
            )
            .def(bp::init<int>(bp::args("n")))
            .def("__len__", &PhotonArray::size)
            .def("reserve", &PhotonArray::reserve)
            .def("setPhoton", &PhotonArray::setPhoton, bp::args("i", "x", "y", "flux"))
            .def("getX", &PhotonArray::getX)
            .def("getY", &PhotonArray::getY)
            .def("getFlux", &PhotonArray::getFlux)
            .def("getTotalFlux", &PhotonArray::getTotalFlux)
            .def("setTotalFlux", &PhotonArray::setTotalFlux)
            .def("append", &PhotonArray::append)
            .def("convolve", &PhotonArray::convolve)
            .def("addTo", 
                 (double(PhotonArray::*)(ImageView<float> &) const)&PhotonArray::addTo,
                 bp::arg("image"),
                 "Add photons' fluxes into image. Returns total flux of photons falling inside "
                 "image bounds.")
            .def("addTo", 
                 (double(PhotonArray::*)(ImageView<double> &) const)&PhotonArray::addTo,
                 bp::arg("image"),
                 "Add photons' fluxes into image. Returns total flux of photons falling inside "
                 "image bounds.")
            ;
    }

};

// Used by multiple profile classes to ensure at most one radius is given.
void checkRadii(const bp::object & r1, const bp::object & r2, const bp::object & r3) {
    int nRad = (r1.ptr() != Py_None) + (r2.ptr() != Py_None) + (r3.ptr() != Py_None);
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
            .def("drawShoot", 
                 (double (SBProfile::*)(Image<U> &, double, UniformDeviate, 
                                        double, double, double, bool) 
                  const)&SBProfile::drawShoot,
                 (bp::arg("image"), bp::arg("N")=0., bp::arg("ud"), bp::arg("dx")=0.,
                  bp::arg("gain")=1., bp::arg("noise")=0., bp::arg("poisson_flux")=true),
                 "Draw object into existing image using photon shooting.\n"
                 "\n"
                 "Setting optional integer arg possionFlux != 0 allows profile flux to vary \n"
                 "according to Poisson statistics for N samples.\n"
                 "\n"
                 "Returns total flux of photons that landed inside image bounds.")
            .def("drawShoot", 
                 (double (SBProfile::*)(ImageView<U>, double, UniformDeviate,
                                        double, double, double, bool)
                  const)&SBProfile::drawShoot,
                 (bp::arg("image"), bp::arg("N")=0., bp::arg("ud"), bp::arg("dx")=0.,
                  bp::arg("gain")=1., bp::arg("noise")=0., bp::arg("poisson_flux")=true),
                 "Draw object into existing image using photon shooting.\n"
                 "\n"
                 "Setting optional integer arg possionFlux != 0 allows profile flux to vary \n"
                 "according to Poisson statistics for N samples.\n"
                 "\n"
                 "Returns total flux of photons that landed inside image bounds.")
            .def("draw", 
                 (double (SBProfile::*)(Image<U> &, double, double, int) const)&SBProfile::draw,
                 (bp::arg("image"), bp::arg("dx")=0., bp::arg("gain")=1., bp::arg("wmult")=1),
                 "Draw in-place, resizing if necessary, and return the summed flux.")
            .def("draw", 
                 (double (SBProfile::*)(ImageView<U> &, double, double, int) const)&SBProfile::draw,
                 (bp::arg("image"), bp::arg("dx")=0., bp::arg("gain")=1., bp::arg("wmult")=1),
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
            .def("nyquistDx", &SBProfile::nyquistDx, "Image pixel spacing that does not alias maxK")
            .def("stepK", &SBProfile::stepK,
                 "Sampling in k space necessary to avoid folding of image in x space")
            .def("isAxisymmetric", &SBProfile::isAxisymmetric)
            .def("hasHardEdges", &SBProfile::hasHardEdges)
            .def("isAnalyticX", &SBProfile::isAnalyticX,
                 "True if real-space values can be determined immediately at any position without\n"
                 " DFT.")
            .def("centroid", &SBProfile::centroid)
            .def("getFlux", &SBProfile::getFlux)
            .def("scaleFlux", &SBProfile::scaleFlux, bp::args("fluxRatio"))
            .def("setFlux", &SBProfile::setFlux, bp::args("flux"))
            .def("applyTransformation", &SBProfile::applyTransformation, bp::args("e"))
            .def("applyShear",
                 (void (SBProfile::*)(Shear))&SBProfile::applyShear,
                 (bp::arg("s")))
            .def("applyRotation", &SBProfile::applyRotation, bp::args("theta"))
            .def("applyShift", &SBProfile::applyShift, bp::args("dx", "dy"))
            .def("shoot", &SBProfile::shoot, bp::args("n", "u"))
            .def("draw", (ImageView<float> (SBProfile::*)(double, double, int) const)&SBProfile::draw,
                 (bp::arg("dx")=0., bp::arg("gain")=1., bp::arg("wmult")=1), "default draw routine")
            ;
        wrapTemplates<float>(pySBProfile);
        wrapTemplates<double>(pySBProfile);
    }

};

struct PySBAdd {

    // This will be wrapped as a Python constructor; it accepts an arbitrary Python iterable.
    static SBAdd * construct(bp::object const & iterable) {
        bp::stl_input_iterator<SBProfile> begin(iterable), end;
        std::list<SBProfile> plist(begin, end);
        return new SBAdd(plist);
    }

    static void wrap() {
        static char const * doc = 
            "Sum of SBProfiles."
            ;
            
        bp::class_< SBAdd, bp::bases<SBProfile> >("SBAdd", doc, bp::no_init)
            // bp tries the overloads in reverse order, so we wrap the most general one first
            // to ensure we try it last
            .def("__init__", bp::make_constructor(&construct, bp::default_call_policies(),
                                                  bp::args("slist")))
            .def(bp::init<const SBProfile &, const SBProfile &>(bp::args("s1", "s2")))
            .def(bp::init<const SBAdd &>())
            ;
    }

};

struct PySBTransform {

    static void wrap() {
        static char const * doc = 
            "SBTransform is an affine transformation of another SBProfile.\n"
            "Origin of original shape will now appear at x0.\n"
            "Flux is NOT conserved in transformation - SB is preserved."
            ;
            
        bp::class_< SBTransform, bp::bases<SBProfile> >("SBTransform", doc, bp::no_init)
            .def(bp::init<const SBProfile &, double, double, double, double, Position<double>, double >(
                     (bp::args("sbin", "mA", "mB", "mC", "mD"),
                      bp::arg("x0")=Position<double>(0.,0.),
                      bp::arg("fluxScaling")=1.)
                 ))
            .def(bp::init<const SBProfile &, const Ellipse &, double>(
                     (bp::arg("sbin"), bp::arg("e")=Ellipse(), bp::arg("fluxScaling")=1.)
                 ))
            .def(bp::init<const SBTransform &>())
            ;
    }

};

struct PySBConvolve {

    // This will be wrapped as a Python constructor; it accepts an arbitrary Python iterable.
    static SBConvolve * construct(bp::object const & iterable, bool real_space) {
        bp::stl_input_iterator<SBProfile> begin(iterable), end;
        std::list<SBProfile> plist(begin, end);
        return new SBConvolve(plist, real_space);
    }

    static void wrap() {
        bp::class_< SBConvolve, bp::bases<SBProfile> >("SBConvolve", bp::no_init)
            // bp tries the overloads in reverse order, so we wrap the most general one first
            // to ensure we try it last
            .def("__init__", 
                 bp::make_constructor(&construct, bp::default_call_policies(), 
                                      (bp::arg("slist"), bp::arg("real_space")=false)
                 ))
            .def(bp::init<const SBProfile &, const SBProfile &, bool>(
                     (bp::args("s1", "s2"), bp::arg("real_space")=false)
                 ))
            .def(bp::init<const SBProfile &, const SBProfile &, const SBProfile &, bool>(
                     (bp::args("s1", "s2", "s3"), bp::arg("real_space")=false)
                 ))
            .def(bp::init<const SBConvolve &>())
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
        const bp::object & half_light_radius,
        const bp::object & sigma,
        const bp::object & fwhm,
        double flux
    ) {
        double s = 1.0;
        checkRadii(half_light_radius, sigma, fwhm);
        if (half_light_radius.ptr() != Py_None) {
            s = bp::extract<double>(half_light_radius) * 0.84932180028801907; // (2\ln2)^(-1/2)
        }
        if (sigma.ptr() != Py_None) {
            s = bp::extract<double>(sigma);
        }
        if (fwhm.ptr() != Py_None) {
            s = bp::extract<double>(fwhm) * 0.42466090014400953; // 1 / (2(2\ln2)^(1/2))
        }
        return new SBGaussian(s, flux);
    }

    static void wrap() {
        bp::class_<SBGaussian,bp::bases<SBProfile> >(
            "SBGaussian",
            "SBGaussian(flux=1., half_light_radius=None, sigma=None, fwhm=None)\n\n"
            "Construct an exponential profile with the given flux and half-light radius,\n"
            "sigma, or FWHM.  Exactly one radius must be provided.\n",
            bp::no_init)
            .def(
                "__init__", bp::make_constructor(
                    &construct, bp::default_call_policies(),
                    (bp::arg("half_light_radius")=bp::object(), bp::arg("sigma")=bp::object(), 
                     bp::arg("fwhm")=bp::object(), bp::arg("flux")=1.))
            )
            .def(bp::init<const SBGaussian &>())
            .def("getSigma", &SBGaussian::getSigma)
            ;
    }
};

struct PySBSersic {

    static SBSersic * construct(
        double n, 
        const bp::object & half_light_radius,
        double flux
    ) {
        if (half_light_radius.ptr() == Py_None) {
            PyErr_SetString(PyExc_TypeError, "No radius parameter given");
            bp::throw_error_already_set();
        }
        return new SBSersic(n, bp::extract<double>(half_light_radius), flux);
    }
    static void wrap() {
        bp::class_<SBSersic,bp::bases<SBProfile> >("SBSersic", bp::no_init)
            .def("__init__",
                 bp::make_constructor(
                     &construct, bp::default_call_policies(),
                     (bp::arg("n"), bp::arg("half_light_radius")=bp::object(), bp::arg("flux")=1.)
                 )
            )
            .def(bp::init<const SBSersic &>())
            .def("getN", &SBSersic::getN)
            .def("getHalfLightRadius", &SBSersic::getHalfLightRadius)
            ;
    }
};

struct PySBExponential {


    static SBExponential * construct(
        const bp::object & half_light_radius,
        const bp::object & scale_radius,
        double flux
    ) {
        double s = 1.0;
        checkRadii(half_light_radius, scale_radius, bp::object());
        if (half_light_radius.ptr() != Py_None) {
            s = bp::extract<double>(half_light_radius) / 1.6783469900166605; // not analytic
        }
        if (scale_radius.ptr() != Py_None) {
            s = bp::extract<double>(scale_radius);
        }
        return new SBExponential(s, flux);
    }

    static void wrap() {
        bp::class_<SBExponential,bp::bases<SBProfile> >(
            "SBExponential",
            "SBExponential(flux=1., half_light_radius=None, scale=None)\n\n"
            "Construct an exponential profile with the given flux and either half-light radius\n"
            "or scale length.  Exactly one radius must be provided.\n",
            bp::no_init)
            .def(
                "__init__", bp::make_constructor(
                    &construct, bp::default_call_policies(),
                    (bp::arg("half_light_radius")=bp::object(), 
                     bp::arg("scale_radius")=bp::object(), (bp::arg("flux")=1.)))
             )
            .def(bp::init<const SBExponential &>())
            .def("getScaleRadius", &SBExponential::getScaleRadius)
            ;
    }
};

struct PySBAiry {
    static void wrap() {
        bp::class_<SBAiry,bp::bases<SBProfile> >("SBAiry", bp::no_init)
            .def(bp::init<double,double,double>(
                    (bp::arg("lam_over_D"), bp::arg("obscuration")=0., bp::arg("flux")=1.))
            )
            .def(bp::init<const SBAiry &>())
            .def("getLamOverD", &SBAiry::getLamOverD)
            .def("getObscuration", &SBAiry::getObscuration)
            ;
    }
};

struct PySBBox {

    static SBBox * construct(
        const bp::object & xw,
        const bp::object & yw,
        double flux
    ) {
        if (xw.ptr() == Py_None || yw.ptr() == Py_None) {
            PyErr_SetString(PyExc_TypeError, "SBBox requires x and y width parameters");
            bp::throw_error_already_set();
        }
        return new SBBox(bp::extract<double>(xw), bp::extract<double>(yw), flux);
    }

    static void wrap() {
        bp::class_<SBBox,bp::bases<SBProfile> >("SBBox", bp::no_init)
            .def("__init__",
                 bp::make_constructor(
                     &construct, bp::default_call_policies(),
                     (bp::arg("xw")=bp::object(), bp::arg("yw")=bp::object(), 
                      bp::arg("flux")=1.))
            )
            .def(bp::init<const SBBox &>())
            .def("getXWidth", &SBBox::getXWidth)
            .def("getYWidth", &SBBox::getYWidth)
            ;
    }
};

struct PySBMoffat {

    static SBMoffat * construct(
        double beta, 
        const bp::object & fwhm,
        const bp::object & scale_radius,
        const bp::object & half_light_radius,
        double trunc, 
        double flux
    ) {
        double s = 1.0;
        checkRadii(half_light_radius, scale_radius, fwhm);
        SBMoffat::RadiusType rType = SBMoffat::FWHM;
        if (fwhm.ptr() != Py_None) {
            s = bp::extract<double>(fwhm);
        }
        if (scale_radius.ptr() != Py_None) {
            s = bp::extract<double>(scale_radius);
            rType = SBMoffat::SCALE_RADIUS;
        }
        if (half_light_radius.ptr() != Py_None) {
            s = bp::extract<double>(half_light_radius);
            rType = SBMoffat::HALF_LIGHT_RADIUS;
        }
        return new SBMoffat(beta, s, rType, trunc, flux);
    }

    static void wrap() {
        bp::class_<SBMoffat,bp::bases<SBProfile> >("SBMoffat", bp::no_init)
            .def("__init__", 
                 bp::make_constructor(
                     &construct, bp::default_call_policies(),
                     (bp::arg("beta"), bp::arg("fwhm")=bp::object(), 
                      bp::arg("scale_radius")=bp::object(),
                      bp::arg("half_light_radius")=bp::object(),
                      bp::arg("trunc")=0., bp::arg("flux")=1.)
                 )
            )
            .def(bp::init<const SBMoffat &>())
            .def("getBeta", &SBMoffat::getBeta)
            .def("getScaleRadius", &SBMoffat::getScaleRadius)
            .def("getFWHM", &SBMoffat::getFWHM)
            .def("getHalfLightRadius", &SBMoffat::getHalfLightRadius)
            ;
    }
};

struct PySBDeVaucouleurs {
    static SBDeVaucouleurs * construct(
        const bp::object & half_light_radius,
        double flux 
    ) {
        if (half_light_radius.ptr() == Py_None) {
            PyErr_SetString(PyExc_TypeError, "No radius parameter given");
            bp::throw_error_already_set();
        }
        return new SBDeVaucouleurs(bp::extract<double>(half_light_radius), flux);
    }

    static void wrap() {
        bp::class_<SBDeVaucouleurs,bp::bases<SBProfile> >(
            "SBDeVaucouleurs",bp::no_init)
            .def("__init__",
                 bp::make_constructor(
                     &construct, bp::default_call_policies(),
                     (bp::arg("half_light_radius")=bp::object(), bp::arg("flux")=1.)
                 )
            )
            .def(bp::init<const SBDeVaucouleurs &>())
            .def("getHalfLightRadius", &SBDeVaucouleurs::getHalfLightRadius)
            ;
    }
};

} // anonymous

void pyExportSBProfile() {
    PySBProfile::wrap();
    PySBAdd::wrap();
    PySBTransform::wrap();
    PySBConvolve::wrap();
    PySBDeconvolve::wrap();
    PySBGaussian::wrap();
    PySBSersic::wrap();
    PySBExponential::wrap();
    PySBAiry::wrap();
    PySBBox::wrap();
    PySBMoffat::wrap();
    PySBDeVaucouleurs::wrap();
    PyPhotonArray::wrap();
}

} // namespace galsim
