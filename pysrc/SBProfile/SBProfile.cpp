#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"
#include "SBProfile.h"

namespace bp = boost::python;

namespace sbp {
namespace {

typedef bp::return_value_policy<bp::manage_new_object> ManageNew;

struct PySBProfile {

    static void wrap() {
        static char const * doc = 
            "SBProfile is an abstract base class represented all of the\n"
            "2d surface brightness that we know how to draw.\n"
            "Every SBProfile knows how to draw an Image<float> of itself in real\n"
            "and k space.  Each also knows what is needed to prevent aliasing\n"
            "or truncation of itself when drawn.\n"
            "\n"
            "Note that when you use the SBProfile::draw() routines you\n"
            "will get an image of **surface brightness** values in each pixel,\n"
            "not the flux that fell into the pixel.  To get flux, you\n"
            "must multiply the image by (dx*dx).\n"
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
            ;

        bp::class_<SBProfile,boost::noncopyable>("SBProfile", doc, bp::no_init)
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
            .def("centroidX", &SBProfile::centroidX)
            .def("centroidY", &SBProfile::centroidY)
            .def("centroid", &SBProfile::centroid)
            .def("getFlux", &SBProfile::getFlux)
            .def("setFlux", &SBProfile::setFlux)
            .def("distort", &SBProfile::distort, bp::args("e"), ManageNew())
            .def("shear", &SBProfile::shear, bp::args("e1", "e2"), ManageNew())
            .def("rotate", &SBProfile::rotate, bp::args("theta"), ManageNew())
            .def("shift", &SBProfile::shift, bp::args("dx", "dy"), ManageNew())
            ;
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
            
        bp::class_<SBAdd>("SBAdd", doc, bp::init<>())
            // bp tries the overloads in reverse order, so we wrap the most general one first
            // to ensure we try it last
            .def("__init__", bp::make_constructor(&construct, bp::default_call_policies(), bp::arg("slist")))
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
            
        bp::class_<SBDistort>("SBDistort", doc, bp::no_init)
            .def(bp::init<const SBProfile &, double, double, double, double, Position<double> >(
                     (bp::args("sbin", "mA", "mB", "mC", "mD"), bp::arg("x0")=Position<double>(0.,0.))
                 ))
            .def(bp::init<const SBProfile &, const Ellipse &>((bp::arg("sbin"), bp::arg("e")=Ellipse())))
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
        bp::class_<SBConvolve>("SBConvolve", bp::init<>())
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

struct PySBGaussian {
    static void wrap() {
        bp::class_<SBGaussian>(
            "SBGaussian",
            bp::init<double,double>((bp::arg("flux")=1., bp::arg("sigma")=1.))
        );
    }
};

struct PySBSersic {
    static void wrap() {
        bp::class_<SBSersic>(
            "SBSersic",
            bp::init<double,double,double>((bp::arg("n"), bp::arg("flux")=1., bp::arg("re")=1.))
        );
    }
};

struct PySBExponential {
    static void wrap() {
        bp::class_<SBExponential>(
            "SBExponential",
            bp::init<double,double>((bp::arg("flux")=1., bp::arg("r0")=1.))
        );
    }
};

struct PySBAiry {
    static void wrap() {
        bp::class_<SBAiry>(
            "SBAiry",
            bp::init<double,double,double>((bp::arg("D")=1., bp::arg("obs")=1., bp::arg("flux")=1.))
        );
    }
};

struct PySBBox {
    static void wrap() {
        bp::class_<SBBox>(
            "SBBox",
            bp::init<double,double,double>((bp::arg("xw")=1., bp::arg("yw")=0., bp::arg("flux")=1.))
        );
    }
};

struct PySBMoffat {
    static void wrap() {
        bp::class_<SBMoffat>(
            "SBMoffat",
            bp::init<double,double,double,double>(
                (bp::arg("beta"), bp::arg("truncationFWHM")=2., bp::arg("flux")=1., bp::arg("re")=1.)
            )
        );
    }
};

struct PySBDeVaucouleurs {
    static void wrap() {
        bp::class_<SBDeVaucouleurs>(
            "SBDeVaucouleurs",
            bp::init<double,double>((bp::arg("flux")=1., bp::arg("r0")=1.))
        );
    }
};

} // anonymous

void pyExportSBProfile() {
    PySBProfile::wrap();
    PySBAdd::wrap();
    PySBDistort::wrap();
    PySBConvolve::wrap();
    PySBGaussian::wrap();
    PySBSersic::wrap();
    PySBExponential::wrap();
    PySBAiry::wrap();
    PySBBox::wrap();
    PySBMoffat::wrap();
    PySBDeVaucouleurs::wrap();
}

} // namespace sbp
