#include "boost/python.hpp"
#include "Shear.h"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyShear {

    static bp::handle<> getMatrix(Shear const & self) {
        static npy_intp dim[2] = {2, 2};
        // Because the C++ version sets references that are passed in, and that's not possible in
        // Python, we wrap this instead, which returns a numpy array.
        double a=0., b=0., c=0.;
        self.getMatrix(a, b, c);
        bp::handle<> r(PyArray_SimpleNew(2, dim, NPY_DOUBLE));
        *reinterpret_cast<double*>(PyArray_GETPTR2(r.get(), 0, 0)) = a;
        *reinterpret_cast<double*>(PyArray_GETPTR2(r.get(), 1, 1)) = b;
        *reinterpret_cast<double*>(PyArray_GETPTR2(r.get(), 0, 1)) = c;
        *reinterpret_cast<double*>(PyArray_GETPTR2(r.get(), 1, 0)) = c;
        return r;
    }

    static void wrap() {
        static char const * doc = 
            "Shear is represented internally by e1 and e2, which are the second-moment\n"
            "definitions: ellipse with axes a & b has e=(a^2-b^2)/(a^2+b^2).\n"
            "But can get/set the ellipticity by two other measures:\n"
            "g is \"reduced shear\" such that g=(a-b)/(a+b)\n"
            "eta is \"conformal shear\" such that a/b = exp(eta).\n"
            "Beta is always the position angle of major axis.\n"
            "FIXME: what convention for position angle?\n"
            "\n"
            "The + and - operators for Shear are overloaded to do\n"
            "Composition: returns ellipticity of\n"
            "circle that is sheared first by RHS and then by\n"
            "LHS Shear.  Note that this addition is\n"
            "***not commutative***!\n"
            "In the += and -= operations, self is LHS\n"
            "and the operand is RHS of + or - .\n"
            ;
            

        bp::class_<Shear>("Shear", doc, bp::init<const Shear &>())
            .def(bp::init<double,double>((bp::arg("e1")=0.,bp::arg("e2")=0.)))
            .def("setE1E2", &Shear::setE1E2, (bp::arg("e1")=0.,bp::arg("e2")=0.),
                 bp::return_self<>())
            .def("setEBeta", &Shear::setEBeta, (bp::arg("e")=0.,bp::arg("beta")=0.),
                 bp::return_self<>())
            .def("setEta1Eta2", &Shear::setEta1Eta2, (bp::arg("eta1")=0.,bp::arg("eta2")=0.),
                 bp::return_self<>())
            .def("setEtaBeta", &Shear::setEtaBeta, (bp::arg("eta")=0.,bp::arg("beta")=0.),
                 bp::return_self<>())
            .def("setG1G2", &Shear::setG1G2, (bp::arg("g1")=0.,bp::arg("g2")=0.),
                 bp::return_self<>())
            .def("getE1", &Shear::getE1)
            .def("getE2", &Shear::getE2)
            .def("getE", &Shear::getE)
            .def("getESq", &Shear::getESq)
            .def("getBeta", &Shear::getBeta)
            .def("getEta", &Shear::getEta)
            .def("getG", &Shear::getG)
            .def(-bp::self)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(
                "rotationWith", &Shear::rotationWith, bp::args("rhs"),
                "Give the rotation angle for self+rhs;\n"
                "the s1 + s2 operation on points in\n"
                "the plane induces a rotation as well as a shear.\n"
                "This tells you what the rotation was for LHS+RHS.\n"
            )
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def(bp::self * bp::other<double>())
            .def(bp::self / bp::other<double>())
            .def(bp::self *= bp::other<double>())
            .def(bp::self /= bp::other<double>())
            .def("fwd", &Shear::fwd<double>, "FIXME: needs documentation!")
            .def("inv", &Shear::inv<double>, "FIXME: needs documentation!")
            .def("getMatrix", &getMatrix)
            .def(str(bp::self))
            .def("assign", &Shear::operator=, bp::return_self<>())
            ;
    }

};

struct PyEllipse {

    static bp::handle<> getMatrix(Ellipse const & self) {
        static npy_intp dim[2] = {2, 2};
        // Because the C++ version sets references that are passed in, and that's not possible in
        // Python, we wrap this instead, which returns a numpy array.
        tmv::Matrix<double> m = self.getMatrix();
        bp::handle<> r(PyArray_SimpleNew(2, dim, NPY_DOUBLE));
        *reinterpret_cast<double*>(PyArray_GETPTR2(r.get(), 0, 0)) = m(0,0);
        *reinterpret_cast<double*>(PyArray_GETPTR2(r.get(), 1, 1)) = m(1,1);
        *reinterpret_cast<double*>(PyArray_GETPTR2(r.get(), 0, 1)) = m(0,1);
        *reinterpret_cast<double*>(PyArray_GETPTR2(r.get(), 1, 0)) = m(1,0);
        return r;
    }

    static void wrap() {
        static char const * doc = 
            "Class to describe transformation from an ellipse\n"
            "with center x0, size exp(mu), and shape s to the unit circle.\n"
            "Map from source plane to image plane is defined as\n"
            "E(x) = T(D(S(x))), where S=shear, D=dilation, T=translation.\n"
            "Conventions for order of compounding, etc., are same as for Shear.\n"
            ;
        bp::class_<Ellipse>("Ellipse", doc, bp::init<const Ellipse &>())
            .def(
                bp::init<double,double,double,double,double>(
                    (bp::arg("e1")=0.,bp::arg("e2")=0.,bp::arg("m")=0.,bp::arg("x")=0.,bp::arg("y")=0.)
                )
            )
            .def(bp::init<const Shear &, double, const Position<double> &>(bp::args("s", "mu", "p")))
            .def(-bp::self)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def("reset", (void (Ellipse::*)(double,double,double,double,double))&Ellipse::reset,
                 (bp::arg("e1")=0.,bp::arg("e2")=0.,bp::arg("m")=0.,bp::arg("x")=0.,bp::arg("y")=0.))
            .def("reset", (void (Ellipse::*)(const Shear &, double, const Position<double>))&Ellipse::reset,
                 bp::args("s", "mu", "p"))
            .def("fwd", &Ellipse::fwd, "FIXME: needs documentation!")
            .def("inv", &Ellipse::inv, "FIXME: needs documentation!")
            .def("setS", &Ellipse::setS, bp::return_self<>())
            .def("setMu", &Ellipse::setMu, bp::return_self<>())
            .def("setX0", &Ellipse::setX0, bp::return_self<>())
            .def("getS", &Ellipse::getS)
            .def("getMu", &Ellipse::getMu)
            .def("getX0", &Ellipse::getX0)
            .def("getMajor", &Ellipse::getMajor, "FIXME: is this semi-major or full major axis?")
            .def("getMinor", &Ellipse::getMinor, "FIXME: is this semi-minor or full minor axis?")
            .def("getBeta", &Ellipse::getMinor, "position angle FIXME: which convention?")
            .def("range", &Ellipse::range, (bp::arg("nSigma")=1.))
            .def("getMatrix", &getMatrix)
            .def(str(bp::self))
            .def("assign", &Ellipse::operator=, bp::return_self<>())
            ;
    }

};

} // anonymous

void pyExportShear() {
    PyShear::wrap();
    PyEllipse::wrap();
}

} // namespace galsim
