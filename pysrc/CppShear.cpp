#include "boost/python.hpp"
#include "CppShear.h"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyCppShear {

    static bp::handle<> getMatrix(CppShear const & self) {
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
            "CppShear is represented internally by e1 and e2, which are the second-moment\n"
            "definitions: ellipse with axes a & b has e=(a^2-b^2)/(a^2+b^2).\n"
            "But can get/set the ellipticity by other measures:\n"
            "g (default constructor) is \"reduced shear\" such that g=(a-b)/(a+b)\n"
            "eta is \"conformal shear\" such that a/b = exp(eta).\n"
            "The constructor takes g1/g2 (reduced shear) only.\n"
            "Beta is always the real-space position angle of major axis.\n"
            "e.g., g1 = g cos(2*Beta), g2 = g sin(2*Beta).\n"
            "\n"
            "The + and - operators for CppShear are overloaded to do\n"
            "Composition: returns ellipticity of\n"
            "circle that is sheared first by RHS and then by\n"
            "LHS CppShear.  Note that this addition is\n"
            "***not commutative***!\n"
            "In the += and -= operations, self is LHS\n"
            "and the operand is RHS of + or - .\n"
            ;
            

        bp::class_<CppShear>("_CppShear", doc, bp::init<const CppShear &>())
            .def(bp::init<double,double>((bp::arg("g1")=0.,bp::arg("g2")=0.)))
            .def("setE1E2", &CppShear::setE1E2, (bp::arg("e1")=0.,bp::arg("e2")=0.),
                 bp::return_self<>())
            .def("setEBeta", &CppShear::setEBeta, (bp::arg("e")=0.,bp::arg("beta")=0.),
                 bp::return_self<>())
            .def("setEta1Eta2", &CppShear::setEta1Eta2, (bp::arg("eta1")=0.,bp::arg("eta2")=0.),
                 bp::return_self<>())
            .def("setEtaBeta", &CppShear::setEtaBeta, (bp::arg("eta")=0.,bp::arg("beta")=0.),
                 bp::return_self<>())
            .def("setG1G2", &CppShear::setG1G2, (bp::arg("g1")=0.,bp::arg("g2")=0.),
                 bp::return_self<>())
            .def("getE1", &CppShear::getE1)
            .def("getE2", &CppShear::getE2)
            .def("getE", &CppShear::getE)
            .def("getESq", &CppShear::getESq)
            .def("getBeta", &CppShear::getBeta)
            .def("getEta", &CppShear::getEta)
            .def("getG", &CppShear::getG)
            .def("getG1", &CppShear::getG1)
            .def("getG2", &CppShear::getG2)
            .def(-bp::self)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(
                "rotationWith", &CppShear::rotationWith, bp::args("rhs"),
                "Give the rotation angle for self+rhs;\n"
                "the s1 + s2 operation on points in\n"
                "the plane induces a rotation as well as a shear.\n"
                "This tells you what the rotation was for LHS+RHS.\n"
            )
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def("fwd", &CppShear::fwd<double>, "FIXME: needs documentation!")
            .def("inv", &CppShear::inv<double>, "FIXME: needs documentation!")
            .def("getMatrix", &getMatrix)
            .def(str(bp::self))
            .def("assign", &CppShear::operator=, bp::return_self<>())
            ;
    }

};

struct PyCppEllipse {

    static bp::handle<> getMatrix(CppEllipse const & self) {
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
            "Conventions for order of compounding, etc., are same as for CppShear.\n"
            ;
        bp::class_<CppEllipse>("_CppEllipse", doc, bp::init<const CppEllipse &>())
            .def(
                 bp::init<const CppShear &, double, const Position<double> &>(
                     (bp::arg("s")=CppShear(), bp::arg("mu")=0., 
                      bp::arg("p")=Position<double>())
                     )
            )
            .def(-bp::self)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def("reset", (void (CppEllipse::*)(const CppShear &, double, const Position<double>))&CppEllipse::reset,
                 bp::args("s", "mu", "p"))
            .def("fwd", &CppEllipse::fwd, "FIXME: needs documentation!")
            .def("inv", &CppEllipse::inv, "FIXME: needs documentation!")
            .def("setS", &CppEllipse::setS, bp::return_self<>())
            .def("setMu", &CppEllipse::setMu, bp::return_self<>())
            .def("setX0", &CppEllipse::setX0, bp::return_self<>())
            .def("getS", &CppEllipse::getS)
            .def("getMu", &CppEllipse::getMu)
            .def("getX0", &CppEllipse::getX0)
            .def("getMajor", &CppEllipse::getMajor, "FIXME: is this semi-major or full major axis?")
            .def("getMinor", &CppEllipse::getMinor, "FIXME: is this semi-minor or full minor axis?")
            .def("getBeta", &CppEllipse::getMinor, "position angle FIXME: which convention?")
            .def("range", &CppEllipse::range, (bp::arg("nSigma")=1.))
            .def("getMatrix", &getMatrix)
            .def(str(bp::self))
            .def("assign", &CppEllipse::operator=, bp::return_self<>())
            ;
    }

};

} // anonymous

void pyExportCppShear() {
    PyCppShear::wrap();
    PyCppEllipse::wrap();
}

} // namespace galsim
