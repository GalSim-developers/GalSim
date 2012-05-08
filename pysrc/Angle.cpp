#include "boost/python.hpp"
#include "Angle.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyAngleUnit {

    static void wrap() {
        bp::class_< AngleUnit >("AngleUnit", bp::no_init)
            .def(bp::init<double>(bp::arg("val")))
            .def(bp::self == bp::self)
            .def(bp::other<double>() * bp::self)
            ;
    }

};

struct PyAngle {

    static void wrap() {
        bp::class_< Angle >("Angle", bp::init<>())
            .def(bp::init<double, AngleUnit>(bp::args("val","unit")))
            .def(bp::init<const Angle&>(bp::args("rhs")))
            .def(bp::self / bp::other<AngleUnit>())
            .def(bp::self *= bp::other<double>())
            .def(bp::self /= bp::other<double>())
            .def(bp::self * bp::other<double>())
            .def(bp::other<double>() * bp::self)
            .def(bp::self / bp::other<double>())
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def(bp::self <= bp::self)
            .def(bp::self < bp::self)
            .def(bp::self >= bp::self)
            .def(bp::self > bp::self)
            .def(str(bp::self))
            ;
    }

};

} // anonymous

void pyExportAngle() {
    PyAngleUnit::wrap();
    PyAngle::wrap();
}

} // namespace galsim
