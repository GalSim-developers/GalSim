#include "boost/python.hpp"
#include "Bounds.h"

namespace bp = boost::python;

#define ADD_CORNER(getter, setter, prop)\
    do {                                                            \
        bp::object fget = bp::make_function(&Bounds<T>::getter);    \
        bp::object fset = bp::make_function(&Bounds<T>::setter);    \
        pyBounds.def(#getter, fget);                                \
        pyBounds.def(#setter, fset);                                \
        pyBounds.add_property(#prop, fget, fset);                   \
    } while (false)

namespace galsim {
namespace {

template <typename T>
struct PyPosition {

    static void wrap(std::string const & suffix) {
        
        bp::class_< Position<T> >(("Position" + suffix).c_str(), bp::no_init)
            .def(bp::init< const Position<T> & >(bp::args("other")))
            .def(bp::init<T,T>((bp::arg("x")=T(0), bp::arg("y")=T(0))))
            .def_readwrite("x", &Position<T>::x)
            .def_readwrite("y", &Position<T>::y)
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(bp::self *= bp::other<T>())
            .def(bp::self /= bp::other<T>())
            .def(bp::self * bp::other<T>())
            .def(bp::self / bp::other<T>())
            .def(-bp::self)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def(str(bp::self))
            .def("assign", &Position<T>::operator=, bp::return_self<>())
            ;
    }

};

template <typename T>
struct PyBounds {

    static void wrap(std::string const & suffix) {
        bp::class_< Bounds<T> > pyBounds(("Bounds" + suffix).c_str(), bp::init<>());
        pyBounds
            .def(bp::init<T,T,T,T>(bp::args("xmin","xmax","ymin","ymax")))
            .def(bp::init< const Position<T> &>(bp::args("pos")))
            .def(bp::init< const Position<T> &, const Position<T> & >(bp::args("pos1", "pos2")))
            .def("isDefined", &Bounds<T>::isDefined)
            .def("center", &Bounds<T>::center)
            .def(bp::self += bp::self)
            .def(bp::self += bp::other< Position<T> >())
            .def(bp::self += bp::other<T>())
            .def("addBorder", &Bounds<T>::addBorder)
            .def("expand", &Bounds<T>::expand, "grow by the given factor about center")
            .def(bp::self & bp::self)
            .def("shift", (void (Bounds<T>::*)(const T, const T))&Bounds<T>::shift,
                 bp::args("dx", "dy"))
            .def("shift", (void (Bounds<T>::*)(Position<T>))&Bounds<T>::shift, bp::args("d"))
            .def("includes", (bool (Bounds<T>::*)(const Position<T> &) const)&Bounds<T>::includes)
            .def("includes", (bool (Bounds<T>::*)(const T, const T) const)&Bounds<T>::includes)
            .def("includes", (bool (Bounds<T>::*)(const Bounds<T> &) const)&Bounds<T>::includes)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def("area", &Bounds<T>::area)
            .def(str(bp::self))
            .def("assign", &Bounds<T>::operator=, bp::return_self<>())
            ;
        ADD_CORNER(getXMin, setXMin, xMin);
        ADD_CORNER(getXMax, setXMax, xMax);
        ADD_CORNER(getYMin, setYMin, yMin);
        ADD_CORNER(getYMax, setYMax, yMax);
    }

};

} // anonymous

void pyExportBounds() {
    PyPosition<double>::wrap("D");
    PyBounds<double>::wrap("D");
    PyPosition<int>::wrap("I");
    PyBounds<int>::wrap("I");
}

} // namespace galsim
