#include "boost/python.hpp"
#include "Image.h"

namespace bp = boost::python;

namespace galsim {
namespace {

template <typename T> struct PyImage;

template <typename T>
struct PyImage {

    template <typename U, typename W>
    static void wrapCommon(W & wrapper) {
        wrapper
            .def(bp::init<int,int>(bp::args("nrows","ncols")))
            .def(bp::init<const Bounds<int> &, T>(
                     (bp::arg("bounds")=Bounds<int>(), bp::arg("initValue")=T(0))
                 ))
            .def(bp::init<Image<U> const &>(bp::args("other")))
            .def("subimage", &Image<U>::subimage, bp::args("bounds"))
            .def("assign", &Image<U>::operator=, bp::return_self<>())
            ;
    }

    static void wrap(std::string const & suffix) {
        
        bp::object getScale = bp::make_function(&Image<const T>::getScale);
        bp::object setScale = bp::make_function(&Image<const T>::setScale);
        bp::object at = bp::make_function(
            &Image<const T>::at,
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
        );
        bp::class_< Image<const T> > pyConstImage(("ConstImage" + suffix).c_str(), bp::no_init);
        wrapCommon<const T>(pyConstImage);
        pyConstImage
            .def("getScale", getScale)
            .def("setScale", setScale)
            .add_property("scale", getScale, setScale)
            .def("duplicate", &Image<const T>::duplicate)
            .def("resize", &Image<const T>::resize, bp::args("bounds"))
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            .def("shift", &Image<const T>::shift, bp::args("dx", "dy"))
            .def("move", &Image<const T>::move, bp::args("x0", "y0"))
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self * bp::self)
            .def(bp::self / bp::self)
            ;

        bp::class_< Image<T>, bp::bases< Image<const T> > > pyImage(("Image" + suffix).c_str(), bp::no_init);
        wrapCommon<T>(pyImage);
        pyImage
            .def("copyFrom", &Image<T>::copyFrom)
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(bp::self *= bp::self)
            .def(bp::self /= bp::self)
            .def("fill", &Image<T>::fill)
            .def(bp::self += bp::other<T>())
            .def(bp::self -= bp::other<T>())
            .def(bp::self *= bp::other<T>())
            .def(bp::self /= bp::other<T>())
            ;

    }

};

} // anonymous

void pyExportImage() {
    PyImage<short>::wrap("S");
    PyImage<int>::wrap("I");
    PyImage<float>::wrap("F");
    PyImage<double>::wrap("D");

}

} // namespace galsim
