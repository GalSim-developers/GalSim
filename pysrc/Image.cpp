#include "boost/python.hpp"
#include "Image.h"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

namespace bp = boost::python;

#define ADD_CORNER(getter, prop)                                      \
    do {                                                                \
        bp::object fget = bp::make_function(&Image<const T>::getter);   \
        pyConstImage.def(#getter, fget);                                \
        pyConstImage.add_property(#prop, fget);                   \
    } while (false)

namespace galsim {
namespace {

template <typename T> struct NumpyTraits;
template <> struct NumpyTraits<npy_short> { static int getCode() { return NPY_SHORT; } };
template <> struct NumpyTraits<npy_int> { static int getCode() { return NPY_INT; } };
template <> struct NumpyTraits<npy_float> { static int getCode() { return NPY_FLOAT; } };
template <> struct NumpyTraits<npy_double> { static int getCode() { return NPY_DOUBLE; } };

template <typename T>
struct PyImage {

    static void destroyCObjectOwner(void * p) {
        boost::shared_ptr<T const> * owner = reinterpret_cast< boost::shared_ptr<T const> *>(p);
        delete owner;
    }

    struct PythonDeleter {
        void operator()(T * p) { owner.reset(); }

        explicit PythonDeleter(PyObject * o) : owner(bp::borrowed(o)) {}

        bp::handle<> owner;
    };

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

    static bp::object getArrayImpl(bp::object self, bool isConst) {

        // --- Try to get cached array ---
        if (PyObject_HasAttrString(self.ptr(), "_array")) return self.attr("_array");

        Image<const T> const & image = bp::extract<Image<const T> const &>(self);

        // --- Create array ---
        int flags = NPY_ALIGNED;
        if (!isConst) flags |= NPY_WRITEABLE;
        npy_intp shape[2] = {
            image.getYMax() - image.getYMin() + 1,
            image.getXMax() - image.getXMin() + 1
        };
        npy_intp strides[2] = { image.getStride() * sizeof(T), sizeof(T), };
        bp::object result(
            bp::handle<>(
                PyArray_New(
                    &PyArray_Type, 2, shape, NumpyTraits<T>::getCode(), strides,
                    const_cast<T*>(image.getData()), sizeof(T), flags, NULL
                )
            )
        );

        // --- Manage ownership ---
        boost::shared_ptr<T const> owner = image.getOwner();
        PythonDeleter * pyDeleter = boost::get_deleter<PythonDeleter>(owner);
        bp::handle<> pyOwner;
        if (pyDeleter) {
            // If memory was original allocated by Python, we use that Python object as the owner...
            pyOwner = pyDeleter->owner;
        } else {
            // ..if not, we put a shared_ptr in an opaque Python object.
            pyOwner = bp::handle<>(
                PyCObject_FromVoidPtr(new boost::shared_ptr<T const>(owner), &destroyCObjectOwner)
            );
        }
        reinterpret_cast<PyArrayObject*>(result.ptr())->base = pyOwner.release();

        self.attr("_array") = result;
        return result;
    }

    static bp::object getArray(bp::object image) { return getArrayImpl(image, false); }
    static bp::object getConstArray(bp::object image) { return getArrayImpl(image, true); }

    static void buildConstructorArgs(
        bp::object const & array, int x0, int y0, bool isConst,
        T * & data, boost::shared_ptr<T> & owner, int & stride, Bounds<int> & bounds
    ) {
        if (!PyArray_Check(array.ptr())) {
            PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument required");
            bp::throw_error_already_set();
        }
        int actualType = PyArray_TYPE(array.ptr());
        int requiredType = NumpyTraits<T>::getCode();
        if (actualType != requiredType) {
            PyErr_SetString(PyExc_ValueError, "numpy.ndarray argument has incorrect data type");
            bp::throw_error_already_set();
        }
        if (PyArray_NDIM(array.ptr()) != 2) {
            PyErr_SetString(PyExc_ValueError, "numpy.ndarray argument has must be 2-d");
            bp::throw_error_already_set();
        }
        if (!isConst && !(PyArray_FLAGS(array.ptr()) & NPY_WRITEABLE)) {
            PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument must be writeable");
            bp::throw_error_already_set();
        }
        if (PyArray_STRIDE(array.ptr(), 1) != sizeof(T)) {
            PyErr_SetString(PyExc_ValueError, "numpy.ndarray argument must have contiguous rows");
            bp::throw_error_already_set();
        }
        stride = PyArray_STRIDE(array.ptr(), 0) / sizeof(T);
        data = reinterpret_cast<T*>(PyArray_DATA(array.ptr()));
        PyObject * pyOwner = PyArray_BASE(array.ptr());
        if (pyOwner) {
            if (PyArray_Check(pyOwner) && PyArray_TYPE(pyOwner) == requiredType) {
                // Not really important, but we try to use the full array for 
                // the owner pointer if this is a subarray, just to be consistent
                // with how it works for subimages.
                // The deleter is really all that matters.
                owner = boost::shared_ptr<T>(
                    reinterpret_cast<T*>(PyArray_DATA(pyOwner)),
                    PythonDeleter(pyOwner)
                );
            } else {
                owner = boost::shared_ptr<T>(
                    reinterpret_cast<T*>(PyArray_DATA(array.ptr())),
                    PythonDeleter(pyOwner)
                );
            }
        } else {
            owner = boost::shared_ptr<T>(
                reinterpret_cast<T*>(PyArray_DATA(array.ptr())),
                PythonDeleter(array.ptr())
            );
        }
        bounds = Bounds<int>(
            x0, x0 + PyArray_DIM(array.ptr(), 1) - 1,
            y0, y0 + PyArray_DIM(array.ptr(), 0) - 1
        );
    }

    static Image<T> * makeFromArray(bp::object const & array, int x0, int y0) {
        Bounds<int> bounds;
        int stride = 0;
        T * data = 0;
        boost::shared_ptr<T> owner;
        buildConstructorArgs(array, x0, y0, false, data, owner, stride, bounds);
        return new Image<T>(data, owner, stride, bounds);
    }

    static Image<const T> * makeConstFromArray(bp::object const & array, int x0, int y0) {
        Bounds<int> bounds;
        int stride = 0;
        T * data = 0;
        boost::shared_ptr<T> owner;
        buildConstructorArgs(array, x0, y0, true, data, owner, stride, bounds);
        return new Image<const T>(data, owner, stride, bounds);
    }

    static void wrap(std::string const & suffix) {
        
        bp::object getScale = bp::make_function(&Image<const T>::getScale);
        bp::object setScale = bp::make_function(&Image<const T>::setScale);
        bp::object at = bp::make_function(
            &Image<const T>::at,
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
        );
        bp::object getBounds = bp::make_function(
            &Image<const T>::getBounds, 
            bp::return_value_policy<bp::copy_const_reference>()
        ); 
        bp::class_< Image<const T> >
            pyConstImage(("ConstImage" + suffix).c_str(), bp::no_init);
        wrapCommon<const T>(pyConstImage);
        pyConstImage
            .def(
                "__init__",
                bp::make_constructor(
                    makeConstFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("x0")=1, bp::arg("y0")=1)
                )
            )
            .add_property("array", &getConstArray)
            .def("getScale", getScale)
            .def("setScale", setScale)
            .add_property("scale", getScale, setScale)
            .def("duplicate", &Image<const T>::duplicate)
            .def("resize", &Image<const T>::resize, bp::args("bounds"))
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            .def("shift", &Image<const T>::shift, bp::args("dx", "dy"))
            .def("move", &Image<const T>::move, bp::args("x0", "y0"))
            .def("getBounds", getBounds)
            .add_property("bounds", getBounds)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self * bp::self)
            .def(bp::self / bp::self)
            ;
        ADD_CORNER(getXMin, xMin);
        ADD_CORNER(getYMin, yMin);
        ADD_CORNER(getXMax, xMax);
        ADD_CORNER(getYMax, yMax);

        bp::class_< Image<T>, bp::bases< Image<const T> > >
            pyImage(("Image" + suffix).c_str(), bp::no_init);
        wrapCommon<T>(pyImage);
        pyImage
            .def(
                "__init__",
                bp::make_constructor(
                    makeFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("x0")=1, bp::arg("y0")=1)
                )
            )
            .add_property("array", &getArray)
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
