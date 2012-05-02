
#include <stdint.h>

#include "boost/python.hpp"
#include "Image.h"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

namespace bp = boost::python;

#define ADD_CORNER(wrapper, getter, prop)                                      \
    do {                                                                \
        bp::object fget = bp::make_function(&BaseImage<T>::getter);   \
        wrapper.def(#getter, fget);                                \
        wrapper.add_property(#prop, fget);                   \
    } while (false)

namespace galsim {
namespace {

template <typename T> struct NumPyTraits;
template <> struct NumPyTraits<int16_t> { static int getCode() { return NPY_INT16; } };
template <> struct NumPyTraits<int32_t> { static int getCode() { return NPY_INT32; } };
template <> struct NumPyTraits<float> { static int getCode() { return NPY_FLOAT32; } };
template <> struct NumPyTraits<double> { static int getCode() { return NPY_FLOAT64; } };

// return the NumPy type for a C++ class (e.g. float -> numpy.float32)
template <typename T>
bp::object getNumPyType() {
    bp::handle<> h(reinterpret_cast<PyObject*>(PyArray_DescrFromType(NumPyTraits<T>::getCode())));
    return bp::object(h).attr("type");
}

template <typename T>
struct PyImage {

    static void destroyCObjectOwner(void * p) {
        boost::shared_ptr<T> * owner = reinterpret_cast< boost::shared_ptr<T> *>(p);
        delete owner;
    }

    struct PythonDeleter {
        void operator()(T * p) { owner.reset(); }

        explicit PythonDeleter(PyObject * o) : owner(bp::borrowed(o)) {}

        bp::handle<> owner;
    };

    static bp::object getArrayImpl(bp::object self, bool isConst) {

        // --- Try to get cached array ---
        if (PyObject_HasAttrString(self.ptr(), "_array")) return self.attr("_array");

        BaseImage<T> const & image = bp::extract<BaseImage<T> const &>(self);

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
                    &PyArray_Type, 2, shape, NumPyTraits<T>::getCode(), strides,
                    const_cast<T*>(image.getData()), sizeof(T), flags, NULL
                )
            )
        );

        // --- Manage ownership ---
        boost::shared_ptr<T> owner = image.getOwner();
        PythonDeleter * pyDeleter = boost::get_deleter<PythonDeleter>(owner);
        bp::handle<> pyOwner;
        if (pyDeleter) {
            // If memory was original allocated by Python, we use that Python object as the owner...
            pyOwner = pyDeleter->owner;
        } else {
            // ..if not, we put a shared_ptr in an opaque Python object.
            pyOwner = bp::handle<>(
                PyCObject_FromVoidPtr(new boost::shared_ptr<T>(owner), &destroyCObjectOwner)
            );
        }
        reinterpret_cast<PyArrayObject*>(result.ptr())->base = pyOwner.release();

        self.attr("_array") = result;
        return result;
    }

    static bp::object getArray(bp::object image) { return getArrayImpl(image, false); }
    static bp::object getConstArray(bp::object image) { return getArrayImpl(image, true); }

    static void buildConstructorArgs(
        bp::object const & array, int xMin, int yMin, bool isConst,
        T * & data, boost::shared_ptr<T> & owner, int & stride, Bounds<int> & bounds
    ) {
        if (!PyArray_Check(array.ptr())) {
            PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument required");
            bp::throw_error_already_set();
        }
        int actualType = PyArray_TYPE(array.ptr());
        int requiredType = NumPyTraits<T>::getCode();
        if (actualType != requiredType) {
            std::ostringstream oss;
            oss<<"numpy.ndarray argument has incorrect data type\n";
            oss<<"T = "<<typeid(T).name()<<"\n";
            oss<<"actualType = "<<actualType<<"\n";
            oss<<"requiredType = "<<requiredType<<"\n";
            oss<<"For reference: \n";
            oss<<"  NPY_SHORT   = "<<NPY_SHORT<<"\n";
            oss<<"  NPY_INT     = "<<NPY_INT<<"\n";
            oss<<"  NPY_INT32   = "<<NPY_INT32<<"\n";
            oss<<"  NPY_LONG    = "<<NPY_LONG<<"\n";
            oss<<"  NPY_FLOAT   = "<<NPY_FLOAT<<"\n";
            oss<<"  NPY_DOUBLE  = "<<NPY_DOUBLE<<"\n";
            oss<<"  sizeof(int16_t) = "<<sizeof(int16_t)<<"\n";
            oss<<"  sizeof(int) = "<<sizeof(int)<<"\n";
            oss<<"  sizeof(long) = "<<sizeof(long)<<"\n";
            oss<<"  sizeof(npy_int16) = "<<sizeof(npy_int16)<<"\n";
            oss<<"  sizeof(npy_int32) = "<<sizeof(npy_int32)<<"\n";
            oss<<"  sizeof(npy_long) = "<<sizeof(npy_long)<<"\n";
            PyErr_SetString(PyExc_ValueError, oss.str().c_str());
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
            xMin, xMin + PyArray_DIM(array.ptr(), 1) - 1,
            yMin, yMin + PyArray_DIM(array.ptr(), 0) - 1
        );
    }

    static ImageView<T> * makeFromArray(bp::object const & array, int xMin, int yMin) {
        Bounds<int> bounds;
        int stride = 0;
        T * data = 0;
        boost::shared_ptr<T> owner;
        buildConstructorArgs(array, xMin, yMin, false, data, owner, stride, bounds);
        return new ImageView<T>(data, owner, stride, bounds);
    }

    static ConstImageView<T> * makeConstFromArray(bp::object const & array, int xMin, int yMin) {
        Bounds<int> bounds;
        int stride = 0;
        T * data = 0;
        boost::shared_ptr<T> owner;
        buildConstructorArgs(array, xMin, yMin, true, data, owner, stride, bounds);
        return new ConstImageView<T>(data, owner, stride, bounds);
    }

    static bp::object wrapImage(std::string const & suffix) {
        
        char const * doc = \
            "Image[SIFD], ImageView[SIFD] and ConstImage[SIFD] are the classes\n"
            "that represent the primary way to pass image data between Python\n"
            "and the GalSim C++ library.\n\n"
            "There is a separate Python class for each C++ template instantiation,\n"
            "and these can be accessed using NumPy types as keys in the Image dict:\n"
            "  ImageS == Image[numpy.int16]\n"
            "  ImageI == Image[numpy.int32]\n"
            "  ImageF == Image[numpy.float32]\n"
            "  ImageD == Image[numpy.float64]\n"
            "\n"
            "An Image can be thought of as containing a 2-d, row-contiguous NumPy\n"
            "array (which it may share with other image views), and origin point, and\n"
            "a pixel scale (the origin and pixel scale are not shared).\n"
            "\n"
            "There are several ways to construct an Image:\n"
            "  Image(ncol, nrow, init_value=0)        # size and initial value - origin @ (1,1)\n"
            "  Image(bounds=BoundsI(), init_value=0)  # bounding box and initial value\n"
            "\n"
            "An Image also has a '.array' attribute that provides a NumPy view into the\n"
            "Image's pixels.  Regardless of how the Image was constructed, this array\n"
            "and the Image will point to the same underlying data, and modifying one\n"
            "will affect the other.\n"
            "\n"
            "Note that both the attribute and the array constructor argument are\n"
            "ordered [y,x], matching the standard NumPy convention, while the Image\n"
            "class's own accessors are all (x,y).\n\n"
            ;

        bp::object getScale = bp::make_function(&BaseImage<T>::getScale);
        bp::object setScale = bp::make_function(&BaseImage<T>::setScale);

        // Need some typedefs and explicit casts here to resolve overloads of methods
        // that have both const and non-const versions:
        typedef const T& (Image<T>::* at_func_type)(const int, const int) const;
        typedef ImageView<T> (Image<T>::* subImage_func_type)(const Bounds<int>&);
        typedef ImageView<T> (Image<T>::* view_func_type)();

        bp::object at = bp::make_function(
            at_func_type(&Image<T>::at),
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
        );
        bp::object getBounds = bp::make_function(
            &BaseImage<T>::getBounds, 
            bp::return_value_policy<bp::copy_const_reference>()
        ); 

        bp::class_< BaseImage<T>, boost::noncopyable >
            pyBaseImage(("BaseImage" + suffix).c_str(), doc, bp::no_init);
        pyBaseImage
            .def("getScale", getScale)
            .def("setScale", setScale)
            .add_property("scale", getScale, setScale)
            .def("subImage", &BaseImage<T>::subImage, bp::args("bounds"))
            .add_property("array", &getConstArray)
            .def("shift", &BaseImage<T>::shift, bp::args("dx", "dy"))
            .def("setOrigin", &BaseImage<T>::setOrigin, bp::args("x0", "y0"))
            .def("setCenter", &BaseImage<T>::setCenter, bp::args("x0", "y0"))
            .def("getBounds", getBounds)
            .add_property("bounds", getBounds)
            ;
        ADD_CORNER(pyBaseImage, getXMin, xMin);
        ADD_CORNER(pyBaseImage, getYMin, yMin);
        ADD_CORNER(pyBaseImage, getXMax, xMax);
        ADD_CORNER(pyBaseImage, getYMax, yMax);
        

        bp::class_< Image<T>, bp::bases< BaseImage<T> > >
            pyImage(("Image" + suffix).c_str(), doc, bp::no_init);
        pyImage
            .def(bp::init<int,int,T>(
                    (bp::args("ncol","nrow"), bp::arg("init_value")=T(0))
            ))
            .def(bp::init<const Bounds<int> &, T>(
                    (bp::arg("bounds")=Bounds<int>(), bp::arg("init_value")=T(0))
            ))
            .def(bp::init<BaseImage<T> const &>(bp::args("other")))
            .def("subImage", subImage_func_type(&Image<T>::subImage), bp::args("bounds"))
            .def("view", view_func_type(&Image<T>::view))
            //.def("assign", &Image<T>::operator=, bp::return_self<>())
            .add_property("array", &getArray)
            // In python, there is no way to have a function return a mutable reference
            // so you can't make im(x,y) = val work correctly.  Thus, the __call__
            // funtion (which is the im(x,y) syntax) is just the const version.
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            .def("setValue", &Image<T>::setValue, bp::args("x","y","value"))
            .def("copyFrom", &Image<T>::copyFrom)
            .def("fill", &Image<T>::fill)
            ;
        
        return pyImage;
    }

    static bp::object wrapImageView(std::string const & suffix) {
        
        char const * doc = \
            "ImageView[SIFD] represents a mutable view of an Image.\n"
            "There is a separate Python class for each C++ template instantiation,\n"
            "and these can be accessed using NumPy types as keys in the ImageView dict:\n"
            "  ImageViewS == ImageView[numpy.int16]\n"
            "  ImageViewI == ImageView[numpy.int32]\n"
            "  ImageViewF == ImageView[numpy.float32]\n"
            "  ImageViewD == ImageView[numpy.float64]\n"
            "From python, the only way to explicitly construct an ImageView is\n"
            "  ImageView(array, xMin=1, yMin=1)       # NumPy array and origin\n"
            "However, they are also the return type of several functions such as\n"
            "  im.view()\n"
            "  im.subImage(bounds)\n"
            "  im[bounds] (equivalent to subImage)\n"
            "  galsim.fits.read(...)\n"
            "\n"
            "The array argument to the constructor must have contiguous values\n"
            "along rows, which should be the case for newly-constructed arrays, but may\n"
            "not be true for some views and generally will not be true for transposes.\n"
            "\n"
            "An ImageView also has a '.array' attribute that provides a NumPy view into the\n"
            "ImageView's pixels.  Regardless of how the ImageView was constructed, this array\n"
            "and the ImageView will point to the same underlying data, and modifying one\n"
            "will affect the other.\n"
            "\n"
            "Note that both the attribute and the array constructor argument are\n"
            "ordered [y,x], matching the standard NumPy convention, while the ImageView\n"
            "class's own accessors are all (x,y).\n\n"
            ;

        bp::object at = bp::make_function(
            &ImageView<T>::at,
            bp::return_value_policy<bp::copy_non_const_reference>(),
            bp::args("x", "y")
        );
        bp::class_< ImageView<T>, bp::bases< BaseImage<T> > >
            pyImageView(("ImageView" + suffix).c_str(), doc, bp::no_init);
        pyImageView
            .def(
                "__init__",
                bp::make_constructor(
                    makeFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xMin")=1, bp::arg("yMin")=1)
                )
            )
            .def(bp::init<ImageView<T> const &>(bp::args("other")))
            .def("subImage", &ImageView<T>::subImage, bp::args("bounds"))
            .def("view", &ImageView<T>::view, bp::return_self<>())
            //.def("assign", &ImageView<T>::operator=, bp::return_self<>())
            .add_property("array", &getArray)
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            .def("setValue", &ImageView<T>::setValue, bp::args("x","y","value"))
            .def("copyFrom", &ImageView<T>::copyFrom)
            .def("fill", &Image<T>::fill)
            ;
        
        return pyImageView;
    }

    static bp::object wrapConstImageView(std::string const & suffix) {
        
        char const * doc = \
            "ConstImageView[SIFD] represents a non-mutable view of an Image.\n"
            "There is a separate Python class for each C++ template instantiation,\n"
            "and these can be accessed using NumPy types as keys in the ConstImageView dict:\n"
            "  ConstImageViewS == ConstImageView[numpy.int16]\n"
            "  ConstImageViewI == ConstImageView[numpy.int32]\n"
            "  ConstImageViewF == ConstImageView[numpy.float32]\n"
            "  ConstImageViewD == ConstImageView[numpy.float64]\n"
            "From python, the only way to explicitly construct an ConstImageView is\n"
            "  ConstImageView(array, xMin=1, yMin=1)       # NumPy array and origin\n"
            "which works just like the version for ImageView except that the resulting \n"
            "object cannot be used to modify the array.\n"
            "\n"
            ;

        bp::object at = bp::make_function(
            &BaseImage<T>::at,
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
        );
        bp::class_< ConstImageView<T>, bp::bases< BaseImage<T> > >
            pyConstImageView(("ConstImageView" + suffix).c_str(), doc, bp::no_init);
        pyConstImageView
            .def(
                "__init__",
                bp::make_constructor(
                    makeConstFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xMin")=1, bp::arg("yMin")=1)
                )
            )
            .def(bp::init<BaseImage<T> const &>(bp::args("other")))
            .def("view", &ConstImageView<T>::view, bp::return_self<>())
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            ;

        return pyConstImageView;
    }

};

} // anonymous

void pyExportImage() {
    bp::dict pyImageDict;  // dict that lets us say "Image[numpy.float32]", etc.

    pyImageDict[getNumPyType<npy_int16>()] = PyImage<int16_t>::wrapImage("S");
    pyImageDict[getNumPyType<npy_int32>()] = PyImage<int32_t>::wrapImage("I");
    pyImageDict[getNumPyType<npy_float32>()] = PyImage<float>::wrapImage("F");
    pyImageDict[getNumPyType<npy_float64>()] = PyImage<double>::wrapImage("D");

    bp::dict pyConstImageViewDict; 

    pyConstImageViewDict[getNumPyType<npy_int16>()] = PyImage<int16_t>::wrapConstImageView("S");
    pyConstImageViewDict[getNumPyType<npy_int32>()] = PyImage<int32_t>::wrapConstImageView("I");
    pyConstImageViewDict[getNumPyType<npy_float32>()] = PyImage<float>::wrapConstImageView("F");
    pyConstImageViewDict[getNumPyType<npy_float64>()] = PyImage<double>::wrapConstImageView("D");

    bp::dict pyImageViewDict;

    pyImageViewDict[getNumPyType<npy_int16>()] = PyImage<int16_t>::wrapImageView("S");
    pyImageViewDict[getNumPyType<npy_int32>()] = PyImage<int32_t>::wrapImageView("I");
    pyImageViewDict[getNumPyType<npy_float32>()] = PyImage<float>::wrapImageView("F");
    pyImageViewDict[getNumPyType<npy_float64>()] = PyImage<double>::wrapImageView("D");

    bp::scope scope;  // a default constructed scope represents the module we're creating
    scope.attr("Image") = pyImageDict;
    scope.attr("ConstImageView") = pyConstImageViewDict;
    scope.attr("ImageView") = pyImageViewDict;
}

} // namespace galsim
