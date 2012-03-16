
// Image<T> class is a 2d array of any class T.  Data are stored 
// in a form that allows rapid iteration along rows, and a number of
// templates are provided that execute operations on all pixels or
// a subset of pixels in one or more images.
//
// Will be linked with Image.o from Image.cpp.  Note that Image types
// float, int, and short are instantiated at end of Image.cpp.  If you
// need other types, you'll have to add them there.
//
// Image is actually a handle that contains a pointer to an ImageHeader<T>
// structure ("header") and an ImageData<T> structure ("data").  Copy
// and assignment semantics are that a new Image refers to same data
// and header structures as the old one.  Link counting insures deletion
// of the header & data structures when they become unused.  To get a
// fresh deep copy of an Image, use Image::duplicate().
//
// The ImageData<T> class should never be needed by the user.  It
// is used by FITSImage class (and maybe other disk image formats) 
// that reads/writes Image objects from disk.
//
// All specifications of pixel areas use Bounds<int> objects - see Bounds.h.
// Bounds<int>(x1,x2,y1,y2) is usual constructor, Bounds<int>() creates a
// null region.
//
// Image::subimage creates a new image that is contained within the original
// image AND SHARES ITS DATA.  Deleting the parent Image before any
// of its derived subimages throws an exception.
//
// Iterators Image::iter and Image::citer are provided to traverse rows 
// of images.  These are only valid to traverse a row, going past end of 
// row will give unpredictable results.  Functions rowBegin() and rowEnd() 
// give bounds for row iteration.  getIter() gets iterator to arbitrary 
// point.  
// Range-checked iterators are ImageChk_iter and ImageChk_citer, which are 
// also typedef'd as Image::checked_iter and checked_citer.  Range-checked 
// access is via Image::at() calls.  Range-checked iterators are used for 
// all calls if
// #define IMAGE_BOUNDS_CHECK
// is compiled in. 
//
// A const Image has read-only header and data access.
//
// Image constructors are:
// Image(ncol, nrows) makes new image with origin at (1,1)
// Image(Bounds<int>) makes new image with arbitrary row/col range
// Image(Bounds<int>, value) makes new image w/all pixels set to value.
//   also a constructor directly from ImageHeader & ImageData structures,
//   which should be used only by FITSImage or routines that build Images.
//
// You can access image elements with (int x, int y) syntax:
//  theImage(4,12)=15.2;
// Many unary and binary arithmetic operations are supplied, with
// templates provided to build others quickly.
//

#ifndef Image_H
#define Image_H

#include <algorithm>
#include <functional>
#include <list>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>

#include <boost/shared_ptr.hpp>

#include "Std.h"
#include "Bounds.h"

namespace galsim {

    // Exception classes:
    class ImageError : public std::runtime_error
    {
    public: 
        ImageError(const std::string& m="") : 
            std::runtime_error("Image Error: " + m) {}

    };

    class ImageBounds : public ImageError 
    {
    public: 
        ImageBounds(const std::string& m="") : 
            ImageError("Access to out-of-bounds pixel " + m) {}

        ImageBounds(const std::string& m, const int min, const int max, const int tried);

        ImageBounds(const int x, const int y, const Bounds<int> b);
    };

    template <typename T> class Image;

    template <typename T>
    class Image<const T> {
    protected:
        
        boost::shared_ptr<T> _owner;  // manages ownership; _owner.get() != _data if subimage
        T * _data;                      // pointer to be used for this image
        int _stride;                    // number of elements between rows (!= width for subimages)
        Bounds<int> _bounds;

        inline int addressPixel(const int y) const {
            return (y - getYMin()) * _stride;
        }
        
        inline int addressPixel(const int x, const int y) const {
            return (x - getXMin()) + addressPixel(y);
        }

        void makeSubimageInPlace(const Bounds<int> & bounds);

    public:

        Image(const int ncol, const int nrow);

        // Default constructor builds a null image:
        explicit Image(const Bounds<int> & bounds=Bounds<int>(), const T initValue=T(0));

        /**
         *  @brief Shallow constructor.
         *
         *  The new image will share pixel data (but nothing else) with the input image.
         */
        Image(const Image& rhs) :
            _owner(rhs._owner), _data(rhs._data), _stride(rhs._stride), _bounds(rhs._bounds)
        {}

        /**
         *  @brief Shallow conversion constructor.
         *
         *  We can construct an Image<const T> from an Image<T>, but not the reverse.
         *
         *  The new image will share pixel data (but nothing else) with the input image.
         */
        Image(const Image<T>& rhs);

        /**
         *  @brief Shallow assignment.
         *
         *  The two images will share pixel data (but nothing else).
         */
        Image & operator=(const Image& rhs) {
            if (&rhs != this) {
                _owner = rhs._owner;
                _data = rhs._data;
                _stride = rhs._stride;
                _bounds = rhs._bounds;
            }
            return *this;
        }

        /**
         *  @brief Shallow conversion assignment.
         *
         *  We can assign an Image<T> to an Image<const T>, but not the reverse.
         *
         *  The two images will share pixel data (but nothing else).
         */
        Image & operator=(const Image<T>& rhs);

        /**
         *  @brief Return a shared pointer that manages the lifetime of the image's pixels.
         *
         *  The actual pointer will point to a parent image rather than the image itself
         *  if this is a subimage.
         */
        boost::shared_ptr<const T> getOwner() const { return _owner; }

        /// @brief Return a pointer to the first pixel in the image.
        const T * getData() const { return _data; }

        /// @brief Return the number of elements between rows in memory.
        int getStride() const { return _stride; }

        /**
         *  @brief Deep copy the image.
         *
         *  The returned image will have the same bounding box and pixel values as this,
         *  but will they will not share data.
         */
        Image<T> duplicate() const;

        /// @brief New image that is a subimage of this (shares pixels)
        Image subimage(const Bounds<int> & bounds) const;

        /**
         *  @brief Resize the image.
         *
         *  Images sharing pixel with this one will cease to share, but will otherwise be unaffected.
         *
         *  The image will be reallocated unless the new bounds are the same size as the current bounds,
         *  and will be left uninitialized.
         */
        void resize(const Bounds<int> & bounds);

        // Shift the bounding box of the image by the given offsets (just shifts the box).
        void shift(int x0, int y0) { _bounds.shift(x0, y0); }

#ifdef IMAGE_BOUNDS_CHECK
        // Element access is checked always
        const T& operator()(const int xpos, const int ypos) const 
        { return at(xpos,ypos); }
#else
        // Unchecked access
        const T& operator()(const int xpos, const int ypos) const 
        { return _data[addressPixel(xpos, ypos)]; }
#endif

        // Element access - checked
        const T& at(const int xpos, const int ypos) const {
            if (!_bounds.includes(xpos, ypos)) throw ImageBounds(xpos, ypos, _bounds);
            return _data[addressPixel(xpos, ypos)];
        }

        typedef const T* Iter;
        Iter rowBegin(int r) const { return _data + addressPixel(r); }
        Iter rowEnd(int r) const { return _data + addressPixel(getXMax() + 1, r); }
        Iter getIter(const int x, const int y) const { return _data + addressPixel(x, y); }

        // bounds access functions
        Bounds<int> const & getBounds() const { return _bounds; }
        int getXMin() const { return _bounds.getXMin(); }
        int getXMax() const { return _bounds.getXMax(); }
        int getYMin() const { return _bounds.getYMin(); }
        int getYMax() const { return _bounds.getYMax(); }

    };

    template <typename T>
    class Image : public Image<const T> {
    public:

        Image(const int ncol, const int nrow) : Image<const T>(ncol, nrow) {}

        explicit Image(const Bounds<int> & bounds=Bounds<int>()) : Image<const T>(bounds) {}

        explicit Image(const Bounds<int> & bounds, const T initValue) :
            Image<const T>(bounds, initValue)
        {}

        Image(const Image& rhs) : Image<const T>(rhs) {}

        Image & operator=(const Image& rhs) {
            Image<const T>::operator=(rhs);
            return *this;
        }

        /**
         *  @brief Return a shared pointer that manages the lifetime of the image's pixels.
         *
         *  The actual pointer will point to a parent image rather than the image itself
         *  if this is a subimage.
         */
        boost::shared_ptr<T> getOwner() const { return this->_owner; }

        /// @brief Return a pointer to the first pixel in the image.
        T * getData() const { return this->_data; }

        /// @brief New image that is a subimage of this (shares pixels)
        Image subimage(const Bounds<int> & bounds) const;

#ifdef IMAGE_BOUNDS_CHECK
        // Element access is checked always
        T& operator()(const int xpos, const int ypos) const 
        { return at(xpos,ypos); }
#else
        // Unchecked access
        T& operator()(const int xpos, const int ypos) const 
        { return this->_data[this->addressPixel(xpos, ypos)]; }
#endif

        // Element access - checked
        T& at(const int xpos, const int ypos) const {
            if (!this->_bounds.includes(xpos, ypos)) throw ImageBounds(xpos, ypos, this->_bounds);
            return this->_data[this->addressPixel(xpos, ypos)];
        }

        typedef T* Iter;
        Iter rowBegin(int r) const { return this->_data + this->addressPixel(r); }
        Iter rowEnd(int r) const {
            return this->_data + this->addressPixel(this->getXMax() + 1, r);
        }
        Iter getIter(const int x, const int y) const {
            return this->_data + this->addressPixel(x, y);
        }

        //@{
        /**
         *  @brief Assignment and augmented assignment with scalars.
         *
         *  We don't overload the actual assignment operator to fill with a scalar, because
         *  the assignment operator that takes an image resets the pointers rather than
         *  copying pixels, and it's nice to be consistent.
         *
         *  Also note that all these are const member functions, but they are only defined on
         *  image's with non-const pixel values, in keeping with the overall "smart-pointer-like"
         *  
         */
        void fill(T x) const;
        Image const & operator+=(T x) const;
        Image const & operator-=(T x) const;
        Image const & operator*=(T x) const;
        Image const & operator/=(T x) const;
        //@}

        /**
         *  @brief Deep-copy pixel values from rhs to this.
         *
         *  Only pixels in the intersection of the images' bounding boxes will be copied.
         */
        void copyFrom(const Image<const T> & rhs);

        //@{
        /**
         *  @brief Augmented assignment.
         *
         *  Only the region in the intersection of the two images will be affected.
         */
        Image const & operator+=(const Image<const T> & rhs) const;
        Image const & operator-=(const Image<const T> & rhs) const;
        Image const & operator*=(const Image<const T> & rhs) const;
        Image const & operator/=(const Image<const T> & rhs) const;
        //@}

        // Image/Image arithmetic binops: output image is intersection
        // of bounds of two input images.  Exception for null output.
        Image<T> operator+(const Image<const T> & rhs) const;
        Image<T> operator-(const Image<const T> & rhs) const;
        Image<T> operator*(const Image<const T> & rhs) const;
        Image<T> operator/(const Image<const T> & rhs) const;

    };

    template <typename T>
    inline Image<const T>::Image(Image<T> const & other) :
        _owner(other.getOwner()),
        _data(other.getData()),
        _stride(other.getStride()),
        _bounds(other.getBounds())
    {}

    template <typename T>
    inline Image<const T> & Image<const T>::operator=(Image<T> const & rhs) {
        if (&rhs != this) {
            _owner = rhs.getOwner();
            _data = rhs.getData();
            _stride = rhs.getStride();
            _bounds = rhs.getBounds();
        }
        return *this;
    }

    //////////////////////////////////////////////////////////////////////////
    // Templates for stepping through image pixels
    //////////////////////////////////////////////////////////////////////////

    /// @brief Call a unary function on each pixel value
    template <typename T, typename Op>
    Op for_each_pixel(const Image<T> & image, Op f) {
        for (int i = image.getYMin(); i <= image.getYMax(); i++)
            f = std::for_each(image.rowBegin(i), image.rowEnd(i), f);
        return f;
    }

    /// @brief Call a unary function on each pixel in a subset of the image.
    template <typename T, typename Op>
    Op for_each_pixel(const Image<T> & image, const Bounds<int> & bounds, Op f) {
        if (!image.getBounds().includes(bounds))
            throw ImageError("for_each_pixel range exceeds image range");
        
        for (int i = bounds.getYMin(); i <= bounds.getYMax(); i++)
            f = std::for_each(image.getIter(bounds.getXMin(),i),
                              image.getIter(bounds.getXMax()+1,i), f);
        return f;
    }

    /// @brief Replace image with a function of its pixel values.
    template <typename T, typename Op>
    Op transform_pixel(const Image<T> & image, Op f) {
        typedef typename Image<T>::Iter Iter;
        for (int y = image.getYMin(); y <= image.getYMax(); y++) {
            const Iter ee = image.rowEnd(y);
            for (Iter it = image.rowBegin(y); it != ee; ++it) 
                *it = f(*it);
        }
        return f;
    }

    /// @brief Replace a subset of the image with a function of its pixel values.
    template <typename T, typename Op>
    Op transform_pixel(const Image<T> & image, const Bounds<int> & bounds, Op f) {
        typedef typename Image<T>::Iter Iter;
        if (!image.getBounds().includes(bounds))
            throw ImageError("transform_pixel range exceeds image range");
        for (int y = bounds.getYMin(); y <= bounds.getYMax(); y++) {
            const Iter ee = image.getIter(bounds.getXMax()+1,y);      
            for (Iter it = image.getIter(bounds.getXMin(),y); it != ee; ++it) 
                *it = f(*it);
        }
        return f;
    }

    /// @brief Add a function of pixel coords to an image.
    template <typename T, typename Op>
    Op add_function_pixel(const Image<T> & image, Op f) {
        typedef typename Image<T>::Iter Iter;
        for (int y = image.getYMin(); y <= image.getYMax(); y++) {
            int x = image.getXMin();
            const Iter ee = image.rowEnd(y);
            for (Iter it = image.rowBegin(y); it != ee; ++it, ++x) 
                *it += f(x,y);
        }
        return f;
    }

    /// @brief Add a function of pixel coords to a subset of an image.
    template <typename T, typename Op>
    Op add_function_pixel(const Image<T> & image, const Bounds<int> & bounds, Op f) {
        typedef typename Image<T>::Iter Iter;
        if (!bounds.isDefined()) return f;
        if (!image.getBounds().includes(bounds))
            throw ImageError("add_function_pixel range exceeds image range");
        for (int y = bounds.getYMin(); y <= bounds.getYMax(); y++) {
            int x = bounds.getXMin();
            const Iter ee = image.getIter(bounds.getXMax()+1,y);      
            for (Iter it = image.getIter(bounds.getXMin(),y); it != ee; ++it, ++x) 
                *it += f(x,y);
        }
        return f;
    }

    /// @brief Replace image with a function of pixel coords.
    template <typename T, typename Op>
    Op fill_pixel(const Image<T> & image, Op f) {
        typedef typename Image<T>::Iter Iter;
        for (int y = image.getYMin(); y <= image.getYMax(); y++) {
            int x = image.getXMin();
            const Iter ee = image.rowEnd(y);      
            for (Iter it = image.rowBegin(y); it != ee; ++it, ++x) 
                *it = f(x,y);
        }
        return f;
    }

    /// @brief Replace subset of an image with a function of pixel coords.
    template <typename T, typename Op>
    Op fill_pixel(const Image<T> & image, const Bounds<int> & bounds, Op f) {
        typedef typename Image<T>::Iter Iter;
        if (!image.getBounds().includes(bounds))
            throw ImageError("add_function_pixel range exceeds image range");
        for (int y = bounds.getYMin(); y <= bounds.getYMax(); y++) {
            int x = bounds.getXMin();
            const Iter ee = image.getIter(bounds.getXMax()+1,y);      
            for (Iter it = image.getIter(bounds.getXMin(),y); it != ee; ++it, ++x) 
                *it = f(x,y);
        }
        return f;
    }

    // Assign function of 2 images to 1st
    template <typename T1, typename T2, typename Op>
    Op transform_pixel(const Image<T1> & image1, const Image<const T2> & image2, Op f) {
        typedef typename Image<T1>::Iter Iter1;
        typedef typename Image<const T2>::Iter Iter2;
        for (int y = image1.getYMin(); y <= image1.getYMax(); y++) {
            Iter2 it2 = image2.getIter(image1.getXMin(),y);
            const Iter1 ee = image1.rowEnd(y);      
            for (Iter1 it1 = image1.rowBegin(y); it1 != ee; ++it1, ++it2)
                *it1 = f(*it1,*it2);
        }
        return f;
    }

    // Assign function of Img2 & Img3 to Img1
    template <typename T1, typename T2, typename T3, typename Op>
    Op transform_pixel(
        const Image<T1> & image1,
        const Image<const T2> & image2, 
        const Image<const T3> & image3,
        Op f
    ) {
        typedef typename Image<T1>::Iter Iter1;
        typedef typename Image<const T2>::Iter Iter2;
        typedef typename Image<const T3>::Iter Iter3;
        for (int y = image1.getYMin(); y <= image1.getYMax(); y++) {
            Iter2 it2 = image2.getIter(image1.getXMin(),y);
            Iter3 it3 = image3.getIter(image1.getXMin(),y);
            const Iter1 ee = image1.rowEnd(y);      
            for (Iter1 it1 = image1.rowBegin(y); it1 != ee; ++it1, ++it2, ++it3) 
                *it1 = f(*it2,*it3);
        }
        return f;
    }

    // Assign function of 2 images to 1st over bounds
    template <typename T1, typename T2, typename Op>
    Op transform_pixel(
        const Image<T1> & image1,
        const Image<T2> & image2,
        const Bounds<int> & bounds,
        Op f
    ) {
        typedef typename Image<T1>::Iter Iter1;
        typedef typename Image<T2>::Iter Iter2;
        if (!image1.getBounds().includes(bounds) || !image2.getBounds().includes(bounds))
            throw ImageError("transform_pixel range exceeds image range");
        for (int y = bounds.getYMin(); y <= bounds.getYMax(); y++) {
            const Iter1 ee = image1.getIter(bounds.getXMax()+1,y);      
            Iter2 it2 = image2.getIter(bounds.getXMin(),y);
            for (Iter1 it1 = image1.getIter(bounds.getXMin(),y); it1 != ee; ++it1, ++it2) 
                *it1 = f(*it1,*it2);
        }
        return f;
    }



} // namespace galsim

#endif
