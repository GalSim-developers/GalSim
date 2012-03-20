// -*- c++ -*-
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

    /// @brief Exception class usually thrown by images.
    class ImageError : public std::runtime_error {
    public: 
        ImageError(const std::string& m="") : 
            std::runtime_error("Image Error: " + m) {}

    };

    /// @brief Exception class thrown when out-of-bounds pixels are accessed on an image.
    class ImageBounds : public ImageError {
    public: 
        ImageBounds(const std::string& m="") : 
            ImageError("Access to out-of-bounds pixel " + m) {}

        ImageBounds(const std::string& m, const int min, const int max, const int tried);

        ImageBounds(const int x, const int y, const Bounds<int> b);
    };

    template <typename T> class Image;

    /**
     *  @brief Image class with const pixels.
     *
     *  @copydetails Image
     */
    template <typename T>
    class Image<const T> {
    protected:
        
        boost::shared_ptr<T> _owner;  // manages ownership; _owner.get() != _data if subimage
        T * _data;                    // pointer to be used for this image
        int _stride;                  // number of elements between rows (!= width for subimages)
        double _scale;                // pixel scale (used by SBPixel and SBProfile; units?!)
        Bounds<int> _bounds;          // bounding box

        inline int addressPixel(const int y) const {
            return (y - getYMin()) * _stride;
        }
        
        inline int addressPixel(const int x, const int y) const {
            return (x - getXMin()) + addressPixel(y);
        }

    public:

        /// @brief Create a new image with origin at (1,1).
        Image(const int ncol, const int nrow);

        /**
         *  @brief Create a new image with the given bounding box and initial value.
         *
         *  If !bounds.isDefined(), the image's data pointer will be null.
         */
        explicit Image(const Bounds<int> & bounds=Bounds<int>(), const T initValue=T(0));

        /**
         *  @brief Construct from external data.
         *
         *  This is mostly intended for use by the Python interface.
         */
        Image(
            const T * data,
            boost::shared_ptr<const T> const & owner, 
            int stride,
            const Bounds<int> & bounds
        );

        /// @brief Return the pixel scale.
        double getScale() const { return _scale; }

        /// @brief Set the pixel scale.
        void setScale(double scale) { _scale = scale; }

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
         *  Images sharing pixel with this one will cease to share, but will otherwise be
         *  unaffected.
         *
         *  The image will be reallocated unless the new bounds are the same size as the current
         *  bounds, and will be left uninitialized.
         */
        void resize(const Bounds<int> & bounds);

        /**
         *  @brief Redefine the image's bounds within a larger image.
         *
         *  This is a more dangerous, in-place version of subimage that also allows you to
         *  increase the bounds (or otherwise move them outside the current bounds to
         *  somewhere else within a parent image).
         *
         *  WARNING: this will modify the image's data pointer to point to a new location,
         *  meaning that it must already be a subimage unless the new bounds are contained
         *  by the current bounds.  But the image has no way of checking whether the new
         *  origin is valid, or even whether it is a subimage at all; use with caution!
         */
        void redefine(const Bounds<int> & bounds);

        /**
         *  @brief Shift the bounding box of the image, changing the logical location of the pixels.
         *
         *  This does not affect subimages.
         */
        void shift(int dx, int dy) { _bounds.shift(dx, dy); }

        /**
         *  @brief Move the origin of the image, changing the logical location of the pixels.
         *
         *  This does not affect subimages.
         */
        void move(int x0, int y0) { shift(x0 - getXMin(), y0 - getYMin()); }

#ifdef IMAGE_BOUNDS_CHECK
        /// Element access is checked always
        const T& operator()(const int xpos, const int ypos) const 
        { return at(xpos,ypos); }
#else
        /// Unchecked element access
        const T& operator()(const int xpos, const int ypos) const 
        { return _data[addressPixel(xpos, ypos)]; }
#endif

        /// Element access - checked
        const T& at(const int xpos, const int ypos) const {
            if (!_bounds.includes(xpos, ypos)) throw ImageBounds(xpos, ypos, _bounds);
            return _data[addressPixel(xpos, ypos)];
        }

        /// @brief Iterator type for pixels within a row (unchecked).
        typedef const T* Iter;

        /// @brief Return an iterator to the beginning of a row.
        Iter rowBegin(int y) const { return _data + addressPixel(y); }

        /// @brief Return an iterator to one-past-the-end of a row.
        Iter rowEnd(int y) const { return _data + addressPixel(getXMax() + 1, y); }

        /// @brief Return an iterator to an arbitrary pixel.
        Iter getIter(const int x, const int y) const { return _data + addressPixel(x, y); }

        /// @brief Return the bounding box of the image.
        Bounds<int> const & getBounds() const { return _bounds; }

        //@{
        /// Convenience accessors for the bounding box corners.
        int getXMin() const { return _bounds.getXMin(); }
        int getXMax() const { return _bounds.getXMax(); }
        int getYMin() const { return _bounds.getYMin(); }
        int getYMax() const { return _bounds.getYMax(); }
        //@}

        //@{
        /**
         *  @brief Binary arithmetic operators.
         *
         *  The output image is the intersection of the bounding boxes of the two images;
         *  returns a null image if there is no intersection.
         */
        Image<T> operator+(const Image<const T> & rhs) const;
        Image<T> operator-(const Image<const T> & rhs) const;
        Image<T> operator*(const Image<const T> & rhs) const;
        Image<T> operator/(const Image<const T> & rhs) const;
        //@}
        
    };

    /**
     *  @brief Image class (non-const).
     *
     *  The Image class is a 2-d array with pixel values stored contiguously in memory along
     *  rows (but not necessarily between rows).  An image's pixel values may be shared between
     *  multiple image objects (with reference counting), and a subimage may share data with
     *  its parent and multiple siblings.  Images may also share pixel values with NumPy arrays.
     *
     *  An Image also contains a bounding box; its origin need not be (0,0) or (1,1).  It also
     *  contains a single floating-point pixel scale, though this may be intepreted differently
     *  in different contexts.
     *
     *  The fact that images share memory makes their constness semantics a lot more complicated.
     *  The copy constructor and the assignment operator (these are implicitly defined by the
     *  compiler) for images are both "shallow" - they simply create new references to the
     *  same underlying memory, or change what memory an image object points at.  This means
     *  the usual constness semantics don't work: if we have a "const Image &", we could
     *  trivially copy it to a non-const Image object that shares data with it, so there's
     *  no point to preventing pixel modifications on const Images.
     *
     *  Instead, the Image<const T> template is specialized, and should be used to represent
     *  Images whose pixels cannot be modified.  Furthermore, Image<T> inherits from
     *  Image<const T>, so you can use an Image<T> anywhere an Image<const T> is expected.
     *  This is similar to how constness semantics work with pointers and smart pointers,
     *  and once you get used to it it makes a lot of sense.
     *
     *  However, it does have some surprising implications:
     *   - Member functions that modify the pixels are marked as const (but are not defined for
     *     Image<const T>).
     *   - Because the main assignment operator is shallow, there is no assignment operator that
     *     accepts a scalar (because it would have to be deep).  Instead, there's a "fill" member
     *     function that sets the entire image to scalar.
     *   - Similarly, there's "copyFrom" member function that does deep assignment of an image.
     *     You can also use "duplicate" to return a new image that is a deep copy.
     *   - The augmented assignment operators ARE deep, and hence behave very differently from
     *     the regular assignment operators.  And, because they just modify pixel values, they're
     *     const member functions and retern const references to this/
     *
     *  Note that the bounding box and the pixel scale are not shared between images, and these
     *  have the regular constness semantics: member functions that modify them are not const.
     *
     *  Image templates for short, int, float, and double are explicitly instantiated in Image.cpp.
     */
    template <typename T>
    class Image : public Image<const T> { 
    // NOTE: Doxygen is confused by this inheritance and warns that it's recursive.
    // It's not recursive; Doxygen just isn't parsing partial specialization.  We
    // could hide the inheritance from Doxygen completely, but that would reduce
    // the quality of the outputs.  I think it's best just to live with the warning.
    public:

        /// @brief Create a new image with origin at (1,1).
        Image(const int ncol, const int nrow) : Image<const T>(ncol, nrow) {}

        /**
         *  @brief Create a new image with the given bounding box and initial value.
         *
         *  If !bounds.isDefined(), the image's data pointer will be null.
         */
        explicit Image(const Bounds<int> & bounds=Bounds<int>(), const T initValue=T(0)) :
            Image<const T>(bounds, initValue)
        {}

        /**
         *  @brief Construct from external data.
         *
         *  This is mostly intended for use by the Python interface.
         */
        Image(T * data, boost::shared_ptr<T> const & owner, int stride, const Bounds<int> & bounds)
            : Image<const T>(data, owner, stride, bounds)
        {}

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
        /// Element access is checked always
        T& operator()(const int xpos, const int ypos) const 
        { return at(xpos,ypos); }
#else
        /// Unchecked access
        T& operator()(const int xpos, const int ypos) const 
        { return this->_data[this->addressPixel(xpos, ypos)]; }
#endif

        /// Element access - checked
        T& at(const int xpos, const int ypos) const {
            if (!this->_bounds.includes(xpos, ypos)) throw ImageBounds(xpos, ypos, this->_bounds);
            return this->_data[this->addressPixel(xpos, ypos)];
        }

        /// @brief Iterator type for pixels within a row (unchecked).
        typedef T* Iter;

        /// @brief Return an iterator to the beginning of a row.
        Iter rowBegin(int r) const { return this->_data + this->addressPixel(r); }

        /// @brief Return an iterator to one-past-the-end of a row.
        Iter rowEnd(int r) const {
            return this->_data + this->addressPixel(this->getXMax() + 1, r);
        }

        /// @brief Return an iterator to an arbitrary pixel.
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
         *  constness semantics.
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
         *  Only pixels in the intersection of the images' bounding boxes will be copied;
         *  silent no-op if there is no intersection.
         */
        void copyFrom(const Image<const T> & rhs);

        //@{
        /**
         *  @brief Augmented assignment.
         *
         *  Only pixels in the intersection of the images' bounding boxes will be affected;
         *  silent no-op if there is no intersection.
         */
        Image const & operator+=(const Image<const T> & rhs) const;
        Image const & operator-=(const Image<const T> & rhs) const;
        Image const & operator*=(const Image<const T> & rhs) const;
        Image const & operator/=(const Image<const T> & rhs) const;
        //@}
    };

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
