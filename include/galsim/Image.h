/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef GalSim_Image_H
#define GalSim_Image_H

#include <algorithm>
#include <functional>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <complex>

// Need this for instantiated types, since we will use int16_t and int32_t
// rather than short and int to explicitly match the python levels numpy.int16 and numpy.int32.
// Note: <cstdint> only really became standard for gcc >= 4.4, so can't use that.
// Hopefully all our compilers will conform to the C99 standard which includes stdint.h.
#include <stdint.h>

#include "Std.h"
#include "Bounds.h"

template <typename T>
struct Traits
{
    enum { isreal = true };
    enum { iscomplex = false };
    typedef T real_type;
    typedef std::complex<T> complex_type;
};

template <typename T>
struct Traits<std::complex<T> >
{
    enum { isreal = false };
    enum { iscomplex = true };
    typedef T real_type;
    typedef std::complex<T> complex_type;
};

namespace galsim {

    template <typename T>
    std::shared_ptr<T> allocateAlignedMemory(int n);

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /**
     *  @brief Exception class usually thrown by images.
     */
    class PUBLIC_API ImageError : public std::runtime_error {
    public:
        ImageError(const std::string& m) : std::runtime_error("Image Error: " + m) {}

    };

    /**
     *  @brief Exception class thrown when out-of-bounds pixels are accessed on an image.
     */
    class PUBLIC_API ImageBoundsError : public ImageError {
    public:
        ImageBoundsError(const std::string& m) :
            ImageError("Access to out-of-bounds pixel " + m) {}

        ImageBoundsError(const std::string& m, int min, int max, int tried);

        ImageBoundsError(int x, int y, const Bounds<int> b);
    };

    //! @endcond


    template <typename T> class PUBLIC_API AssignableToImage;
    template <typename T> class PUBLIC_API BaseImage;
    template <typename T> class PUBLIC_API ImageAlloc;
    template <typename T> class PUBLIC_API ImageView;
    template <typename T> class PUBLIC_API ConstImageView;

    template <typename T1>
    class ReturnSecond
    {
    public:
        T1 operator()(T1, T1 v) const { return v; }
    };

    /**
     *  @brief AssignableToImage is a base class for anything that can be assigned to
     *         an Image.
     *
     *  It is the base class for both BaseImage, which has all the real image types
     *  and various composite types that are used to make im3 = im1 + im2 efficient.
     */
    template <typename T>
    class PUBLIC_API AssignableToImage
    {
    public:

        /**
         *  @brief Destructor is public and virtual
         */
        virtual ~AssignableToImage() {}

        /**
         *  @brief Assign values to an ImageView
         */
        virtual void assignTo(ImageView<T> rhs) const = 0;

        /**
         *  @brief Return the bounding box of the image.
         */
        const Bounds<int>& getBounds() const { return _bounds; }

    protected:

        Bounds<int> _bounds;          // bounding box

        /**
         *  @brief Constructor is protected since this is an abstract base class.
         */
        AssignableToImage(const Bounds<int>& b) : _bounds(b) {}

        /**
         *  @brief Copy constructor also protected.
         */
        AssignableToImage(const AssignableToImage<T>& rhs) : _bounds(rhs._bounds) {}

    private:
        /**
         *  @brief op= is invalid.  So private and undefined.
         */
        void operator=(const AssignableToImage<T>&);

    };

    /**
     *  @brief BaseImage class defines the interface of what you can do with Images
     *         without modifying pixels
     *
     *  BaseImage is the base class for the other Image types:
     *  - ImageAlloc
     *  - ImageView
     *  - ConstImageView
     *
     *  You should never need to declare an object of this class directly, using
     *  instead the various derived classes.  However, if you are using an image
     *  as a parameter to a function in a non-modifying context, then it is
     *  convenient to declare the parameter "const BaseImage<T>&" so that any
     *  of the various image types can be used without requiring any casts.
     */
    template <typename T>
    class PUBLIC_API BaseImage : public AssignableToImage<T>
    {
    public:

        /**
         *  @brief Destructor is virtual and public.
         *
         *  Note: There are no public constructors, since this is an abstract
         *  base class.  But the destructor should be public (and virtual).
         *
         *  Nothing special needs to be done here, since shared_ptr takes care
         *  of deleting the data if this is the last thing to own the data.
         */
        virtual ~BaseImage() {}

        /**
         *  @brief Return a shared pointer that manages the lifetime of the image's pixels.
         *
         *  The actual pointer will point to a parent image rather than the image itself
         *  if this is a subimage.
         */
        shared_ptr<T> getOwner() const { return _owner; }

        /**
         *  @brief Return a pointer to the first pixel in the image.
         */
        const T* getData() const { return _data; }

        /**
         *  @brief Return how many data elements are currently allocated in memory.
         *
         *  This is usually the same as getBounds().area(), but it may not be if the image
         *  has been resized.
         */
        ptrdiff_t getNElements() const { return _nElements; }

        /**
         *  @brief Return the number of elements between rows in memory.
         */
        int getStride() const { return _stride; }

        /**
         *  @brief Return the number of elements between cols in memory.
         */
        int getStep() const { return _step; }

        /**
         *  @brief Return the number of columns in the image
         */
        int getNCol() const { return _ncol; }

        /**
         *  @brief Return the number of rows in the image
         */
        int getNRow() const { return _nrow; }

        /**
         *  @brief Return the number of columns to skip at the end of each row when iterating.
         */
        int getNSkip() const { return _stride - _ncol*_step; }

        /**
         *  @brief Return whether the data is contiguous in memory.
         *
         *  Shorthand for:
         *  (getStride() == getNCol()) or equivalently (getNSkip() == 0)
         */
        bool isContiguous() const
        { return (_step == 1 && _stride == _ncol); }

        /**
         *  @brief Deep copy the image.
         *
         *  The returned image will have the same bounding box and pixel values as this,
         *  but will they will not share data.
         */
        ImageAlloc<T> copy() const
        { return ImageAlloc<T>(*this); }

        /**
         *  @brief Create a new view of the image.
         *
         *  The returned image will share data with this image
         */
        ConstImageView<T> view() const { return ConstImageView<T>(*this); }

        /**
         *  @brief New image that is a subimage of this (shares pixels)
         */
        ConstImageView<T> subImage(const Bounds<int>& bounds) const;

        /**
         *  @brief im[bounds] is another syntax for making a sub-image
         */
        ConstImageView<T> operator[](const Bounds<int>& bounds) const
        { return subImage(bounds); }

        /**
         *  @brief Return the smallest bounds that includes all non-zero elements of the image.
         */
        Bounds<int> nonZeroBounds() const;

        /**
         *  @brief Shift the bounding box of the image, changing the logical location of the pixels
         *
         *  xmin_new = xmin + dx
         *  xmax_new = xmax + dx
         *  ymin_new = ymin + dy
         *  ymax_new = ymax + dy
         */
        void shift(const Position<int>& delta) { this->_bounds.shift(delta); }

        /**
         *  @brief Return the bounding box of the image.
         */
        // (Repeat this here for the sake of the boost python wrapping, so we don't have to
        // wrap AssignableToImage.)
        const Bounds<int>& getBounds() const { return this->_bounds; }

        //@{
        /**
         *  @brief Convenience accessors for the bounding box corners.
         */
        int getXMin() const { return this->_bounds.getXMin(); }
        int getXMax() const { return this->_bounds.getXMax(); }
        int getYMin() const { return this->_bounds.getYMin(); }
        int getYMax() const { return this->_bounds.getYMax(); }
        //@}

        //@{
        /**
         *  @brief Unchecked element access
         */
        const T& operator()(int xpos, int ypos) const { return _data[addressPixel(xpos, ypos)]; }
        const T& operator()(const Position<int>& pos) const { return operator()(pos.x,pos.y); }
        //@}

        //@{
        /**
         *  @brief Element access - checked
         */
        const T& at(int xpos, int ypos) const;
        const T& at(const Position<int>& pos) const { return at(pos.x,pos.y); }
        //@}

        //@{
        /**
         *  @brief Return a pointer to the data at an arbitrary pixel.
         */
        const T* getPtr(int x, int y) const { return _data + addressPixel(x, y); }
        const T* getPtr(const Position<int>& pos) const { return getPtr(pos.x,pos.y); }
        //@}

        /**
         *  @brief BaseImage's assignTo just uses the normal copyFrom method.
         */
        void assignTo(ImageView<T> rhs) const { rhs.copyFrom(*this); }

        /**
         *  @brief Return the sum of the elements in the image.
         */
        T sumElements() const;

        /**
         *  @brief Return the maximum absolute value in the image.
         */
        typename Traits<T>::real_type maxAbsElement() const;

    protected:

        shared_ptr<T> _owner;  // manages ownership; _owner.get() != _data if subimage
        T* _data;                     // pointer to be used for this image
        ptrdiff_t _nElements;         // number of elements allocated in memory
        int _step;                    // number of elements between cols (normally 1)
        int _stride;                  // number of elements between rows (!= width for subimages)
        int _ncol;                    // number of columns
        int _nrow;                    // number of rows

        inline int addressPixel(int y) const
        { return (y - this->getYMin()) * _stride; }

        inline int addressPixel(int x, int y) const
        { return (x - this->getXMin()) * _step + addressPixel(y); }

        /**
         *  @brief Constructor is protected since a BaseImage is a virtual base class.
         */
        BaseImage(T* data, ptrdiff_t nElements, shared_ptr<T> owner,
                  int step, int stride, const Bounds<int>& b) :
            AssignableToImage<T>(b),
            _owner(owner), _data(data), _nElements(nElements),
            _step(step), _stride(stride),
            _ncol(b.getXMax()-b.getXMin()+1), _nrow(b.getYMax()-b.getYMin()+1)
        { if (_nElements == 0) _nElements = _ncol * _nrow; }

        /**
         *  @brief Copy constructor also protected
         *
         *  This does the trivial copy of the values.  Valid for ImageView
         *  and ConstImageView, but not ImageAlloc.
         */
        BaseImage(const BaseImage<T>& rhs) :
            AssignableToImage<T>(rhs),
            _owner(rhs._owner), _data(rhs._data), _nElements(rhs._nElements),
            _step(rhs._step), _stride(rhs._stride), _ncol(rhs._ncol), _nrow(rhs._nrow)
        {}

        /**
         *  @brief Also have a constructor that just takes a bounds.
         *
         *  This constructor allocates new memory for the data array for these bounds.
         *  This is only used by the ImageAlloc<T> derived class, but it turns out to be
         *  convenient to have the functionality here instead of in ImageAlloc.
         *
         *  If the bounds are not defined, then the _data pointer is 0.
         *  Most often, this is used for default-constructing an ImageAlloc which is then
         *  resized later.
         */
        BaseImage(const Bounds<int>& b);

        /**
         *  @brief Allocate new memory for the image
         *
         *  This is used to implement both the above constructor and ImageAlloc<T>'s
         *  resize function.
         */
        void allocateMem();

    private:
        /**
         *  @brief op= is invalid.  So private and undefined.
         */
        void operator=(const BaseImage<T>&);

    };

    /**
     *  @brief ConstImageView class views an Image in a read-only context
     *
     *  Read-only only refers to the data values.  The bounds may be changed.
     */
    template <typename T>
    class PUBLIC_API ConstImageView : public BaseImage<T>
    {
    public:

        /**
         *  @brief Direct constructor given all the necessary information
         */
        ConstImageView(T* data, const shared_ptr<T>& owner, int step, int stride,
                       const Bounds<int>& b) :
            BaseImage<T>(data,0,owner,step,stride,b) {}

        /**
         *  @brief Copy Constructor from a BaseImage makes a new view of the same data
         */
        ConstImageView(const BaseImage<T>& rhs) : BaseImage<T>(rhs) {}

        /**
         *  @brief Repeat the same copy constructor functionality for ConstImageView
         *         to make sure the compiler doesn't make the default one.
         */
        ConstImageView(const ConstImageView<T>& rhs) : BaseImage<T>(rhs) {}

        /**
         *  @brief View just returns itself.
         */
        ConstImageView<T> view() const { return ConstImageView<T>(*this); }

    private:
        /**
         *  @brief op= is invalid so private and undefined.
         */
        void operator=(const ConstImageView<T>& rhs);
    };

    /**
     *  @brief ImageView class is a mutable view of an Image
     *
     *  The copy constructor is shallow, so ImageView's can be cheaply returned by value.
     *  The data values persist until the last view of some data goes out of scope.
     *
     *  The op= is deep, though.  This is the intuitive behavior, so you can write
     *  something like
     *
     *  im1[bounds] = im2
     *
     *  and the data in im2 will be copied to the sub-image of im1.
     *
     *  Also note that through the python interface, we can make an ImageView that
     *  views a numpy array rather than anything that was created as an ImageAlloc.
     *  We have some tricky stuff in pysrc/Image.cpp to get the C++ shared_ptr to
     *  interact correctly with numpy's reference counting so the data are deleted
     *  when the last numpy array _or_ ImageView finally goes out of scope.
     *
     *  You could do the same thing within the C++ layer too.  You would just have
     *  to provide a shared_ptr explicitly to set up the ownership.
     */
    template <typename T>
    class PUBLIC_API ImageView : public BaseImage<T>
    {
    public:

        /**
         *  @brief Direct constructor given all the necessary information
         */
        ImageView(T* data, const shared_ptr<T>& owner, int step, int stride,
                  const Bounds<int>& b, int nElements=0) :
            BaseImage<T>(data, nElements, owner, step, stride, b) {}

        /**
         *  @brief Shallow copy constructor.
         *
         *  The original image and its copy will share pixel values, but their bounding
         *  boxes will not be shared (even though they will be set to the same values initially).
         */
        ImageView(const ImageView<T>& rhs) : BaseImage<T>(rhs) {}

        /**
         *  @brief Shallow copy constructor from ImageAlloc.
         *
         *  The original image and its copy will share pixel values, but their bounding
         *  boxes will not be shared (even though they will be set to the same values initially).
         */
        ImageView(ImageAlloc<T>& rhs) : BaseImage<T>(rhs) {}

        /**
         *  @brief Deep assignment operator.
         *
         *  The bounds must be commensurate (i.e. the same shape).
         *  If not, an exception will be thrown.
         */
        ImageView<T>& operator=(const AssignableToImage<T>& rhs)
        { if (this != &rhs) rhs.assignTo(*this); return *this; }

        /**
         *  @brief Repeat for ImageView to prevent compiler from making the default op=
         */
        ImageView<T>& operator=(const ImageView<T>& rhs)
        { if (this != &rhs) copyFrom(rhs); return *this; }

        /**
         *  @brief Allow copy from a different type
         */
        template <typename U>
        ImageView<T>& operator=(const BaseImage<U>& rhs)
        { if (this != &rhs) copyFrom(rhs); return *this; }

        //@{
        /**
         *  @brief Assignment with a scalar.
         */
        void fill(T x);
        ImageView<T>& operator=(T x) { fill(x); return *this; }
        void setZero() { fill(T(0)); }
        //@}

        /**
         * @brief Set each element to its inverse: im(i,j) = 1/im(i,j)
         *
         * Note that if an element is zero, then this function quietly returns its inverse as zero.
         */
        void invertSelf();

        /**
         *  @brief Return a pointer to the first pixel in the image.
         *
         *  This overrides the version in BaseImage, since this one returns a non-const
         *  pointer.  (T*, not const T*)
         */
        T* getData() { return this->_data; }

        /**
         *  @brief View just returns itself.
         */
        ImageView<T> view() { return ImageView<T>(*this); }

        /**
         *  @brief New image that is a subimage of this (shares pixels)
         */
        ImageView<T> subImage(const Bounds<int>& bounds);

        /**
         *  @brief im[bounds] is another syntax for making a sub-image
         */
        ImageView<T> operator[](const Bounds<int>& bounds)
        { return subImage(bounds); }

        //@{
        /**
         *  @brief Unchecked access
         */
        T& operator()(int xpos, int ypos)
        { return this->_data[this->addressPixel(xpos, ypos)]; }
        T& operator()(const Position<int>& pos) { return operator()(pos.x,pos.y); }
        //@}

        //@{
        /**
         *  @brief Element access - checked
         */
        T& at(int xpos, int ypos);
        T& at(const Position<int>& pos) { return at(pos.x,pos.y); }
        //@}

        /**
         *  @brief Another way to set a value.  Equivalent to im(x,y) = value.
         *
         *  The python layer can't implement the im(x,y) = value syntax, so
         *  we need something else to set a single pixel.
         *  This function is unnecessary at the C++ level, but in the interest of
         *  trying to keep the two layers as close as possible, we might as well include it.
         *
         *  Note: This uses the checked element access.
         */
        void setValue(int x, int y, T value)
        { at(x,y) = value; }

        /**
         *  @brief Deep-copy pixel values from rhs to this.
         *
         *  The bounds must be commensurate (i.e. the same shape).
         *  If not, an exception will be thrown.
         */
        void copyFrom(const BaseImage<T>& rhs);

        /**
         *  @brief Deep copy may be from a different type of image.
         *
         *  Do this inline, so we don't have to worry about instantiating all pairs of types.
         */
        template <class U>
        void copyFrom(const BaseImage<U>& rhs)
        {
            if (!this->_bounds.isSameShapeAs(rhs.getBounds()))
                throw ImageError("Attempt im1 = im2, but bounds not the same shape");
            transform_pixel(*this, rhs, ReturnSecond<T>());
        }
    };

    /**
     *  @brief ImageAlloc class
     *
     *  The ImageAlloc class is a 2-d array with pixel values stored contiguously in memory along
     *  rows (but not necessarily between rows).  An image's pixel values may be shared between
     *  multiple image objects (with reference counting), and a subimage may share data with
     *  its parent and multiple siblings.  ImageAllocs may also share pixel values with NumPy
     *  arrays when the allocation happens in the C++ layer.
     *
     *  An ImageAlloc also contains a bounding box; its origin need not be (0,0) or (1,1).
     *
     *  The const semantics for this are pretty normal.  You cannot change either the
     *  pixel values or the ancillary information (like bounds) for a const ImageAlloc,
     *  while you can change things about a non-const ImageAlloc.
     *
     *  ImageAlloc templates for uint16_t, uint32_t, int16_t, int32_t, float, and double are
     *  explicitly instantiated in Image.cpp.
     */
    template <typename T>
    class PUBLIC_API ImageAlloc : public BaseImage<T>
    {
    public:

        /**
         * @brief Default constructor leaves the image's data pointer as null.
         */
        ImageAlloc() : BaseImage<T>(Bounds<int>()) {}

        /**
         *  @brief Create a new image with origin at (1,1).
         *
         *  An exception is thrown if ncol or nrow <= 0
         */
        ImageAlloc(int ncol, int nrow);

        /**
         *  @brief Create a new image with origin at (1,1), intialized with some init_value
         *
         *  An exception is thrown if ncol or nrow <= 0
         */
        ImageAlloc(int ncol, int nrow, T init_value);

        /**
         *  @brief Create a new image with the given bounding box
         */
        ImageAlloc(const Bounds<int>& bounds);

        /**
         *  @brief Create a new image with the given bounding box and initial value.
         */
        ImageAlloc(const Bounds<int>& bounds, T init_value);

        /**
         *  @brief Deep copy constructor.
         */
        ImageAlloc(const ImageAlloc<T>& rhs) : BaseImage<T>(rhs._bounds)
        { copyFrom(rhs); }

        /**
         *  @brief Can construct from any AssignableToImage
         */
        ImageAlloc(const AssignableToImage<T>& rhs) : BaseImage<T>(rhs.getBounds())
        { rhs.assignTo(view()); }

        /**
         *  @brief If rhs is a BaseImage, then type doesn't have to match.
         */
        template <typename U>
        ImageAlloc(const BaseImage<U>& rhs) : BaseImage<T>(rhs.getBounds())
        { copyFrom(rhs); }

        /**
         *  @brief Deep assignment operator.
         *
         *  The bounds must be commensurate (i.e. the same shape).
         *  If not, an exception will be thrown.
         */
        ImageAlloc<T>& operator=(const AssignableToImage<T>& rhs)
        { if (this != &rhs) rhs.assignTo(view()); return *this; }

        /**
         *  @brief Repeat for ImageAlloc to prevent compiler from making the default op=
         */
        ImageAlloc<T>& operator=(const ImageAlloc<T>& rhs)
        { if (this != &rhs) copyFrom(rhs); return *this; }

        /**
         *  @brief Copy from BaseImage allowed for different types.
         */
        template <typename U>
        ImageAlloc<T>& operator=(const BaseImage<U>& rhs)
        { copyFrom(rhs); return *this; }

        //@{
        /**
         *  @brief Assignment with a scalar.
         */
        void fill(T x) { view().fill(x); }
        ImageAlloc<T>& operator=(T x) { fill(x); return *this; }
        void setZero() { fill(T(0)); }
        //@}

        /**
         * @brief Set each element to its inverse: im(i,j) = 1/im(i,j)
         *
         * Note that if an element is zero, then this function quietly returns its inverse as zero.
         */
        void invertSelf() { view().invertSelf(); }

        /**
         *  @brief Resize the image to a new bounds.  The values are left uninitialized.
         *
         *  Any views that share data with this ImageAlloc are still valid and still
         *  share data with each other, but the tie to this ImageAlloc is severed.
         *
         *  This typically allocates new memory for the array.  The only
         *  exception is if the new size is _smaller_ than currently and there
         *  are no other views of the data.
         */
        void resize(const Bounds<int>& new_bounds);

        //@{
        /**
         *  @brief Return a pointer to the first pixel in the image.
         */
        T* getData() { return this->_data; }
        const T* getData() const { return this->_data; }
        //@}

        //@{
        /**
         *  @brief Make a view of this image
         */
        ImageView<T> view()
        {
            return ImageView<T>(this->_data, this->_owner, this->_step, this->_stride,
                                this->_bounds, this->_nElements);
        }
        ConstImageView<T> view() const { return ConstImageView<T>(*this); }
        //@}

        //@{
        /**
         *  @brief New image that is a subimage of this (shares pixels)
         */
        ImageView<T> subImage(const Bounds<int>& bounds)
        { return view().subImage(bounds); }
        ConstImageView<T> subImage(const Bounds<int>& bounds) const
        { return view().subImage(bounds); }
        //@}

        //@{
        /**
         *  @brief im[bounds] is another syntax for making a sub-image
         */
        ImageView<T> operator[](const Bounds<int>& bounds)
        { return subImage(bounds); }
        ConstImageView<T> operator[](const Bounds<int>& bounds) const
        { return subImage(bounds); }
        //@}

        //@{
        /**
         *  @brief Unchecked access
         */
        T& operator()(int xpos, int ypos)
        { return this->_data[this->addressPixel(xpos, ypos)]; }
        const T& operator()(int xpos, int ypos) const
        { return this->_data[this->addressPixel(xpos, ypos)]; }
        T& operator()(const Position<int>& pos) { return operator()(pos.x,pos.y); }
        const T& operator()(const Position<int>& pos) const { return operator()(pos.x,pos.y); }
        //@}

        //@{
        /**
         *  @brief Element access - checked
         */
        T& at(int xpos, int ypos);
        const T& at(int xpos, int ypos) const;
        T& at(const Position<int>& pos) { return at(pos.x,pos.y); }
        const T& at(const Position<int>& pos) const { return at(pos.x,pos.y); }
        //@}

        /**
         *  @brief Another way to set a value.  Equivalent to im(x,y) = value.
         *
         *  The python layer can't implement the im(x,y) = value syntax, so
         *  we need something else to set a single pixel.
         *  This function is unnecessary at the C++ level, but in the interest of
         *  trying to keep the two layers as close as possible, we might as well include it.
         *
         *  Note: This uses the checked element access.
         */
        void setValue(int x, int y, T value)
        { at(x,y) = value; }

        /**
         *  @brief Deep-copy pixel values from rhs to this.
         *
         *  The bounds must be commensurate (i.e. the same shape).
         *  If not, an exception will be thrown.
         */
        template <typename U>
        void copyFrom(const BaseImage<U>& rhs) { view().copyFrom(rhs); }
    };

    /**
     * @brief A helper function that will return the smallest 2^n or 3x2^n value that is
     * even and >= the input integer.
     */
    PUBLIC_API int goodFFTSize(int input);


    /**
     *  @brief Perform a 2D FFT from real space to k-space.
     */
    template <typename T>
    PUBLIC_API void rfft(
        const BaseImage<T>& in, ImageView<std::complex<double> > out,
        bool shift_in=true, bool shift_out=true);

    /**
     *  @brief Perform a 2D inverse FFT from k-space to real space.
     */
    template <typename T>
    PUBLIC_API void irfft(
        const BaseImage<T>& in, ImageView<double> out,
        bool shift_in=true, bool shift_out=true);

    /**
     *  @brief Perform a 2D FFT from complex space to k-space or the inverse.
     */
    template <typename T>
    PUBLIC_API void cfft(
        const BaseImage<T>& in, ImageView<std::complex<double> > out,
        bool inverse, bool shift_in=true, bool shift_out=true);

    /**
     *  @brief Wrap the full image onto a subset of the image and return that subset.
     *
     *  This is used to alias the data of a k-space image before doing the FFT to real space.
     */
    template <typename T>
    PUBLIC_API void wrapImage(
        ImageView<T> im, const Bounds<int>& bounds, bool hermx, bool hermy);

    /**
     *  @brief Set each element to its inverse: im(i,j) = 1/im(i,j)
     *
     *  Note that if an element is zero, then this function quietly returns its inverse as zero.
     */
    template <typename T>
    PUBLIC_API void invertImage(ImageView<T> im);




} // namespace galsim

#include "ImageArith.h"

#endif
