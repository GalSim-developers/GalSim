// -*- c++ -*-
#ifndef Image_H
#define Image_H

#include <algorithm>
#include <functional>
#include <typeinfo>
#include <stdexcept>
#include <string>

// Need this for instantiated types, since we will use int16_t and int32_t
// rather than short and int to explicitly match the python levels numpy.int16 and numpy.int32.
// Note: <cstdint> only really became standard for gcc >= 4.4, so can't use that.
// Hopefully all our compilers will conform to the C99 standard which includes stdint.h.
#include <stdint.h>

#include <boost/shared_ptr.hpp>

#include "Std.h"
#include "Bounds.h"
#include "ImageArith.h"

namespace galsim {

    template <typename T> class AssignableToImage;
    template <typename T> class BaseImage;
    template <typename T> class Image;
    template <typename T> class ImageView;
    template <typename T> class ConstImageView;

    /**
     *  @brief AssignableToImage is a base class for anything that can be assigned to
     *         an Image.
     *
     *  It is the base class for both BaseImage, which has all the real image types
     *  and various composite types that are used to make im3 = im1 + im2 efficient.
     */
    template <typename T>
    class AssignableToImage
    {
    public:

        /**
         *  @brief Destructor is public and virtual
         */
        virtual ~AssignableToImage() {}

        /**
         *  @brief Assign values to an ImageView
         */
        virtual void assignTo(const ImageView<T>& rhs) const = 0;

        /**
         *  @brief Return the bounding box of the image.
         */
        const Bounds<int>& getBounds() const { return this->_bounds; }

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
     *  - Image
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
    class BaseImage : public AssignableToImage<T> 
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
         *  @brief Return the pixel scale.
         */
        double getScale() const { return _scale; }

        /**
         *  @brief Return a shared pointer that manages the lifetime of the image's pixels.
         *
         *  The actual pointer will point to a parent image rather than the image itself
         *  if this is a subimage.
         */
        boost::shared_ptr<T> getOwner() const { return _owner; }

        /**
         *  @brief Return a pointer to the first pixel in the image.
         */
        const T* getData() const { return _data; }

        /**
         *  @brief Return the number of elements between rows in memory.
         */
        int getStride() const { return _stride; }

        /**
         *  @brief Return whether the data is contiguous in memory.
         *
         *  Shorthand for:
         *  (getStride() == getBounds().getXMax() - getBounds().getXMin() + 1)
         */
        bool isContiguous() const
        { return (getStride() == this->getBounds().getXMax() - this->getBounds().getXMin() + 1); }

        /**
         *  @brief Deep copy the image.
         *
         *  The returned image will have the same bounding box and pixel values as this,
         *  but will they will not share data.
         */
        Image<T> copy() const
        { return Image<T>(*this); }

        /**
         *  @brief Create a new view of the image.
         *
         *  The returned image will share data with this image
         */
        ConstImageView<T> view() const
        { return ConstImageView<T>(*this); }

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
         *  @brief Shift the bounding box of the image, changing the logical location of the pixels.
         *
         *  xMin_new = xMin + dx
         *  xMax_new = xMax + dx
         *  yMin_new = yMin + dy
         *  yMax_new = yMax + dy
         */
        void shift(int dx, int dy) { this->_bounds.shift(dx, dy); }

        /**
         *  @brief Move the origin of the image, changing the logical location of the pixels.
         *
         *  (x0,y0) becomes the new lower-left corner of the image.
         *
         *  xMin_new = x0
         *  xMax_new = x0 + xMax - xMin
         *  yMin_new = y0
         *  yMax_new = y0 + yMax - yMin
         */
        void setOrigin(int x0, int y0) { shift(x0 - this->getXMin(), y0 - this->getYMin()); }

        /**
         *  @brief Set the pixel scale 
         */
        void setScale(double scale) { _scale = scale; }

        /**
         *  @brief Move the center of the image, changing the logical location of the pixels.
         *
         *  (x0,y0) becomes the new center of the image if the x and y ranges are odd.
         *  If the x range is even, then the new center will be x0 + 1/2.
         *  Likewisw for y.
         *
         *  xMin_new = x0 - (xMax - xMin)/2
         *  xMax_new = xMin_new + xMax - xMin
         *  yMin_new = y0 - (yMax - yMin)/2
         *  yMax_new = yMin_new + yMax - yMin
         */
        void setCenter(int x0, int y0) 
        { 
            shift(x0 - (this->getXMax()+this->getXMin())/2 ,
                  y0 - (this->getYMax()+this->getYMin())/2 ); 
        }

        /**
         *  @brief Return the bounding box of the image.
         */
        const Bounds<int>& getBounds() const { return AssignableToImage<T>::getBounds(); }

        //@{
        /**
         *  @brief Convenience accessors for the bounding box corners.
         */
        int getXMin() const { return getBounds().getXMin(); }
        int getXMax() const { return getBounds().getXMax(); }
        int getYMin() const { return getBounds().getYMin(); }
        int getYMax() const { return getBounds().getYMax(); }
        //@}
        


#ifdef IMAGE_BOUNDS_CHECK
        /**
         *  @brief Element access is checked always
         */
        const T& operator()(int xpos, int ypos) const 
        { return at(xpos,ypos); }
#else
        /**
         *  @brief Unchecked element access
         */
        const T& operator()(int xpos, int ypos) const 
        { return _data[addressPixel(xpos, ypos)]; }
#endif

        /**
         *  @brief Element access - checked
         */
        const T& at(int xpos, int ypos) const;

        /**
         *  @brief const_iterator type for pixels within a row (unchecked).
         */
        typedef const T* const_iterator;

        /** 
         *  @brief Return an iterator to the beginning of a row.
         */
        const_iterator rowBegin(int y) const 
        { return _data + addressPixel(y); }

        /**
         *  @brief Return an iterator to one-past-the-end of a row.
         */
        const_iterator rowEnd(int y) const 
        { return _data + addressPixel(this->getXMax() + 1, y); }

        /**
         *  @brief Return an iterator to an arbitrary pixel.
         */
        const_iterator getIter(int x, int y) const 
        { return _data + addressPixel(x, y); }

        /**
         *  @brief BaseImage's assignTo just uses the normal copyFrom method.
         */
        void assignTo(const ImageView<T>& rhs) const
        { rhs.copyFrom(*this); }

    protected:

        boost::shared_ptr<T> _owner;  // manages ownership; _owner.get() != _data if subimage
        T * _data;                    // pointer to be used for this image
        int _stride;                  // number of elements between rows (!= width for subimages)
        double _scale;                // pixel scale (used by SBInterpolatedImage and SBProfile;
                                      // units?!)

        inline int addressPixel(int y) const
        { return (y - this->getYMin()) * _stride; }
        
        inline int addressPixel(int x, int y) const
        { return (x - this->getXMin()) + addressPixel(y); }

        /**
         *  @brief Constructor is protected since a BaseImage is a virtual base class.
         */
        BaseImage(T* data, boost::shared_ptr<T> owner, int stride, const Bounds<int>& b, 
                  double scale=1.) :
            AssignableToImage<T>(b), _owner(owner), _data(data), _stride(stride), _scale(scale) {}

        /**
         *  @brief Copy constructor also protected
         *
         *  This does the trivial copy of the values.  Valid for ImageView
         *  and ConstImageView, but not Image.
         */
        BaseImage(const BaseImage<T>& rhs) :
            AssignableToImage<T>(rhs),
            _owner(rhs._owner), _data(rhs._data), _stride(rhs._stride), _scale(rhs._scale) {}

        /**
         *  @brief Also have a constructor that just takes a bounds.  
         *
         *  This constructor allocates new memory for the data array for these bounds.
         *  This is only used by the Image<T> derived class, but it turns out to be 
         *  convenient to have the functionality here instead of in Image.
         *
         *  If the bounds are not defined, then the _data pointer is 0.
         *  Most often, this is used for default-constructing an Image which is then
         *  resized later.
         */
        BaseImage(const Bounds<int>& b, double scale=1.);

        /**
         *  @brief Allocate new memory for the image
         *
         *  This is used to implement both the above constructor and Image<T>'s 
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
    class ConstImageView : public BaseImage<T> 
    {
    public:

        /**
         *  @brief Direct constructor given all the necessary information
         */
        ConstImageView(T* data, const boost::shared_ptr<T>& owner, int stride,
                       const Bounds<int>& b, double scale=1.) :
            BaseImage<T>(data,owner,stride,b,scale) {}

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
        const ConstImageView<T>& view() const { return *this; }
 
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
     *  views a numpy array rather than anything that was creates as an Image.
     *  We have some tricky stuff in pysrc/Image.cpp to get the C++ shared_ptr to
     *  interact correctly with numpy's reference counting so the data are deleted
     *  when the last numpy array _or_ ImageView finally goes out of scope.
     *
     *  You could do the same thing within the C++ layer too.  You would just have
     *  to provide a shared_ptr explicitly to set up the ownership.
     *
     *  Note that the const-ness here is slightly odd.  const for ImageView refers
     *  to whether the data it points to is const.  Also, the bounds.
     *  So most things that modify data are labeled const.  If you want something
     *  that is not allowed to modify data, you want to make a ConstImageView.
     *  (Or you can just cast this as a BaseImage<T>, which will also work.)
     */
    template <typename T>
    class ImageView : public BaseImage<T> 
    {
    public:

        /**
         *  @brief Direct constructor given all the necessary information
         */
        ImageView(T* data, const boost::shared_ptr<T>& owner, int stride, const Bounds<int>& b,
                  double scale=1.) :
            BaseImage<T>(data, owner, stride, b, scale) {}

        /**
         *  @brief Shallow copy constructor.
         *
         *  The original image and its copy will share pixel values, but their bounding
         *  boxes and scales will not be shared (even though they will be set to the same
         *  values initially).
         */
        ImageView(const ImageView<T>& rhs) : BaseImage<T>(rhs) {}

        /**
         *  @brief Shallow copy constructor from Image.
         *
         *  The original image and its copy will share pixel values, but their bounding
         *  boxes and scales will not be shared (even though they will be set to the same
         *  values initially).
         */
        ImageView(Image<T>& rhs) : BaseImage<T>(rhs) {}

        /**
         *  @brief Deep assignment operator.
         *
         *  The bounds must be commensurate (i.e. the same shape).
         *  If not, an exception will be thrown.
         */
        const ImageView<T>& operator=(const AssignableToImage<T>& rhs) const 
        { if (this != &rhs) rhs.assignTo(*this); return *this; }

        /**
         *  @brief Repeat for ImageView to prevent compiler from making the default op=
         */
        const ImageView<T>& operator=(const ImageView<T>& rhs) 
        { if (this != &rhs) copyFrom(rhs); return *this; }

        //@{
        /**
         *  @brief Assignment with a scalar.
         */
        void fill(T x) const;
        const ImageView<T>& operator=(T x) const { fill(x); return *this; }
        //@}

        /**
         *  @brief Return a pointer to the first pixel in the image.
         *
         *  This overrides the version in BaseImage, since this one returns a non-const
         *  pointer.  (T*, not const T*)
         */
        T* getData() const { return this->_data; }

        /**
         *  @brief View just returns itself.
         */
        const ImageView<T>& view() const
        { return *this; }
 
        /**
         *  @brief New image that is a subimage of this (shares pixels)
         */
        ImageView<T> subImage(const Bounds<int>& bounds) const;

        /**
         *  @brief im[bounds] is another syntax for making a sub-image
         */
        ImageView<T> operator[](const Bounds<int>& bounds) const
        { return subImage(bounds); }


#ifdef IMAGE_BOUNDS_CHECK
        /** 
         *  @brief Element access is checked always
         */
        T& operator()(int xpos, int ypos) const 
        { return at(xpos,ypos); }
#else
        /**
         *  @brief Unchecked access
         */
        T& operator()(int xpos, int ypos) const 
        { return this->_data[this->addressPixel(xpos, ypos)]; }
#endif

        /**
         *  @brief Element access - checked
         */
        T& at(int xpos, int ypos) const;

        /**
         *  @brief Another way to set a value.  Equivalent to im(x,y) = value.
         *
         *  The python layer can't implement the im(x,y) = value syntax, so 
         *  we need something else to set a single pixel.  
         *  This function is unnecessary at the C++ level, but in the interest of 
         *  trying to keep the two layers as close as possible, we might as well include it.
         */
        void setValue(int x, int y, T value)
        { (*this)(x,y) = value; }

        /**
         *  @brief iterator type for pixels within a row (unchecked).
         */
        typedef T* iterator;

        /**
         *  @brief Return an iterator to the beginning of a row.
         */
        iterator rowBegin(int r) const { return this->_data + this->addressPixel(r); }

        /**
         *  @brief Return an iterator to one-past-the-end of a row.
         */
        iterator rowEnd(int r) const 
        { return this->_data + this->addressPixel(this->getXMax() + 1, r); }

        /**
         *  @brief Return an iterator to an arbitrary pixel.
         */
        iterator getIter(int x, int y) const 
        { return this->_data + this->addressPixel(x, y); }

        /**
         *  @brief Deep-copy pixel values from rhs to this.
         *
         *  The bounds must be commensurate (i.e. the same shape).
         *  If not, an exception will be thrown.
         */
        void copyFrom(const BaseImage<T>& rhs) const;
    };

    /**
     *  @brief Image class
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
     *  The const semantics for this are pretty normal.  You cannot change either the 
     *  pixel values or the ancillary information (like bounds) for a const Image,
     *  while you can change things about a non-const Image.
     *
     *  Image templates for int16_t, int32_t, float, and double are explicitly instantiated 
     *  in Image.cpp.
     */
    template <typename T>
    class Image : public BaseImage<T> 
    {
    public:

        /**
         *  @brief Create a new image with origin at (1,1).
         *
         *  An exception is thrown if ncol or nrow <= 0
         */
        Image(int ncol, int nrow, T init_value = T(0));

        /**
         *  @brief Create a new image with the given bounding box and initial value.
         *
         *  If !bounds.isDefined(), the image's data pointer will be null.
         *  Note: This is also effectively the default constructor Image().
         */
        explicit Image(const Bounds<int>& bounds = Bounds<int>(),
                       T init_value = T(0));

        /**
         *  @brief Deep copy constructor.
         */
        Image(const Image<T>& rhs) : BaseImage<T>(rhs._bounds, rhs._scale) 
        { copyFrom(rhs); }

        /**
         *  @brief Can construct from any AssignableToImage
         */
        Image(const AssignableToImage<T>& rhs) : BaseImage<T>(rhs.getBounds()) 
        { rhs.assignTo(view()); }

        /**
         *  @brief If rhs is a BaseImage, then also get the scale
         */
        Image(const BaseImage<T>& rhs) : BaseImage<T>(rhs.getBounds(), rhs.getScale())
        { copyFrom(rhs); }

        /**
         *  @brief Deep assignment operator.
         *
         *  The bounds must be commensurate (i.e. the same shape).
         *  If not, an exception will be thrown.
         */
        Image<T>& operator=(const AssignableToImage<T>& rhs)
        { if (this != &rhs) rhs.assignTo(view()); return *this; }

        /**
         *  @brief Repeat for Image to prevent compiler from making the default op=
         */
        Image<T>& operator=(const Image<T>& rhs)
        { if (this != &rhs) view().copyFrom(rhs); return *this; }

        //@{
        /**
         *  @brief Assignment with a scalar.
         */
        void fill(T x) { view().fill(x); }
        Image<T>& operator=(T x) { fill(x); return *this; }
        //@}

        /**
         *  @brief Resize the image to a new bounds.  The values are left uninitialized.
         *
         *  Any views that share data with this Image are still valid and still
         *  share data with each other, but the tie to this Image is severed.
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
            return ImageView<T>(this->_data, this->_owner, this->_stride,
                                this->_bounds, this->_scale); 
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

#ifdef IMAGE_BOUNDS_CHECK
        //@{
        /**
         *  @brief Element access is checked always
         */
        T& operator()(int xpos, int ypos)
        { return at(xpos,ypos); }
        const T& operator()(int xpos, int ypos) const 
        { return at(xpos,ypos); }
        //@}
#else
        //@{
        /**
         *  @brief Unchecked access
         */
        T& operator()(int xpos, int ypos)
        { return this->_data[this->addressPixel(xpos, ypos)]; }
        const T& operator()(int xpos, int ypos) const 
        { return this->_data[this->addressPixel(xpos, ypos)]; }
        //@}
#endif

        //@{
        /**
         *  @brief Element access - checked
         */
        T& at(int xpos, int ypos);
        const T& at(int xpos, int ypos) const;
        //@}

        /**
         *  @brief Another way to set a value.  Equivalent to im(x,y) = value.
         *
         *  The python layer can't implement the im(x,y) = value syntax, so 
         *  we need something else to set a single pixel.  
         *  This function is unnecessary at the C++ level, but in the interest of 
         *  trying to keep the two layers as close as possible, we might as well include it.
         */
        void setValue(int x, int y, T value)
        { (*this)(x,y) = value; }

        //@{
        /**
         *  @brief Iterator type for pixels within a row (unchecked).
         */
        typedef T* iterator;
        typedef const T* const_iterator;
        //@}

        //@{
        /**
         *  @brief Return an iterator to the beginning of a row.
         */
        iterator rowBegin(int r)
        { return this->_data + this->addressPixel(r); }
        const_iterator rowBegin(int r) const 
        { return this->_data + this->addressPixel(r); }
        //@}

        //@{
        /**
         *  @brief Return an iterator to one-past-the-end of a row.
         */
        iterator rowEnd(int r)
        { return this->_data + this->addressPixel(this->getXMax() + 1, r); }
        const_iterator rowEnd(int r) const 
        { return this->_data + this->addressPixel(this->getXMax() + 1, r); }
        //@}

        //@{
        /**
         *  @brief Return an iterator to an arbitrary pixel.
         */
        iterator getIter(int x, int y)
        { return this->_data + this->addressPixel(x, y); }
        const_iterator getIter(int x, int y) const 
        { return this->_data + this->addressPixel(x, y); }
        //@}

        /**
         *  @brief Deep-copy pixel values from rhs to this.
         *
         *  The bounds must be commensurate (i.e. the same shape).
         *  If not, an exception will be thrown.
         */
        void copyFrom(const BaseImage<T>& rhs) { view().copyFrom(rhs); }
    };

} // namespace galsim

#endif
