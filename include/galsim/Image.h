
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
//**********************  ImageHeader ****************************
// Contains auxiliary information for Images.  This includes:
// a list of COMMENT strings
// a list of HISTORY strings
// a list of other keyword-indexed records, a la FITS headers.
//
// ???? Add WCS information to Header structure ???
//
// Header records have a keyword string, a value, and optional comment 
// and units strings.  Individual records have a base class
//   HdrRecordBase
// and the derived classes are
//   HdrRecordNull (no value)
//   HdrRecord<T>  (value of type T).
// Header keywords are case-insensitive and at least for FITS are
// limited to 8 characters.
//
// Usually the client will not construct ImageHeaders, but always get
// them from Images.  The most common methods will be:
//   append("keyword",value,"comment")
//        ...to append a new keyword/value pair to the header.
//   replace("keyword",value,"comment")
//        ...replaces old keyword header, or appends if no old one
//   getValue("keyword", value)
//        ...returns true & fills the value if keyword is in header
//           returns false if keyword is not already in header.
//   addComment("comment") 
//        ...appends a new COMMENT record.
//   addHistory("history")  is a new HISTORY entry.
//
// Less frequently the client will use a list-oriented access to all
// the header records.
// There is an internal pointer to the header record list.
// It is manipulated by rewind(), atEnd(), incr().  Pointer to the
// "current" record is current().  find() moves the pointer to next
// record that matches a keyword.
// append() or insert() add new records at end or at current pointer.
// You can call either of these with a keyword,value pair, and the
// correct type for HdrRecord<T> will be inferred from the value.
// erase() gets rid of either a certain keyword, or the current record.
//
// clear() flushes all records, plus the HISTORY and COMMENT lists.
// size() is total number of records, HISTORY, and COMMENT entries.
//
// A copy or assignment of an ImageHeader is a deep copy (as long as
// all the header types T are).  ImageHeader owns all the HdrRecords
// and will delete them upon deletion of hte ImageHeader.
//
// Each ImageHeader keeps an "isAltered"
// flag so one can note whether it is unchanged since its creation or
// last call to notAltered().
//
// You can't erase individual comments or history entries.
//****************************************************************/

#ifndef Image_H
#define Image_H

#include <algorithm>
#include <functional>
#include <list>
#include <sstream>
#include <typeinfo>
#include <string>

#include <boost/shared_array.hpp>

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

    //////////////////////////////////////////////////////////////////////////
    // Templates for stepping through image pixels
    //////////////////////////////////////////////////////////////////////////
    // Execute function on each pixel value
    template <class Img, class Op>
    Op for_each_pixel(Img I, Op f) 
    {
        for (int i=I.YMin(); i<=I.YMax(); i++)
            f=for_each(I.rowBegin(i), I.rowEnd(i), f);
        return f;
    }

    // Execute function on a range of pixels
    template <class Img, class Op>
    Op for_each_pixel(Img I, Bounds<int> b, Op& f) 
    {
        if (!I.getBounds().includes(b))
            throw ImageError("for_each_pixel range exceeds image range");
        for (int i=b.getYMin(); i<=b.getYMax(); i++)
            f=for_each(I.getIter(b.getXMin(),i), 
                       I.getIter(b.getXMax()+1,i), f);
        return f;
    }

    // Replace image with function of itself
    template <class Img, class Op>
    Op transform_pixel(Img I, Op f) 
    {
        for (int y=I.YMin(); y<=I.YMax(); y++) {
            typename Img::iter ee=I.rowEnd(y);      
            for (typename Img::iter it=I.rowBegin(y);
                 it!=ee; 
                 ++it) 
                *it=f(*it);
        }
        return f;
    }

    // Replace image with function of itself over range
    template <class Img, class Op>
    Op transform_pixel(Img I, Bounds<int> b, Op f) 
    {
        if (!I.getBounds().includes(b))
            throw ImageError("transform_pixel range exceeds image range");
        for (int y=b.getYMin(); y<=b.getYMax(); y++) {
            typename Img::iter ee=I.getIter(b.getXMax()+1,y);      
            for (typename Img::iter it=I.getIter(b.getXMin(),y);
                 it!=ee; 
                 ++it) 
                *it=f(*it);
        }
        return f;
    }

    // Add function of pixel coords to image
    template <class Img, class Op>
    Op add_function_pixel(Img I, Op f) 
    {
        for (int y=I.YMin(); y<=I.YMax(); y++) {
            int x=I.XMin();
            typename Img::iter ee=I.rowEnd(y);      
            for (typename Img::iter it=I.rowBegin(y);
                 it!=ee; 
                 ++it, ++x) 
                *it+=f(x,y);
        }
        return f;
    }

    // Add function of pixel coords to image over a range
    template <class Img, class Op>
    Op add_function_pixel(Img I, Bounds<int> b, Op f) 
    {
        if (b && !I.getBounds().includes(b))
            throw ImageError("add_function_pixel range exceeds image range");
        for (int y=b.getYMin(); y<=b.getYMax(); y++) {
            int x=b.getXMin();
            typename Img::iter ee=I.getIter(b.getXMax()+1,y);      
            for (typename Img::iter it=I.getIter(b.getXMin(),y);
                 it!=ee; 
                 ++it, ++x) 
                *it+=f(x,y);
        }
        return f;
    }

    // Replace image with function of pixel coords
    template <class Img, class Op>
    Op fill_pixel(Img I, Op f) 
    {
        for (int y=I.YMin(); y<=I.YMax(); y++) {
            int x=I.XMin();
            typename Img::iter ee=I.rowEnd(y);      
            for (typename Img::iter it=I.rowBegin(y);
                 it!=ee; 
                 ++it, ++x) 
                *it=f(x,y);
        }
        return f;
    }

    // Replace image with function of pixel coords, over specified bounds
    template <class Img, class Op>
    Op fill_pixel(Img I, Bounds<int> b, Op f) 
    {
        if (!I.getBounds().includes(b))
            throw ImageError("add_function_pixel range exceeds image range");
        for (int y=b.getYMin(); y<=b.getYMax(); y++) {
            int x=b.getXMin();
            typename Img::iter ee=I.getIter(b.getXMax()+1,y);      
            for (typename Img::iter it=I.getIter(b.getXMin(),y);
                 it!=ee; 
                 ++it, ++x) 
                *it=f(x,y);
        }
        return f;
    }

    // Assign function of 2 images to 1st
    template <class Img1, class Img2, class Op>
    Op transform_pixel(Img1 I1, const Img2 I2, Op f) 
    {
        for (int y=I1.YMin(); y<=I1.YMax(); y++) {
            typename Img2::citer it2=I2.getIter(I1.XMin(),y);
            typename Img1::iter ee=I1.rowEnd(y);      
            for (typename Img1::iter it1=I1.rowBegin(y); 
                 it1!=ee; ++it1, ++it2) *it1=f(*it1,*it2);
        }
        return f;
    }

    // Assign function of Img2 & Img3 to Img1
    template <class Img1, class Img2, class Img3, class Op>
    Op transform_pixel(Img1 I1, const Img2 I2, const Img3 I3, Op f) 
    {
        for (int y=I1.YMin(); y<=I1.YMax(); y++) {
            typename Img2::citer it2=I2.getIter(I1.XMin(),y);
            typename Img3::citer it3=I3.getIter(I1.XMin(),y);
            typename Img1::iter ee=I1.rowEnd(y);      
            for (typename Img1::iter it1=I1.rowBegin(y);
                 it1!=ee; 
                 ++it1, ++it2, ++it3) 
                *it1=f(*it2,*it3);
        }
        return f;
    }

    // Assign function of 2 images to 1st over bounds
    template <class Img1, class Img2, class Op>
    Op transform_pixel(Img1 I1, const Img2 I2, Op f, Bounds<int> b) 
    {
        if (!I1.getBounds().includes(b) || !I2.getBounds().includes(b))
            throw ImageError("transform_pixel range exceeds image range");
        for (int y=b.getYMin(); y<=b.getYMax(); y++) {
            typename Img1::iter ee=I1.getIter(b.getXMax()+1,y);      
            typename Img2::citer it2=I2.getIter(b.getXMin(),y);
            for (typename Img1::iter it1=I1.getIter(b.getXMin(),y);
                 it1!=ee; 
                 ++it1, ++it2) 
                *it1=f(*it1,*it2);
        }
        return f;
    }


    ////////////////////////////////////////////////////////////////
    // The pixel data structure - never used by outside programs
    ////////////////////////////////////////////////////////////////
    template <class T>
    class ImageData 
    {
    public:
        // Create:
        // image with unspecified data values:
        ImageData(const Bounds<int> inBounds) ;

        // image filled with a scalar:
        ImageData(const Bounds<int> inBounds, const T initValue) ;

        // image for which the data array has been set up by someone else:
        ImageData(const Bounds<int> inBounds, T** rptrs, bool _contig=false);

        ~ImageData();

        // This routine used by some other object rearranging the storage
        // array.  If ownRowPointers was set, then this routine deletes the
        // old array and assumes responsibility for deleting the new one.
        void replaceRowPointers(T** newRptrs) const 
        {
            if (ownRowPointers) delete[] (rowPointers + bounds.getYMin()); 
            rowPointers=newRptrs;
        }

        // Make a new ImageData that is subimage of this one.  Data will be
        // shared in memory, just pixel bounds are different.  subimages
        // a.k.a. children must be destroyed before the parent.
        ImageData* subimage(const Bounds<int> bsub);
        const ImageData* const_subimage(const Bounds<int> bsub) const;  //return const
        const ImageData* subimage(const Bounds<int> bsub) const;  //if this const
        void deleteSubimages() const; // delete all living subimages

        // Create a new subimage that is duplicate of this ones data
        ImageData* duplicate() const;

        // Resize the image - this will throw exception if data array
        // is not owned or if there are subimages in use.
        // Data are undefined afterwards
        void resize(const Bounds<int> newBounds);

        // Copy in the data (and size) from another image.
        void copyFrom(const ImageData<T>& rhs);

        // move origin of coordinates, preserving data.  Throws exception
        // if data are not owned or if there are subimages in use.
        void shift(int x0, int y0);

        // Element access - unchecked
        const T& operator()(const int xpos, const int ypos) const 
        { return *location(xpos,ypos); }
        T& operator()(const int xpos, const int ypos) 
        { return *location(xpos,ypos); }

        // Element access - checked
        const T& at(const int xpos, const int ypos) const 
        {
            if (!bounds.includes(xpos,ypos)) throw ImageBounds(xpos,ypos,bounds);
            return *location(xpos,ypos);
        }

        T& at(const int xpos, const int ypos) 
        {
            if (!bounds.includes(xpos,ypos)) throw ImageBounds(xpos,ypos,bounds);
            return *location(xpos,ypos);
        }

        // give pointer to a pixel in the storage array, 
        // for use by routines that buffer image data for us.
        T* location(const int xpos, const int ypos) const 
        { return *(rowPointers+ypos)+xpos; } 

        // Access functions
        Bounds<int> getBounds() const { return bounds; }
        bool contiguousData() const { return isContiguous; }

    private:
        // image which will be a subimage of a parent:
        ImageData(const Bounds<int> inBounds, const ImageData<T>* _parent);

        // No inadvertent copying allowed! Use copyFrom() to be explicit.
        ImageData(const ImageData& );
        ImageData& operator=(const ImageData&);

        Bounds<int> bounds;

        const ImageData<T>* parent; // If this is a subimage, what's parent?

        // Does this object own (i.e. have responsibility for destroying):
        bool ownDataArray; // the actual data array
        bool ownRowPointers; // the rowpointer array

        mutable T** rowPointers; // Pointers to start of the data rows

        mutable bool  isContiguous; // Set if entire image is contiguous in memory

        //list of subimages of this (sub)image:
        mutable std::list<const ImageData<T>*> children; 

        // class utility functions:
        void acquireArrays(Bounds<int> inBounds);
        void discardArrays();
        void unlinkChild(const ImageData<T>* child) const;
        void linkChild(const ImageData<T>* child) const;
    };


    //////////////////////////////////////////////////////////////////////////
    // The Image handle:  this is what outside programs use.
    //////////////////////////////////////////////////////////////////////////

    template <class T>
    class Image 
    {
    private:
        T * _data;
        Position<int> _bounds;
        boost::shared_array<T> _owner;
    public:
        Image(const int ncol, const int nrow) :
            D(new ImageData<T>(Bounds<int>(1,ncol,1,nrow))), H(new ImageHeader()),
            dcount(new int(1)), hcount(new int(1)) 
        {}

        // Default constructor builds a null image:
        explicit Image(const Bounds<int> inBounds=Bounds<int>()) : 
            D(new ImageData<T>(inBounds)), H(new ImageHeader()),
            dcount(new int(1)), hcount(new int(1)) 
        {}

        explicit Image(const Bounds<int> inBounds, const T initValue) : 
            D(new ImageData<T>(inBounds, initValue)), H(new ImageHeader()),
            dcount(new int(1)), hcount(new int(1)) 
        {}

        Image(const Image& rhs) : 
            D(rhs.D), H(rhs.H), dcount(rhs.dcount), hcount(rhs.hcount) 
        { (*dcount)++; (*hcount)++; }

        Image& operator=(const Image& rhs) 
        {
            // Note no assignment of const image to non-const image. ???
            if (&rhs == this) return *this;
            if (D!=rhs.D) {
                if (--(*dcount)==0) { delete D; delete dcount; }
                D = rhs.D; dcount=rhs.dcount; (*dcount)++;
            }
            if (H!=rhs.H) {
                if (--(*hcount)==0) { delete H; delete hcount; }
                H = rhs.H; hcount=rhs.hcount; (*hcount)++;
            }
            return *this;
        }

        ~Image() 
        {
            if (--(*dcount)==0) { delete D; delete dcount; }
            if (--(*hcount)==0) { delete H; delete hcount; }
        }

        // Constructor for use by other image-manipulation routines:
        // Create from a data and a header object: note that both will be
        // deleted when this object is deleted unless [dh]count are given.  
        Image(ImageData<T>* Din, ImageHeader* Hin, int* _dc=0, int* _hc=0) : 
            D(Din), H(Hin), dcount(_dc), hcount(_hc) 
        {
            if (!_dc) dcount = new int(0);
            if (!_hc) hcount = new int(0);
            (*dcount)++; (*hcount)++; 
        }

        // Make this image (or just data) be a duplicate of another's.
        // Note this can change size, which is illegal if there exist
        // open subimages.  All Images that refer to same data are changed.
        void copyDataFrom(const Image& rhs) { D->copyFrom(*(rhs.D)); }

        void copyFrom(const Image& rhs) 
        {
            *H = *(rhs.H);
            D->copyFrom(*(rhs.D));
        }

        // Create new image with fresh copies of data & header
        Image duplicate() const;

        // New image that is subimage of this (shares pixels & header data)
        Image subimage(const Bounds<int> bsub);
        const Image subimage(const Bounds<int> bsub) const ;

        // Resize the image - will throw if data aren't owned or if subimages
        // exist.  Note all Images sharing this ImageData will be affected.
        // Data are destroyed in the process
        void resize(const Bounds<int> newBounds) { D->resize(newBounds); }

        // Shift origin of image - same caveats apply as above
        void shift(int x0, int y0) { D->shift(x0,y0); }

#ifdef IMAGE_BOUNDS_CHECK
        // Element access is checked always
        const T& operator()(const int xpos, const int ypos) const 
        { return at(xpos,ypos); }

        T& operator()(const int xpos, const int ypos) 
        { return at(xpos,ypos); }
#else
        // Unchecked access
        const T& operator()(const int xpos, const int ypos) const 
        { return (*D)(xpos,ypos); }

        T& operator()(const int xpos, const int ypos) 
        { return (*D)(xpos,ypos); }
#endif

        // Element access - checked
        const T& at(const int xpos, const int ypos) const 
        { return D->at(xpos,ypos); }

        T& at(const int xpos, const int ypos) 
        { return D->at(xpos,ypos); }

        typedef T* iter;
        typedef const T* citer;
        iter rowBegin(int r) { return &(*D)(XMin(),r); }
        citer rowBegin(int r) const { return &(*D)(XMin(),r); }
        iter rowEnd(int r) { return &(*D)(XMax()+1,r); }
        citer rowEnd(int r) const { return &(*D)(XMax()+1,r); }
        iter getIter(const int x, const int y) { return &(*D)(x,y); }
        citer getIter(const int x, const int y) const { return &(*D)(x,y); }

        // bounds access functions
        Bounds<int> getBounds() const { return D->getBounds(); }
        int XMin() const { return D->getBounds().getXMin(); }
        int XMax() const { return D->getBounds().getXMax(); }
        int YMin() const { return D->getBounds().getYMin(); }
        int YMax() const { return D->getBounds().getYMax(); }

        // Image/scalar arithmetic operations
        void  operator+=(T x) { transform_pixel(*this, bind2nd(std::plus<T>(),x)); }
        void  operator-=(T x) { transform_pixel(*this, bind2nd(std::minus<T>(),x)); }
        void  operator*=(T x) { transform_pixel(*this, bind2nd(std::multiplies<T>(),x)); }
        void  operator/=(T x) { transform_pixel(*this, bind2nd(std::divides<T>(),x)); }
        void  operator-() { transform_pixel(*this, std::negate<T>()); }

        class ConstReturn 
        {
        public: 
            ConstReturn(const T v): val(v) {}
            T operator()(const T dummy) const { return val; }
        private:
            T val;
        };

        void  operator=(const T val) { transform_pixel(*this, ConstReturn(val)); }

        // Image/Image arithmetic ops: rhs must include this bounds
        void  operator+=(const Image<T> rhs);
        void  operator-=(const Image<T> rhs);
        void  operator*=(const Image<T> rhs);
        void  operator/=(const Image<T> rhs);

        // Image/Image arithmetic binops: output image is intersection
        // of bounds of two input images.  Exception for null output.
        Image<T> operator+(const Image<T> rhs) const;
        Image<T> operator-(const Image<T> rhs) const;
        Image<T> operator*(const Image<T> rhs) const;
        Image<T> operator/(const Image<T> rhs) const;

    };

}

#endif
