
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

#include "Std.h"
#include "Bounds.h"
#include "FITStypes.h"

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

    class ImageHeaderError : public std::runtime_error
    {
    public:
        ImageHeaderError(const std::string m="") : 
            std::runtime_error("Image Header Error: " + m) {}
    };


    //////////////////////////////////////////////////////////////////////////
    // Auxiliary information held for all images
    //////////////////////////////////////////////////////////////////////////

    // First the classes that are individual ImageHeader records:
    std::string KeyFormat(const std::string input);

    // Base class for all header entries
    class HdrRecordBase 
    {
    public:
        HdrRecordBase(const std::string _kw, const std::string _com="", const std::string _un="") : 
            keyword(KeyFormat(_kw)), comment(_com), units(_un) {}

        virtual ~HdrRecordBase() {}
        virtual HdrRecordBase* duplicate() const 
        { return new HdrRecordBase(*this); }

        bool matchesKey(const std::string _k) const 
        { return keyword==KeyFormat(_k); }

        std::string getComment() const { return comment; }
        void setComment(const std::string _c) { comment=_c; }

        std::string getUnits() const { return units; }
        void setUnits(const std::string _u) { units=_u; }

        std::string getKeyword() const { return keyword; }
        void setKeyword(const std::string _k) { keyword=KeyFormat(_k); }

        virtual void reset(
            const std::string _kw, const std::string _com="", const std::string _un="") 
        { keyword=_kw; comment=_com; units=_un; }

        // Two functions needed for easy interface to CFITSIO:
        virtual DataType dataType() const { return Tnull; }

        // return (void *) pointer to the value, if any
        virtual void* voidPtr() { return 0; }
        virtual const void* voidPtr() const { return 0; }

        // Set value for this entry from string; return true on failure
        virtual bool setValueString(const std::string _v) { return false; }
        virtual std::string getValueString() const { return ""; }

        std::string writeCard() const;

    protected:
        std::string keyword;
        std::string comment;
        std::string units;
    };


    class HdrRecordNull : public HdrRecordBase 
    {
    public:
        // no additional data over base class
        HdrRecordNull(const std::string _kw, const std::string _com="", const std::string _un="") : 
            HdrRecordBase(_kw,_com,_un) {}

        virtual HdrRecordNull* duplicate() const 
        { return new HdrRecordNull(*this); }
    };

    // Header Record that holds arbitary data class:
    template <class T>
    class HdrRecord : public HdrRecordBase 
    {
    private:
        T val;
        mutable std::string valString; //string representation of value

    public:
        HdrRecord(
            const std::string _kw, const T _val,
            const std::string _com="", const std::string _un="") : 
            HdrRecordBase(_kw, _com, _un), val(_val) 
        {}

        virtual HdrRecord* duplicate() const 
        { return new HdrRecord(*this); }

        T& Value() { return val; }
        const T& Value() const { return val; }

        void* voidPtr() { return static_cast<void *> (&val); }
        const void* voidPtr() const { return static_cast<const void *> (&val); }

        bool setValueString(const std::string _v) 
        {
            std::istringstream iss(_v.c_str());
            std::string leftover;
            return !(iss >> val) || (iss >> leftover);
        };

        std::string getValueString() const 
        {
            std::ostringstream os;
            os  << val; // ??? compiler bug here ???
            return os.str();
        }

        DataType dataType() const 
        { return MatchType<T>(); }
    };

    //specializations for bool
    template<>
    std::string HdrRecord<bool>::getValueString() const; 

    template<>
    bool HdrRecord<bool>::setValueString(const std::string _v);

    //and for string - enclose in quotes
    template<>
    std::string HdrRecord<std::string>::getValueString() const;

    // ??? bother to fix lack of decimal point on float/double?


    ///////////////////////////////////////////////////////////////
    // Now the class for ImageHeader itself:
    class ImageHeader 
    {
    private: 
        mutable std::list<HdrRecordBase*> hlist;
        mutable std::list<HdrRecordBase*>::iterator hptr;  //current record
        bool isAltered;
        std::list<std::string> lcomment; //Comment and History strings
        std::list<std::string> lhistory;

    public:
        ImageHeader(): hlist(), hptr(hlist.begin()), isAltered(false) {}

        ImageHeader(const ImageHeader& rhs) 
        {
            copyFrom(rhs);
            isAltered = false;
        }

        void copyFrom(const ImageHeader& rhs) 
        {
            hlist.clear(); lcomment.clear(); lhistory.clear();
            std::list<HdrRecordBase*>::const_iterator rhsptr;
            for (rhsptr=rhs.hlist.begin(); rhsptr!=rhs.hlist.end(); ++rhsptr)
                hlist.push_back( (*rhsptr)->duplicate());
            hptr = hlist.begin();
            std::list<std::string>::const_iterator sptr;
            for (sptr=rhs.lcomment.begin(); sptr!=rhs.lcomment.end(); ++sptr)
                lcomment.push_back(*sptr);
            for (sptr=rhs.lhistory.begin(); sptr!=rhs.lhistory.end(); ++sptr)
                lhistory.push_back(*sptr);
            touch();
        }

        ImageHeader& operator=(const ImageHeader& rhs) 
        {
            if (this==&rhs) return *this;
            copyFrom(rhs);
            return *this;
        }

        ~ImageHeader() 
        {
            for (hptr=hlist.begin(); hptr!=hlist.end(); ++hptr)
                delete *hptr;
        }

        ImageHeader* duplicate() const 
        { return new ImageHeader(*this); }

        // Clear all header records, plus comments & history
        void clear() 
        {
            hlist.clear();
            hptr=hlist.begin(); 
            lcomment.clear();
            lhistory.clear();
            touch();
        }

        void reset() { clear(); }

        // History/Comment accessors
        const std::list<std::string>& comments() const { return lcomment; }
        const std::list<std::string>& history() const { return lhistory; }

        void addComment(const std::string s) { lcomment.push_back(s); touch(); }
        void addHistory(const std::string s) { lhistory.push_back(s); touch(); }

        bool isChanged() const { return isAltered; }  //changed since creation?
        void notAltered() { isAltered=false; } //reset altered flag
        void touch() { isAltered=true; }

        // Manipulate the pointer to current header record:
        void rewind() const { hptr=hlist.begin(); }
        bool atEnd() const { return hptr==hlist.end(); }
        int size() const 
        { return hlist.size() + lcomment.size() + lhistory.size(); }

        HdrRecordBase* current() { return *hptr; }
        void incr() const { ++hptr; }
        const HdrRecordBase* current() const { return *hptr; }

        // Append contents of another header to this one
        void operator+=(const ImageHeader& rhs) 
        {
            if (this==&rhs) return;
            std::list<HdrRecordBase*>::const_iterator rptr;
            for (rptr=rhs.hlist.begin(); rptr!=rhs.hlist.end(); ++rptr)
                hlist.push_back( (*rptr)->duplicate());
            lcomment.insert(lcomment.end(), rhs.lcomment.begin(), rhs.lcomment.end());
            lhistory.insert(lhistory.end(), rhs.lhistory.begin(), rhs.lhistory.end());
            touch();
        }

        // Add/remove header records, by base class ptr or keyword
        void append(HdrRecordBase* record) { hlist.push_back(record); touch(); }
        void insert(HdrRecordBase* record) { hlist.insert(hptr,record); touch(); }

        void erase() { delete *hptr; hptr=hlist.erase(hptr); touch(); }

        void erase(const std::string kw) 
        {
            if (find(kw)) {
                delete *hptr; 
                hptr=hlist.erase(hptr); 
                touch();
            } else {
                throw ImageHeaderError("Cannot find record with keyword " + kw);
            }
        }

        template <class T>
        void append(
            const std::string keyword, const T& value, 
            const std::string comment="", const std::string units="") 
        {
            hlist.push_back(new HdrRecord<T>(keyword, value, comment,units));
            touch();
        }

        template <class T>
        void replace(
            const std::string keyword, const T& value, 
            const std::string comment="", const std::string units="") 
        {
            try { 
                erase(keyword);
            } catch (ImageHeaderError& i) {}
            append(keyword, value, comment, units);
        }

        void appendNull(const std::string keyword, const std::string comment="") 
        {
            hlist.push_back(new HdrRecordNull(keyword, comment));
            touch();
        }

        template <class T>
        void insert(
            const std::string keyword, const T& value, 
            const std::string comment="", const std::string units="") 
        {
            hlist.insert(hptr, new HdrRecord<T>(keyword, value, comment,units));
            touch();
        }

        void insertNull(const std::string keyword, const std::string comment="") 
        {
            hlist.insert(hptr, new HdrRecordNull(keyword, comment));
            touch();
        }

        HdrRecordBase* find(const std::string keyword) 
        {
            std::list<HdrRecordBase*>::iterator start(hptr);
            touch(); // ?? note header is marked as altered just for returning
            // a non-const pointer to header record.
            for ( ; hptr!=hlist.end(); ++hptr)
                if ((*hptr)->matchesKey(keyword)) return *hptr;
            // search from beginning to starting point
            for (hptr=hlist.begin(); hptr!=hlist.end() && hptr!=start; ++hptr)
                if ((*hptr)->matchesKey(keyword)) return *hptr;
            return 0; //null pointer if nothing found
        }

        const HdrRecordBase* findConst(const std::string keyword) const 
        {
            std::list<HdrRecordBase*>::iterator start(hptr);
            for ( ; hptr!=hlist.end(); ++hptr)
                if ((*hptr)->matchesKey(keyword)) return *hptr;
            // search from beginning to starting point
            for (hptr=hlist.begin(); hptr!=hlist.end() && hptr!=start; ++hptr)
                if ((*hptr)->matchesKey(keyword)) return *hptr;
            return 0; //null pointer if nothing found
        }

        const HdrRecordBase* find(const std::string keyword) const 
        { return findConst(keyword); }

        // Get/set the value of an existing record.  Bool returns false if
        // keyword doesn't exist or does not match type of argument.
        template <class T> 
        bool getValue(const std::string keyword, T& outVal) const;

        template <class T> 
        bool setValue(const std::string keyword, const T& inVal);


    };  

    template <class T> 
    bool ImageHeader::getValue(const std::string keyword, T& outVal) const 
    {
        const HdrRecordBase* b=findConst(keyword);
        if (!b) return false;
        const HdrRecord<T> *dhdr;
        dhdr = dynamic_cast<const HdrRecord<T>*> (b);
        if (!dhdr) return false;
        outVal = dhdr->Value();
        return true;
    }

    template <class T> 
    bool ImageHeader::setValue(const std::string keyword, const T& inVal) 
    {
        HdrRecordBase* b=find(keyword);
        if (!b) return false;
        HdrRecord<T> *dhdr;
        dhdr = dynamic_cast<HdrRecord<T>*> (b);
        if (!dhdr) return false;
        dhdr->Value() = inVal;
        touch();
        return true;
    }


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
            int x=b.getXMin();
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
    // Checked iterator for images
    //////////////////////////////////////////////////////////////////////////
    // ??? need conversion from iter to const iter??
    template <class T>
    class ImageChk_iter 
    {
    private:
        ImageData<T>* I;
        T* ptr;
        int col; //keep track of column number
    public:
        ImageChk_iter(ImageData<T>* ii, const int x, const int y) : I(ii), col(x) 
        {
            if (y<I->getBounds().getYMin() || y>I->getBounds().getYMax()) {
                throw ImageBounds(
                    "row", I->getBounds().getYMin(), I->getBounds().getYMax(), y);
            }
            ptr = &( (*I)(x,y));
        }

        T& operator*() const 
        {
            if (col<I->getBounds().getXMin() || col>I->getBounds().getXMax()) {
                throw ImageBounds(
                    "column", I->getBounds().getXMin(), I->getBounds().getXMax(), col);
            }
            return *ptr;
        }

        ImageChk_iter operator++() { ++ptr; ++col; return *this; }
        ImageChk_iter operator--() { ++ptr; ++col; return *this; }
        ImageChk_iter operator+=(int i) { ptr+=i; col+=i; return *this; }
        ImageChk_iter operator-=(int i) { ptr-=i; col-=i; return *this; }
        
        bool operator<(const ImageChk_iter rhs) const { return ptr<rhs.ptr; }
        bool operator<=(const ImageChk_iter rhs) const { return ptr<=rhs.ptr; }
        bool operator>(const ImageChk_iter rhs) const { return ptr>rhs.ptr; }
        bool operator>=(const ImageChk_iter rhs) const { return ptr>=rhs.ptr; }
        bool operator==(const ImageChk_iter rhs) const { return ptr==rhs.ptr; }
        bool operator!=(const ImageChk_iter rhs) const { return ptr!=rhs.ptr; }
    };

    template <class T>
    class ImageChk_citer 
    {
    private:
        const ImageData<T>* I;
        const T* ptr;
        int col; //keep track of column number

    public:
        ImageChk_citer(const ImageData<T>* ii, const int x, const int y) : I(ii), col(x) 
        {
            if (y<I->getBounds().getYMin() || y>I->getBounds().getYMax()) {
                throw ImageBounds(
                    "row", I->getBounds().getYMin(), I->getBounds().getYMax(), y);
            }
            ptr = &( (*I)(x,y));
        }

        const T& operator*() const 
        {
            if (col<I->getBounds().getXMin() || col>I->getBounds().getXMax()) {
                throw ImageBounds(
                    "column", I->getBounds().getXMin(), I->getBounds().getXMax(), col);
            }
            return *ptr;
        }

        ImageChk_citer operator++() { ++ptr; ++col; return *this; }
        ImageChk_citer operator--() { ++ptr; ++col; return *this; }
        ImageChk_citer operator+=(int i) { ptr+=i; col+=i; return *this; }
        ImageChk_citer operator-=(int i) { ptr-=i; col-=i; return *this; }

        bool operator<(const ImageChk_citer rhs) const { return ptr<rhs.ptr; }
        bool operator<=(const ImageChk_citer rhs) const { return ptr<=rhs.ptr; }
        bool operator>(const ImageChk_citer rhs) const { return ptr>rhs.ptr; }
        bool operator>=(const ImageChk_citer rhs) const { return ptr>=rhs.ptr; }
        bool operator==(const ImageChk_citer rhs) const { return ptr==rhs.ptr; }
        bool operator!=(const ImageChk_citer rhs) const { return ptr!=rhs.ptr; }
    };


    //////////////////////////////////////////////////////////////////////////
    // The Image handle:  this is what outside programs use.
    //////////////////////////////////////////////////////////////////////////

    template <class T>
    class Image 
    {
    private:
        ImageData<T>* D; //pixel data
        ImageHeader* H; //the "header"
        mutable int* dcount;  // link count for the data structure
        mutable int* hcount;

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
        Image(ImageData<T>* Din, ImageHeader* Hin, int* _dc=new int(0), int* _hc=new int(0)) : 
            D(Din), H(Hin), dcount(_dc), hcount(_hc) 
        { (*dcount)++; (*hcount)++; }

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

        bool isLastData() const { return *dcount==1; }
        bool isLastHeader() const { return *hcount==1; }

        ImageHeader* header() { return H; }
        const ImageHeader* header() const { return H; }

        ImageData<T>* data() { return D; }
        const ImageData<T>* data() const { return D; }

        // Get/set the value of header records.  Bool returns false if
        // keyword doesn't exist or does not match type of argument.
        template <class U> 
        bool getHdrValue(const std::string keyword, U& outVal) const 
        { return header()->getValue(keyword, outVal); }

        template <class U> 
        bool setHdrValue(const std::string keyword, const U& inVal) 
        { return header()->setValue(keyword,inVal); }

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

        // iterators, rowBegin()/end()
        typedef ImageChk_iter<T> checked_iter;
        checked_iter Chk_iter(const int x, const int y) 
        { return checked_iter(D,x,y); }

        typedef ImageChk_citer<T> checked_citer;
        checked_citer Chk_citer(const int x, const int y) const 
        { return checked_citer(D,x,y); }

#ifdef IMAGE_BOUNDS_CHECK
        typedef checked_iter iter;
        typedef checked_citer citer;
        iter rowBegin(int r) { return Chk_iter(XMin(), r); }
        citer rowBegin(int r) const { return Chk_citer(XMin(), r); }
        iter rowEnd(int r) { return Chk_iter(XMax()+1, r); }
        citer rowEnd(int r) const { return Chk_citer(XMax()+1, r ); }
        iter getIter(const int x, const int y) { return Chk_iter(x,y); }
        citer getIter(const int x, const int y) const { return Chk_citer(x,y); }
#else
        typedef T* iter;
        typedef const T* citer;
        iter rowBegin(int r) { return &(*D)(XMin(),r); }
        citer rowBegin(int r) const { return &(*D)(XMin(),r); }
        iter rowEnd(int r) { return &(*D)(XMax()+1,r); }
        citer rowEnd(int r) const { return &(*D)(XMax()+1,r); }
        iter getIter(const int x, const int y) { return &(*D)(x,y); }
        citer getIter(const int x, const int y) const { return &(*D)(x,y); }
#endif

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
