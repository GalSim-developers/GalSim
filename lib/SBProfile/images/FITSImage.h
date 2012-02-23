// ??? There is a problem with trying to create an extension in an
// empty FITS file (e.g. after opening FITSFile with Overwrite),
// probably when trying to count headers. ?????
//
//   FITSImage<T> is a class for a FITS image that is on disk.  FITS
//   files can hold multiple "Header Data Units" (HDUs), of which the
//  first (number 1) must be an image and any of the others can be an image.
//  When creating a FITSImage object, you give it the filename of the FITS
//  file and an optional HDU number or name.  The opening and closing of
//  the FITS file, getting proper extension, etc., will be transparent
//  to you, and any changes you make to the data will be automatically
//  written back to the disk file.  

//  A buffering system is built in which  allows:
//  *creating an arbitrary number of FITSImages at any time
//  *reducing disk reads/writes when many Images are used or
//  extracted from a single disk file.
//  *swapping out buffers not recently used.
//  The target size for the buffer of a single FITSImage can be set by
//  FITSImage::suggestBufferSize().  The max amount of memory to be
//  used for ALL the buffers combined is set by FITSImage:totalMemoryTarget().
//  When sum of buffers will exceed
//  this, buffers not currently in use are closed to make room,
//  but can't close if Image<> structures still point to them.
//  The memory limits are exceeded if the set of open Images requires
//  this.  

//  FITS files are opened and closed as needed automatically.

//  On creating a FITSImage, one can specify img::Flags to control
//  whether the disk file is writable.  If the image is to be read-only, 
//  an exception is thrown if it does not exist.  A read-write FITSImage
//  will be created if the FITS file does not exist.  If the file does
//  exist but the requested HDU does not exist, an exception will be
//  thrown.  The CreateImage flag can be set to cause a new image HDU
//  to be created.  By default the HDU is 1, the "primary HDU".

//  FITSImage<T> is a template for images with data type T.  If the
//  type T does not match the native type of the FITS disk image, 
//  conversions will be done implicitly (by CFITSIO)

//  Once the FITSImage is created, you can access its header info via
//  FITSImage::header()
//  To use image data, you create Image<T> objects via
//  FITSImage::use()  or useConst()
//   Images obtained with the use() methods are tied to the disk data;
//   any changes to these images are written back to the disk.  Destroying
//   the FITSImage before destroying all Images in use throws an exception.
//   All the Images in use() are sharing data and headers.  Destroy
//   Images when you are done using them to free up memory buffer space.
//  FITSImage::extract() gives back an Image that is a COPY of the disk
//   data and header.  Changes to an extracted image are NOT mirrored back
//   to disk.
//  Both use() and extract() pull out the whole image by default.  Bounds
//  may be specified, you get back the intersection of our requested
//  area and the area that actually exists on disk.  Null intersection 
//  throws an exception.

//  FITSImage::flush() writes all buffers back to disk for safety.
//  FITSImage::isNull() reports whether there are any pixels in the 
//    FITSImage (by default, new extensions are created with zero dimension.
//  FITSImage::resize() changes the image dimensions, throwing an exception
//    if any images are in use() at the time.

// Opening the same FITS image extension with more than one
// FITSImage object will be recognized if the same filename is
// used.  But opening with varying permissions or different
// filenames may cause buffers to get out of synch so is ill-advised.

#ifndef FITSIMAGE_H
#define FITSIMAGE_H

//#define FITSDEBUG

#include <list>

#include "FITS.h"
#include "Image.h"

namespace galsim {

    const int DEFAULT_IMAGE_BUFFER_SIZE=32; //Megabytes of data to buffer per img
    const int DEFAULT_TOTAL_BUFFER_SIZE=128; //Megabytes of data to buffer, total

    ////////////////////////////////////////////////////////////////
    // A base class used only to allow a common memory pool for all
    // FITSImage data types.  Pointer to this class can serve as a means to
    // maintain lists of images of different datatypes.
    ////////////////////////////////////////////////////////////////

    class FITSImageBase 
    {
    public:
        virtual ~FITSImageBase() {}
    protected:
        virtual bool freeBuffer() const=0;
        //Get rid of Image<> objects; returns true if any still linked to buffer:
        virtual bool purgeImages() const=0;
        static  long  totalMemoryInUse;
        static  int   totalMemoryTarget;
        static  std::list<const FITSImageBase*> imgQ;
        typedef std::list<const FITSImageBase*>::iterator fptr;
        //Delete buffers until addsize is available
        static  void makeRoom(long addsize);
    };

    ////////////////////////////////////////////////////////////
    // Here is the FITSImage class
    ////////////////////////////////////////////////////////////
    template <class T=float>
    class FITSImage : public FITSImageBase 
    {
    public:
        FITSImage(const std::string fname, const Flags f=ReadOnly, const int HDUnum=1);
        FITSImage(const std::string fname, const Flags f, const std::string HDUname);
        // ??? need constructor for new/blank image/extension of desired size
        ~FITSImage();

        typedef T  value_type;

        ImageHeader* header() { loadHeader(); return hptr; }
        const ImageHeader* header() const { loadHeader(); return hptr; }

        std::string filename() const { return parent.getFilename(); }

        Image<T> extract(Bounds<int> b) const;
        Image<T> extract() const; // Get full image
        Image<T> use();
        Image<T> use(const Bounds<int> b);

        const Image<T> useConst() const;
        const Image<T> useConst(const Bounds<int> b) const;
        const Image<T> use() const { return useConst(); }
        const Image<T> use(const Bounds<int> b) const { return useConst(b); }

        void write(const Image<T> I);
        void write(const Image<T> I, const Bounds<int> b);
        void flush() const { flushData(); flushHeader(); }
        void flush() { flushData(); flushHeader(); }

        Bounds<int> getBounds() const { return diskBounds; }

        bool isNull() const { return !diskBounds; }  //no data for image?

        void suggestBufferSize(const int MBytes) const;

        void bufferEntireImage() const;

        // set (or read, w/o argument) limit on total memory usage
        static int totalMemoryLimit(const int MBytes=0);

        // Return the data type in which the image was written
        DataType getNativeType() const { return nativeType; }

        // Change image size - destroys data (header is kept).
        void resize(const Bounds<int> newBounds);

        // Replace extension header & image with that of an entire Image
        // (If the FITSImage has an EXTNAME and the source Image does not,
        // it keeps the old extension name.)
        void copy(const Image<T> I);
        
        // Copy directly from another FITS extension.
        void copy(const FITSImage<T>& fI);

        // Give this extension a name
        void renameExtension(const std::string newext);

        // Convenience functions to make new FITSImage that is a COPY of
        // the specified Image (both data and header).  The FITS image is
        // NOT a mirror of the input image, it's a duplicate.
        // The ReadWrite and CreateImage flags are implicit here.
        static void writeToFITS(
            const std::string fname, const Image<T> imageIn,
            const int HDUnum=1);
        static void writeToFITS(
            const std::string fname, const Image<T> imageIn,
            const std::string HDUname);

    private:
        // private copy/assignment to prevent use - pointer/ref passing only:
        FITSImage(const FITSImage& rhs);
        void operator=(const FITSImage& rhs);

        mutable FITSFile parent; //FITSFiles don't like being const
        int HDUnumber;
        Bounds<int> diskBounds;
        mutable Bounds<int>  alterableBounds;
        DataType nativeType;

        mutable std::list<Image<T> > imgList;
        typedef typename std::list<Image<T> >::iterator imgptr;
        mutable T* buffer;
        mutable int bufferTarget;
        mutable long bufferSize;
        mutable long bufferRows;
        mutable Bounds<int> bufferBounds;
        mutable int firstRowOffset; //What row of buffer holds bottom of bounds?

        mutable ImageHeader* hptr;
        mutable int* hcount; //link count for the header

        void allocateBuffer(const Bounds<int> b) const;
        void flushData() const;
        void touch() const;
        Bounds<int> desiredBounds(const Bounds<int> b) const;
        void bufferMustSpan(const Bounds<int> b) const;
        void readRows(const int ymin, const int ymax, bool useExisting) const;

        // Note that WriteRows is const even though it changes disk data.
        // It is only used internally, the public routines for writing will
        // preclude writing to const images.
        void writeRows(const int ymin, const int ymax) const;

        T* bufferLocation(const int xpos, const int ypos) const;
        T** makeRowPointers(const Bounds<int> b) const;

        // Get rid of unused Image<> objects; 
        // returns true if any still linked to buffer:
        bool purgeImages() const;

        // Close out the data buffer.  Returns true if there are Images still attached.
        bool freeBuffer() const;

        void loadHeader() const; // read header from file.
        void flushHeader(); // Write header back to file,  and
        void flushHeader() const; // remove it from memory if not in use.
    };

} 

#endif
