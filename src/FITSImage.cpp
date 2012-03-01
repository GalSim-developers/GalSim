// FITS Image manipulation routines.
#include "FITSImage.h"

#include <typeinfo>
#include <cstring> 

namespace galsim {

    const int MAX_IMAGE_DIMENSIONS=10; //maximal number of img dimensions


    // Shared static structures
    std::list<const FITSImageBase*> FITSImageBase::imgQ;
    int FITSImageBase::totalMemoryTarget=DEFAULT_TOTAL_BUFFER_SIZE;
    long  FITSImageBase::totalMemoryInUse=0;

    /////////////////////////////////////////////////////////////////
    // Reading & Writing FITS headers to ImageHeader structures
    /////////////////////////////////////////////////////////////////

    // These are the FITS Header keywords that specify the extensions
    // and the image size.  They will NOT be saved into our ImageHeader
    // structure, as they are read/written automatically when maintaining
    // the HDUs and images.
    const char *SpecialKeys[]={ 
        "SIMPLE","BITPIX","NAXIS","NAXIS1","NAXIS2",
        "EXTEND","XTENSION","PCOUNT","GCOUNT"
    };
    int nSpecialKeys=9;

    // read a full header from FITS extension (assumed current HDU)

    ImageHeader* ReadFITSHeader(fitsfile *fptr) 
    {
        ImageHeader* ih=new ImageHeader();

        int status(0);
        int nkeys;
        fits_get_hdrspace(fptr, &nkeys, NULL, &status);

        char keyword[FLEN_CARD];
        char comment[FLEN_CARD];
        char value[FLEN_CARD];
        char units[FLEN_CARD];
        char vtype;

        for (int ikey=1; ikey<=nkeys; ikey++) {
            fits_read_keyn(fptr,ikey,keyword,value,comment,&status);
            if (strlen(value)>0) {
                fits_get_keytype(value, &vtype, &status);
                fits_read_key_unit(fptr,keyword,units,&status);
            } else {
                vtype='N';
                units[0]=0;
            }
            if (status) throw_CFITSIO("ReadFITSHeader collecting all keys");
            // Make the HdrRecord: 
            HdrRecordBase* hh;
            bool badstring(false);
            std::string vstring=value;
            switch (vtype) {
              case 'N': 
                   hh = new HdrRecordNull(keyword, comment);
                   break;
              case 'C':
                   if (vstring[0]!='\'' || vstring[vstring.size()-1]!='\'')
                       std::cerr << "no quotes on string: [" << vstring << "]" << std::endl;
                   hh = new HdrRecord<std::string>(
                       keyword, vstring.substr(1,vstring.size()-2), comment, units);
                   break;
              case 'L':
                   hh = new HdrRecord<bool>(keyword, false, comment, units);
                   badstring = hh->setValueString(vstring);
                   break;
              case 'I':
                   hh = new HdrRecord<int>(keyword, 0, comment, units);
                   badstring = hh->setValueString(vstring);
                   break;
              case 'F':
                   hh = new HdrRecord<double>(keyword, 0., comment, units);
                   badstring = hh->setValueString(vstring);
                   break;
              case 'X':
                   hh = new HdrRecord<std::complex<double> >(
                       keyword, std::complex<double>(), comment, units);
                   badstring = hh->setValueString(vstring);
                   break;
              default:
                   throw ImageHeaderError(
                       "Header vstring [" + vstring + "] of unknown type " + vtype);
            }
            if (badstring) {
                throw ImageHeaderError(
                    "Header vstring [" + vstring + "] could not be interpeted as type " + vtype);
            }
            // Skip keywords that are part of FITS extension/image definitions
            bool isSpecial=false;
            for (int i=0; !isSpecial && i<nSpecialKeys; i++) {
                isSpecial = isSpecial || hh->matchesKey(SpecialKeys[i]);
            }
            if (isSpecial) {
                delete hh;
            } else if (hh->matchesKey("COMMENT")) {
                ih->addComment(hh->getComment());
                delete hh;
            } else if (hh->matchesKey("HISTORY")) {
                ih->addHistory(hh->getComment());
                delete hh;
            } else {
                ih->append(hh);
            }
        }
        return ih;
    }

    // Append contents of ImageHeader to header of a FITS file's
    // current HDU:
    void WriteFITSHeader(fitsfile *fptr, ImageHeader *ih) 
    {
        int status(0);
        char keyword[FLEN_CARD];
        char comment[FLEN_CARD];
        char vstring[FLEN_CARD];
        char units[FLEN_CARD];

        // First write all the Records
        for (ih->rewind(); !ih->atEnd(); ih->incr()) {
            const HdrRecordBase* hh=ih->current();
            DataType t=hh->dataType();
            strncpy(keyword,hh->getKeyword().c_str(),sizeof(keyword));
            strncpy(comment,hh->getComment().c_str(),sizeof(comment));
            strncpy(units,hh->getUnits().c_str(),sizeof(units));

            switch (t) {
              case Tnull:
                   fits_write_key_null(fptr, keyword, comment, &status);
                   break;
              case Tstring:
                   { 
                       const HdrRecord<std::string>* hs =
                           dynamic_cast<const HdrRecord<std::string>*> (hh);
                       strncpy(vstring,hs->Value().c_str(),sizeof(vstring));
                       vstring[sizeof(vstring)-1]=0;
                   }
                   fits_write_key_str(fptr, keyword, vstring, comment, &status);
                   if (!hh->getUnits().empty())
                       fits_write_key_unit(fptr, keyword, units, &status);
                   break;
              default:
                   {
                       // nasty cast needed for CFITSIO:
                       void *vv = const_cast<void *> (hh->voidPtr());
                       fits_write_key(fptr, t, keyword, vv, comment, &status);
                   }
                   if (!hh->getUnits().empty())
                       fits_write_key_unit(fptr, keyword, units, &status);
            }

        }

        // Then write all the COMMENT fields
        std::list<std::string>::const_iterator sptr;
        for (sptr=ih->comments().begin(); sptr!=ih->comments().end(); ++sptr) {
            strncpy(comment, sptr->c_str(), sizeof(comment));
            comment[sizeof(comment)-1]=0;
            fits_write_comment(fptr, comment, &status);
        }

        // And HISTORY strings
        for (sptr=ih->history().begin(); sptr!=ih->history().end(); ++sptr) {
            strncpy(comment, sptr->c_str(), sizeof(comment));
            comment[sizeof(comment)-1]=0;
            fits_write_history(fptr, comment, &status);
        }
        if (status) throw_CFITSIO("WriteFITSHeader");
    }

    // Clear a FITS header of all optional keywords
    void ClearFITSHeader(fitsfile *fptr) 
    {
        const char *incl[]={"*"};
        char card[FLEN_CARD];
        int status(0);
        int nkeys, nextkey;
        fits_read_record(fptr, 0, card, &status);  //rewind header
        while (!status) {
            fits_find_nextkey(
                fptr, const_cast<char**>(incl), 1,
                const_cast<char**>(SpecialKeys), nSpecialKeys,
                card, &status);
            fits_get_hdrpos(fptr, &nkeys, &nextkey, &status);
            if (status) break;
            assert (nextkey>1);
            fits_delete_record(fptr, nextkey-1, &status);
        }
        if (status!=KEY_NO_EXIST) throw_CFITSIO("ClearFITSHeader");
        else fits_clear_errmsg();
    }

    // Connect the FITSImage disk files to ImageHeader structures
    template <class T>
    void FITSImage<T>::loadHeader() const 
    {
        if (hcount) return; //already have it.
#ifdef FITSDEBUG
        std::cerr << "Loading header from disk for " << parent.getFilename() 
            << " HDU #" << HDUnumber << std::endl;
#endif

        // go to correct HDU
        int status(0);
        fits_movabs_hdu(parent.getFitsptr(), HDUnumber, NULL, &status);
        if (status) 
            throw_CFITSIO("loadHeader getting HDU for " + parent.getFilename());

        hptr = ReadFITSHeader(parent.getFitsptr());
        hcount = new int(1);
        hptr->notAltered(); //clear the "alteration" flag
    }

    template <class T>
    void FITSImage<T>::flushHeader() 
    {
#ifdef FITSDEBUG
        std::cerr << "flushHeader() for " << parent.getFilename() 
            << " HDU #" << HDUnumber << std::endl;
#endif
        if (!hcount) return; //no header loaded
        if (hptr->isChanged()) {
            int status(0);
            // Header has changed, write back to file
            if (!(parent.getFlags() & ReadWrite))
                throw FITSError(
                    "attempt to write altered header to FITS file" + parent.getFilename());
            // go to correct HDU
            fits_movabs_hdu(parent.getFitsptr(), HDUnumber, NULL, &status);
            if (status) 
                throw_CFITSIO("flushHeader getting HDU for " + parent.getFilename());
            ClearFITSHeader(parent.getFitsptr());
            WriteFITSHeader(parent.getFitsptr(), hptr);
            hptr->notAltered();  //reset the alteration flag
        }
        if (*hcount==1) {
            delete hptr;
            delete hcount;
            hcount=0;
        }
    }

    // flush for a const FITSImage should not involve any altered header
    template <class T>
    void FITSImage<T>::flushHeader() const 
    {
#ifdef FITSDEBUG
        std::cerr << "const flushHeader() for " << parent.getFilename() 
            << " HDU #" << HDUnumber << std::endl;
#endif
        if (!hcount) return; //no header loaded
        if (hptr->isChanged()) 
            throw FITSError("const FITSImage has altered ImageHeader, " + parent.getFilename());
        if (*hcount==1) {
            delete hptr;
            delete hcount;
            hcount=0;
        }
    }

    /////////////////////////////////////////////////////////////////
    // Constructors/Destructors for the FITSImage objects
    /////////////////////////////////////////////////////////////////
    template <class T>
    void FITSImage<T>::writeToFITS(
        const std::string fname, const Image<T> imageIn, const int HDUnum) 
    {
        FITSImage<T> fi(fname, ReadWrite+CreateImage, HDUnum);
        fi.copy(imageIn);
    }

    template <class T>
    void FITSImage<T>::writeToFITS(
        const std::string fname, const Image<T> imageIn, const std::string HDUname) 
    {
        FITSImage<T> fi(fname, ReadWrite+CreateImage, HDUname);
        fi.copy(imageIn);
    }

    template <class T>
    FITSImage<T>::FITSImage(const std::string fname, const Flags f, const int HDUnum) :
        parent(fname, Flags(f & ~(Overwrite+CreateImage))), 
        HDUnumber(HDUnum), buffer(0), bufferTarget(DEFAULT_IMAGE_BUFFER_SIZE),
        bufferBounds(), hcount(0) 
    {
        // Open image, insure this extension is in fact a 2d image, 
        // and read axis data, etc.
        if (f & Overwrite)
            throw FITSError("Cannot open FITSImage with Overwrite flag");
        if (HDUnumber < 1 || HDUnumber > parent.HDUCount()) {
            // ??? add HDU numbers to these messages ???
            if ( (f & CreateImage) && (f & ReadWrite)) {
                // create needed null-image extension(s)
                int n=parent.HDUCount();
                int naxis=0; //note zero-dimenisional images now.
                long naxes[MAX_IMAGE_DIMENSIONS];
                int status(0);
                while (n<HDUnumber) {
#ifdef FITSDEBUG
                    std::cerr << "creating image extension " << n << std::endl;
#endif
                    fits_create_img(
                        parent.getFitsptr(), DataType_to_Bitpix(Tfloat),
                        naxis, naxes, &status);
                    if (status) 
                        throw_CFITSIO("Constructor creating image for " + parent.getFilename());
                    n++;
                }
                parent.flush();
                assert(parent.HDUCount()==HDUnumber);
            } else {
                throw FITSError("Requested non-existent HDU for " + fname);
            }
        }
        if (parent.getHDUType(HDUnumber) != HDUImage)
            throw FITSError("FITSImage requested extension that is not an image");

        int naxis, bitpix;
        long naxes[MAX_IMAGE_DIMENSIONS];
        int status(0);
        fits_get_img_param(
            parent.getFitsptr(), MAX_IMAGE_DIMENSIONS, &bitpix, &naxis, naxes, &status);
        if (status) throw_CFITSIO("Constructor get_img_param for " + parent.getFilename());

        if (naxis==2) {
            // a good 2d image
            nativeType = Bitpix_to_DataType(bitpix);
            diskBounds = Bounds<int>( int(1), naxes[0], int(1), naxes[1]);
        } else if (naxis==0) {
            // No image data, set this up as null image
            nativeType = Bitpix_to_DataType(bitpix);
            diskBounds = Bounds<int>();   //null bounds by default
        } else {
            throw FITSError("Image is not 2d.");
        }  
    }

    // Open with extension name
    template <class T>
    FITSImage<T>::FITSImage(const std::string fname, const Flags f, const std::string HDUname) :
        parent(fname, Flags(f & ~(Overwrite+CreateImage))), 
        buffer(0), bufferTarget(DEFAULT_IMAGE_BUFFER_SIZE),
        bufferBounds(), hcount(0) 
    {
        // Open image, insure this extension is in fact a 2d image, 
        // and read axis data, etc.
        if (f & Overwrite)
            throw FITSError("Cannot open FITSImage with Overwrite flag");
        if (parent.getHDUType(HDUname, HDUnumber) != HDUImage) {
            if (HDUnumber<=0) {
                if ( (f & CreateImage) && (f & ReadWrite)) {
                    // create needed null-image extension to give this name
                    int naxis=0; //note zero-dimensional images now.
                    long naxes[MAX_IMAGE_DIMENSIONS];
                    int status(0);
#ifdef FITSDEBUG
                    std::cerr << "creating FITSImage HDUname " << HDUname << std::endl;
#endif
                    fits_movabs_hdu(
                        parent.getFitsptr(), parent.HDUCount(), NULL, &status);
                    fits_create_img(
                        parent.getFitsptr(), DataType_to_Bitpix(Tfloat), naxis, naxes, &status);
                    parent.flush();
                    HDUnumber = parent.HDUCount();
                    fits_movabs_hdu(
                        parent.getFitsptr(), HDUnumber, NULL, &status);
                    char nname[FLEN_VALUE];
                    strncpy(nname,HDUname.c_str(), sizeof(nname));
                    nname[sizeof(nname)-1]=0;

                    //give extension desired name
                    fits_write_key(
                        parent.getFitsptr(), Tstring, "EXTNAME", nname, NULL, &status); 
                    if (status) throw_CFITSIO("Constructor locating HDU for "
                                              + fname);
                } else {
                    throw FITSError("Requested non-existent HDU [" + HDUname
                                    + "] for " + fname);
                }
            } else {
                throw FITSError("FITSImage HDU  ["
                                + HDUname + "] of file "
                                + fname + "is not an image");
            }
        }
        int naxis, bitpix;
        long naxes[MAX_IMAGE_DIMENSIONS];
        int status(0);
        fits_get_img_param(
            parent.getFitsptr(), MAX_IMAGE_DIMENSIONS, &bitpix, &naxis, naxes, &status);
        if (status) 
            throw_CFITSIO("Constructor 2 get_img_param for " + parent.getFilename());
        if (naxis==2) {
            // a good 2d image
            nativeType = Bitpix_to_DataType(bitpix);
            diskBounds = Bounds<int>( int(1), naxes[0], int(1), naxes[1]);
        } else if (naxis==0) {
            // No image data, set this up as null image
            nativeType = Bitpix_to_DataType(bitpix);
            diskBounds = Bounds<int>();   //null bounds by default
        } else {
            throw FITSError("Image is not 2d.");
        }  
    }

    template <class T>
    FITSImage<T>::~FITSImage() 
    {
#ifdef FITSDEBUG
        std::cerr << "Destroying FITSImage " << parent.getFilename() 
            << " HDU #" << HDUnumber << std::endl;
#endif
        if (freeBuffer()) {
            if (!std::uncaught_exception())
                throw FITSError("Destroying FITSImage " + parent.getFilename() + 
                                "with Images still in use");
        }
        flushHeader();
        if (hcount && !std::uncaught_exception()) {
            throw FITSError(
                "Header for " + parent.getFilename() + "still linked upon FITSImage destruction");
        }
    }

    template <class T>
    bool FITSImage<T>::freeBuffer() const 
    {
        flushData(); //write any data (or headers) back to disk
        if (!buffer) return false; //nothing to do if no buffer.
        // Return TRUE if still images using these data; in most cases this
        // is cause for an exception, unless resizing the buffer.
        bool IfImagesLeft = purgeImages();
#ifdef FITSDEBUG
        std::cerr << "  About to delete buffer for " << parent.getFilename() 
            << " HDU #" << HDUnumber << std::endl;
#endif
        delete[] buffer; buffer=0; 
        bufferBounds = Bounds<int>(); //assign null bounds
        totalMemoryInUse -= bufferSize*sizeof(T);
        //unlink this from list of buffers
        fptr me=find(imgQ.begin(), imgQ.end(), this);
        if (me==imgQ.end()) throw FITSError("Can't find buffer on imgQ");
        imgQ.erase(me);
        return IfImagesLeft;
    }

    // Change image size - destroys data (header is kept).
    template <class T>
    void FITSImage<T>::resize(const Bounds<int> newBounds) 
    {
        if (purgeImages()) {
            throw FITSError(
                "Cannot resize FITSImage for " + parent.getFilename()
                + " with Images currently open");
        }
        freeBuffer();
        int status(0);
        int naxis(2);
        long naxes[MAX_IMAGE_DIMENSIONS];
        int bitpix = DataType_to_Bitpix(MatchType<T>());
        if (newBounds) {
            // Image is defined, make 2d
            if (newBounds.getXMin()!=1 ||
                newBounds.getYMin()!=1)
                throw FITSError("Bounds of resized FITSImage <"
                                + parent.getFilename() + 
                                "> do not start at (1,1)");
            naxes[0] = newBounds.getXMax();
            naxes[1] = newBounds.getYMax();
        } else {
            // redefine as a zero-dimensional image
            naxis=0;
        }
        // go to correct HDU
        fits_movabs_hdu(parent.getFitsptr(), HDUnumber, NULL, &status);
        parent.flush();
        fits_resize_img(parent.getFitsptr(), bitpix, naxis, naxes, &status);
        if (status) throw_CFITSIO("resize() on " + parent.getFilename());
        parent.flush();
        if (status) throw_CFITSIO("resize() flush on " + parent.getFilename());
        diskBounds = newBounds;
    }

    // Get rid of otherwise-unused Image<> structures.
    // Then tell whether any are left
    template <class T>
    bool FITSImage<T>::purgeImages() const 
    {
        imgptr i;
        for (i=imgList.begin(); i!=imgList.end(); ++i) {
            if (i->isLastData()) {
                imgList.erase(i--);
            }
        }
        return imgList.size() > 0;
    }

    // Promote this FITSImage to the front of the buffer-usage queue.
    // Back of queue contains longest-dormant buffers
    template <class T>
    void FITSImage<T>::touch() const 
    {
        if (!buffer) return; //If not using buffer then not on queue at all.
        fptr me;
        me = find(imgQ.begin(), imgQ.end(), this);
        if (me!=imgQ.end()) imgQ.erase(me);
        imgQ.push_front(this);
    }

    void FITSImageBase::makeRoom(long addsize) 
    {
        // starting at back of buffer queue, try freeing buffers
        // until there is room for requested memory
        std::list<const FITSImageBase*>::iterator qptr=imgQ.end();
        while ( (addsize + totalMemoryInUse > totalMemoryTarget* 1024L * 1024L)
                && qptr!=imgQ.begin()) {
            qptr--;
            if (!(*qptr)->purgeImages()) {
                // Free to get rid of this buffer - freeBuffer() removes from imgQ
                // Tricky business here: freeBuffer will remove this one from imgQ,
                // so be careful of what happens to the iterator!
                if (qptr==imgQ.begin()) {
                    (*qptr)->freeBuffer();
                    qptr = imgQ.begin();
                } else {
                    (*(qptr--))->freeBuffer();
                }

            } // if buffer can be purged

        } // while over quota
    }

    // Allocate space for a new buffer of size elements
    template<class T>
    void FITSImage<T>::allocateBuffer(const Bounds<int> b) const 
    {
        if (buffer) {
            throw FITSError(
                "allocateBuffer() called when buffer!=0 for " + parent.getFilename());
        }
        if (!b) {
            throw FITSError(
                "allocateBuffer() called with invalid bounds for " + parent.getFilename());
        }
        long size = b.getXMax() - b.getXMin() + 1;
        size *= b.getYMax() - b.getYMin() + 1;
        makeRoom(size*sizeof(T));
#ifdef FITSDEBUG
        std::cerr << "  About to allocate buffer for " << parent.getFilename() 
            << " HDU #" << HDUnumber
            << " with size " << size
            << std::endl;
#endif
        buffer = new T[size]; //catch memory failure here??? while loop?
        bufferSize = size;
        bufferRows = b.getYMax() - b.getYMin() + 1;
        alterableBounds = Bounds<int>(); //set to nil region
        totalMemoryInUse += bufferSize*sizeof(T);
        touch();
    }

    template <class T>
    void FITSImage<T>::suggestBufferSize(const int MBytes) const 
    { bufferTarget = MBytes; }
    template <class T>
    int FITSImage<T>::totalMemoryLimit(const int MBytes) 
    {
        if (MBytes>0) totalMemoryTarget = MBytes;
        return totalMemoryTarget;
    }

    // Flush our buffer to CFITSIO and flush CFITSIO to disk
    template <class T>
    void FITSImage<T>::flushData() const 
    {
        if (alterableBounds) {
            // If any of the buffer could have changed, write it back
            writeRows(alterableBounds.getYMin(),alterableBounds.getYMax());
            purgeImages(); 
            //Alterable region is now union of all open Images.  Note that
            // I am counting readonly Images here too since there's
            // no easy way to tell ???
            imgptr i;
            alterableBounds=Bounds<int>();
            for (i=imgList.begin(); i!=imgList.end(); ++i)
                alterableBounds += i->getBounds();
        }
        parent.flush();
    }

    template <class T>
    void FITSImage<T>::bufferEntireImage() const 
    {
        // Change suggested size to full image, next read will pull it all.
        if (!diskBounds) return;
        long size = diskBounds.getXMax() - diskBounds.getXMin()+1;
        size *= diskBounds.getYMax() - diskBounds.getYMin()+1;
        size *=sizeof(T);
        size /= 1024L * 1024L;
        size += 1;
        bufferTarget = static_cast<int>(size);
    }

    // Get a range of rows from disk into current data buffer.
    template <class T>
    void FITSImage<T>::readRows(const int ymin, const int ymax, bool useCurrent) const
    {
#ifdef FITSDEBUG
        std::cerr << "    readRows(" << ymin << "," << ymax << ") for image "
            << parent.getFilename()
            << " HDU #" << HDUnumber 
            << std::endl;
#endif
        if (!buffer || (useCurrent && !bufferBounds)
            || ymin < diskBounds.getYMin()
            || ymax > diskBounds.getYMax()
            || (ymax-ymin+1) > bufferRows ) 
            throw FITSError("Bad bounds or absent buffer in readRows()");

        // Move the FITSFile to the proper HDU
        int status(0);
        long firstpix[2];
        long nelements;
        long xsize = diskBounds.getXMax()-diskBounds.getXMin() + 1;
        int readmin, readmax;
        firstpix[0]=diskBounds.getXMin();
        fits_movabs_hdu( parent.getFitsptr(), HDUnumber, NULL, &status);
        if (status) 
            throw_CFITSIO("bufferEntireImage() locating HDU on " + parent.getFilename());

        // If new region will not overlap old data at all, just start over
        if (!bufferBounds 
            || ymin + bufferRows-1 < bufferBounds.getYMin()
            || ymax - bufferRows + 1 > bufferBounds.getYMax()) {
            useCurrent = false;
        }

        if (!useCurrent) {
            // Ignore/Toss current data, just fill the array as desired
            firstRowOffset = 0;
            firstpix[1]=ymin;
            nelements = (ymax-ymin+1)*xsize;
            fits_read_pix(
                parent.getFitsptr(), MatchType<T>(), firstpix, nelements, NULL,
                buffer, NULL, &status);
            bufferBounds=Bounds<int>(diskBounds.getXMin(), diskBounds.getXMax(), ymin, ymax);
        } else {
            // Keep as much of current data in place as possible.

            // Any data to be read BELOW current buffer range?
            readmin = ymin;
            readmax = bufferBounds.getYMin()-1;
            if (readmin <= readmax) {
                assert(readmax - readmin + 1 < bufferRows);
                int firstBuffRow = bufferBounds.getYMin() - firstRowOffset;

                if (readmin < firstBuffRow) {
                    // Data to read in will go under bottom of the buffer.
                    // First get the part at bottom of buffer
                    firstpix[1]=firstBuffRow;
                    nelements = (readmax-firstBuffRow+1)*xsize;
                    if (nelements>0) {
                        fits_read_pix(
                            parent.getFitsptr(), MatchType<T>(),
                            firstpix, nelements, NULL, buffer, NULL, &status);
                    }
                    // Now get the lower row range, to store at top of buffer
                    firstpix[1] = readmin;
                    nelements = (firstBuffRow - readmin)*xsize;
                    firstRowOffset = readmin + bufferRows - firstBuffRow;
                    T* target = buffer + xsize *  firstRowOffset;
                    fits_read_pix(
                        parent.getFitsptr(), MatchType<T>(),
                        firstpix, nelements, NULL, target , NULL, &status);
                } else {
                    // Data will fit continguously below existing data
                    firstpix[1] = readmin;
                    nelements = (readmax - readmin + 1) *
                        (diskBounds.getXMax() - diskBounds.getXMin() + 1);
                    firstRowOffset = readmin - firstBuffRow;
                    T* target = buffer + xsize*firstRowOffset;
                    fits_read_pix(
                        parent.getFitsptr(), MatchType<T>(),
                        firstpix, nelements, NULL, target , NULL, &status);
                }
                // At this point, ymin is stored in firstRowOffset row of buffer.
                bufferBounds.setYMin(ymin);
                // and we may have overwritten the previous higher rows, so update
                // YMax:
                if (bufferBounds.getYMax() - bufferBounds.getYMin() > bufferRows-1)
                    bufferBounds.setYMax(bufferBounds.getYMin() + bufferRows - 1);
            }
            // Any data to be read ABOVE current buffer range?
            readmin = bufferBounds.getYMax()+1;
            readmax = ymax;
            if (readmin <= readmax) {
                assert(readmax - readmin + 1 < bufferRows);
                int lastBuffRow = bufferBounds.getYMin() - firstRowOffset + bufferRows - 1;
                // put lastBuffRow in or above range to be read here
                while (lastBuffRow < readmin) lastBuffRow+=bufferRows;
                // Check for wrap around
                if (readmax > lastBuffRow && readmin <= lastBuffRow) {
                    // Data to read in will wrap around the buffer.
                    // First get the part at end of buffer
                    firstpix[1]=readmin;
                    nelements = (lastBuffRow-readmin+1)*xsize;
                    T* target = buffer + xsize*
                        (bufferRows - 1 - lastBuffRow + readmin);
                    fits_read_pix(
                        parent.getFitsptr(), MatchType<T>(),
                        firstpix, nelements, NULL, target, NULL, &status);
                    // Now get the last row range, which will wrap around
                    firstpix[1] = lastBuffRow+1;
                    nelements = (readmax - lastBuffRow)*xsize;
                    fits_read_pix(
                        parent.getFitsptr(), MatchType<T>(),
                        firstpix, nelements, NULL, buffer , NULL, &status);
                } else {
                    // Data will fit continguously 
                    firstpix[1] = readmin;
                    nelements = (readmax - readmin + 1)*xsize;
                    T* target;
                    target = buffer + xsize*
                        (bufferRows - 1 - lastBuffRow + readmin);
                    fits_read_pix(
                        parent.getFitsptr(), MatchType<T>(),
                        firstpix, nelements, NULL, target , NULL, &status);
                }
                bufferBounds.setYMax(ymax);
                // and we may have overwritten the previous lowest rows, so update
                // YMin and firstRowOffset
                if (bufferBounds.getYMax() - bufferBounds.getYMin() > bufferRows-1) {
                    bufferBounds.setYMin(bufferBounds.getYMax() - bufferRows + 1);
                    firstRowOffset = bufferRows - lastBuffRow + bufferBounds.getYMin() - 1;
                    while (firstRowOffset < 0) firstRowOffset += bufferRows;
                }
            }
        }
        if (status) throw_CFITSIO("readRows() on " + parent.getFilename());
    }

    /////////////////////////////////////////////////////////////
    // Write a range of rows from data buffer back to disk.
    /////////////////////////////////////////////////////////////
    template <class T>
    void FITSImage<T>::writeRows(const int ymin, const int ymax) const 
    {
#ifdef FITSDEBUG
        std::cerr << "writeRows(" << ymin << "," << ymax << ") for image "
            << parent.getFilename() 
            << "HDU #" << HDUnumber
            << std::endl;
#endif
        if (!buffer || 
            ymin < diskBounds.getYMin() || ymax > diskBounds.getYMax() || 
            ymin < bufferBounds.getYMin() || ymax > bufferBounds.getYMax()) {
            throw FITSError("Bad bounds or absent buffer in writeRows()");
        }

        // Move the FITSFile to the proper HDU
        int status(0);
        long firstpix[2];
        long nelements;
        long xsize = diskBounds.getXMax() -diskBounds.getXMin() + 1;

        firstpix[0]=diskBounds.getXMin();
        fits_movabs_hdu( parent.getFitsptr(), HDUnumber, NULL, &status);
        if (status) 
            throw_CFITSIO("writeRows() moving to HDU on " + parent.getFilename());

        // See if the region to write wraps around the buffer
        int lastBuffRow = bufferBounds.getYMin() - firstRowOffset + bufferRows -1;
        if (ymin <= lastBuffRow && ymax > lastBuffRow) {
            // Need to do the writing in 2 sections.  First the lower rows:
            firstpix[1]=ymin;
            nelements = (lastBuffRow-ymin+1)*xsize;
            fits_write_pix(
                parent.getFitsptr(), MatchType<T>(), firstpix, nelements, 
                bufferLocation(diskBounds.getXMin(), ymin), &status);

            // Now the upper range, which wraps around to beginning of buffer
            firstpix[1] = lastBuffRow+1;
            nelements = (ymax - lastBuffRow)*xsize;
            fits_write_pix(
                parent.getFitsptr(), MatchType<T>(), firstpix, nelements,
                bufferLocation(diskBounds.getXMin(), lastBuffRow+1), &status);
        } else {
            // Region to write is contiguous
            firstpix[1]=ymin;
            nelements = (ymax-ymin+1)*xsize;
            fits_write_pix(
                parent.getFitsptr(), MatchType<T>(), firstpix, nelements, 
                bufferLocation(diskBounds.getXMin(),ymin), &status);
        }
        if (status) throw_CFITSIO("writeRows() on " + parent.getFilename());
    }

    // If we want buffer to include bounds b, see what the buffer should
    // be. Desired bounds are a region of suggested size centered on
    // those required.
    template <class T>
    Bounds<int> FITSImage<T>::desiredBounds(const Bounds<int> b) const 
    {
        // See what the total buffered area must be
        Bounds<int> required=b;
        if (!purgeImages() && !b) {
            // Neither existing images nor this request need any area.
            return required;
        }

        // Always get full rows
        required.setXMin(diskBounds.getXMin());
        required.setXMax(diskBounds.getXMax());
        imgptr i;
        for (i=imgList.begin(); i!=imgList.end(); ++i)
            required += i->getBounds();

        // Keep the same thing if we already have required region
        if (buffer && bufferBounds.includes(required)) return bufferBounds;

        // Error if asking for outside the image
        if (!diskBounds.includes(required))
            throw FITSError("Request for buffer outside FITS image area");

        long desiredRows = bufferTarget * 1024L * 1024L;
        desiredRows /= (diskBounds.getXMax() - diskBounds.getXMin()+1)
            *sizeof(T);
        // Insure required region is included
        desiredRows = std::max(desiredRows, static_cast<long> (
                required.getYMax() - required.getYMin() + 1));
        // And don't need more than entire image
        desiredRows = std::min(desiredRows,  static_cast<long> (
                diskBounds.getYMax() - diskBounds.getYMin() + 1));

        // Center region, place within bounds
        int firstRow = (required.getYMax() + required.getYMin()
                        - desiredRows) / 2;
        firstRow = std::min(firstRow,required.getYMin());
        firstRow = std::max(firstRow,diskBounds.getYMin());

        int lastRow = firstRow + desiredRows - 1;
        lastRow = std::max(lastRow, required.getYMax());
        lastRow = std::min(lastRow, diskBounds.getYMax());
        firstRow = lastRow - desiredRows + 1;

#ifdef FITSDEBUG
        std::cerr << "desired rows are " << firstRow
            << " to " << lastRow
            << " in FITSImage " << parent.getFilename()
            << " HDU #" << HDUnumber
            << std::endl;
#endif
        return Bounds<int>(diskBounds.getXMin(),diskBounds.getXMax(), firstRow, lastRow);
    }

    //////////////////////////
    //Manipulate buffer so that it includes target region
    template <class T>
    void FITSImage<T>::bufferMustSpan(const Bounds<int> b) const 
    {
        if (!b) throw FITSError("Invalid bounds in bufferMustSpan()");
        assert(diskBounds.includes(b));

        // Choose favored region to buffer.
        Bounds<int> desired=desiredBounds(b);

        // Nothing further if we're already buffered
        if (buffer && bufferBounds.includes(desired)) return;

        long desiredSize = desired.getXMax() - desired.getXMin() + 1;
        desiredSize *= desired.getYMax() - desired.getYMin() + 1;

        // If there is no current buffer, get what's desired
        if (!buffer) {
            allocateBuffer(desired);
            // Read full desired region, ignore previous.
            readRows(desired.getYMin(), desired.getYMax(), false);
            return;
        } else if ( bufferSize < desiredSize || desiredSize < 0.8*bufferSize) {
            // Current buffer is too small or substantially oversize;
            // get a new buffer.
            freeBuffer(); //This flushes any altered data to disk.
            allocateBuffer(desired);
            readRows(desired.getYMin(), desired.getYMax(), false);
            for (imgptr i=imgList.begin(); i!=imgList.end(); ++i) {
                // give new rowPointer array to every active Image's ImageData.
                T** newrpt = makeRowPointers(i->getBounds());
                i->data()->replaceRowPointers(newrpt);
            }
            return;
        } else if (desired & bufferBounds) {
            // buffer is correct size, has some overlap with desired region:

            // flush buffered region that might be written over:
            if (alterableBounds && 
                (desired.getYMax() < alterableBounds.getYMax()) )
                writeRows(desired.getYMax()+1, alterableBounds.getYMax());
            if (alterableBounds && 
                (desired.getYMin() > alterableBounds.getYMin()) )
                writeRows(alterableBounds.getYMin(), desired.getYMin()-1);
            // Shrink alterable region to be within target
            alterableBounds = alterableBounds & desired;

            // then read in regions that are now needed:
            if (desired.getYMin() < bufferBounds.getYMin()) {
                readRows(desired.getYMin(), bufferBounds.getYMin()-1, true);
            }
            if (desired.getYMax() > bufferBounds.getYMax()) {
                readRows(bufferBounds.getYMax()+1, desired.getYMax(), true);
            }
        } else {
            // Buffer is correct size but disjoint from desired data:
            flushData();
            // read new data
            readRows(desired.getYMin(), desired.getYMax(), false);
        }
    }

    // Return pointer to the location of a given coord in the buffer
    template <class T>
    T* FITSImage<T>::bufferLocation(const int xpos, const int ypos) const 
    {
        if (!buffer) return 0;
        int row = ypos - bufferBounds.getYMin() + firstRowOffset;
        row %= bufferRows; //wrap around buffer
        long rowlen = bufferBounds.getXMax() - bufferBounds.getXMin() +1;
        return buffer + row*rowlen + (xpos - bufferBounds.getXMin());
    }

    // Make a new RowPointer array for some image subsection stored in buffer
    template <class T>
    T** FITSImage<T>::makeRowPointers(const Bounds<int> b) const 
    {
        // check bounds
        if (!buffer || !b || !bufferBounds.includes(b)) {
            throw FITSError("makeRowPointers to data not in buffer");
        }
        T** rptr = new T*[b.getYMax()-b.getYMin()+1]; //??catch memory failure
        T** dptr=rptr;
        for (int y=b.getYMin(); y<=b.getYMax(); ++y, ++dptr)
            *dptr = bufferLocation(0,y);
        return rptr - b.getYMin();
    }

    // Make a new Image that is a copy of part of this image.
    template <class T>
    Image<T> FITSImage<T>::extract(Bounds<int> b) const 
    {
        Bounds<int> get = b & diskBounds;
        if (!get) 
            throw FITSError("Attempt to extract outside of FITSImage bounds");
        // Make new Image data structure, which will own all the data arrays.
        ImageData<T>* idata = new ImageData<T>(get);
        loadHeader();
        ImageHeader* ihdr = new ImageHeader(*hptr);

        // Fill the ImageData structure from appropriate source:
        if (get==diskBounds && !bufferBounds.includes(get)) {
            // Read entire image directly to the extracted image, no buffer
            long nelements = (diskBounds.getYMax()-diskBounds.getYMin() + 1);
            nelements *= (diskBounds.getXMax()-diskBounds.getXMin() + 1);

            long firstpix[2]={diskBounds.getXMin(),diskBounds.getYMin()};
            int status(0);
            // Assuming new ImageData stores data contiguously!!! ***
            fits_read_pix(
                parent.getFitsptr(), MatchType<T>(), firstpix, nelements, NULL,
                idata->location(get.getXMin(), get.getYMin()), NULL, &status);
            if (status) 
                throw_CFITSIO("extract() contigous read_pix on " + parent.getFilename());
        } else {
            // Build image from buffered data, copy row by row
            touch();
            bufferMustSpan(get);
            long nbytes=sizeof(T)*(get.getXMax()-get.getXMin()+1);
            for (int y=get.getYMin(); y<=get.getYMax(); ++y) {
                memcpy(
                    idata->location(get.getXMin(), y), bufferLocation(get.getXMin(), y), nbytes);
            }
        }
        return Image<T>(idata,ihdr);
    }

    // Or the entire image:
    template <class T>
    Image<T> FITSImage<T>::extract() const 
    { return extract(diskBounds); }

    template <class T>
    Image<T> FITSImage<T>::use(const Bounds<int> b) 
    {
        Bounds<int> get = b & diskBounds;
        if (!get) 
            throw FITSError("Attempt to use() outside of FITSImage bounds");
        if (!(parent.getFlags() & ReadWrite))
            throw FITSError("Attempt to use() writable Image from ReadOnly FITS");
        // Build image from buffered data, copy row by row
        touch();
        bufferMustSpan(get);
        T** rptrs = makeRowPointers(get);
        // Make new ImageData and ImageHeader structures, then create Image
        ImageData<T>* idata = new ImageData<T>(get, rptrs, false);
        loadHeader();
        Image<T> imgout(idata,hptr, new int(0), hcount);
        // Add to list of in-use images
        imgList.push_front(imgout);
        alterableBounds += get;
        return imgout;
    }

    // Get a const Image back
    template <class T>
    const Image<T> FITSImage<T>::useConst(const Bounds<int> b) const 
    {
        Bounds<int> get = b & diskBounds;
        if (!get) 
            throw FITSError("Attempt to use() outside of FITSImage bounds");
        // Build image from buffered data, copy row by row
        touch();
        bufferMustSpan(get);

        T** rptrs = makeRowPointers(get);
        // Make new ImageData and Image structures
        // Make new ImageData and ImageHeader structures, then create Image
        ImageData<T>* idata = new ImageData<T>(get, rptrs, false);
        loadHeader();
        const Image<T> imgout(idata, hptr, new int(0), hcount);
        // Add to list of in-use images - in this case, don't mark alterableBounds
        imgList.push_front(imgout);
        return imgout;
    }

    // Full images by default:
    template <class T>
    Image<T> FITSImage<T>::use() { return use(diskBounds);}

    template <class T>
    const Image<T> FITSImage<T>::useConst() const { return useConst(diskBounds);}

    // Write data from an external Image into this array.
    // Only the subsection b will be written in, and it must
    // fit within the existing DiskImage.

    template <class T>
    void FITSImage<T>::write(const Image<T> I) {write(I, I.getBounds());}

    template <class T>
    void FITSImage<T>::write(const Image<T> I, const Bounds<int> b) 
    {
        if (!(parent.getFlags() & ReadWrite))
            throw FITSError("Attempt to write() to read-only FITSImage()");
        if (!(I.getBounds().includes(b) && diskBounds.includes(b)))
            throw FITSError("Attempt to write() beyond bounds of image");

        // Choose favored region to buffer.
        Bounds<int> desired=desiredBounds(b);

        long desiredSize = desired.getXMax() - desired.getXMin() + 1;
        desiredSize *= desired.getYMax() - desired.getYMin() + 1;

        int status(0);

        // If buffering this would make something too large, then think
        // about writing directly to the disk file:
        if (desiredSize > bufferTarget) {
            // Move FITS pointer to proper HDU
            fits_movabs_hdu( parent.getFitsptr(), HDUnumber, NULL, &status);
            if (!buffer) {
                // write full image directly to disk.
                // if contiguous this can be a single call:
                // isContiguous needs a check...
                if (I.data()->contiguousData() && b.getXMin()==diskBounds.getXMin()
                    && b.getXMax()==diskBounds.getXMax()) {
                    long nelements=(b.getXMax()-b.getXMin()+1);
                    nelements *= b.getYMax() - b.getYMin() + 1;
                    long firstpix[2]={b.getXMin(), b.getYMin()};
                    fits_write_pix(
                        parent.getFitsptr(), MatchType<T>(), firstpix, nelements, 
                        I.data()->location(b.getXMin(),b.getYMin()), &status);
                } else {
                    //write to CFITSIO row by row
                    long nelements=(b.getXMax()-b.getXMin()+1);
                    long firstpix[2]={b.getXMin(), b.getYMin()};

                    for (int y=b.getYMin(); y<=b.getYMax(); y++) {
                        // write row to disk
                        firstpix[1] = y;
                        fits_write_pix(
                            parent.getFitsptr(), MatchType<T>(), firstpix, nelements, 
                            I.data()->location(b.getXMin(),y), &status);
                    }
                }
            } else {
                // write to disk those parts not in buffer
                // write the rest to present buffer
                touch();
                alterableBounds += (b & bufferBounds); //mark as changed
                long nelements=(b.getXMax()-b.getXMin()+1);
                long nbytes=sizeof(T)*nelements;
                long firstpix[2]={b.getXMin(), b.getYMin()};

                for (int y=b.getYMin(); y<=b.getYMax(); y++) {
                    if (bufferBounds.includes(b.getXMin(), y)) {
                        // write row to buffer
                        memcpy(bufferLocation(b.getXMin(), y),
                               I.data()->location(b.getXMin(), y),
                               nbytes);
                    } else {
                        // write row to disk
                        firstpix[1] = y;
                        fits_write_pix(
                            parent.getFitsptr(), MatchType<T>(), firstpix, nelements, 
                            I.data()->location(b.getXMin(),y), &status);
                    }
                }
            }
        } else {
            // Write the whole thing to a buffer
            touch();
            bufferMustSpan(b);
            alterableBounds += b;
            // note some inefficiency here, we might have just read
            // things we're going to overwrite now. ???
            // copy data to buffer row by row
            long nbytes=sizeof(T)*(b.getXMax()-b.getXMin()+1);
            for (int y=b.getYMin(); y<=b.getYMax(); ++y) {
                memcpy(
                    bufferLocation(b.getXMin(), y), I.data()->location(b.getXMin(), y), nbytes);
            }
        }
        if (status) throw_CFITSIO("write() on " + parent.getFilename());
    }

    template <class T>
    void FITSImage<T>::copy(const Image<T> I) 
    {
        resize(I.getBounds()); //This will check that no Images are outstanding
        write(I);

        // Keep old extension name if there is no new one.
        std::string keyw="EXTNAME";
        std::string oldName;
        bool hasOldName = header()->getValue(keyw, oldName);
        header()->clear();
        (*header())+= *I.header();
        // Restore previous extension name if there is no new one:
        if (hasOldName && !header()->find(keyw)) header()->append(keyw,oldName);
    }

    //note I don't use CFITSIO hdu_copy calls here for copying HDUs because they
    // might not work for copy from one extension to another within image, and they
    // also always put the new one at the end of the extensions.
    template <class T>
    void FITSImage<T>::copy(const FITSImage<T>& fI) 
    {
        resize(fI.getBounds()); //This will check that no Images are outstanding
        if (!fI.isNull()) {
            const Image<T> ii=fI.useConst();
            write(ii);
        }
        std::string keyw="EXTNAME";
        std::string oldName;
        bool hasOldName = header()->getValue(keyw, oldName);
        header()->clear();
        (*header())+= *fI.header();
        // Restore previous extension name if there is no new one:
        if (hasOldName && !header()->find(keyw))
            header()->append(keyw,oldName);
    }

    template <class T>
    void FITSImage<T>::renameExtension(const std::string newext) 
    {
        const std::string keyw="EXTNAME";
        if (header()->find(keyw)) {
            header()->setValue(keyw,newext);
        } else {
            header()->append(new HdrRecord<std::string>(keyw,newext,"FITS Extension name"));
        }
    }

    template class FITSImage<float>;
    template class FITSImage<int>;
    template class FITSImage<short>;

}
