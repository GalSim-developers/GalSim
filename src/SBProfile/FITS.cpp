
#include "FITS.h"
#include <algorithm>
#include <iostream>
#include <cstring> 

namespace sbp {

    int FITS_fitsiofile::filesOpen=0;

    std::list<FITS_fitsiofile*> FITS_fitsiofile::fileQ;

    std::list<FITSFile*> FITSFile::ffList;

    // Predicate for filename hunting:
    bool FilenameMatch(const FITSFile* f1, const FITSFile* f2) 
    { return f1->getFilename()==f2->getFilename(); }

    // Function to unroll the CFITSIO error stack into exception string
    void throw_CFITSIO(const std::string m1) 
    {
        std::string m= m1 + " CFITSIO Error: ";
        char ebuff[FLEN_ERRMSG];
        while (fits_read_errmsg(ebuff)) {m+= "\n\t"; m+=ebuff;}
        // Do not throw if we are already unwinding stack from
        // another thrown exception:
        if (std::uncaught_exception()) {
            std::cerr << "During exception processing: " << m << std::endl;
        } else {
            throw FITSError(m);
        }
    }

    void flushFITSErrors() 
    { fits_clear_errmsg(); }

    // FITSFile constructor:  open the file to test for existence
    FITSFile::FITSFile(const std::string& fname, Flags f) 
    {
        // Is this a duplicate filename?
        lptr twin;
        for (twin = ffList.begin(); 
             twin!=ffList.end() && (*twin)->getFilename()!=fname; ++twin);
        if (twin==ffList.end()) {
            // Unique filename:  open new CFITSIO handle
            ffile = new FITS_fitsiofile(fname, f);
            pcount = new int(1);
            ffList.push_front(this);
        } else {
            // Duplicate filename:
            if (f & Overwrite) {
                throw FITSError(
                    "Overwrite specified on FITS file " + fname + " that is already open.");
            } else if (f != (*twin)->ffile->getFlags()) {
                // differing flags, we should just open a new copy and let
                // CFITSIO do the confusing stuff
                ffile = new FITS_fitsiofile(fname, f);
                pcount = new int(1);
                ffList.push_front(this);
            } else {
                // true duplicate, just add new reference to same handle
                ffile = (*twin)->ffile;
                pcount = (*twin)->pcount;
                (*pcount)++;
                ffList.push_front(this);
            }
        }
    }

    FITSFile::~FITSFile() 
    {
        // Find and remove ourself from the fflist:
        lptr me = find(ffList.begin(), ffList.end(), this);
        // Don't throw another exception if already unwinding one.
        if (me==ffList.end() && !std::uncaught_exception()) 
            throw FITSError("Did not find open object in ffList!");
        ffList.erase(me);
        (*pcount)--;
        if (*pcount==0) {
            delete ffile;
            delete pcount;
        }
    }

    //////////////////////////////////////////////////////////////////
    // Open/close the CFITSIO files themselves.  Automatically
    // keep number of open files under some limit.
    //////////////////////////////////////////////////////////////////

    // FITS_fitsiofile constructor:  open the file to test for existence
    FITS_fitsiofile::FITS_fitsiofile(const std::string& fname, Flags f) : 
        filename(fname), flags(f), fitsptr(0) 
    { useThis(); }

    FITS_fitsiofile::~FITS_fitsiofile() 
    {
        // Find and remove ourself from the fileQ:
        lptr me = find(fileQ.begin(), fileQ.end(), this);
        if (me==fileQ.end() && !std::uncaught_exception()) 
            throw FITSError("Did not find open file in fileQ!");
        fileQ.erase(me);
        // close file with CFITSIO
        closeFile();
    }

    // Open a file if it's not already open.  Close the open file used longest
    // ago if there will be too many files open.
    // Note that CFITSIO is prepared to deal with multiple calls to open
    // the same actual file, so we don't have to worry about that case.
    void FITS_fitsiofile::useThis() 
    {
        static int hashReportInterval=1000; //issue warning for file hashing.
        // set to zero to disable this reporting.
        static int hashCount=0; //count # times need to close a file
        if (fitsptr) {
            // already open; push to front of Q for recent usage
            if (*fileQ.begin() != this) {
                lptr me = find(fileQ.begin(), fileQ.end(), this);
                fileQ.erase(me); fileQ.push_front(this);
            }
        } else {
            // Desired file is not open.  Do we need to close another first?
            if (filesOpen >= MAX_FITS_FILES_OPEN) {
                // ??? wait for an open to fail to do this???
                // ??? or a while loop looking for the CFITSIO status code???
                // choose the one to close - last in queue that's open
                rptr oldest;
                for (oldest=fileQ.rbegin(); 
                     oldest!=fileQ.rend() && (*oldest)->fitsptr==0;
                     oldest++) ;
                if (oldest==fileQ.rend())
                    throw FITSError("Screwup in finding stale file to close");
                (*oldest)->closeFile();
                hashCount++;
                if (hashReportInterval!=0 && (hashCount%hashReportInterval)==0)
                    std::cerr << "WARNING: possible FITS_fitsiofile hashing, "
                        << hashCount
                        << " files closed so far"
                        << std::endl;
            }
            openFile();
            // If successfully opened, put at front of usage queue.
            lptr me = find(fileQ.begin(), fileQ.end(), this);
            if (me!=fileQ.end()) fileQ.erase(me);
            fileQ.push_front(this);
        }
    }

    void FITS_fitsiofile::openFile() 
    {
        int status=0;
        std::string openname;

        if (flags & ReadWrite) {
            // Open for writing - if not overwriting, try opening existing
            // image first.
            if (!(flags &  Overwrite)) {
#ifdef FITSDEBUG
                std::cerr << "fits_open_file " << filename << " R/W" << std::endl;
#endif
                fits_open_file(&fitsptr, filename.c_str(), READWRITE, &status);
                if (status==0) { filesOpen++; return;}
                status = 0;
                fits_clear_errmsg();
            }
            // Try creating new file; overwrite if desired
            if (flags &  Overwrite) {
                openname = "!" + filename;
            } else {
                openname = filename;
            }
#ifdef FITSDEBUG
            std::cerr << "fits_create_file " << filename << std::endl;
#endif
            fits_create_file(&fitsptr, openname.c_str(), &status);
            if (status!=0) {
                char ebuff[256];
                fits_read_errmsg(ebuff);
                std::string m(ebuff);
                throw FITSCantOpen(openname, m);
            }
            // On success, clear the Overwrite flag
            flags = Flags(flags & ~Overwrite);
        } else {
            // Open for reading, error if file does not exist
#ifdef FITSDEBUG
            std::cerr << "fits_open_file " << filename << " RO " << std::endl;
#endif
            fits_open_file(&fitsptr, filename.c_str(), READONLY, &status);
            if (status!=0) {
                char ebuff[256];
                fits_read_errmsg(ebuff);
                std::string m(ebuff);
                throw FITSCantOpen(filename, m);
            }
        }
        filesOpen++;
    }

    void FITS_fitsiofile::closeFile() 
    {
        if (fitsptr==0) return;
        int status=0;
#ifdef FITSDEBUG
        std::cerr << "fits_close_file " << filename << std::endl;
#endif
        fits_close_file(fitsptr, &status);
        if (status && !std::uncaught_exception()) 
            if (status) throw_CFITSIO("closeFile() on " 
                                      + getFilename());
        fitsptr = 0;
        filesOpen--;
    }

    void FITS_fitsiofile::flush() 
    {
        if (fitsptr==0) return; //No need to flush if no CFITSIO buffer
        int status=0;
        fits_flush_file(fitsptr, &status);
        if (status) throw_CFITSIO("flushing " + filename);
    }

    /////////////////////////////////////////////////////////////////////////
    // Access FITSFile information
    /////////////////////////////////////////////////////////////////////////

    int FITSFile::HDUCount() 
    {
        int status=0, count;
        fits_get_num_hdus(getFitsptr(), &count, &status);
        if (status) throw_CFITSIO("HDUCount() on " + getFilename());
        return count;
    }

    HDUType FITSFile::getHDUType(const int HDUnumber) 
    {
        int status=0, retval;
        fits_movabs_hdu(getFitsptr(), HDUnumber, &retval, &status);
        if (status) throw_CFITSIO("getHDUType() on " + getFilename());
        return HDUType(retval);
    }

    HDUType FITSFile::getHDUType(const std::string HDUname, int& HDUnum) 
    {
        int status=0, retval;
        char ff[80];
        strncpy(ff, HDUname.c_str(), sizeof(ff)); ff[sizeof(ff)-1]=0;
        fits_movnam_hdu(getFitsptr(), ANY_HDU, ff, 0, &status);
        if (status==BAD_HDU_NUM) {
            //name does not exist, return 0
            HDUnum = 0;
            fits_clear_errmsg();
            return HDUAny;
        }
        fits_get_hdu_num(getFitsptr(), &HDUnum);
        fits_get_hdu_type(getFitsptr(), &retval, &status);
        if (status) throw_CFITSIO("getHDUType() by name on " + getFilename());
        return HDUType(retval);
    }
    
    HDUType FITSFile::getHDUType(const std::string HDUname) 
    {
        int junk;
        return getHDUType(HDUname, junk);
    }

}

