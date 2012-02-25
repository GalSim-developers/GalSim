
// FITS.h:  manipulations of FITS files, particularly the image
// extensions.  Buffering of data and file handles intended to allow 
// efficient use of a large number of image files/extensions
// simultaneously.

// The FITStypes.h file contains various enumerators and flags to
// be used with FITS files.

// The user will only need the class FITSFile (and the enumerators here).
// The class FITS_fitsiofile represents an opened cfitsio file.  
// FITSFile is basically a handle for FITS_fitsiofile, and
// several FITSFile's may correspond to the same FITS_fitsiofile.
// FITSFile's that refer to the same filename with the same permissions
// are automatically referred to the same FITS_fitsiofile.

// These routines automatically keep track of which and how many
// files are open, and will close the last-used one when access to
// an unopened file is needed and the quota of open files has already
// been reached.  The idea is that the user just opens all of them
// and doesn't worry about it.  If there is frequent access to a large 
// number of files, then there could be hashing.  A FITS_fitsiofile
// has fitsptr=0 if it is not currently opened.

#ifndef GBFITS_H
#define GBFITS_H

#include <list>
#include <string>

#include "Std.h"
#include "FITStypes.h"

namespace sbp {

    // Utility function to throw FITSError and dump CFITSIO message stack.
    void throw_CFITSIO(const std::string m="");

    // And one to flush the error message stack
    void flushFITSErrors();

    // Class representing an opened CFITSIO file.
    class FITS_fitsiofile 
    {
    public:
        FITS_fitsiofile(const std::string& fname, Flags f=ReadOnly);
        ~FITS_fitsiofile();
        // Asking for its CFITSIO pointer reopens the file
        fitsfile* getFitsptr() { useThis(); return fitsptr; }
        std::string getFilename() { return filename; }
        Flags getFlags() { return flags; }
        void flush();
    private:
        std::string filename;
        Flags  flags;
        fitsfile *fitsptr;

        // A list is kept of all open fitsio files.  Most recently
        // accessed is head of list.
        static int filesOpen;
        static std::list<FITS_fitsiofile*> fileQ;
        typedef std::list<FITS_fitsiofile*>::iterator lptr;
        typedef std::list<FITS_fitsiofile*>::reverse_iterator rptr;

        void useThis(); //Make sure file is open, close others if needed
        void openFile(); //Execute the CFITSIO operations to open/close
        void closeFile();
    };

    class FITSFile 
    {
    public:
        FITSFile(const std::string& fname, Flags f=ReadOnly);
        ~FITSFile();
        fitsfile* getFitsptr() { return ffile->getFitsptr(); }
        int HDUCount() ;
        HDUType getHDUType(const int HDUnumber) ;
        HDUType getHDUType(const std::string HDUname);
        HDUType getHDUType(const std::string HDUname, int& HDUnum); //return number too
        std::string getFilename() const { return ffile->getFilename(); }
        void flush() { ffile->flush(); }
        Flags getFlags() const { return ffile->getFlags(); }
    private:
        FITS_fitsiofile* ffile;
        int  *pcount;

        // Keep a list of all opened FITSFile's.  This is so we can check when
        // a new FITSFile is asking for same physical file as an existing one.
        static std::list<FITSFile*> ffList;
        typedef std::list<FITSFile*>::iterator lptr;
        typedef std::list<FITSFile*>::reverse_iterator rptr;
    };

}

#endif
