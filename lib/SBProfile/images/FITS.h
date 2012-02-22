// 	$Id: FITS.h,v 1.15 2011/07/20 17:38:51 dgru Exp $	
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
#include <cstring>
#include "Std.h"
using std::list;

#include "FITStypes.h"

namespace FITS {

  // Utility function to throw FITSError and dump CFITSIO message stack.
  void throw_CFITSIO(const string m="");

  // And one to flush the error message stack
  void flushFITSErrors();

  // Class representing an opened CFITSIO file.
  class FITS_fitsiofile {
  public:
    FITS_fitsiofile(const string& fname, Flags f=ReadOnly);
    ~FITS_fitsiofile();
    // Asking for its CFITSIO pointer reopens the file
    fitsfile* getFitsptr() {useThis(); return fitsptr;}
    string getFilename() {return filename;}
    Flags getFlags() {return flags;}
    void flush();
  private:
    string filename;
    Flags  flags;
    fitsfile *fitsptr;

    // A list is kept of all open fitsio files.  Most recently
    // accessed is head of list.
    static int filesOpen;
    static list<FITS_fitsiofile*> fileQ;
    typedef list<FITS_fitsiofile*>::iterator lptr;
    typedef list<FITS_fitsiofile*>::reverse_iterator rptr;

    void useThis();	//Make sure file is open, close others if needed
    void openFile();	//Execute the CFITSIO operations to open/close
    void closeFile();
  };

  class FITSFile {
  public:
    FITSFile(const string& fname, Flags f=ReadOnly);
    ~FITSFile();
    fitsfile* getFitsptr() {return ffile->getFitsptr();}
    int HDUCount() ;
    HDUType getHDUType(const int HDUnumber) ;
    HDUType getHDUType(const string HDUname);
    HDUType getHDUType(const string HDUname, int &HDUnum); //return number too
    string getFilename() const {return ffile->getFilename();}
    void flush() {ffile->flush();}
    Flags getFlags() const {return ffile->getFlags();}
  private:
    FITS_fitsiofile* ffile;
    int  *pcount;

    // Keep a list of all opened FITSFile's.  This is so we can check when
    // a new FITSFile is asking for same physical file as an existing one.
    static list<FITSFile*> ffList;
    typedef list<FITSFile*>::iterator lptr;
    typedef list<FITSFile*>::reverse_iterator rptr;
  };

} // end namespace FITS
using FITS::FITSFile;

#endif
