
// Definitions, includes, and constants for FITS files and CFITSIO.

#ifndef FITSTYPES_H
#define FITSTYPES_H
#include <typeinfo>
#include <stdexcept>
#include <complex>
#include "fitsio.h"

#include "Std.h"

namespace galsim {

    //const int MAX_FITS_FILES_OPEN=NIOBUF;
    const int MAX_FITS_FILES_OPEN=40;

    const bool CLongIsFITSLongLong = (sizeof(long)*8==LONGLONG_IMG);
    const bool CIntIsFITSLongLong = (sizeof(int)*8==LONGLONG_IMG);
    const bool CIntIsFITSLong = (sizeof(int)*8==LONG_IMG);
    const bool CIntIsFITSShort = (sizeof(int)*8==SHORT_IMG);

    class FITSError : public std::runtime_error
    {
    public: 
        FITSError(const std::string& m="") : std::runtime_error("FITS Error: " + m) {}
    };

    class FITSCantOpen : public FITSError 
    {
    public:
        FITSCantOpen(const std::string& fname, const std::string& m="") : 
            FITSError("Cannot open FITS file " + fname + ": " + m) {}
    };

    // Make enumerators out of the CFITSIO constants we'll need
    class Flags 
    {
    private:
        int f;
    public:
        explicit Flags(int i) : f(i) {}
        Flags(const Flags& rhs) : f(rhs.f) {}

        const Flags& operator=(const Flags& rhs) { f=rhs.f; return *this; }

        Flags operator+(const Flags rhs) const { return Flags(f+rhs.f); }
        Flags operator|(const Flags rhs) const { return Flags(f|rhs.f); }
        Flags operator&(const Flags rhs) const { return Flags(f&rhs.f); }
        Flags operator~() const { return Flags(~f); }

        bool operator==(const Flags rhs) const { return f == rhs.f; }
        bool operator!=(const Flags rhs) const { return f != rhs.f; }

        operator bool() const { return f!=0; }
        int getInt() const { return f; }
    };

    const Flags ReadOnly(0);
    const Flags ReadWrite(1);
    const Flags Overwrite(2);
    const Flags CreateImage(4);

    enum HDUType { 
        HDUImage=IMAGE_HDU, 
        HDUAsciiTable=ASCII_TBL,
        HDUBinTable=BINARY_TBL,
        HDUAny = ANY_HDU 
    };

    // CFITSIO data types 
    enum DataType {
        Tnull, 
        Tbit = TBIT, 
        Tbyte = TBYTE, 
        Tlogical = TLOGICAL, 
        Tstring = TSTRING, 
        Tushort = TUSHORT, 
        Tshort = TSHORT,
        Tuint = TUINT,
        Tint = TINT,
        Tulong = TULONG,
        Tlong = TLONG,
        Tlonglong = TLONGLONG,
        Tfloat = TFLOAT,
        Tdouble = TDOUBLE,
        Tcomplex = TCOMPLEX,
        Tdblcomplex=TDBLCOMPLEX
    };

    // Handy function stolen from CCFits, uses RTTI to give back the 
    // enum code for a data type.
    template <typename T>
    inline DataType MatchType()
    {
        if ( typeid(T) == typeid(double) ) return Tdouble;
        if ( typeid(T) == typeid(float) ) return Tfloat;
        if ( typeid(T) == typeid(std::complex<float>) ) return Tcomplex;
        if ( typeid(T) == typeid(std::complex<double>) ) return Tdblcomplex;
        if ( typeid(T) == typeid(std::string) ) return Tstring;
        if ( typeid(T) == typeid(int) )  return Tint;
        if ( typeid(T) == typeid(unsigned int) ) return Tuint;
        if ( typeid(T) == typeid(short) ) return Tshort;
        if ( typeid(T) == typeid(unsigned short) ) return Tushort;
        if ( typeid(T) == typeid(bool) ) return Tlogical;
        if ( typeid(T) == typeid(unsigned char) ) return Tbyte;
        // Check definition of long int against FITS standards.  Note
        // not doing this for unsigned long 
        if ( typeid(T) == typeid(long) ) {
            if (CLongIsFITSLongLong) return Tlonglong;
            else return Tlong;
        }
        if ( typeid(T) == typeid(unsigned long) ) return Tulong;
        throw FITSError("Invalid data type for FITS Image I/O");    
    }

    // Convert to/from BITPIX keywords to data types
    inline DataType Bitpix_to_DataType(const int bitpix) 
    {
        if (bitpix==BYTE_IMG)   return Tbyte;
        if (bitpix==SHORT_IMG)  return Tshort;
        if (bitpix==LONG_IMG)   return Tint;
        if (bitpix==FLOAT_IMG)  return Tfloat;
        if (bitpix==DOUBLE_IMG) return Tdouble;
        if (bitpix==USHORT_IMG) return Tushort;
        if (bitpix==ULONG_IMG)  return Tulong;
        if (bitpix==LONGLONG_IMG) return Tlonglong;
        throw FITSError("Unknown BITPIX value");
    }

    inline int DataType_to_Bitpix(const DataType dt) 
    {
        if (dt==Tbyte) return BYTE_IMG;
        if (dt==Tshort) return SHORT_IMG;
        if (dt==Tlong) return LONG_IMG;
        if (dt==Tlonglong) return LONGLONG_IMG;
        if (dt==Tfloat) return FLOAT_IMG;
        if (dt==Tdouble) return DOUBLE_IMG;
        if (dt==Tushort) return USHORT_IMG;
        if (dt==Tulong) return ULONG_IMG;

        if (dt==Tint && CIntIsFITSShort) return SHORT_IMG;
        if (dt==Tint && CIntIsFITSLong) return LONG_IMG;
        if (dt==Tint && CIntIsFITSLongLong) return LONGLONG_IMG;
        if (dt==Tuint && (sizeof(uint)==USHORT_IMG/8)) return USHORT_IMG;
        if (dt==Tuint && (sizeof(uint)==ULONG_IMG/8)) return ULONG_IMG;

        throw FITSError("Datatype cannot be converted to BITPIX");
    }

}

#endif  //FITSTYPES_H
