
// Parameter file/structure stuff

#ifndef PSET_H
#define PSET_H

#include <string>
#include <list>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#include "StringStuff.h"

namespace sbp {

    // Parameter setting class
    // Pset is a collection of parameters attached to variables in the code.

    // Pset class can set any of its variables by reading from an input stream.
    // Each line of the input stream is assumed to have the following format:
    // <key> [=] <value> [comment]
    // key is the name of the parameter
    // optional = sign follows
    // value is ASCII representation of the value
    // comment is ; or # character followed by stuff to ignore.
    // 
    // Blank lines are ignored, as are lines starting with comment characters.
    // White space always allowed between elements of a line.
    // String values can be enclosed in quotes, otherwise everything on
    //   the line up to comment delimiter is assumed to be part of string.
    // Trailing white space is stripped from non-quoted strings.

    //********************************************************************
    // Exception classes:
    //********************************************************************

    class PsetError : public std::runtime_error 
    {
    public:
        PsetError(const std::string& m="") : std::runtime_error("Pset error: " + m) {};
    };

    class PsetNoDefault : public PsetError 
    {
    public:
        PsetNoDefault(const std::string& key="") :
            PsetError("No default for keyword ->" + key + "<-") {};
    };

    class PsetOutOfBounds : public PsetError 
    {
    public:
        PsetOutOfBounds() : PsetError("Value out of bounds") {};
        PsetOutOfBounds(const std::string& key, const std::string& val) : 
            PsetError("Keyword ->" + key + "<- has value ->" 
                      + val + "<- out of bounds") {};
    };

    class PsetUnboundedQuote : public PsetError 
    {
    public:
        PsetUnboundedQuote() : PsetError("Quoted string not closed") {};
        PsetUnboundedQuote(const std::string& line) :
            PsetError("Quoted string not closed: " + line) {};
    };

    class PsetKeywordNotFound : public PsetError 
    {
    public:
        PsetKeywordNotFound(const std::string& key) : 
            PsetError("Keyword ->" + key + "<- not found") {};
    };

    class PsetFormatError : public PsetError 
    {
    public:
        PsetFormatError(const std::string& key, const std::string& val) :
            PsetError("Format error for value ->" + val 
                      + "<- of keyword ->" + key + "<-") {};
    };

    //********************************************************************
    // Pset Member classes
    //********************************************************************

    // Base class for a parameter set member.  Also works as 
    // class with no value.
    class PsetMember 
    {
    public:
        enum flags { 
            hasDefault=1, hasLowerBound=2, openLowerBound=4, 
            hasUpperBound=8, openUpperBound=16
        };

        std::string keyword;
        PsetMember(const char *k, const int _f=0, const char *c="") : 
            keyword(k), f(flags(_f)), comment(c) {}

        virtual ~PsetMember() {}

        virtual void setValue(const std::string& valstring) {}
        virtual void setDefault() {}

        virtual void dump(std::ostream& os) const 
        {
            os << std::setw(12) << keyword.c_str() << "= " 
                << std::setw(20) << ""
                << ";" << comment  << std::endl;
        }

        flags getFlags() const { return f; }
        virtual PsetMember* clone() { return new PsetMember(*this); }

    protected:
        flags f;
        std::string comment;
    };

    // General type:
    template <class T>
    class PsetMem : public PsetMember 
    {
    public:
        PsetMem(const char *k, T* vptr, const int _f=0, const char *c="",
                const T& d=0, const T& lo=0, const T& up=0) : 
            PsetMember(k, _f, c), valueptr(vptr), defval(d), lowerBound(lo), upperBound(up) {}

        virtual PsetMem* clone() { return new PsetMem(*this); }

        void setValue(const std::string& valstring);
        void setDefault() { *valueptr=defval; }

        virtual void dump(std::ostream& os) const 
        {
            os.setf(std::ios::left, std::ios::adjustfield);
            os << std::setw(12) << keyword.c_str() << "= " 
                << std::setw(20) << *valueptr
                << ";" << comment  << std::endl;
        }

    private:
        T* valueptr;
        T defval;
        T lowerBound;
        T upperBound;
    };

    // specialization for a string:
    template <>
    class PsetMem<std::string> : public PsetMember 
    {
    public:
        PsetMem(
            const char *k, std::string* vptr, 
            const int _f=0, const char *c="", const char *d="") : 
            PsetMember(k, _f, c), valueptr(vptr), defval(d) 
        {}

        virtual PsetMem* clone() { return new PsetMem(*this); }

        void setValue(const std::string& valstring) 
        {
            if (valstring.length()>0) *valueptr=valstring;
            else if (f & hasDefault) setDefault(); 
            else throw PsetNoDefault(keyword); 
        }

        void setDefault() { *valueptr=defval; }

        virtual void dump(std::ostream& os) const 
        {
            os.setf(std::ios::left, std::ios::adjustfield);
            os << std::setw(12) << keyword.c_str() << "= " //??? not doing what it should...
                << std::setw(20) << valueptr->c_str() 
                << ";" << comment  << std::endl;
        }

    private:
        std::string *valueptr;
        std::string defval;
    };

    // specialization for a boolean:
    template <>
    class PsetMem<bool> : public PsetMember 
    {
    public:
        PsetMem(const char *k, bool* vptr, const int _f=0, const char *c="",
                const bool d=false) : 
            PsetMember(k, _f, c), valueptr(vptr), defval(d) {}

        virtual PsetMem* clone() { return new PsetMem(*this); }

        void setValue(const std::string& valstring) 
        {
            if (valstring.empty()) {
                if (f & hasDefault) setDefault(); 
                else throw PsetNoDefault(keyword);
            }
            if (valstring=="1" || nocaseEqual(valstring, "t") || 
                nocaseEqual(valstring, "true")) {
                *valueptr = true;
            } else if (valstring=="0" || nocaseEqual(valstring, "f") || 
                       nocaseEqual(valstring, "false")) {
                *valueptr = false;
            } else {
                throw PsetFormatError(keyword,valstring);
            }
        }

        void setDefault() { *valueptr=defval; }
        virtual void dump(std::ostream& os) const 
        {
            os.setf(std::ios::left, std::ios::adjustfield);
            os << std::setw(12) << keyword.c_str() << "= " 
                << std::setw(20) << (*valueptr ? "T " : "F ")
                << ";" << comment  << std::endl;
        }

    private:
        bool *valueptr;
        bool defval;
    };

    // function class for testing keyword equivalence.
    // Case will not matter.
    class PsetKeywordTest 
    {
    public:
        PsetKeywordTest(const std::string& r) : rhs() 
        {
            for (std::string::const_iterator p=r.begin(); p!=r.end(); ++p)
                rhs+=toupper(*p);
        }
        bool operator()(const PsetMember *ps) const 
        {
            if (ps->keyword.length()!=rhs.length()) return false;
            std::string::const_iterator pr=rhs.begin();
            std::string::const_iterator pl=ps->keyword.begin();
            while (pr!=rhs.end()) {
                if ( static_cast<char> (toupper(*pl)) != *pr) return false;
                ++pl; ++pr;
            }
            return true;
        }
    private:
        std::string rhs;
    };

    template <class T>
    void PsetMem<T>::setValue(const std::string& valstring) 
    {
        if (valstring.length()==0) {
            if (f & hasDefault) { setDefault(); return; }
            else throw PsetNoDefault(keyword);
        }
        std::istringstream is(valstring);
        T thisval;
        is >> thisval;
        if (is.fail()) throw PsetFormatError(keyword,valstring);
        if (f & hasLowerBound) {
            if ( thisval < lowerBound 
                 || ( (f&openLowerBound) && thisval == lowerBound) )
                throw PsetOutOfBounds(keyword, valstring);
        }
        if (f & hasUpperBound) {
            if ( thisval > upperBound 
                 || ( (f&openUpperBound) && thisval == upperBound) )
                throw PsetOutOfBounds(keyword, valstring);
        }
        *valueptr = thisval;
    }

    //********************************************************************
    // Pset is a list of members:
    //********************************************************************

    //A list of PsetMembers comprises a Pset.
    class Pset 
    {
    public:
        Pset() : l() {};
        Pset(Pset& rhs) : l() { copyList(rhs.l); }
        ~Pset() { killList(); }

        template <class T>
        void addMember(
            const char *k, T* vptr, const int _f=0, const char *c="",
            const T& d=0, const T& lo=0, const T& up=0)  
        {
            l.push_back( 
                static_cast<PsetMember*> ( new PsetMem<T>(k, vptr, _f, c, d, lo, up) ) ); 
        }

        // string specialization; note set up to use string literals as input:
        void addMember(
            const char *k, std::string* vptr, const int _f=0, const char *c="", const char *d="")  
        {
            l.push_back( 
                static_cast<PsetMember*> ( new PsetMem<std::string>(k, vptr, _f, c, d) ) ); 
        }

        // bool specialization - no upper or lower
        void addMember(
            const char *k, bool* vptr, const int _f=0, const char *c="", const bool d=false)  
        {
            l.push_back( 
                static_cast<PsetMember*> ( new PsetMem<bool>(k, vptr, _f, c, d) ) ); 
        }

        void addMemberNoValue(const char *k, const int _f=0, const char *c="");

        void setDefault() 
        { for (iter p=l.begin(); p!=l.end(); ++p) (*p)->setDefault(); }

        void dump(std::ostream& os) const 
        { for (citer p=l.begin(); p!=l.end(); ++p) (*p)->dump(os); }

        void setKeyValue(const std::string& keyword, const std::string& value) {
            iter p=find_if(l.begin(), l.end(), PsetKeywordTest(keyword));
            if (p==l.end()) throw PsetKeywordNotFound(keyword);
            (*p)->setValue(value);
        }

        // read many lines of specification, glean all useful ones.
        // throw KeywordNotFound only after going through all lines.
        void setStream(std::istream& is) 
        {
            std::string line, keyword, value;
            PsetKeywordNotFound ex("none");
            bool thrown=false;
            while (is) {
                getline(is,line);
                if (read_keyvalue(line, keyword, value)) {
                    try { 
                        setKeyValue(keyword,value);
                    } catch (PsetKeywordNotFound e) {
                        ex = e; thrown=true;
                    }
                }
            }
            if (thrown) throw ex;
        }

    private:
        std::list<PsetMember*> l;
        typedef std::list<PsetMember*>::iterator iter;
        typedef std::list<PsetMember*>::const_iterator citer;
        void killList() 
        { 
            for (iter p=l.begin(); p!=l.end(); ++p)
                delete *p; 
            l.clear();
        }

        void copyList( std::list<PsetMember*>& rhs) 
        { 
            for (citer p=rhs.begin(); p!=rhs.end(); ++p)
                l.push_back( (*p)->clone());
        }

        static bool read_keyvalue(const std::string& in, std::string& keyword, std::string& value);
    };

}

#endif
