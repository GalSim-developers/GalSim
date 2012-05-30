
// Classes to represent lookup tables.
// A is the argument class, which must have ordering
// operations, and +-*/ for interpolation.
// D is the value class, which must have + and * operations
// to permit interpolation.

#ifndef TABLE_H
#define TABLE_H

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include "Std.h"

namespace galsim {

    // Exception classes:
    class TableError : public std::runtime_error 
    {
    public:
        TableError(const std::string& m="") : std::runtime_error("Table Error: " +m) {}
    };

    class TableOutOfRange : public TableError 
    {
    public:
        TableOutOfRange() : TableError("Argument out of range") {}
    };

    class TableReadError : public TableError 
    {
    public:
        TableReadError(const std::string& c) : TableError("Data read error for line ->"+c) {}
    };

    // Table element:
    template<class V, class A>
    class TableEntry 
    {
    public:
        TableEntry(A a, V v) : arg(a), val(v) {}
        A arg;
        V val;
        bool operator==(const TableEntry rhs) const { return arg==rhs.arg; }
        bool operator==(const A rhs) const { return arg==rhs; }
        bool operator!=(const TableEntry rhs) const { return arg!=rhs.arg; }
        bool operator!=(const A rhs) const { return arg!=rhs; }
        bool operator>(const TableEntry rhs) const { return arg>rhs.arg; }
        bool operator>(const A rhs) const { return arg>rhs; }
        bool operator<(const TableEntry rhs) const { return arg<rhs.arg; }
        bool operator<(const A rhs) const { return arg<rhs; }
        bool operator<=(const TableEntry rhs) const { return arg<=rhs.arg; }
        bool operator<=(const A rhs) const { return arg>=rhs; }
        bool operator>=(const TableEntry rhs) const { return arg>=rhs.arg; }
        bool operator>=(const A rhs) const { return arg>=rhs; }
    };

    // The Table itself:
    // Derive from Function1d if that has been defined, otherwise not needed.
    template<class V, class A>
    class Table
#ifdef FUNCTION1D_H
    : public Function1d<V,A> 
#endif
    {
    public:
        enum interpolant { linear, spline, floor, ceil };

        //Construct empty table
        Table(interpolant i=linear) : iType(i), isReady(false) {} 

        //Table from two arrays:
        Table(const A* argvec, const V* valvec, int N, interpolant in=linear);
        Table(const std::vector<A>& a, const std::vector<V>& v, interpolant in=linear);

        Table(std::istream& is, interpolant in=linear) : iType(in), isReady(false)
        { read(is); }

        void clear() { v.clear(); isReady=false; }
        void read(std::istream& is);

        void addEntry(const A a, const V v); //new element for table.

        V operator() (const A a) const; //lookup & interp. function value.

        V lookup(const A a) const; //interp, but exception if beyond bounds

        int size() const { return v.size(); } //size of table

        //Smallest argument
        A argMin() const 
        { 
            setup();
            return _argMin();
        }
        //Largest argument
        A argMax() const 
        { 
            setup();
            return _argMax();
        }

        template <class T>
        void TransformVal(T& xfrm) 
        {
            for (iter p=v.begin(); p!=v.end(); ++p) p->val = xfrm(p->arg, p->val);
            isReady=false; setup();
        }

        template <class T>
        void TransformArg(T& xfrm) 
        {
            for (iter p=v.begin(); p!=v.end(); ++p)
                p->arg = xfrm(p->arg, p->val);
            isReady=false; setup();
        }

        void dump() const 
        {
            setup(); 
            for (citer p=v.begin(); p!=v.end(); ++p) 
                std::cout << p->arg << " " << p->val << std::endl; 
        }

    private:
        typedef TableEntry<V,A> Entry;
        typedef typename std::vector<Entry>::const_iterator citer;
        typedef typename std::vector<Entry>::iterator iter;

        interpolant iType;
        mutable bool isReady; //Flag if table has been prepped.
        mutable bool equalSpaced; //Flag set if arguments are nearly equally spaced.
        mutable A dx; // ...in which case this is argument interval
        mutable int lastIndex; //Index for last lookup into table.

        mutable std::vector<Entry> v;
        mutable std::vector<V> y2; //vector of 2nd derivs for spline

        // Private versions that don't check for a null table:
        A _argMin() const { return v.front().arg; }
        A _argMax() const { return v.back().arg; }

        //get index to 1st element >= argument.  Can throw the exception here.
        int upperIndex(const A a) const;

        void sortIt() const { std::sort(v.begin(), v.end()); }
        void setup() const; //Do any necessary preparation;
        void setupSpline() const; // Calculate the y2 vector

        //Interpolate value btwn p & --p:
        mutable V (*interpolate)(A a, int i, const std::vector<Entry>& v,
                                 const std::vector<V>& y2);

        static V linearInterpolate(A a, int i, const std::vector<Entry>& v,
                                   const std::vector<V>& y2);
        static V splineInterpolate(A a, int i, const std::vector<Entry>& v,
                                   const std::vector<V>& y2);
        static V floorInterpolate(A a, int i, const std::vector<Entry>& v,
                                  const std::vector<V>& y2);
        static V ceilInterpolate(A a, int i, const std::vector<Entry>& v,
                                 const std::vector<V>& y2);

    };

}

#endif
