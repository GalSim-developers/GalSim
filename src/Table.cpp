#include "TMV.h"
#include "TMV_SymBand.h"
#include "Table.h"
#include <cmath>
#include <vector>

#include <iostream>

namespace galsim {

    // Look up an index.  Use STL binary search; maybe faster to use
    template<class V, class A>
    int Table<V,A>::upperIndex(const A a) const 
    {
        if (a<_argMin() || a>_argMax()) throw TableOutOfRange();
        // Go directly to index if arguments are regularly spaced.
        if (equalSpaced) {
            int index = int( std::ceil( (a-_argMin()) / dx) );
            if (index >= int(v.size())) --index; // in case of rounding error
            if (index == 0) ++index;
            // check if we need to move ahead or back one step due to rounding errors
            while (a > v[index].arg) ++index;
            while (a < v[index-1].arg) --index;
            return index;
        } else {
            // Warning: I think all of this is correct, but as far as I can tell, 
            // this branch is never tested in our unit tests.  So if someone decides
            // to use Table without equally spaced entries, this function should
            // be checkout it in more detail.
            assert(lastIndex >= 1);
            assert(lastIndex < int(v.size()));

            if ( a < v[lastIndex-1].arg ) {
                assert(lastIndex-2 >= 0);
                // Check to see if the previous one is it.
                if (a >= v[lastIndex-2].arg) return --lastIndex; 
                else {
                    // Look for the entry from 0..lastIndex-1:
                    Entry e(a,0); 
                    iter p = std::upper_bound(v.begin(), v.begin()+lastIndex-1, e);
                    assert(p != v.begin());
                    assert(p != v.begin()+lastIndex-1);
                    lastIndex = p-v.begin();
                    return lastIndex;
                }
            } else if (a > v[lastIndex].arg) {
                assert(lastIndex+1 < int(v.size()));
                // Check to see if the next one is it.
                if (a <= v[lastIndex+1].arg) return ++lastIndex;
                else {
                    // Look for the entry from lastIndex..end
                    Entry e(a,0); 
                    iter p = std::lower_bound(v.begin()+lastIndex+1, v.end(), e);
                    assert(p != v.begin()+lastIndex+1);
                    assert(p != v.end());
                    lastIndex = p-v.begin();
                    return lastIndex;
                }
            } else {
                // Then lastIndex is correct.
                return lastIndex;
            }
        }
    }

    //new element for table.
    template<class V, class A>
    void Table<V,A>::addEntry(const A _arg, const V _val) 
    {
        Entry e(_arg,_val);
        v.push_back(e);
        isReady = false; //re-sort array next time used
    }

    template<class V, class A>
    Table<V,A>::Table(const A* argvec, const V* valvec, int N, interpolant in) :
        iType(in), isReady(false)
    {
        v.reserve(N);
        const A* aptr;
        const V* vptr;
        int i;
        for (i=0, aptr=argvec, vptr=valvec; i<N; i++, aptr++, vptr++) {
            Entry e(*aptr,*vptr);
            v.push_back(e);
        }
    }

    template<class V, class A>
    Table<V,A>::Table(const std::vector<A>& aa, const std::vector<V>& vv, interpolant in) : 
        iType(in), isReady(false)
    {
        v.reserve(aa.size());
        if (vv.size() != aa.size()) 
            throw TableError("input vector lengths don't match");
        typename std::vector<A>::const_iterator aptr=aa.begin();
        typename std::vector<V>::const_iterator vptr=vv.begin();
        for (size_t i=0; i<aa.size(); i++, ++aptr, ++vptr) {
            Entry e(*aptr,*vptr);
            v.push_back(e);
        }
    }

    //lookup & interp. function value. - this one returns 0 out of bounds.
    template<class V, class A>
    V Table<V,A>::operator() (const A a) const 
    {
        setup(); //do any necessary prep
        try {
            int i = upperIndex(a);
            return interpolate(a,i,v,y2);
        } catch (TableOutOfRange) {
            return static_cast<V> (0);
        }
    }

    //lookup & interp. function value.
    template<class V, class A>
    V Table<V,A>::lookup(const A a) const 
    {
        setup();
        int i = upperIndex(a);
        return interpolate(a,i,v,y2);
    }

    template<class V, class A>
    V Table<V,A>::linearInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& )
    {
        A h = v[i].arg - v[i-1].arg;
        A aa = (v[i].arg - a) / h;
        A bb = 1. - aa;
        return aa*v[i-1].val + bb*v[i].val;
    }

    template<class V, class A>
    V Table<V,A>::splineInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& y2)
    {
#if 0
        // Direct calculation saved for comparison:
        A h = v[i].arg - v[i-1].arg;
        A aa = (v[i].arg - a)/h;
        A bb = 1. - aa;
        return aa*v[i-1].val +bb*v[i].val +
            ((aa*aa*aa-aa)*y2[i-1]+(bb*bb*bb-bb)*y2[i]) *
            (h*h)/6.0;
#else
        // Factor out h factors, so only need 1 division by h.
        // Also, use the fact that bb = h-aa to simplify the calculation. 
        A h = v[i].arg - v[i-1].arg;
        A aa = (v[i].arg - a);
        A bb = h-aa;
        return ( aa*v[i-1].val + bb*v[i].val -
                 (1./6.) * aa * bb * ( (aa+h)*y2[i-1] +
                                       (bb+h)*y2[i]) ) / h;
#endif
    }

    template<class V, class A>
    V Table<V,A>::floorInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& )
    {
        // On entry, it is only guaranteed that v[i-1].arg <= a <= v[i].arg.
        // Normally those ='s are ok, but for floor and ceil we make the extra
        // check to see if we should choose the opposite bound.
        if (v[i].arg == a) return v[i].val;
        else return v[i-1].val; 
    }

    template<class V, class A>
    V Table<V,A>::ceilInterpolate(
        A a, int i, const std::vector<Entry>& v, const std::vector<V>& )
    {
        if (v[i-1].arg == a) return v[i-1].val;
        return v[i].val; 
    }

    template<class V, class A>
    void Table<V,A>::read(std::istream& is) 
    {
        std::string line;
        const std::string comments="#;!"; //starts comment
        V vv;
        A aa;
        while (is) {
            getline(is,line);
            // skip leading white space:
            size_t i;
            for (i=0;  isspace(line[i]) && i<line.length(); i++) ;
            // skip line if blank or just comment
            if (i==line.length()) continue;
            if (comments.find(line[i])!=std::string::npos) continue;
            // try reading arg & val from line:
            std::istringstream iss(line);
            iss >> aa >> vv;
            if (iss.fail()) throw TableReadError(line) ;
            addEntry(aa,vv);
        }
    }

    // Do any necessary setup of the table before using
    template<class V, class A>
    void Table<V,A>::setup() const 
    {
        if (isReady) return;

        if (v.size() <= 1) 
            throw TableError("Trying to use a null Table (need at least 2 entries)");

        sortIt();
        lastIndex = 1; // Start back at the beginning for the next search.

        // See if arguments are equally spaced
        // ...within this fractional error:
        const double tolerance = 0.01;
        dx = (v.back().arg - v.front().arg) / (v.size()-1);
        if (dx == 0.) 
            throw TableError("First and last Table entry are equal.");
        equalSpaced = true;
        for (int i=1; i<int(v.size()); i++) {
            if ( std::abs( ((v[i].arg-v[0].arg)/dx - i)) > tolerance) equalSpaced = false;
            if (v[i].arg == v[i-1].arg)
                throw TableError("Table has repeated arguments.");
        }

        switch (iType) {
          case linear:
               interpolate = &Table<V,A>::linearInterpolate;
               break;
          case spline : 
               setupSpline();
               interpolate = &Table<V,A>::splineInterpolate;
               break;
          case floor:
               interpolate = &Table<V,A>::floorInterpolate;
               break;
          case ceil:
               interpolate = &Table<V,A>::ceilInterpolate;
               break;
          default:
               throw TableError("interpolation method not yet implemented");
        }
        isReady = true;
    }

    template <class V, class A>
    void Table<V,A>::setupSpline() const
    {
        /**
         * Calculate the 2nd derivatives of the natural cubic spline.
         *
         * Here we follow the broad procedure outlined in this technical note by Jim
         * Armstrong, freely available online:
         * http://www.algorithmist.net/spline.html
         * 
         * The system we solve is equation [7].  In our adopted notation u_i are the diagonals
         * of the matrix M, and h_i the off-diagonals.  y'' is z_i and the rhs = v_i.
         *
         * For table sizes larger than the fully trivial (2 or 3 elements), we use the
         * symmetric tridiagonal matrix solution capabilities of MJ's TMV library.
         */
        // Set up the 2nd-derivative table for splines
        int n = v.size();
        y2.resize(n);
        // End points 2nd-derivatives zero for natural cubic spline 
        y2[0] = V(0);
        y2[n-1] = V(0);
        // For 3 points second derivative at i=1 is simple
        if (n == 3){

            y2[1] = 3.*((v[2].val - v[1].val) / (v[2].arg - v[1].arg) - 
                        (v[1].val - v[0].val) / (v[1].arg - v[0].arg)) / (v[2].arg - v[0].arg);

        } else {  // For 4 or more points we use the TMV symmetric tridiagonal matrix solver

            tmv::SymBandMatrix<V> M(n-2, 1);
            for (int i=1; i<=n-3; i++){
                M(i, i-1) = v[i+1].arg - v[i].arg;
            }
            tmv::Vector<V> rhs(n-2);
            for (int i=1; i<=n-2; i++){
                M(i-1, i-1) = 2. * (v[i+1].arg - v[i-1].arg);
                rhs(i-1) = 6. * ( (v[i+1].val - v[i].val) / (v[i+1].arg - v[i].arg) -
                                  (v[i].val - v[i-1].val) / (v[i].arg - v[i-1].arg) );
            }
            tmv::Vector<V> solution(n-2);
            solution = rhs / M;   // solve the tridiagonal system of equations
            for (int i=1; i<=n-2; i++){
                y2[i] = solution[i-1];
            }
        }
    }

    template class Table<double,double>;

}
