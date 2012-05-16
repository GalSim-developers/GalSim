#include "TMV.h"
#include "TMV_SymBand.h"
#include "Table.h"
#include <cmath>
#include <vector>

namespace galsim {

    // Look up an index.  Use STL binary search; maybe faster to use
    template<class V, class A>
    typename Table<V,A>::iter Table<V,A>::upperIndex(const A a) const 
    {
        setup();
        if (v.size()==0 || a<argMin())  throw TableOutOfRange();
        // Go directly to index if arguments are regularly spaced.
        if (equalSpaced) {
            int index = static_cast<int> ( std::ceil( (a-argMin()) / dx) );
            if (index >= int(v.size())) throw TableOutOfRange();
            // check if we need to move ahead or back one step due to rounding errors
            if (a > v[index].arg) { 
                ++index;
                if (index >= int(v.size())) throw TableOutOfRange();
            } else if (index>0 && a<v[index-1].arg) {
                --index;
            }
            lastIndex = index;  //interpolate() uses lastIndex
            return v.begin() + index;
        }
        // First see if the previous index is still ok
        if (lastIndex>0 && lastIndex<int(v.size())) {
            iter p = (v.begin()+lastIndex);
            if ( (p->arg >= a) && (a > (p-1)->arg) ) return p;
        }

        // This STL algorithm uses binary search to get 1st element >= ours.
        Entry e(a,0); 
        iter p = std::lower_bound(v.begin(), v.end(), e);
        // bounds check
        if (p==v.end()) throw TableOutOfRange();
        lastIndex = p-v.begin();
        return p;
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
        v(), iType(in), y2() 
    {
        v.reserve(N);
        const A* aptr;
        const V* vptr;
        int i;
        for (i=0, aptr=argvec, vptr=valvec; i<N; i++, aptr++, vptr++) {
            Entry e(*aptr,*vptr);
            v.push_back(e);
        }
        isReady = false; //set flag for setup next use.
    }

    template<class V, class A>
    Table<V,A>::Table(const std::vector<A>& aa, const std::vector<V>& vv, interpolant in) : 
        v(), iType(in), y2() 
    {
        v.reserve(aa.size());
        if (vv.size()<aa.size()) 
            throw TableError("input vector lengths don't match");
        typename std::vector<A>::const_iterator aptr=aa.begin();
        typename std::vector<V>::const_iterator vptr=vv.begin();
        for (size_t i=0; i<aa.size(); i++, ++aptr, ++vptr) {
            Entry e(*aptr,*vptr);
            v.push_back(e);
        }
        isReady = false;
    }

    //lookup & interp. function value. - this one returns 0 out of bounds.
    template<class V, class A>
    V Table<V,A>::operator() (const A a) const 
    {
        try {
            citer p1(upperIndex(a));
            return interpolate(a,p1);
        } catch (TableOutOfRange) {
            return static_cast<V> (0);
        }
    }

    //lookup & interp. function value.
    template<class V, class A>
    V Table<V,A>::lookup(const A a) const 
    {
        citer p1(upperIndex(a));
        return interpolate(a,p1);
    }

    template<class V, class A>
    V Table<V,A>::interpolate(const A a, const citer p1) const 
    {
        setup(); //do any necessary prep
        // First case when there is for single-point table
        if (v.size()==1) {
            return p1->val;
        } else if (iType==linear) {
            if (p1==v.begin())  return p1->val;
            citer p0 = p1-1;
            if (p1->arg==p0->arg) return p0->val;
            double frac=(a - p0->arg) / (p1->arg - p0->arg);
            return frac*p1->val + (1-frac) * p0->val;
        } else if (iType==spline) {
            if (p1==v.begin())  return p1->val;
            citer p0 = p1-1;
            A h = p1->arg-p0->arg;
            A aa=(p1->arg - a)/h;
            A bb=(a - p0->arg)/h;
            return aa*p0->val +bb*p1->val +
                ((aa*aa*aa-aa)*y2[lastIndex-1]+(bb*bb*bb-bb)*y2[lastIndex])
                * (h*h)/6.0;
        } else if (iType==floor) {
            if (p1==v.begin()) {
                return p1->val;
            } else {
                citer p2 = p1;
                return (--p2)->val;
            }
        } else if (iType==ceil) {
            return p1->val;
        } else {
            throw TableError("interpolation method not yet implemented");
        }
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
        equalSpaced = false;
        sortIt();
        if (v.size()<=1) {
            // Nothing to do if the table is degenerate
            isReady = true;
            return;
        }

        // See if arguments are equally spaced
        // ...within this fractional error:
        const double tolerance = 0.01;
        dx = (v.back().arg - v.front().arg) / (v.size()-1);
        equalSpaced = true;
        for (int i=1; i<int(v.size()); i++) {
            if ( std::abs( ((v[i].arg-v[0].arg)/dx - i)) > tolerance) {
                equalSpaced = false;
                break;
            }
        }

        if (iType==spline) {
            // Set up the 2nd-derivative table for splines
            int n = v.size();
            if (n<2) throw TableError("Spline Table with only 1 entry");
            y2.resize(n);
            // End points 2nd-derivatives zero for natural cubic spline 
            y2[0] = static_cast<V>(0);
            y2[n-1] = static_cast<V>(0);
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
                for (int i=1; i<=n-2; i++);{
                    M(i-1, i-1) = 2. * (v[i+1].arg - v[i-1].arg);
                    rhs(i-1) = 6. * ( (v[i+1].val - v[i].val) / (v[i+1].arg - v[i].arg) -
                                      (v[i].val - v[i-1].val) / (v[i].arg - v[i-1].arg) );
                }
                tmv::Vector<V> solution(n-2);
                solution = rhs / M;
                for (int i=1; i<=n-2; i++){
                    y2[i] = solution[i-1];
                }
            }
            isReady = true;
            return;
        } else {
            // Nothing to do for any other interpolant
            isReady = true;
            return;
        }
    }

    template class Table<double,double>;

}
