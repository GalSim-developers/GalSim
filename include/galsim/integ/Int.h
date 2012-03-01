#ifndef INT_H
#define INT_H

#include <functional>
#include <vector>
#include <queue>
#include <map>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>
#include <ostream>
#include <complex>
#include <stdexcept>


#include "MoreFunctional.h"
#include "IntGKPData10.h"

// Basic Usage:
//
// First, define a function object, which should derive from 
// std::unary_function<double,double>.  For example, to integrate a
// Gaussian, use something along the lines of this:
//
// class Gauss :
//     public std::unary_function<double,double>
// {   
// public :
// 
//     Gauss(double _mu, double _sig) :   
//         mu(_mu), sig(_sig), sigsq(_sig*_sig) {}
// 
//     double operator()(double x) const
//     {
//         const double SQRTTWOPI = 2.50662827463;
//         return exp(-pow(x-mu,2)/2./sigsq)/SQRTTWOPI/sig;
//     }
// 
//   private :
//       double mu,sig,sigsq;
// };
//
//
// Then to perform an integral, over the range (min ... max) you would write:
//
// double integ1 = integ::int1d(Gauss(mu,sigma),min,max);
//
// e.g. integ::int1d(Gauss(0.,1.),0.,1.)
//
// should yield a value of 0.68.
//
// If either min or max are > 1.e10 or < -1.e10, then these values are taken to be
// infinity / -infinity, rather than the actual value.  
// So to integrate from 0 to infinity:
//
// double integ2 = integ::int1d(Gauss(0.,2.),0.,1.e100);
//
// which should yield a value of 0.5.  Or, you can also use the variable integ::MOCK_INF, 
// which might be clearer.
//
// There are two final arguments, which we've omitted so far, that can be used to specify
// the precision required.  First the relative error, then the absolute error.
// The defaults are 1.e-6 and 1.e-15 respectively, which are generally fine for most
// purposes, but you can specify different values if you prefer.
//
// The absolute error only comes into play for results which are close to 
// 0 to prevent requiring an error of 0 for integrals which evaluate to 0 
// or very close to it.
//
//
//
// Advanced Usage:
//
// Sometimes it is useful to provide more information into how to split up a region
// when the GKP algorithm fails to converge on the whole region.  To do this,
// we need to start by making an IntRegion object.  e.g. the above integ1 integral
// could use:
//
// integ::IntRegion reg1(0.,1.);
//
// And the above call could instead be called as:
//
// int1d(Gauss(0.,1.), reg1);
//
// For that integral, there is nothing weird going on, so the default works fine.
// But when an integration fails to converge with the usual GKP algorithm,
// it splits the region into 2 (or more) and tries again with each sub-region.
// The default is to just bisect the region (or something similarly smart for
// infinite regions), but if you know of a good place to split the region,
// you can tell it using the method addSplit(x).
// For example, if your integral has a singularity at 1/3, then it would help the 
// program a lot to split there, so you can add a split point:
//
// reg1.addSplit(1./3.);
//
// Zeros of the integrand can also be good choices for splitting.
//
// In addition to the integral being returned from int1d, int2d, or int3d as
// the return value, the value is also stored in the region itself. 
// You can access it using:
//
// reg.Area();
//
// There is also an estimate of the error in the value:
//
// reg.Err();
//
// (It is intended to be an overestimate of the actual error, 
// but it doesn't always get it completely right.)
//
//
// 
// Two- and Three-Dimensional Integrals:
//
// These are slightly more complicated.  The easiest case is when the
// bounds of the integral are a rectangle or 3d box.  In this case,
// you can still use the regular IntRegion.  The only new thing then
// is the definition of the function.  For example, to integrate 
// int(3x^2 + xy + y , x=0..1, y=0..1):
//
// struct Integrand :
//     public std::binary_function<double,double,double>
// {
//     double operator()(double x, double y) const { return x*(3.*x + y) + y; }
// };
//
// integ::IntRegion<double> reg3(0.,1.);
// double integ3 = int2d(Integrand(),reg3,reg3);
//
// (Which should give 1.75 as the result.)
// 
//
//

namespace galsim {
namespace integ {

    const double MOCK_INF = 1.e10;

    struct IntFailure : public std::runtime_error
    {
        IntFailure(const std::string& s) : 
            std::runtime_error(s.c_str())
        {}
    };

#ifdef NDEBUG
#define integ_dbg1 if (false) (*dbgout)
#define integ_dbg2 if (false) (*reg.dbgout)
#define integ_dbg3 if (false) (*tempreg.dbgout)
#else
#define integ_dbg1 if (dbgout) (*dbgout)
#define integ_dbg2 if (reg.dbgout) (*reg.dbgout)
#define integ_dbg3 if (tempreg.dbgout) (*tempreg.dbgout)
#endif

    template <class T> 
    inline T norm(const T& x) { return x*x; }
    using std::norm;
    template <class T> 
    inline T real(const T& x) { return x; }
    using std::real;

    //#define COUNTFEVAL 
    // If defined, then count the number of function evaluations

#ifdef COUNTFEVAL
    int nfeval = 0;
#endif

    template <class T> 
    struct IntRegion 
    {

    public:
        IntRegion(const T _a, const T _b, std::ostream* _dbgout=0) :
            a(_a), b(_b), error(0.), area(0), dbgout(_dbgout) {}
        bool operator<(const IntRegion<T>& r2) const 
        { return error < r2.error; }
        bool operator>(const IntRegion<T>& r2) const 
        { return error > r2.error; }
        void SubDivide(std::vector<IntRegion<T> >* children) 
        {
            assert(children->size() == 0);
            if (splitpoints.size() == 0) Bisect();
            if (splitpoints.size() > 1) 
                std::sort(splitpoints.begin(),splitpoints.end());

#if 0
            if (a>splitpoints[0] || b < splitpoints.back()) {
                std::cerr<<"a,b = "<<a<<','<<b<<std::endl;
                std::cerr<<"splitpoints = ";
                for(size_t i=0;i<splitpoints.size();i++) 
                    std::cerr<<splitpoints[i]<<"  "; 
                std::cerr<<std::endl;
            }
#endif
            assert(splitpoints[0] >= a);
            assert(splitpoints.back() <= b);
            children->push_back(IntRegion<T>(a,splitpoints[0],dbgout));
            for(size_t i=1;i<splitpoints.size();i++) {
                children->push_back(
                    IntRegion<T>(splitpoints[i-1],splitpoints[i],dbgout));
            }
            children->push_back(IntRegion<T>(splitpoints.back(),b,dbgout));
        }
        void Bisect() { splitpoints.push_back((a+b)/2.); }
        void AddSplit(const T x) { splitpoints.push_back(x); }
        size_t NSplit() const { return splitpoints.size(); }

        const T& Left() const { return a; }
        const T& Right() const { return b; }
        const T& Err() const { return error; }
        const T& Area() const { return area; }
        void SetArea(const T& a, const T& e) { area = a; error = e; }

    private:
        T a,b,error,area;
        std::vector<T> splitpoints;

    public:
        std::ostream* dbgout;
    };

    template <class T> 
    inline T Epsilon() 
    { return std::numeric_limits<T>::epsilon(); }
    template <class T> 
    inline T MinRep() 
    { return std::numeric_limits<T>::min(); }

    template <class T> 
    inline T rescale_error (T err, const T& resabs, const T& resasc)
    {
        if (resasc != 0. && err != 0.) {
            const T scale = (200. * err / resasc);
            if (scale < 1.) err = resasc * scale * sqrt(scale) ;
            else err = resasc ;
        }
        if (resabs > MinRep<T>() / (50. * Epsilon<T>())) {
            const T min_err = 50. * Epsilon<T>() * resabs;
            if (min_err > err) err = min_err;
        }
        return err;
    }

    template <class UF> 
    inline bool IntGKPNA(
        const UF& func, IntRegion<typename UF::result_type>& reg,
        const typename UF::result_type epsrel, 
        const typename UF::result_type epsabs,
        std::map<typename UF::result_type,
        typename UF::result_type>* fxmap=0)
    {
        // A non-adaptive integration of the function f over the region reg.
        // The algorithm computes first a Gaussian quadrature value
        // then successive Kronrod/Patterson extensions to this result.
        // The functions terminates when the difference between successive
        // approximations (rescaled according to rescale_error) is less than 
        // either epsabs or epsrel * I, where I is the latest estimate of the 
        // integral.
        // The order of the Gauss/Kronron/Patterson scheme is determined
        // by which file is included above.  Currently schemes starting 
        // with order 1 and order 10 are calculated.  There seems to be 
        // little practical difference in the integration times using 
        // the two schemes, so I haven't bothered to calculate any more.
        typedef typename UF::result_type T;
        const T a = reg.Left();
        const T b = reg.Right();

        const T half_length =  0.5 * (b - a);
        const T abs_half_length = fabs (half_length);
        const T center = 0.5 * (b + a);
        const T f_center = func(center);
#ifdef COUNTFEVAL
        nfeval++;
#endif

        assert(gkp_wb<T>(0).size() == gkp_x<T>(0).size()+1);
        T area1 = gkp_wb<T>(0).back() * f_center;
        std::vector<T> fv1, fv2;
        fv1.reserve(2*gkp_x<T>(0).size()+1);
        fv2.reserve(2*gkp_x<T>(0).size()+1);
        for (size_t k=0; k<gkp_x<T>(0).size(); k++) {
            const T abscissa = half_length * gkp_x<T>(0)[k];
            const T fval1 = func(center - abscissa);
            const T fval2 = func(center + abscissa);
            area1 += gkp_wb<T>(0)[k] * (fval1+fval2);
            fv1.push_back(fval1);
            fv2.push_back(fval2);
            if (fxmap) {
                (*fxmap)[center-abscissa] = fval1;
                (*fxmap)[center+abscissa] = fval2;
            }
        }
#ifdef COUNTFEVAL
        nfeval+=gkp_x<T>(0).size()*2;
#endif

        integ_dbg2<<"level 0 rule: area = "<<area1<<std::endl;

        T err=0; 
        bool calcabsasc = true;
        T resabs=0., resasc=0.;
        for (int level=1; level<NGKPLEVELS; level++) {
            assert(gkp_wa<T>(level).size() == fv1.size());
            assert(gkp_wa<T>(level).size() == fv2.size());
            assert(gkp_wb<T>(level).size() == gkp_x<T>(level).size()+1);
            T area2 = gkp_wb<T>(level).back() * f_center;
            // resabs = approximation to integral of abs(f)
            if (calcabsasc) resabs = fabs(area2);
            for (size_t k=0; k<fv1.size(); k++) {
                area2 += gkp_wa<T>(level)[k] * (fv1[k]+fv2[k]);
                if (calcabsasc) 
                    resabs += gkp_wa<T>(level)[k] *
                        (fabs(fv1[k]) + fabs(fv2[k]));
            }
            for (size_t k=0; k<gkp_x<T>(level).size(); k++) {
                const T abscissa = half_length * gkp_x<T>(level)[k];
                const T fval1 = func(center - abscissa);
                const T fval2 = func(center + abscissa);
                const T fval = fval1 + fval2;
                area2 += gkp_wb<T>(level)[k] * fval;
                if (calcabsasc) 
                    resabs += gkp_wb<T>(level)[k] * (fabs(fval1) + fabs(fval2));
                fv1.push_back(fval1);
                fv2.push_back(fval2);
                if (fxmap) {
                    (*fxmap)[center-abscissa] = fval1;
                    (*fxmap)[center+abscissa] = fval2;
                }
            }
#ifdef COUNTFEVAL
            nfeval+=gkp_x<T>(level).size()*2;
#endif
            if (calcabsasc) {
                const T mean = area1*T(0.5);
                // resasc = approximation to the integral of abs(f-mean) 
                resasc = gkp_wb<T>(level).back() * fabs(f_center-mean);
                for (size_t k=0; k<gkp_wa<T>(level).size(); k++) {
                    resasc += gkp_wa<T>(level)[k] * 
                        (fabs(fv1[k]-mean) + fabs(fv2[k]-mean));
                }
                for (size_t k=0; k<gkp_x<T>(level).size(); k++) {
                    resasc += gkp_wb<T>(level)[k] * 
                        (fabs(fv1[k]-mean) + fabs(fv2[k]-mean));
                }
                resasc *= abs_half_length ;
                resabs *= abs_half_length;
            }
            area2 *= half_length;
            err = rescale_error (fabs(area2-area1), resabs, resasc) ;
            if (err < resasc) calcabsasc = false;

            integ_dbg2<<"at level "<<level<<" area2 = "<<area2;
            integ_dbg2<<" +- "<<err<<std::endl;

            //   test for convergence.
            if (err < epsabs || err < epsrel * fabs (area2)) {
                reg.SetArea(area2,err);
                return true;
            }
            area1 = area2;
        }

        // failed to converge 
        reg.SetArea(area1,err);

        integ_dbg2<<"Failed to reach tolerance with highest-order GKP rule";

        return false;
    }

    template <class UF> 
    inline void IntGKP(
        const UF& func, IntRegion<typename UF::result_type>& reg,
        const typename UF::result_type epsrel,
        const typename UF::result_type epsabs,
        std::map<typename UF::result_type,
        typename UF::result_type>* fxmap=0)
    {
        // An adaptive integration algorithm which computes the integral of f
        // over the region reg.
        // First the non-adaptive GKP algorithm is tried.
        // If that is not accurate enough (according to the absolute and
        // relative accuracies, epsabs and epsrel),
        // the region is split in half, and each new region is integrated.
        // The routine continues by successively splitting the subregion
        // which gave the largest absolute error until the integral converges.
        //
        // The area and estimated error are returned as reg.Area() and reg.Err()
        // If desired, *retx and *retf return std::vectors of x,f(x) 
        // respectively
        // They only include the evaluations in the non-adaptive pass, so they
        // do not give an accurate estimate of the number of function 
        // evaluations.
        typedef typename UF::result_type T;
        integ_dbg2<<"Start IntGKP\n";

        assert(epsabs >= 0.);
        assert(epsrel > 0.);

        // perform the first integration 
        bool done = IntGKPNA(func, reg, epsrel, epsabs, fxmap);
        if (done) return;

        integ_dbg2<<"In adaptive GKP, failed first pass... subdividing\n";
        integ_dbg2<<"Intial range = "<<reg.Left()<<".."<<reg.Right()<<std::endl;

        int roundoff_type1 = 0, error_type = 0;
        T roundoff_type2 = 0.;
        size_t iteration = 1;

        std::priority_queue<IntRegion<T>,std::vector<IntRegion<T> > > allregions;
        allregions.push(reg);
        T finalarea = reg.Area();
        T finalerr = reg.Err();
        T tolerance= std::max(epsabs, epsrel * fabs(finalarea));
        assert(finalerr > tolerance);

        while(!error_type && finalerr > tolerance) {
            // Bisect the subinterval with the largest error estimate 
            integ_dbg2<<"Current answer = "<<finalarea<<" +- "<<finalerr;
            integ_dbg2<<"  (tol = "<<tolerance<<")\n";
            IntRegion<T> parent = allregions.top(); 
            allregions.pop();
            integ_dbg2<<"Subdividing largest error region ";
            integ_dbg2<<parent.Left()<<".."<<parent.Right()<<std::endl;
            integ_dbg2<<"parent area = "<<parent.Area();
            integ_dbg2<<" +- "<<parent.Err()<<std::endl;
            std::vector<IntRegion<T> > children;
            parent.SubDivide(&children);
            // For "GKP", there are only two, but for GKPOSC, there is one 
            // for each oscillation in region

            // Try to do at least 3x better with the children
            T factor = 3*children.size()*finalerr/tolerance;
            T newepsabs = fabs(parent.Err()/factor);
            T newepsrel = newepsabs/fabs(parent.Area());
            integ_dbg2<<"New epsabs,rel = "<<newepsabs<<','<<newepsrel;
            integ_dbg2<<"  ("<<children.size()<<" children)\n";

            T newarea = T(0.0);
            T newerror = 0.0;
            for(size_t i=0;i<children.size();i++) {
                IntRegion<T>& child = children[i];
                integ_dbg2<<"Integrating child "<<child.Left();
                integ_dbg2<<".."<<child.Right()<<std::endl;
                bool converged;
                converged = IntGKPNA(func, child, newepsrel, newepsabs);
                integ_dbg2<<"child ("<<i+1<<'/'<<children.size()<<") ";
                if (converged) {
                    integ_dbg2<<" converged."; 
                } else {
                    integ_dbg2<<" failed.";
                }
                integ_dbg2<<"  Area = "<<child.Area()<<
                    " +- "<<child.Err()<<std::endl;

                newarea += child.Area();
                newerror += child.Err();
            }
            integ_dbg2<<"Compare: newerr = "<<newerror;
            integ_dbg2<<" to parent err = "<<parent.Err()<<std::endl;

            finalerr += (newerror - parent.Err());
            finalarea += newarea - parent.Area();

            T delta = parent.Area() - newarea;
            if (newerror <= parent.Err() && fabs (delta) <=  parent.Err()
                && newerror >= 0.99 * parent.Err()) {
                integ_dbg2<<"roundoff type 1: delta/newarea = ";
                integ_dbg2<<fabs(delta)/fabs(newarea);
                integ_dbg2<<", newerror/error = "<<
                    newerror/parent.Err()<<std::endl;
                roundoff_type1++;
            }
            if (iteration >= 10 && newerror > parent.Err() && 
                fabs(delta) <= newerror-parent.Err()) {
                integ_dbg2<<"roundoff type 2: newerror/error = ";
                integ_dbg2<<newerror/parent.Err()<<std::endl;
                roundoff_type2+=std::min(newerror/parent.Err()-1.,T(1.));
            }

            tolerance = std::max(epsabs, epsrel * fabs(finalarea));
            if (finalerr > tolerance) {
                if (roundoff_type1 >= 200) {
                    error_type = 1;	// round off error 
                    integ_dbg2<<"GKP: Round off error 1\n";
                }
                if (roundoff_type2 >= 200.) {
                    error_type = 2;	// round off error 
                    integ_dbg2<<"GKP: Round off error 2\n";
                }
                const double parent_size = parent.Right()-parent.Left();
                const double reg_size = reg.Right()-parent.Left();
                if (fabs(parent_size / reg_size) < Epsilon<double>()) {
                    error_type = 3; // found singularity
                    integ_dbg2<<"GKP: Probable singularity\n";
                }
            }
            for(size_t i=0;i<children.size();i++) allregions.push(children[i]);
            iteration++;
        } 

        // Recalculate finalarea in case there are any slight rounding errors
        finalarea=0.; finalerr=0.;
        while (!allregions.empty()) {
            const IntRegion<T>& r=allregions.top();
            finalarea += r.Area();
            finalerr += r.Err();
            allregions.pop();
        }
        reg.SetArea(finalarea,finalerr);

        if (error_type == 1) {
            std::ostringstream s;
            s << "Type 1 roundoff's = "<<roundoff_type1;
            s << ", Type 2 = "<<roundoff_type2<<std::endl;
            s << "Roundoff error 1 prevents tolerance from being achieved ";
            s << "in IntGKP\n";
            throw IntFailure(s.str());
        } else if (error_type == 2) {
            std::ostringstream s;
            s << "Type 1 roundoff's = "<<roundoff_type1;
            s << ", Type 2 = "<<roundoff_type2<<std::endl;
            s << "Roundoff error 2 prevents tolerance from being achieved ";
            s << "in IntGKP\n";
            throw IntFailure(s.str());
        } else if (error_type == 3) {
            std::ostringstream s;
            s << "Bad integrand behavior found in the integration interval ";
            s << "in IntGKP\n";
            throw IntFailure(s.str());
        }
    }

    const double DEFRELERR = 1.e-6;
    const double DEFABSERR = 1.e-15;

    template <class UF> 
    struct AuxFunc1 : // f(1/x-1) for int(a..infinity)
        public std::unary_function<typename UF::argument_type,
        typename UF::result_type> 
    {
    public:
        AuxFunc1(const UF& _f) : f(_f) {}
        typename UF::result_type operator()(
            typename UF::argument_type x) const 
        { return f(1./x-1.)/(x*x); }
    private:
        const UF& f;
    };

    template <class UF> 
    AuxFunc1<UF> inline Aux1(UF uf) 
    { return AuxFunc1<UF>(uf); }

    template <class UF> 
    struct AuxFunc2 : // f(1/x+1) for int(-infinity..b)
        public std::unary_function<typename UF::argument_type,
        typename UF::result_type> 
    {
    public:
        AuxFunc2(const UF& _f) : f(_f) {}
        typename UF::result_type operator()(
            typename UF::argument_type x) const 
        { return f(1./x+1.)/(x*x); }
    private:
        const UF& f;
    };

    template <class UF> AuxFunc2<UF> 
    inline Aux2(UF uf) 
    { return AuxFunc2<UF>(uf); }

    template <class UF> 
    inline typename UF::result_type int1d(
        const UF& func, 
        IntRegion<typename UF::result_type>& reg,
        const typename UF::result_type& relerr=DEFRELERR,
        const typename UF::result_type& abserr=DEFABSERR)
    {
        typedef typename UF::result_type T;

        integ_dbg2<<"start int1d: "<<reg.Left()<<".."<<reg.Right()<<std::endl;

        if ((reg.Left() <= -MOCK_INF && reg.Right() > 0) ||
            (reg.Right() >= MOCK_INF && reg.Left() < 0)) { 
            reg.AddSplit(0);
        }

        if (reg.NSplit() > 0) {
            std::vector<IntRegion<T> > children;
            reg.SubDivide(&children);
            integ_dbg2<<"Subdivided into "<<children.size()<<" children\n";
            T answer=0;
            T err=0;
            for(size_t i=0;i<children.size();i++) {
                IntRegion<T>& child = children[i];
                integ_dbg2<<"i = "<<i;
                integ_dbg2<<": bounds = "<<child.Left()<<
                    ','<<child.Right()<<std::endl;
                answer += int1d(func,child,relerr,abserr);
                err += child.Err();
                integ_dbg2<<"subint = "<<child.Area()<<
                    " +- "<<child.Err()<<std::endl;
            }
            reg.SetArea(answer,err);
            return answer;
        } else {
            if (reg.Left() <= -MOCK_INF) {
                integ_dbg2<<"left = -infinity, right = "<<
                    reg.Right()<<std::endl;
                assert(reg.Right() <= 0.);
                IntRegion<T> modreg(1./(reg.Right()-1.),0.,reg.dbgout);
                IntGKP(Aux2<UF>(func),modreg,relerr,abserr);
                reg.SetArea(modreg.Area(),modreg.Err());
            } else if (reg.Right() >= MOCK_INF) {
                integ_dbg2<<"left = "<<reg.Left()<<", right = infinity\n";
                assert(reg.Left() >= 0.);
                IntRegion<T> modreg(0.,1./(reg.Left()+1.),reg.dbgout);
                IntGKP(Aux1<UF>(func),modreg,relerr,abserr);
                reg.SetArea(modreg.Area(),modreg.Err());
            } else {
                integ_dbg2<<"left = "<<reg.Left();
                integ_dbg2<<", right = "<<reg.Right()<<std::endl;
                IntGKP(func,reg,relerr,abserr);
            }
            integ_dbg2<<"done int1d  answer = "<<reg.Area();
            integ_dbg2<<" +- "<<reg.Err()<<std::endl;
            return reg.Area();
        }
    }

    template <class UF> 
    inline typename UF::result_type int1d(
        const UF& func, 
        typename UF::result_type min, typename UF::result_type max,
        const typename UF::result_type& relerr=DEFRELERR,
        const typename UF::result_type& abserr=DEFABSERR)
    {
        IntRegion<typename UF::result_type> reg(min,max);
        return int1d(func,reg,relerr,abserr); 
    }

    template <class BF, class YREG> 
    class Int2DAuxType : 
        public std::unary_function<typename BF::first_argument_type,
        typename BF::result_type> 
    {
    public:
        Int2DAuxType(const BF& _func, const YREG& _yreg,
                     const typename BF::result_type& _relerr,
                     const typename BF::result_type& _abserr) :
            func(_func),yreg(_yreg),relerr(_relerr),abserr(_abserr) 
        {}

        typename BF::result_type operator()(
            typename BF::first_argument_type x) const 
        {
            typename YREG::result_type tempreg = yreg(x);
            typename BF::result_type result = 
                int1d(bind21(func,x),tempreg,relerr,abserr);
            integ_dbg3<<"Evaluated int2dAux at x = "<<x;
            integ_dbg3<<": f = "<<result<<" +- "<<tempreg.Err()<<std::endl;
            return result;
        } 

    private:
        const BF& func;
        const YREG& yreg;
        typename BF::result_type relerr,abserr;
    };

    template <class BF, class YREG> 
    inline typename BF::result_type int2d(
        const BF& func,
        IntRegion<typename BF::result_type>& reg,
        const YREG& yreg, const typename BF::result_type& relerr=DEFRELERR,
        const typename BF::result_type& abserr=DEFABSERR)
    {
        integ_dbg2<<"Starting int2d: range = ";
        integ_dbg2<<reg.Left()<<".."<<reg.Right()<<std::endl;
        Int2DAuxType<BF,YREG> faux(func,yreg,relerr*1.e-3,abserr*1.e-3);
        typename BF::result_type answer = int1d(faux,reg,relerr,abserr);
        integ_dbg2<<"done int2d  answer = "<<answer<<
            " +- "<<reg.Err()<<std::endl;
        return answer;
    }

    template <class TF, class YREG, class ZREG> 
    class Int3DAuxType : 
        public std::unary_function<typename TF::firstof3_argument_type,
        typename TF::result_type> 
    {
    public:
        Int3DAuxType(const TF& _func, const YREG& _yreg, const ZREG& _zreg, 
                     const typename TF::result_type& _relerr,
                     const typename TF::result_type& _abserr) :
            func(_func),yreg(_yreg),zreg(_zreg),relerr(_relerr),abserr(_abserr) 
        {}

        typename TF::result_type operator()(
            typename TF::firstof3_argument_type x) const 
        {
            typename YREG::result_type tempreg = yreg(x);
            typename TF::result_type result = 
                int2d(bind31(func,x),tempreg,bind21(zreg,x),relerr,abserr);
            integ_dbg3<<"Evaluated int3dAux at x = "<<x;
            integ_dbg3<<": f = "<<result<<" +- "<<tempreg.Err()<<std::endl;
            return result;
        }

    private:
        const TF& func;
        const YREG& yreg;
        const ZREG& zreg;
        typename TF::result_type relerr,abserr;
    };

    template <class TF, class YREG, class ZREG> 
    inline typename TF::result_type int3d(
        const TF& func,
        IntRegion<typename TF::result_type>& reg,
        const YREG& yreg, const ZREG& zreg,
        const typename TF::result_type& relerr=DEFRELERR,
        const typename TF::result_type& abserr=DEFABSERR)
    {
        integ_dbg2<<"Starting int3d: range = ";
        integ_dbg2<<reg.Left()<<".."<<reg.Right()<<std::endl;
        Int3DAuxType<TF,YREG,ZREG> faux(
            func,yreg,zreg,relerr*1.e-3,abserr*1.e-3);
        typename TF::result_type answer = int1d(faux,reg,relerr,abserr);
        integ_dbg2<<"done int3d  answer = "<<answer<<
            "+- "<<reg.Err()<<std::endl;
        return answer;
    }

    // Helpers for constant regions for int2d, int3d:

    template <class T> 
    struct ConstantReg1 : public std::unary_function<T, IntRegion<T> >
    {
        ConstantReg1(T a,T b) : ir(a,b) {}
        ConstantReg1(const IntRegion<T>& r) : ir(r) {}
        IntRegion<T> operator()(T x) const { return ir; }
        IntRegion<T> ir;
    };

    template <class T> 
    struct ConstantReg2 : public std::binary_function<T, T, IntRegion<T> >
    {
        ConstantReg2(T a,T b) : ir(a,b) {}
        ConstantReg2(const IntRegion<T>& r) : ir(r) {}
        IntRegion<T> operator()(T x, T y) const { return ir; }
        IntRegion<T> ir;
    };

    template <class BF> 
    inline typename BF::result_type int2d(
        const BF& func,
        IntRegion<typename BF::result_type>& reg,
        IntRegion<typename BF::result_type>& yreg,
        const typename BF::result_type& relerr=DEFRELERR,
        const typename BF::result_type& abserr=DEFABSERR)
    { 
        return int2d(
            func,reg,
            ConstantReg1<typename BF::result_type>(yreg),
            relerr,abserr); 
    }

    template <class BF>
    inline typename BF::result_type int2d(
        const BF& func,
        typename BF::result_type xmin, typename BF::result_type xmax, 
        typename BF::result_type ymin, typename BF::result_type ymax, 
        const typename BF::result_type& relerr=DEFRELERR,
        const typename BF::result_type& abserr=DEFABSERR)
    { 
        IntRegion<typename BF::result_type> xreg(xmin,xmax);
        IntRegion<typename BF::result_type> yreg(ymin,ymax);
        return int2d(func,xreg,yreg,relerr,abserr);
    }

    template <class TF> 
    inline typename TF::result_type int3d(
        const TF& func,
        IntRegion<typename TF::result_type>& reg,
        IntRegion<typename TF::result_type>& yreg,
        IntRegion<typename TF::result_type>& zreg,
        const typename TF::result_type& relerr=DEFRELERR,
        const typename TF::result_type& abserr=DEFABSERR)
    {
        return int3d(
            func,reg,
            ConstantReg1<typename TF::result_type>(yreg),
            ConstantReg2<typename TF::result_type>(zreg),
            relerr,abserr);
    }

    template <class TF>
    inline typename TF::result_type int3d(
        const TF& func,
        typename TF::result_type xmin, typename TF::result_type xmax, 
        typename TF::result_type ymin, typename TF::result_type ymax, 
        typename TF::result_type zmin, typename TF::result_type zmax, 
        const typename TF::result_type& relerr=DEFRELERR,
        const typename TF::result_type& abserr=DEFABSERR)
    { 
        IntRegion<typename TF::result_type> xreg(xmin,xmax);
        IntRegion<typename TF::result_type> yreg(ymin,ymax);
        IntRegion<typename TF::result_type> zreg(zmin,zmax);
        return int2d(func,xreg,yreg,zreg,relerr,abserr);
    }

}} 

#endif
