
/** 
 * @file Int.h
 *
 * @brief A set of functions for doing integrals of functions which can be evaluated at 
 *        arbitrary integrands.  
 *        Uses an adaptive Gauss-Kronrod-Patterson algorithm.
 *
 *
 * Basic Usage:
 * 
 *     First, define a function object, which should derive from 
 *     std::unary_function<double,double>.  For example, to integrate a
 *     Gaussian, use something along the lines of this:
 * 
 *     class Gauss :
 *         public std::unary_function<double,double>
 *     {   
 *     public :
 *     
 *         Gauss(double _mu, double _sig) :   
 *             mu(_mu), sig(_sig), sigsq(_sig*_sig) {}
 *     
 *         double operator()(double x) const
 *         {
 *             const double SQRTTWOPI = 2.50662827463;
 *             return exp(-pow(x-mu,2)/2./sigsq)/SQRTTWOPI/sig;
 *         }
 *     
 *       private :
 *           double mu,sig,sigsq;
 *     };
 * 
 * 
 *     Then to perform an integral, over the range (min ... max) you would write:
 * 
 *     double integ1 = integ::int1d(Gauss(mu,sigma),min,max);
 * 
 *     e.g. integ::int1d(Gauss(0.,1.),0.,1.)
 * 
 *     should yield a value of 0.68.
 * 
 *     If either min or max are > 1.e10 or < -1.e10, then these values are taken to be
 *     infinity / -infinity, rather than the actual value.  
 *     So to integrate from 0 to infinity:
 * 
 *     double integ2 = integ::int1d(Gauss(0.,2.),0.,1.e100);
 * 
 *     which should yield a value of 0.5.  Or, you can also use the variable integ::MOCK_INF, 
 *     which might be clearer.
 * 
 *     There are two final arguments, which we've omitted so far, that can be used to specify
 *     the precision required.  First the relative error, then the absolute error.
 *     The defaults are 1.e-6 and 1.e-15 respectively, which are generally fine for most
 *     purposes, but you can specify different values if you prefer.
 * 
 *     The absolute error only comes into play for results which are close to 
 *     0 to prevent requiring an error of 0 for integrals which evaluate to 0 
 *     or very close to it.
 * 
 * 
 * 
 * Advanced Usage:
 *
 *     Sometimes it is useful to provide more information into how to split up a region
 *     when the GKP algorithm fails to converge on the whole region.  To do this,
 *     we need to start by making an IntRegion object.  e.g. the above integ1 integral
 *     could use:
 *
 *     integ::IntRegion reg1(0.,1.);
 *
 *     And the above call could instead be called as:
 *
 *     int1d(Gauss(0.,1.), reg1);
 *
 *     For that integral, there is nothing weird going on, so the default works fine.
 *     But when an integration fails to converge with the usual GKP algorithm,
 *     it splits the region into 2 (or more) and tries again with each sub-region.
 *     The default is to just bisect the region (or something similarly smart for
 *     infinite regions), but if you know of a good place to split the region,
 *     you can tell it using the method addSplit(x).
 *     For example, if your integral has a singularity at 1/3, then it would help the 
 *     program a lot to split there, so you can add a split point:
 *
 *     reg1.addSplit(1./3.);
 *
 *     Zeros of the integrand can also be good choices for splitting.
 *
 *     In addition to the integral being returned from int1d, int2d, or int3d as
 *     the return value, the value is also stored in the region itself. 
 *     You can access it using:
 *
 *     reg.getArea();
 *
 *     There is also an estimate of the error in the value:
 *
 *     reg.getErr();
 *
 *     (It is intended to be an overestimate of the actual error, 
 *     but it doesn't always get it completely right.)
 *
 *
 *
 * Two- and Three-Dimensional Integrals:
 *
 *     These are slightly more complicated.  The easiest case is when the
 *     bounds of the integral are a rectangle or 3d box.  In this case,
 *     you can still use the regular IntRegion.  The only new thing then
 *     is the definition of the function.  For example, to integrate 
 *     int(3x^2 + xy + y , x=0..1, y=0..1):
 *
 *     struct Integrand :
 *         public std::binary_function<double,double,double>
 *     {
 *         double operator()(double x, double y) const { return x*(3.*x + y) + y; }
 *     };
 *
 *     integ::IntRegion<double> reg3(0.,1.);
 *     double integ3 = int2d(Integrand(),reg3,reg3);
 *
 *     (Which should give 1.75 as the result.)
 *
 */


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
namespace galsim {
namespace integ {

    const double MOCK_INF = 1.e100;  ///< May be used to indicate infinity in integration regions.
    const double DEFRELERR = 1.e-6;  ///< The default target relative error if not specified.
    const double DEFABSERR = 1.e-15; ///< The default target absolute error if not specified.


    /// An exception type thrown if the integrator encounters a problem.
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

#ifdef COUNTFEVAL
    int nfeval = 0;  ///< If COUNTFEVAL is defined, this counts the number of function evaluations
#endif

    /**
     * @brief A type that encapsulates everything known about the integral in a region.
     *
     * The constructor of IntRegion takes the minimum and maximum values for the region,
     * along with an optional ostream for outputing diagnostic information.
     *
     * After the integration is done, the IntRegion will also hold the estimate of the 
     * integral's value over the region, along with an estimate of the error.
     */
    template <class T> 
    struct IntRegion 
    {

    public:
        /**
         * @brief Constructor taking the bounds of the region and (optionally) 
         *        an ostream for outputing diagnostic information.
         *
         * Note that normally a < b.  However, a > b is also valid.
         * If a > b, then the integral will be from right to left, which is just
         * the negative of the integral from b to a.
         */
        IntRegion(
            const T _a,  ///< The left end of the region
            const T _b,  ///< The right end of the region
            std::ostream* _dbgout=0  ///< An optional ostream for diagnostic info
        ) :
            a(_a), b(_b), error(0.), area(0), dbgout(_dbgout) {}


        /// op< sorts by the error estimate
        bool operator<(const IntRegion<T>& r2) const 
        { return error < r2.error; }

        /// op> sorts by the error estimate
        bool operator>(const IntRegion<T>& r2) const 
        { return error > r2.error; }

        /** 
         * @brief Subdivide a region according to the current split points or biset
         *
         * If there are no split points set yet, then this will bisect the region.
         * However, you may also set split points using addSplit(x). 
         * This is worth doing if you know about any discontinuities, zeros or poles 
         * in the function you are integrating.
         */
        void subDivide(std::vector<IntRegion<T> >* children) 
        {
            assert(children->size() == 0);
            if (splitpoints.size() == 0) bisect();
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

        /// Set a split point at the bisection
        void bisect() { splitpoints.push_back((a+b)/2.); }

        /**
         * @brief Add a split point to the current list to be used by the next subDivide call
         *
         * This is worth doing if you know about any discontinuities, zeros or poles 
         * in the function you are integrating.
         */
        void addSplit(const T x) { splitpoints.push_back(x); }

        /// Get the number of split points currently set.
        size_t getNSplit() const { return splitpoints.size(); }

        /// Get the left end of the region
        const T& left() const { return a; }

        /// Get the right end of the region
        const T& right() const { return b; }

        /// Get the current error estimate
        const T& getErr() const { return error; }

        /// Get the current estimate of the integral over the region
        const T& getArea() const { return area; }

        /// Set a new estimate of the area and error
        void setArea(const T& a, const T& e) { area = a; error = e; }

    private:
        T a,b,error,area;
        std::vector<T> splitpoints;

    public:
        std::ostream* dbgout;
    };

    /// Rescale the error if int |f| dx or int |f-mean| dx are too large
    template <class T> 
    inline T rescaleError(
        T err, ///< The current estimate of the error
        const T& int_abs,     ///< An estimate of int |f| dx
        const T& int_absdiff ///< An estimate of int |f-mean| dx
    )
    {
        const int eps = std::numeric_limits<T>::epsilon();
        const int minrep = std::numeric_limits<T>::min();

        if (int_absdiff != 0. && err != 0.) {
            const T scale = (200. * err / int_absdiff);
            if (scale < 1.) err = int_absdiff * scale * sqrt(scale) ;
            else err = int_absdiff ;
        }
        if (int_abs > minrep / (50. * eps)) {
            const T min_err = 50. * eps * int_abs;
            if (min_err > err) err = min_err;
        }
        return err;
    }

    /**
     * @brief Non-adaptive GKP integration
     *
     * A non-adaptive integration of the function f over the region reg.
     *
     * The algorithm computes first a Gaussian quadrature value
     * then successive Kronrod/Patterson extensions to this result.
     * The functions terminates when the difference between successive
     * approximations (rescaled according to rescaleError) is less than 
     * either abserr or relerr * I, where I is the latest estimate of the 
     * integral.
     *
     * The order of the Gauss/Kronron/Patterson scheme is determined
     * by which file is included above.  Currently schemes starting 
     * with order 1 and order 10 are calculated.  There seems to be 
     * little practical difference in the integration times using 
     * the two schemes, so I haven't bothered to calculate any more.
     */
    template <class UF> 
    inline bool intGKPNA(
        const UF& func, ///< The function to integrate
        IntRegion<typename UF::result_type>& reg, ///< The region with the bounds
        const typename UF::result_type relerr,  ///< The target relative error
        const typename UF::result_type abserr,  ///< The target absolute error
        std::map<typename UF::result_type,typename UF::result_type>* fxmap=0 ///< Known results
    )
    {
        typedef typename UF::result_type T;
        const T a = reg.left();
        const T b = reg.right();

        const T half_length =  0.5 * (b - a);
        const T abs_half_length = std::abs(half_length);
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
        bool calc_int_abs = true;
        T int_abs=0., int_absdiff=0.;
        for (int level=1; level<NGKPLEVELS; level++) {
            assert(gkp_wa<T>(level).size() == fv1.size());
            assert(gkp_wa<T>(level).size() == fv2.size());
            assert(gkp_wb<T>(level).size() == gkp_x<T>(level).size()+1);
            T area2 = gkp_wb<T>(level).back() * f_center;
            // int_abs = approximation to integral of abs(f)
            if (calc_int_abs) int_abs = std::abs(area2);
            for (size_t k=0; k<fv1.size(); k++) {
                area2 += gkp_wa<T>(level)[k] * (fv1[k]+fv2[k]);
                if (calc_int_abs) 
                    int_abs += gkp_wa<T>(level)[k] *
                        (std::abs(fv1[k]) + std::abs(fv2[k]));
            }
            for (size_t k=0; k<gkp_x<T>(level).size(); k++) {
                const T abscissa = half_length * gkp_x<T>(level)[k];
                const T fval1 = func(center - abscissa);
                const T fval2 = func(center + abscissa);
                const T fval = fval1 + fval2;
                area2 += gkp_wb<T>(level)[k] * fval;
                if (calc_int_abs) 
                    int_abs += gkp_wb<T>(level)[k] * (std::abs(fval1) + std::abs(fval2));
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
            if (calc_int_abs) {
                const T mean = area1*T(0.5);
                // int_absdiff = approximation to the integral of abs(f-mean) 
                int_absdiff = gkp_wb<T>(level).back() * std::abs(f_center-mean);
                for (size_t k=0; k<gkp_wa<T>(level).size(); k++) {
                    int_absdiff += gkp_wa<T>(level)[k] * 
                        (std::abs(fv1[k]-mean) + std::abs(fv2[k]-mean));
                }
                for (size_t k=0; k<gkp_x<T>(level).size(); k++) {
                    int_absdiff += gkp_wb<T>(level)[k] * 
                        (std::abs(fv1[k]-mean) + std::abs(fv2[k]-mean));
                }
                int_absdiff *= abs_half_length ;
                int_abs *= abs_half_length;
            }
            area2 *= half_length;
            err = rescaleError(std::abs(area2-area1), int_abs, int_absdiff) ;
            if (err < int_absdiff) calc_int_abs = false;

            integ_dbg2<<"at level "<<level<<" area2 = "<<area2;
            integ_dbg2<<" +- "<<err<<std::endl;

            //  Test for convergence.
            if (err < abserr || err < relerr * std::abs(area2)) {
                // Converged.  Return current estimate.
                reg.setArea(area2,err);
                return true;
            }
            area1 = area2;
        }

        // Failed to converge.  Return with current estimate of area and error
        reg.setArea(area1,err);

        integ_dbg2<<"Failed to reach tolerance with highest-order GKP rule";

        return false;
    }

    /**
     * @brief Adaptive GKP integration
     *
     * An adaptive integration algorithm which computes the integral of f
     * over the region reg.
     *
     * First the non-adaptive GKP algorithm is tried.
     *
     * If that is not accurate enough (according to the absolute and
     * relative accuracies, abserr and relerr), the region is split in half, 
     * and each new region is integrated.
     *
     * The routine continues by successively splitting the subregion
     * which gave the largest absolute error until the integral converges.
     *
     * The area and estimated error are returned as reg.getArea() and reg.getErr()
     *
     * If desired, *fxmap returns a std::map of x,f(x) values that were calculated
     * during the integration process.
     */
    template <class UF> 
    inline void intGKP(
        const UF& func, IntRegion<typename UF::result_type>& reg,
        const typename UF::result_type relerr,
        const typename UF::result_type abserr,
        std::map<typename UF::result_type,
        typename UF::result_type>* fxmap=0)
    {
        typedef typename UF::result_type T;
        const int eps = std::numeric_limits<T>::epsilon();

        integ_dbg2<<"Start intGKP\n";

        assert(abserr >= 0.);
        assert(relerr > 0.);

        // Perform the first integration 
        bool done = intGKPNA(func, reg, relerr, abserr, fxmap);
        if (done) return;

        integ_dbg2<<"In adaptive GKP, failed first pass... subdividing\n";
        integ_dbg2<<"Intial range = "<<reg.left()<<".."<<reg.right()<<std::endl;

        int roundoff_type1 = 0, error_type = 0;
        T roundoff_type2 = 0.;
        size_t iteration = 1;

        // Keep track of all subdivision in a priority_queue.  
        // The top() is always the largest value, and pop() removes it.
        // We define < and > for IntRegoins such that the "largest" is the one 
        // with the largest current error estimate.  This is the next one to be split 
        // if we need to split further.
        std::priority_queue<IntRegion<T>,std::vector<IntRegion<T> > > allregions;
        allregions.push(reg);
        T finalarea = reg.getArea();
        T finalerr = reg.getErr();
        T tolerance= std::max(abserr, relerr * std::abs(finalarea));
        assert(finalerr > tolerance);

        while(!error_type && finalerr > tolerance) {
            // Bisect the subinterval with the largest error estimate 
            integ_dbg2<<"Current answer = "<<finalarea<<" +- "<<finalerr;
            integ_dbg2<<"  (tol = "<<tolerance<<")\n";
            IntRegion<T> parent = allregions.top(); 
            allregions.pop();
            integ_dbg2<<"Subdividing largest error region ";
            integ_dbg2<<parent.left()<<".."<<parent.right()<<std::endl;
            integ_dbg2<<"parent area = "<<parent.getArea();
            integ_dbg2<<" +- "<<parent.getErr()<<std::endl;
            std::vector<IntRegion<T> > children;
            parent.subDivide(&children);
            // For "GKP", there are only two, but for GKPOSC, there is one 
            // for each oscillation in region

            // Try to do at least 3x better with the children
            T factor = 3*children.size()*finalerr/tolerance;
            T newabserr = std::abs(parent.getErr()/factor);
            T newrelerr = newabserr/std::abs(parent.getArea());
            integ_dbg2<<"New abserr,rel = "<<newabserr<<','<<newrelerr;
            integ_dbg2<<"  ("<<children.size()<<" children)\n";

            T newarea = T(0.0);
            T newerror = 0.0;
            for(size_t i=0;i<children.size();i++) {
                IntRegion<T>& child = children[i];
                integ_dbg2<<"Integrating child "<<child.left();
                integ_dbg2<<".."<<child.right()<<std::endl;
                bool converged;
                converged = intGKPNA(func, child, newrelerr, newabserr);
                integ_dbg2<<"child ("<<i+1<<'/'<<children.size()<<") ";
                if (converged) {
                    integ_dbg2<<" converged."; 
                } else {
                    integ_dbg2<<" failed.";
                }
                integ_dbg2<<"  Area = "<<child.getArea()<<
                    " +- "<<child.getErr()<<std::endl;

                newarea += child.getArea();
                newerror += child.getErr();
            }
            integ_dbg2<<"Compare: newerr = "<<newerror;
            integ_dbg2<<" to parent err = "<<parent.getErr()<<std::endl;

            finalerr += (newerror - parent.getErr());
            finalarea += newarea - parent.getArea();

            T delta = parent.getArea() - newarea;
            if (newerror <= parent.getErr() && std::abs(delta) <= parent.getErr()
                && newerror >= 0.99 * parent.getErr()) {
                integ_dbg2<<"roundoff type 1: delta/newarea = ";
                integ_dbg2<<std::abs(delta)/std::abs(newarea);
                integ_dbg2<<", newerror/error = "<<
                    newerror/parent.getErr()<<std::endl;
                roundoff_type1++;
            }
            if (iteration >= 10 && newerror > parent.getErr() && 
                std::abs(delta) <= newerror-parent.getErr()) {
                integ_dbg2<<"roundoff type 2: newerror/error = ";
                integ_dbg2<<newerror/parent.getErr()<<std::endl;
                roundoff_type2+=std::min(newerror/parent.getErr()-1.,T(1.));
            }

            tolerance = std::max(abserr, relerr * std::abs(finalarea));
            if (finalerr > tolerance) {
                if (roundoff_type1 >= 200) {
                    error_type = 1;	// round off error 
                    integ_dbg2<<"GKP: Round off error 1\n";
                }
                if (roundoff_type2 >= 200.) {
                    error_type = 2;	// round off error 
                    integ_dbg2<<"GKP: Round off error 2\n";
                }
                const double parent_size = parent.right()-parent.left();
                const double reg_size = reg.right()-parent.left();
                if (std::abs(parent_size / reg_size) < eps) {
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
            finalarea += r.getArea();
            finalerr += r.getErr();
            allregions.pop();
        }
        reg.setArea(finalarea,finalerr);

        if (error_type == 1) {
            std::ostringstream s;
            s << "Type 1 roundoff's = "<<roundoff_type1;
            s << ", Type 2 = "<<roundoff_type2<<std::endl;
            s << "Roundoff error 1 prevents tolerance from being achieved ";
            s << "in intGKP\n";
            throw IntFailure(s.str());
        } else if (error_type == 2) {
            std::ostringstream s;
            s << "Type 1 roundoff's = "<<roundoff_type1;
            s << ", Type 2 = "<<roundoff_type2<<std::endl;
            s << "Roundoff error 2 prevents tolerance from being achieved ";
            s << "in intGKP\n";
            throw IntFailure(s.str());
        } else if (error_type == 3) {
            std::ostringstream s;
            s << "Bad integrand behavior found in the integration interval ";
            s << "in intGKP\n";
            throw IntFailure(s.str());
        }
    }

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

    /// Perform a 1-dimensional integral using an IntRegion
    template <class UF> 
    inline typename UF::result_type int1d(
        const UF& func,  ///< The function to be integrated(may be a function object)
        IntRegion<typename UF::result_type>& reg, ///< The region of the integration
        const typename UF::result_type& relerr=DEFRELERR, ///< The target relative error
        const typename UF::result_type& abserr=DEFABSERR  ///< The target absolute error
    )
    {
        typedef typename UF::result_type T;

        integ_dbg2<<"start int1d: "<<reg.left()<<".."<<reg.right()<<std::endl;

        if ((reg.left() <= -MOCK_INF && reg.right() > 0) ||
            (reg.right() >= MOCK_INF && reg.left() < 0)) { 
            reg.addSplit(0);
        }

        if (reg.getNSplit() > 0) {
            std::vector<IntRegion<T> > children;
            reg.subDivide(&children);
            integ_dbg2<<"Subdivided into "<<children.size()<<" children\n";
            T answer=0;
            T err=0;
            for(size_t i=0;i<children.size();i++) {
                IntRegion<T>& child = children[i];
                integ_dbg2<<"i = "<<i;
                integ_dbg2<<": bounds = "<<child.left()<<
                    ','<<child.right()<<std::endl;
                answer += int1d(func,child,relerr,abserr);
                err += child.getErr();
                integ_dbg2<<"subint = "<<child.getArea()<<
                    " +- "<<child.getErr()<<std::endl;
            }
            reg.setArea(answer,err);
            return answer;
        } else {
            if (reg.left() <= -MOCK_INF) {
                integ_dbg2<<"left = -infinity, right = "<<
                    reg.right()<<std::endl;
                assert(reg.right() <= 0.);
                IntRegion<T> modreg(1./(reg.right()-1.),0.,reg.dbgout);
                intGKP(Aux2<UF>(func),modreg,relerr,abserr);
                reg.setArea(modreg.getArea(),modreg.getErr());
            } else if (reg.right() >= MOCK_INF) {
                integ_dbg2<<"left = "<<reg.left()<<", right = infinity\n";
                assert(reg.left() >= 0.);
                IntRegion<T> modreg(0.,1./(reg.left()+1.),reg.dbgout);
                intGKP(Aux1<UF>(func),modreg,relerr,abserr);
                reg.setArea(modreg.getArea(),modreg.getErr());
            } else {
                integ_dbg2<<"left = "<<reg.left();
                integ_dbg2<<", right = "<<reg.right()<<std::endl;
                intGKP(func,reg,relerr,abserr);
            }
            integ_dbg2<<"done int1d  answer = "<<reg.getArea();
            integ_dbg2<<" +- "<<reg.getErr()<<std::endl;
            return reg.getArea();
        }
    }

    /// Perform a 1-dimensional integral using simple min/max values for the region
    template <class UF> 
    inline typename UF::result_type int1d(
        const UF& func,  ///< The function to be integrated (may be a function object)
        typename UF::result_type min, ///< The lower bound of the integration
        typename UF::result_type max, ///< The upper bound of the integration
        const typename UF::result_type& relerr=DEFRELERR, ///< The target relative error
        const typename UF::result_type& abserr=DEFABSERR  ///< The target absolute error
    )
    {
        IntRegion<typename UF::result_type> reg(min,max);
        return int1d(func,reg,relerr,abserr); 
    }

    template <class BF, class YREG> 
    class Int2DAuxType : 
        public std::unary_function<typename BF::first_argument_type,typename BF::result_type> 
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
            integ_dbg3<<": f = "<<result<<" +- "<<tempreg.getErr()<<std::endl;
            return result;
        } 

    private:
        const BF& func;
        const YREG& yreg;
        typename BF::result_type relerr,abserr;
    };

    /// Perform a 2-dimensional integral
    template <class BF, class YREG> 
    inline typename BF::result_type int2d(
        const BF& func,  ///< The function to be integrated (may be a function object)
        IntRegion<typename BF::result_type>& reg,  ///< The region of the inner integral
        const YREG& yreg, ///< yreg(x) is the region of the outer integral at x
        const typename BF::result_type& relerr=DEFRELERR, ///< The target relative error
        const typename BF::result_type& abserr=DEFABSERR  ///< The target absolute error
    )
    {
        integ_dbg2<<"Starting int2d: range = ";
        integ_dbg2<<reg.left()<<".."<<reg.right()<<std::endl;
        Int2DAuxType<BF,YREG> faux(func,yreg,relerr*1.e-3,abserr*1.e-3);
        typename BF::result_type answer = int1d(faux,reg,relerr,abserr);
        integ_dbg2<<"done int2d  answer = "<<answer<<
            " +- "<<reg.getErr()<<std::endl;
        return answer;
    }

    template <class TF, class YREG, class ZREG> 
    class Int3DAuxType : 
        public std::unary_function<typename TF::firstof3_argument_type,typename TF::result_type> 
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
            integ_dbg3<<": f = "<<result<<" +- "<<tempreg.getErr()<<std::endl;
            return result;
        }

    private:
        const TF& func;
        const YREG& yreg;
        const ZREG& zreg;
        typename TF::result_type relerr,abserr;
    };

    /// Perform a 3-dimensional integral
    template <class TF, class YREG, class ZREG> 
    inline typename TF::result_type int3d(
        const TF& func,  ///< The function to be integrated (may be a function object)
        IntRegion<typename TF::result_type>& reg,  ///< The region of the inner integral
        const YREG& yreg, ///< yreg(x) is the region of the middle integral at x
        const ZREG& zreg, ///< zreg(x)(y) is the region of the outer integral at x,y
        const typename TF::result_type& relerr=DEFRELERR, ///< The target relative error
        const typename TF::result_type& abserr=DEFABSERR  ///< The target absolute error
    )
    {
        integ_dbg2<<"Starting int3d: range = ";
        integ_dbg2<<reg.left()<<".."<<reg.right()<<std::endl;
        Int3DAuxType<TF,YREG,ZREG> faux(
            func,yreg,zreg,relerr*1.e-3,abserr*1.e-3);
        typename TF::result_type answer = int1d(faux,reg,relerr,abserr);
        integ_dbg2<<"done int3d  answer = "<<answer<<
            "+- "<<reg.getErr()<<std::endl;
        return answer;
    }

    // Helpers for constant regions for int2d, int3d:

    template <class T> 
    struct ConstantReg1 : 
        public std::unary_function<T, IntRegion<T> >
    {
        ConstantReg1(T a,T b) : ir(a,b) {}
        ConstantReg1(const IntRegion<T>& r) : ir(r) {}
        IntRegion<T> operator()(T x) const { return ir; }
        IntRegion<T> ir;
    };

    template <class T> 
    struct ConstantReg2 : 
        public std::binary_function<T, T, IntRegion<T> >
    {
        ConstantReg2(T a,T b) : ir(a,b) {}
        ConstantReg2(const IntRegion<T>& r) : ir(r) {}
        IntRegion<T> operator()(T x, T y) const { return ir; }
        IntRegion<T> ir;
    };

    /// Perform a 3-dimensional integral using constant IntRegions for both regions
    /// (i.e. the integral is over a square)
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

    /// Perform a 2-dimensional integral using simple min/max values for borh regions
    /// (i.e. the integral is over a square)
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

    /// Perform a 3-dimensional integral using constant IntRegions for all regions
    /// (i.e. the integral is over a cube)
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

    /// Perform a 3-dimensional integral using simple min/max values for all regions
    /// (i.e. the integral is over a cube)
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
