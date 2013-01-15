// -*- c++ -*-
#ifndef CORRELATEDNOISE_IMPL_H
#define CORRELATEDNOISE_IMPL_H

//#define DEBUGLOGGING

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
/*
 * There are three levels of verbosity which can be helpful when debugging, which are written as
 * dbg, xdbg, xxdbg (all defined in Std.h).
 * It's Mike's way to have debug statements in the code that are really easy to turn on and off.
 *
 * If DEBUGLOGGING is #defined, then these write out to *dbgout, according to the value of 
 * verbose_level.
 * dbg requires verbose_level >= 1
 * xdbg requires verbose_level >= 2
 * xxdbg requires verbose_level >= 3
 * If DEBUGLOGGING is not defined, the all three becomes just `if (false) std::cerr`,
 * so the compiler parses the statement fine, but trivially optimizes the code away, so there is no
 * efficiency hit from leaving them in the code.
 */
#endif

/**
 * @file CorrelatedNoise.h @brief Contains a class definition for handling the correlation
 * properties of noise in Images.
 */
#include "SBInterpolatedImageImpl.h"
#include "CorrelatedNoise.h"
#include "SBAddImpl.h"

namespace galsim {

    class CorrelationFunction::CorrelationFunctionImpl: 
    public SBInterpolatedImage::SBInterpolatedImageImpl
    {
        public:

            template <typename T> 
            CorrelationFunctionImpl(
                const BaseImage<T>& image, 
                boost::shared_ptr<Interpolant2d> xInterp,
                boost::shared_ptr<Interpolant2d> kInterp,
                double dx, double pad_factor);

            double xValue(const Position<double>& p) const;

            std::complex<double> kValue(const Position<double>& p) const;

        private:

            // Copy constructor and op= are undefined.
            CorrelationFunctionImpl(const CorrelationFunctionImpl& rhs);
            void operator=(const CorrelationFunctionImpl& rhs);

            const int _Ni; // dimension of input lookup table for correlation function along i
            const int _Nj; // ditto for j

            void initialize(); ///< Put code common to both constructors here.

    };

    class AddCorrelationFunction::AddCorrelationFunctionImpl:
    public SBAdd::SBAddImpl
    {
        public:

            AddCorrelationFunctionImpl(const CorrelationFunction& c1, const CorrelationFunction& c2)
            { add(c1); add(c2); initialize(); }

            AddCorrelationFunctionImpl(const std::list<CorrelationFunction>& clist)
            {
                for (ConstIter cptr = clist.begin(); cptr!=clist.end(); ++cptr)
                    add(*cptr);
                initialize();
            }

            ~AddCorrelationFunctionImpl() {}

            void add(const CorrelationFunction& rhs);

        typedef std::list<CorrelationFunction>::iterator Iter;
        typedef std::list<CorrelationFunction>::const_iterator ConstIter;

        private:

            /// @brief The plist content is a pointer to a fresh copy of the summands.
            std::list<CorrelationFunction> _plist; 

            // Copy constructor and op= are undefined.
            AddImpl(const AddCorrelationFunctionImpl& rhs);
            void operator=(const AddCorrelationFunctionImpl& rhs);

    };

}
#endif
