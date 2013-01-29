// -*- c++ -*-
#ifndef SBADD_IMPL_H
#define SBADD_IMPL_H

#include "SBProfileImpl.h"
#include "SBAdd.h"

namespace galsim {

    class SBAdd::SBAddImpl : public SBProfileImpl
    {
    public:
        SBAddImpl(const SBProfile& s1, const SBProfile& s2)
        { add(s1); add(s2); initialize(); }

        SBAddImpl(const std::list<SBProfile>& slist)
        {
            for (ConstIter sptr = slist.begin(); sptr!=slist.end(); ++sptr)
                add(*sptr); 
            initialize();
        }

        ~SBAddImpl() {}

        void add(const SBProfile& rhs);

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const { return _maxMaxK; }
        double stepK() const { return _minStepK; }

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        { 
            xmin = integ::MOCK_INF; xmax = -integ::MOCK_INF; 
            for (ConstIter pptr = _plist.begin(); pptr!=_plist.end(); ++pptr) {
                double xmin_1, xmax_1;
                pptr->getXRange(xmin_1,xmax_1,splits);
                if (xmin_1 < xmin) xmin = xmin_1;
                if (xmax_1 > xmax) xmax = xmax_1;
            }
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            ymin = integ::MOCK_INF; ymax = -integ::MOCK_INF; 
            for (ConstIter pptr = _plist.begin(); pptr!=_plist.end(); ++pptr) {
                double ymin_1, ymax_1;
                pptr->getYRange(ymin_1,ymax_1,splits);
                if (ymin_1 < ymin) ymin = ymin_1;
                if (ymax_1 > ymax) ymax = ymax_1;
            }
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            ymin = integ::MOCK_INF; ymax = -integ::MOCK_INF; 
            for (ConstIter pptr = _plist.begin(); pptr!=_plist.end(); ++pptr) {
                double ymin_1, ymax_1;
                pptr->getYRangeX(x,ymin_1,ymax_1,splits);
                if (ymin_1 < ymin) ymin = ymin_1;
                if (ymax_1 > ymax) ymax = ymax_1;
            }
        }

        bool isAxisymmetric() const { return _allAxisymmetric; }
        bool hasHardEdges() const { return _anyHardEdges; }
        bool isAnalyticX() const { return _allAnalyticX; }
        bool isAnalyticK() const { return _allAnalyticK; }

        Position<double> centroid() const 
        { return Position<double>(_sumfx / _sumflux, _sumfy / _sumflux); }

        double getFlux() const { return _sumflux; }

        /**
         * @brief Shoot photons through this SBAdd.
         *
         * SBAdd will divide the N photons among its summands with probabilities proportional to
         * their integrated (absolute) fluxes.  Note that the order of photons in output array will
         * not be random as different summands' outputs are simply concatenated.
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        /**
         * @brief Give total positive flux of all summands
         *
         * Note that `getPositiveFlux()` return from SBAdd may not equal the integral of positive
         * regions of the image, because summands could have positive and negative regions
         * cancelling each other.  Rather it will be the sum of the `getPositiveFlux()` of all the
         * images.
         * @returns Total positive flux of all summands
         */
        double getPositiveFlux() const;

        /** @brief Give absolute value of total negative flux of all summands
         *
         * Note that `getNegativeFlux()` return from SBAdd may not equal the integral of negative
         * regions of the image, because summands could have positive and negative regions
         * cancelling each other. Rather it will be the sum of the `getNegativeFlux()` of all the
         * images.
         * @returns Absolute value of total negative flux of all summands
         */
        double getNegativeFlux() const;

        // Overrides for better efficiency
        void fillKGrid(KTable& kt) const;
        void fillXGrid(XTable& xt) const;

        typedef std::list<SBProfile>::iterator Iter;
        typedef std::list<SBProfile>::const_iterator ConstIter;

    protected:  // This is protected since we want inheritance by AddCorrelationFunctionImpl

        /// @brief The plist content is a pointer to a fresh copy of the summands.
        std::list<SBProfile> _plist; 
        double _sumflux; ///< Keeps track of the cumulated flux of all summands.
        double _sumfx; ///< Keeps track of the cumulated `fx` of all summands.
        double _sumfy; ///< Keeps track of the cumulated `fy` of all summands.
        double _maxMaxK; ///< Keeps track of the cumulated `maxK()` of all summands.
        double _minStepK; ///< Keeps track of the cumulated `minStepK()` of all summands.
        double _minMinX; ///< Keeps track of the cumulated `minX()` of all summands.
        double _maxMaxX; ///< Keeps track of the cumulated `maxX()` of all summands.
        double _minMinY; ///< Keeps track of the cumulated `minY()` of all summands.
        double _maxMaxY; ///< Keeps track of the cumulated `maxY()` of all summands.

        /// @brief Keeps track of the cumulated `isAxisymmetric()` properties of all summands.
        bool _allAxisymmetric;

        /// @brief Keeps track of whether any summands have hard edges.
        bool _anyHardEdges;

        /// @brief Keeps track of the cumulated `isAnalyticX()` property of all summands. 
        bool _allAnalyticX; 

        /// @brief Keeps track of the cumulated `isAnalyticK()` properties of all summands.
        bool _allAnalyticK; 

        void initialize();  ///< Sets all private book-keeping variables to starting state.

        // Copy constructor and op= are undefined.
        SBAddImpl(const SBAddImpl& rhs);
        void operator=(const SBAddImpl& rhs);
    };

}

#endif // SBADD_IMPL_H

