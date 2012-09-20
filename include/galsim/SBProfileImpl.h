// -*- c++ -*-
#ifndef SBPROFILE_IMPL_H
#define SBPROFILE_IMPL_H

#include "SBProfile.h"
#include "FFT.h"
#include "integ/Int.h"

namespace galsim {

    class SBProfile::SBProfileImpl
    {
    public:

        // Constructor doesn't do anything
        SBProfileImpl() {}

        // Virtual destructor
        virtual ~SBProfileImpl() {}

        // Pure virtual functions:
        virtual double xValue(const Position<double>& p) const =0;
        virtual std::complex<double> kValue(const Position<double>& k) const =0; 
        virtual double maxK() const =0; 
        virtual double stepK() const =0;
        virtual bool isAxisymmetric() const =0;
        virtual bool hasHardEdges() const =0;
        virtual bool isAnalyticX() const =0; 
        virtual bool isAnalyticK() const =0; 
        virtual Position<double> centroid() const = 0;
        virtual double getFlux() const =0; 
        virtual boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const=0;

        // Functions with default implementations:
        virtual void getXRange(double& xmin, double& xmax, std::vector<double>& /*splits*/) const 
        { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; }

        virtual void getYRange(double& ymin, double& ymax, std::vector<double>& /*splits*/) const 
        { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }

        virtual void getYRangeX(
            double /*x*/, double& ymin, double& ymax, std::vector<double>& splits) const 
        { getYRange(ymin,ymax,splits); }

        virtual double getPositiveFlux() const { return getFlux()>0. ? getFlux() : 0.; }

        virtual double getNegativeFlux() const { return getFlux()>0. ? 0. : -getFlux(); }

        // Utility for drawing into Image data structures.
        // returns flux integral
        template <typename T>
        double fillXImage(ImageView<T>& image, double gain) const  
        { return doFillXImage(image, gain); }

        // Utility for drawing a k grid into FFT data structures 
        virtual void fillKGrid(KTable& kt) const;

        // Utility for drawing an x grid into FFT data structures 
        virtual void fillXGrid(XTable& xt) const;

        // Virtual functions cannot be templates, so to make fillXImage work like a virtual
        // function, we have it call these, which need to include all the types of Image
        // that we want to use.
        //
        // Then in the derived class, these functions should call a template version of 
        // fillXImage in that derived class that implements the functionality you want.
        virtual double doFillXImage(ImageView<float>& image, double gain) const
        { return doFillXImage2(image,gain); }
        virtual double doFillXImage(ImageView<double>& image, double gain) const
        { return doFillXImage2(image,gain); }

        // Here in the base class, we need yet another name for the version that actually
        // implements this as a template:
        template <typename T>
        double doFillXImage2(ImageView<T>& image, double gain) const;

    private:
        // Copy constructor and op= are undefined.
        SBProfileImpl(const SBProfileImpl& rhs);
        void operator=(const SBProfileImpl& rhs);
    };

}

#endif // SBPROFILE_IMPL_H

