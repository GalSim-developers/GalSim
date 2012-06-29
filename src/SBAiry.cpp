
//#define DEBUGLOGGING

#include "SBAiry.h"
#include "SBAiryImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    SBAiry::SBAiry(double lam_over_D, double obscuration, double flux) :
        SBProfile(new SBAiryImpl(lam_over_D, obscuration, flux)) {}

    SBAiry::SBAiry(const SBAiry& rhs) : SBProfile(rhs) {}

    SBAiry::~SBAiry() {}

    double SBAiry::getLamOverD() const 
    {
        assert(dynamic_cast<const SBAiryImpl*>(_pimpl.get()));
        return dynamic_cast<const SBAiryImpl&>(*_pimpl).getLamOverD(); 
    }

    double SBAiry::getObscuration() const 
    {
        assert(dynamic_cast<const SBAiryImpl*>(_pimpl.get()));
        return dynamic_cast<const SBAiryImpl&>(*_pimpl).getObscuration(); 
    }

    SBAiry::SBAiryImpl::SBAiryImpl(double lam_over_D, double obscuration, double flux) :
        _lam_over_D(lam_over_D), 
        _D(1. / lam_over_D), 
        _obscuration(obscuration), 
        _flux(flux), 
        _Dsq(_D*_D), _obssq(_obscuration*_obscuration),
        _inv_Dsq_pisq(1. / (_Dsq * M_PI * M_PI)),
        _xnorm(flux * _Dsq),
        _knorm(flux / (M_PI * (1.-_obssq))),
        _info(nmap.get(_obscuration,_obssq))
    {}

    SBAiry::InfoBarn SBAiry::nmap;

    // This is a scale-free version of the Airy radial function.
    // Input radius is in units of lambda/D.  Output normalized
    // to integrate to unity over input units.
    double SBAiry::AiryRadialFunction::operator()(double radius) const 
    {
        double nu = radius*M_PI;
        // Taylor expansion of j1(u)/u = 1/2 - 1/16 x^2 + ...
        // We can truncate this to 1/2 when neglected term is less than xvalue_accuracy
        // (relative error, so divide by 1/2)
        // xvalue_accurace = 1/8 x^2
        const double thresh = sqrt(8.*sbp::xvalue_accuracy);
        double xval;
        if (nu < thresh) {
            // lim j1(u)/u = 1/2
            xval =  (1.-_obssq);
        } else {
            // See Schroeder eq (10.1.10)
            xval = 2.*( j1(nu) - _obscuration*j1(_obscuration*nu)) / nu ; 
        }
        xval*=xval;
        // Normalize to give unit flux integrated over area.
        xval *= _norm;
        return xval;
    }

    double SBAiry::SBAiryImpl::xValue(const Position<double>& p) const 
    {
        double r = sqrt(p.x*p.x+p.y*p.y) * _D;
        return _xnorm * _info->xValue(r);
    }

    double SBAiry::AiryInfo::xValue(double r) const 
    { return _radial(r); }

    std::complex<double> SBAiry::SBAiryImpl::kValue(const Position<double>& k) const
    {
        double ksq_over_pisq = (k.x*k.x+k.y*k.y) * _inv_Dsq_pisq;
        // calculate circular FT(PSF) on p'=(x',y')
        return _knorm * _info->kValue(ksq_over_pisq);
    }

    // Set maxK to hard limit for Airy disk.
    double SBAiry::SBAiryImpl::maxK() const 
    { return 2.*M_PI*_D; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBAiry::SBAiryImpl::stepK() const
    { return _info->stepK() * _D; }

    double SBAiry::AiryInfo::chord(double r, double h, double rsq, double hsq) const 
    {
        if (r==0.) 
            return 0.;
        else if (r >= h && h >= 0.) 
            return rsq*std::asin(h/r) -h*sqrt(rsq-hsq);
        else if (r<h) 
            throw SBError("Airy calculation r<h");
        else 
            throw SBError("Airy calculation (r||h)<0");
    }

    /* area inside intersection of 2 circles radii r & s, seperated by t*/
    double SBAiry::AiryInfo::circle_intersection(
        double r, double s, double rsq, double ssq, double tsq) const 
    {
        assert(r >= s);
        assert(s >= 0.);
        double rps_sq = (r+s)*(r+s);
        if (tsq >= rps_sq) return 0.;
        double rms_sq = (r-s)*(r-s);
        if (tsq <= rms_sq) return M_PI*ssq;

        /* in between we calculate half-height at intersection */
        double hsq = 0.5*(rsq + ssq) - (tsq*tsq + rps_sq*rms_sq)/(4.*tsq);
        if (hsq<0.) throw SBError("Airy calculation half-height invalid");
        double h = sqrt(hsq);

        if (tsq < rsq - ssq) 
            return M_PI*ssq - chord(s,h,ssq,hsq) + chord(r,h,rsq,hsq);
        else
            return chord(s,h,ssq,hsq) + chord(r,h,rsq,hsq);
    }

    /* area inside intersection of 2 circles both with radius r, seperated by t*/
    double SBAiry::AiryInfo::circle_intersection(double r, double rsq, double tsq) const 
    {
        assert(r >= 0.);
        if (tsq >= 4.*rsq) return 0.;
        if (tsq == 0.) return M_PI*rsq;

        /* in between we calculate half-height at intersection */
        double hsq = rsq - tsq/4.;
        if (hsq<0.) throw SBError("Airy calculation half-height invalid");
        double h = sqrt(hsq);

        return 2.*chord(r,h,rsq,hsq);
    }

    /* area of two intersecting identical annuli */
    double SBAiry::AiryInfo::annuli_intersect(
        double r1, double r2, double r1sq, double r2sq, double tsq) const 
    {
        assert(r1 >= r2);
        return circle_intersection(r1,r1sq,tsq)
            - 2. * circle_intersection(r1,r2,r1sq,r2sq,tsq)
            +  circle_intersection(r2,r2sq,tsq);
    }

    // Beam pattern of annular aperture, in k space, which is just the
    // autocorrelation of two annuli.
    // Unnormalized -- value at k=0 is Pi * (1-obs^2)
    double SBAiry::AiryInfo::kValue(double ksq_over_pisq) const 
    { return annuli_intersect(1.,_obscuration,1.,_obssq,ksq_over_pisq); }

    // Constructor to initialize Airy constants and k lookup table
    SBAiry::AiryInfo::AiryInfo(double obscuration, double obssq) : 
        _obscuration(obscuration), 
        _obssq(obssq),
        _radial(_obscuration,_obssq)
    {
        dbg<<"Initializing AiryInfo for obs = "<<obscuration<<", obssq = "<<obssq<<std::endl;
        // Calculate stepK:
        // Schroeder (10.1.18) gives limit of EE at large radius.
        // This stepK could probably be relaxed, it makes overly accurate FFTs.
        double R = 1. / (sbp::alias_threshold * 0.5 * M_PI * M_PI * (1.-_obscuration));
        // Use at least 5 lam/D
        R = std::max(R,5.);
        _stepk = M_PI / R;
    }

    boost::shared_ptr<PhotonArray> SBAiry::SBAiryImpl::shoot(int N, UniformDeviate u) const
    {
        dbg<<"Airy shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        boost::shared_ptr<PhotonArray> result=_info->shoot(N, u);
        // Then rescale for this flux & size
        result->scaleFlux(_flux);
        result->scaleXY(1./_D);
        dbg<<"Airy Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> SBAiry::AiryInfo::shoot(int N, UniformDeviate u) const
    {
        // Use the OneDimensionalDeviate to sample from scale-free distribution
        checkSampler();
        assert(_sampler.get());
        return _sampler->shoot(N, u);
    }

    void SBAiry::AiryInfo::checkSampler() const 
    {
        if (_sampler.get()) return;
        std::vector<double> ranges(1,0.);
        // Break Airy function into ranges that will not have >1 extremum:
        double rmin = 1.1 - 0.5*_obscuration;
        // Use Schroeder (10.1.18) limit of EE at large radius.
        // to stop sampler at radius with EE>(1-shoot_flux_accuracy)
        double rmax = 2./(sbp::shoot_flux_accuracy * M_PI*M_PI * (1.-_obscuration));
        dbg<<"Airy sampler\n";
        dbg<<"obsc = "<<_obscuration<<std::endl;
        dbg<<"rmin = "<<rmin<<std::endl;
        dbg<<"rmax = "<<rmax<<std::endl;
        ranges.reserve(int(floor((rmax-rmin+2)/0.5+0.5)));
        for(double r=rmin; r<=rmax; r+=0.5) ranges.push_back(r);
        _sampler.reset(new OneDimensionalDeviate(_radial, ranges, true));
    }
}
